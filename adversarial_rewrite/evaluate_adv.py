# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DownstreamModel import DownstreamModel


ALLOWED_CUDA_IDS = {0, 1, 2}
ENCODER_ROOTS = {
    "llama2": "llama2_embedding",
    "bert": "bert_embedding",
    "roberta": "roberta_embedding",
}
DEFAULT_MODELS = ("qwen3.5-9b", "llama3.1-8b")
DEFAULT_ROUNDS = ("r1_trust", "r2_urgency", "r3_emotion")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLMEmbed checkpoints on adversarial rewrite tensors."
    )
    parser.add_argument(
        "--binary_checkpoint",
        default="checkpoints/fraud_binary_20260621_170229.pt",
    )
    parser.add_argument(
        "--multi_checkpoint",
        default="checkpoints/fraud_multi_20260621_170229.pt",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--rounds", nargs="+", default=list(DEFAULT_ROUNDS))
    parser.add_argument("--cuda_no", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="adversarial_data/evaluation")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def validate_cuda_id(cuda_no):
    if cuda_no not in ALLOWED_CUDA_IDS:
        allowed = ", ".join(str(item) for item in sorted(ALLOWED_CUDA_IDS))
        raise ValueError(f"GPU id {cuda_no} is not allowed; use one of: {allowed}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_tensor_task(task, split):
    paths = {
        encoder: Path(root) / task / "dataset_tensor"
        for encoder, root in ENCODER_ROOTS.items()
    }
    llama = torch.load(paths["llama2"] / f"{split}_sents.pt", map_location="cpu")
    bert = torch.load(paths["bert"] / f"{split}_sents.pt", map_location="cpu")
    roberta = torch.load(paths["roberta"] / f"{split}_sents.pt", map_location="cpu")
    labels = torch.load(paths["llama2"] / f"{split}_labels.pt", map_location="cpu")
    metadata_path = paths["llama2"] / f"{split}_metadata.json"
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    count = int(labels.shape[0])
    for name, tensor in [("llama2", llama), ("bert", bert), ("roberta", roberta)]:
        if int(tensor.shape[0]) != count:
            raise ValueError(
                f"{task}: {name} count {tensor.shape[0]} != labels {count}"
            )
    return llama, bert, roberta, labels.long(), metadata


def make_loader(task, split, batch_size, seed):
    llama, bert, roberta, labels, metadata = load_tensor_task(task, split)
    dataset = TensorDataset(llama, bert, roberta, labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, labels, metadata


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    seed = checkpoint.get("seed", 42)
    set_seed(seed)
    model = DownstreamModel(checkpoint["class_num"], checkpoint["SIGMA"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch_size = checkpoint.get("batch_size", 1024)
    return checkpoint, model, seed, batch_size


def predict(loader, device, model, loss_fn):
    total_loss = 0.0
    total_pred, total_y, total_prob = [], [], []
    batch_count = 0
    started = time.perf_counter()
    for batch in tqdm(loader, desc="eval"):
        batch_l, batch_b, batch_r, batch_y = batch
        batch_l = batch_l.to(device)
        batch_b = batch_b.to(device)
        batch_r = batch_r.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            pred = model(batch_l.float(), batch_b.float(), batch_r.float())
            loss = loss_fn(pred, batch_y)
        total_loss += float(loss.detach().cpu())
        total_prob.append(pred.detach().cpu())
        total_pred.append(torch.max(pred, 1).indices.detach().cpu())
        total_y.append(batch_y.detach().cpu())
        batch_count += 1

    labels = torch.cat(total_y)
    preds = torch.cat(total_pred)
    probs = torch.cat(total_prob)
    return {
        "avg_loss": total_loss / max(batch_count, 1),
        "labels": labels,
        "preds": preds,
        "probs": probs,
        "runtime_seconds": time.perf_counter() - started,
    }


def binary_metrics(labels, preds):
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    positive_mask = labels_np == 1
    false_negative = int(np.sum((labels_np == 1) & (preds_np == 0)))
    positive_count = int(np.sum(positive_mask))
    pred_fraud = int(np.sum(preds_np == 1))
    return {
        "sample_count": int(labels_np.shape[0]),
        "label_counts": dict(sorted(Counter(labels_np.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds_np.tolist()).items())),
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
        "positive_count": positive_count,
        "predicted_fraud": pred_fraud,
        "false_negative": false_negative,
        "false_negative_rate": false_negative / positive_count if positive_count else 0.0,
        "positive_recall": recall_score(labels_np, preds_np, zero_division=0),
    }


def binary_attack_metrics(labels, baseline_preds, adv_preds):
    labels_np = labels.numpy()
    base_np = baseline_preds.numpy()
    adv_np = adv_preds.numpy()
    eligible = (labels_np == 1) & (base_np == 1)
    success = eligible & (adv_np == 0)
    eligible_count = int(np.sum(eligible))
    success_count = int(np.sum(success))
    return {
        "asr_denominator": eligible_count,
        "attack_success": success_count,
        "asr": success_count / eligible_count if eligible_count else 0.0,
    }


def per_class_accuracy(labels, preds):
    result = {}
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    for klass in sorted(set(labels_np.tolist())):
        mask = labels_np == klass
        result[str(int(klass))] = {
            "count": int(np.sum(mask)),
            "accuracy": float(np.mean(preds_np[mask] == labels_np[mask])),
        }
    return result


def multi_metrics(labels, preds, class_num):
    labels_np = labels.numpy()
    preds_np = preds.numpy()
    return {
        "sample_count": int(labels_np.shape[0]),
        "label_counts": dict(sorted(Counter(labels_np.tolist()).items())),
        "prediction_counts": dict(sorted(Counter(preds_np.tolist()).items())),
        "accuracy": accuracy_score(labels_np, preds_np),
        "micro_f1": f1_score(labels_np, preds_np, average="micro"),
        "macro_f1": f1_score(labels_np, preds_np, average="macro", zero_division=0),
        "macro_precision": precision_score(
            labels_np, preds_np, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            labels_np, preds_np, average="macro", zero_division=0
        ),
        "per_class_accuracy": per_class_accuracy(labels, preds),
        "confusion_matrix": confusion_matrix(
            labels_np, preds_np, labels=list(range(class_num))
        ).tolist(),
    }


def add_per_class_drop(current, baseline):
    output = dict(current)
    drops = {}
    for klass, item in current["per_class_accuracy"].items():
        base_acc = baseline["per_class_accuracy"].get(klass, {}).get("accuracy")
        if base_acc is None:
            continue
        drops[klass] = {
            "baseline_accuracy": base_acc,
            "current_accuracy": item["accuracy"],
            "drop": base_acc - item["accuracy"],
        }
    output["per_class_drop_vs_baseline"] = drops
    return output


def save_prediction_records(path, labels, preds, probs, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, (label, pred, prob) in enumerate(
        zip(labels.tolist(), preds.tolist(), probs.tolist())
    ):
        row = {
            "index": idx,
            "label": int(label),
            "prediction": int(pred),
            "probabilities": prob,
        }
        if metadata is not None and idx < len(metadata):
            row["metadata"] = metadata[idx]
        rows.append(row)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def format_float(value):
    return f"{value:.4f}"


def write_markdown(path, summary):
    lines = ["# Adversarial Evaluation Results", ""]
    lines.append(f"- Started: `{summary['started_at']}`")
    lines.append(f"- Finished: `{summary['finished_at']}`")
    lines.append(f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`")
    lines.append("")

    binary_base = summary["binary_baseline"]
    lines.extend(
        [
            "## Binary Baseline",
            "",
            "| Task | Samples | Accuracy | Precision | Recall | F1 | FNR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| `fraud_binary` | {binary_base['sample_count']} | "
                f"{format_float(binary_base['accuracy'])} | "
                f"{format_float(binary_base['precision'])} | "
                f"{format_float(binary_base['recall'])} | "
                f"{format_float(binary_base['f1'])} | "
                f"{format_float(binary_base['false_negative_rate'])} |"
            ),
            "",
            "## Binary Adversarial",
            "",
            "| Rewrite model | Round | Samples | Accuracy | Precision | Recall | F1 | FNR | ASR | Attack success |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["binary_adv"]:
        m = item["metrics"]
        a = item["attack"]
        lines.append(
            f"| `{item['rewrite_model']}` | `{item['round']}` | {m['sample_count']} | "
            f"{format_float(m['accuracy'])} | {format_float(m['precision'])} | "
            f"{format_float(m['recall'])} | {format_float(m['f1'])} | "
            f"{format_float(m['false_negative_rate'])} | {format_float(a['asr'])} | "
            f"{a['attack_success']}/{a['asr_denominator']} |"
        )

    multi_base = summary["multi_baseline"]
    lines.extend(
        [
            "",
            "## Multi-Class Baseline",
            "",
            "| Task | Samples | Accuracy | micro-F1 | macro-F1 | macro-Precision | macro-Recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| `fraud_multi` | {multi_base['sample_count']} | "
                f"{format_float(multi_base['accuracy'])} | "
                f"{format_float(multi_base['micro_f1'])} | "
                f"{format_float(multi_base['macro_f1'])} | "
                f"{format_float(multi_base['macro_precision'])} | "
                f"{format_float(multi_base['macro_recall'])} |"
            ),
            "",
            "## Multi-Class Adversarial",
            "",
            "| Rewrite model | Round | Samples | Accuracy | micro-F1 | macro-F1 | macro-Precision | macro-Recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["multi_adv"]:
        m = item["metrics"]
        lines.append(
            f"| `{item['rewrite_model']}` | `{item['round']}` | {m['sample_count']} | "
            f"{format_float(m['accuracy'])} | {format_float(m['micro_f1'])} | "
            f"{format_float(m['macro_f1'])} | {format_float(m['macro_precision'])} | "
            f"{format_float(m['macro_recall'])} |"
        )

    lines.extend(["", "## Runtime", "", "| Stage | Seconds | Note |", "| --- | ---: | --- |"])
    for item in summary.get("runtime_notes", []):
        seconds = item.get("seconds")
        seconds_text = f"{seconds:.2f}" if isinstance(seconds, (int, float)) else "-"
        lines.append(f"| {item['stage']} | {seconds_text} | {item['note']} |")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_runtime_notes(path):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    args = parse_args()
    validate_cuda_id(args.cuda_no)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    device = f"cuda:{args.cuda_no}"
    loss_fn = nn.CrossEntropyLoss().to(device)

    binary_ckpt, binary_model, binary_seed, binary_batch = load_model(
        args.binary_checkpoint, device
    )
    binary_batch_size = args.batch_size or binary_batch
    loader, binary_labels, binary_metadata = make_loader(
        "fraud_binary", args.split, binary_batch_size, binary_seed
    )
    binary_base_pred = predict(loader, device, binary_model, loss_fn)
    binary_base_metrics = binary_metrics(
        binary_base_pred["labels"], binary_base_pred["preds"]
    )
    binary_base_metrics["avg_loss"] = binary_base_pred["avg_loss"]
    binary_base_metrics["runtime_seconds"] = binary_base_pred["runtime_seconds"]

    if args.save_predictions:
        save_prediction_records(
            output_dir / "predictions" / "fraud_binary_baseline.json",
            binary_base_pred["labels"],
            binary_base_pred["preds"],
            binary_base_pred["probs"],
            binary_metadata,
        )

    binary_adv = []
    for model_slug in args.models:
        for round_slug in args.rounds:
            task = f"fraud_binary_adv_{model_slug}_{round_slug}"
            loader, labels, metadata = make_loader(
                task, args.split, binary_batch_size, binary_seed
            )
            pred = predict(loader, device, binary_model, loss_fn)
            metrics = binary_metrics(pred["labels"], pred["preds"])
            metrics["avg_loss"] = pred["avg_loss"]
            metrics["runtime_seconds"] = pred["runtime_seconds"]
            attack = binary_attack_metrics(
                pred["labels"], binary_base_pred["preds"], pred["preds"]
            )
            row = {
                "rewrite_model": model_slug,
                "round": round_slug,
                "task": task,
                "metrics": metrics,
                "attack": attack,
            }
            binary_adv.append(row)
            if args.save_predictions:
                save_prediction_records(
                    output_dir / "predictions" / f"{task}.json",
                    pred["labels"],
                    pred["preds"],
                    pred["probs"],
                    metadata,
                )

    del binary_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multi_ckpt, multi_model, multi_seed, multi_batch = load_model(
        args.multi_checkpoint, device
    )
    multi_batch_size = args.batch_size or multi_batch
    loader, multi_labels, multi_metadata = make_loader(
        "fraud_multi", args.split, multi_batch_size, multi_seed
    )
    multi_base_pred = predict(loader, device, multi_model, loss_fn)
    multi_base_metrics = multi_metrics(
        multi_base_pred["labels"], multi_base_pred["preds"], multi_ckpt["class_num"]
    )
    multi_base_metrics["avg_loss"] = multi_base_pred["avg_loss"]
    multi_base_metrics["runtime_seconds"] = multi_base_pred["runtime_seconds"]

    if args.save_predictions:
        save_prediction_records(
            output_dir / "predictions" / "fraud_multi_baseline.json",
            multi_base_pred["labels"],
            multi_base_pred["preds"],
            multi_base_pred["probs"],
            multi_metadata,
        )

    multi_adv = []
    for model_slug in args.models:
        for round_slug in args.rounds:
            task = f"fraud_multi_adv_{model_slug}_{round_slug}"
            loader, labels, metadata = make_loader(
                task, args.split, multi_batch_size, multi_seed
            )
            pred = predict(loader, device, multi_model, loss_fn)
            metrics = multi_metrics(
                pred["labels"], pred["preds"], multi_ckpt["class_num"]
            )
            metrics = add_per_class_drop(metrics, multi_base_metrics)
            metrics["avg_loss"] = pred["avg_loss"]
            metrics["runtime_seconds"] = pred["runtime_seconds"]
            row = {
                "rewrite_model": model_slug,
                "round": round_slug,
                "task": task,
                "metrics": metrics,
            }
            multi_adv.append(row)
            if args.save_predictions:
                save_prediction_records(
                    output_dir / "predictions" / f"{task}.json",
                    pred["labels"],
                    pred["preds"],
                    pred["probs"],
                    metadata,
                )

    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": time.perf_counter() - started,
        "device": device,
        "binary_checkpoint": args.binary_checkpoint,
        "multi_checkpoint": args.multi_checkpoint,
        "models": args.models,
        "rounds": args.rounds,
        "binary_baseline": binary_base_metrics,
        "binary_adv": binary_adv,
        "multi_baseline": multi_base_metrics,
        "multi_adv": multi_adv,
    }

    json_path = output_dir / "adv_evaluation_summary.json"
    md_path = output_dir / "adv_evaluation_summary.md"
    json_path.write_text(
        json.dumps(to_builtin(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(md_path, summary)
    print(f"summary_json: {json_path}")
    print(f"summary_md: {md_path}")
    print(f"runtime_seconds: {summary['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
