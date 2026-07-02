# -*- coding: utf-8 -*-
import argparse
import json
from collections import Counter
from pathlib import Path

import torch


ENCODER_ROOTS = {
    "llama2": "llama2_embedding",
    "bert": "bert_embedding",
    "roberta": "roberta_embedding",
}

DEFAULT_ROUNDS = ("r1_trust", "r2_urgency", "r3_emotion")
DEFAULT_TASKS = ("fraud_binary", "fraud_multi")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify adversarial LLMEmbed tensor shape and label consistency."
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--rounds", nargs="+", default=list(DEFAULT_ROUNDS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--summary_json",
        default="adversarial_data/validated/adv_tensor_summary.json",
    )
    return parser.parse_args()


def load_tensor(path):
    return torch.load(path, map_location="cpu")


def verify_task(root, model_slug, round_slug, task, split):
    rows = {}
    expected_count = None
    label_counts = None
    errors = []

    for encoder, encoder_root in ENCODER_ROOTS.items():
        tensor_dir = (
            Path(root)
            / encoder_root
            / f"{task}_adv_{model_slug}_{round_slug}"
            / "dataset_tensor"
        )
        sent_path = tensor_dir / f"{split}_sents.pt"
        label_path = tensor_dir / f"{split}_labels.pt"
        metadata_path = tensor_dir / f"{split}_metadata.json"
        if not sent_path.exists() or not label_path.exists():
            errors.append(f"{encoder}:missing_tensor")
            continue

        sents = load_tensor(sent_path)
        labels = load_tensor(label_path)
        if sents.shape[0] != labels.shape[0]:
            errors.append(f"{encoder}:sent_label_count_mismatch")

        current_count = int(labels.shape[0])
        current_label_counts = dict(sorted(Counter(labels.tolist()).items()))
        if expected_count is None:
            expected_count = current_count
            label_counts = current_label_counts
        elif current_count != expected_count:
            errors.append(f"{encoder}:count_mismatch")
        if label_counts is not None and current_label_counts != label_counts:
            errors.append(f"{encoder}:label_distribution_mismatch")

        metadata_count = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_count = len(metadata)
            if metadata_count != current_count:
                errors.append(f"{encoder}:metadata_count_mismatch")

        rows[encoder] = {
            "sents_shape": list(sents.shape),
            "labels_shape": list(labels.shape),
            "label_counts": current_label_counts,
            "metadata_count": metadata_count,
        }

    return {
        "model": model_slug,
        "round": round_slug,
        "task": task,
        "ok": not errors and len(rows) == len(ENCODER_ROOTS),
        "errors": errors,
        "encoders": rows,
    }


def main():
    args = parse_args()
    summary = []
    for model_slug in args.models:
        for round_slug in args.rounds:
            for task in args.tasks:
                item = verify_task(args.root, model_slug, round_slug, task, args.split)
                summary.append(item)
                status = "ok" if item["ok"] else "failed"
                print(f"{status}: {model_slug} {round_slug} {task}")
                if item["errors"]:
                    print("  errors:", ", ".join(item["errors"]))

    output_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary_json: {output_path}")


if __name__ == "__main__":
    main()
