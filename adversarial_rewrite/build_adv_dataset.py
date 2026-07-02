# -*- coding: utf-8 -*-
import argparse
import gc
import json
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import trange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
)


ROUND_FIELDS = (
    "round_1_trust",
    "round_2_urgency",
    "round_3_emotion",
)

ALLOWED_CUDA_IDS = {0, 1, 2}
DEFAULT_BINARY_DATASET_PATH = os.environ.get("FRAUD_BINARY_DATASET", "dataset/fraud_binary")

ROUND_SLUGS = {
    "round_0_original": "r0_original",
    "round_1_trust": "r1_trust",
    "round_2_urgency": "r2_urgency",
    "round_3_emotion": "r3_emotion",
}

ENCODER_ROOTS = {
    "llama2": "llama2_embedding",
    "bert": "bert_embedding",
    "roberta": "roberta_embedding",
}

MODEL_PATHS = {
    "llama2": os.environ.get("LLAMA2_MODEL_PATH", "hf_model/llama2-7b-chat-hf"),
    "bert": os.environ.get("BERT_MODEL_PATH", "hf_model/bert-large-uncased"),
    "roberta": os.environ.get("ROBERTA_MODEL_PATH", "hf_model/roberta-large"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build LLMEmbed adversarial dataset tensors from validated rewrite JSON. "
            "Binary adversarial tasks keep the complete original fraud_binary split: "
            "label=0 samples stay unchanged and label=1 samples are replaced by "
            "the selected rewrite round. Multi-class tasks keep the rewritten fraud "
            "subset with its original fraud type labels."
        )
    )
    parser.add_argument("--rewrite_json", required=True)
    parser.add_argument("--rewrite_model_slug", default=None)
    parser.add_argument(
        "--encoder",
        default="all",
        choices=["all", "llama2", "bert", "roberta"],
    )
    parser.add_argument("--cuda_no", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device, for example cuda:0 or cpu. Defaults to cuda:{cuda_no}.",
    )
    parser.add_argument("--rounds", nargs="+", default=list(ROUND_FIELDS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--binary_dataset_path", default=DEFAULT_BINARY_DATASET_PATH)
    parser.add_argument("--bert_batch_size", type=int, default=512)
    parser.add_argument("--roberta_batch_size", type=int, default=512)
    parser.add_argument("--llama2_batch_size", type=int, default=8)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--allow_invalid", action="store_true")
    parser.add_argument(
        "--binary_positive_only",
        action="store_true",
        help=(
            "Legacy mode: save only rewritten positive fraud samples for the binary "
            "task. Default keeps the full binary split including label=0 samples."
        ),
    )
    return parser.parse_args()


def validate_cuda_id(cuda_no):
    if cuda_no not in ALLOWED_CUDA_IDS:
        allowed = ", ".join(str(item) for item in sorted(ALLOWED_CUDA_IDS))
        raise ValueError(f"GPU id {cuda_no} is not allowed; use one of: {allowed}")


def resolve_device(args):
    if args.device:
        if args.device.startswith("cuda:"):
            validate_cuda_id(int(args.device.split(":", 1)[1]))
        elif args.device == "cuda":
            validate_cuda_id(0)
        return args.device
    validate_cuda_id(args.cuda_no)
    return f"cuda:{args.cuda_no}"


def safe_model_slug(value):
    safe = value.replace(":", "-").replace("/", "-").replace("\\", "-")
    safe = safe.replace(" ", "_")
    return safe


def infer_model_slug(path, records):
    for record in records:
        model = record.get("rewrite_model")
        if model:
            return safe_model_slug(model)
    name = Path(path).name
    suffix = "_fraud_test_rewrites.json"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return safe_model_slug(Path(name).stem)


def load_rewrite_records(path, allow_invalid, max_items):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("rewrite_json must contain a JSON list")

    selected = []
    for record in records:
        if allow_invalid or record.get("is_valid") is True:
            selected.append(record)
        if max_items is not None and len(selected) >= max_items:
            break
    if not selected:
        raise ValueError("no rewrite records selected")
    return selected


def load_binary_split(dataset_path, split):
    dataset = load_from_disk(dataset_path)
    if split not in dataset:
        raise ValueError(f"split not found in {dataset_path}: {split}")
    split_data = dataset[split]
    return list(split_data["text"]), [int(label) for label in split_data["label"]]


def build_round_task_inputs(records, round_field, binary_texts, binary_labels, positive_only):
    binary_adv_texts = list(binary_texts)
    binary_metadata = [
        {
            "output_index": idx,
            "binary_index": idx,
            "source_index": None,
            "multi_index": None,
            "round": round_field,
            "rewrite_model": None,
            "source": "fraud_binary_original",
            "was_rewritten": False,
        }
        for idx in range(len(binary_texts))
    ]

    multi_texts = []
    multi_labels = []
    multi_metadata = []
    positive_binary_indices = []
    seen_binary_indices = set()

    for output_index, record in enumerate(records):
        text = str(record.get(round_field, "")).strip()
        if not text:
            raise ValueError(
                f"empty text for index={record.get('index')} round={round_field}"
            )

        binary_index = int(record.get("binary_index", record.get("index")))
        if binary_index < 0 or binary_index >= len(binary_texts):
            raise ValueError(
                f"binary_index out of range: {binary_index} for round={round_field}"
            )
        if int(binary_labels[binary_index]) != 1:
            raise ValueError(
                f"binary_index={binary_index} is not a positive fraud sample"
            )
        if binary_index in seen_binary_indices:
            raise ValueError(f"duplicate binary_index in rewrite JSON: {binary_index}")
        seen_binary_indices.add(binary_index)

        binary_adv_texts[binary_index] = text
        positive_binary_indices.append(binary_index)
        rewrite_meta = {
            "output_index": output_index,
            "source_index": record.get("index"),
            "binary_index": binary_index,
            "multi_index": record.get("multi_index"),
            "round": round_field,
            "rewrite_model": record.get("rewrite_model"),
            "source": "rewrite_json",
            "was_rewritten": True,
        }
        binary_metadata[binary_index] = rewrite_meta
        multi_texts.append(text)
        multi_labels.append(int(record["multi_label"]))
        multi_metadata.append(rewrite_meta)

    expected_positive_count = sum(1 for label in binary_labels if int(label) == 1)
    if len(records) == expected_positive_count and len(seen_binary_indices) != expected_positive_count:
        raise ValueError(
            f"expected {expected_positive_count} unique positive samples, "
            f"got {len(seen_binary_indices)}"
        )

    if positive_only:
        binary_adv_texts = [binary_adv_texts[idx] for idx in positive_binary_indices]
        binary_task_labels = [1 for _ in positive_binary_indices]
        binary_metadata = [binary_metadata[idx] for idx in positive_binary_indices]
        multi_rep_indices = list(range(len(positive_binary_indices)))
    else:
        binary_task_labels = list(binary_labels)
        multi_rep_indices = positive_binary_indices

    return {
        "binary_texts": binary_adv_texts,
        "binary_labels": binary_task_labels,
        "binary_metadata": binary_metadata,
        "multi_texts": multi_texts,
        "multi_labels": multi_labels,
        "multi_metadata": multi_metadata,
        "multi_rep_indices": multi_rep_indices,
    }


def collect_round(records, round_field):
    texts = []
    binary_labels = []
    multi_labels = []
    metadata = []
    for output_index, record in enumerate(records):
        text = str(record.get(round_field, "")).strip()
        if not text:
            raise ValueError(
                f"empty text for index={record.get('index')} round={round_field}"
            )
        texts.append(text)
        binary_labels.append(int(record.get("binary_label", 1)))
        multi_labels.append(int(record["multi_label"]))
        metadata.append(
            {
                "output_index": output_index,
                "source_index": record.get("index"),
                "binary_index": record.get("binary_index"),
                "multi_index": record.get("multi_index"),
                "round": round_field,
                "rewrite_model": record.get("rewrite_model"),
            }
        )
    return texts, binary_labels, multi_labels, metadata


def extract_bert(texts, device, batch_size):
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS["bert"])
    model = BertModel.from_pretrained(MODEL_PATHS["bert"]).to(device)
    model.eval()

    reps = []
    for start in trange(0, len(texts), batch_size, desc="bert"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            reps.append(model(**encoded).pooler_output.cpu())
    output = torch.cat(reps)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def extract_roberta(texts, device, batch_size):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATHS["roberta"])
    model = RobertaModel.from_pretrained(MODEL_PATHS["roberta"]).to(device)
    model.eval()

    reps = []
    for start in trange(0, len(texts), batch_size, desc="roberta"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            reps.append(outputs.last_hidden_state[:, 0, :].cpu())
    output = torch.cat(reps)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def extract_llama2(texts, device, batch_size):
    model_path = MODEL_PATHS["llama2"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
        "output_hidden_states": True,
    }
    model_config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        config=model_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.eval()

    reps = []
    for start in trange(0, len(texts), batch_size, desc="llama2"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            max_length=256,
            padding="max_length",
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            layer_reps = []
            for layer in range(-1, -6, -1):
                layer_reps.append(torch.mean(outputs.hidden_states[layer], axis=1))
            reps.append(torch.stack(layer_reps, axis=1).cpu())
    output = torch.cat(reps)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def save_tensor_set(
    output_root,
    encoder,
    task,
    split,
    round_slug,
    model_slug,
    sentence_reps,
    labels,
    metadata,
):
    task_name = f"{task}_adv_{model_slug}_{round_slug}"
    tensor_dir = (
        Path(output_root)
        / ENCODER_ROOTS[encoder]
        / task_name
        / "dataset_tensor"
    )
    tensor_dir.mkdir(parents=True, exist_ok=True)
    torch.save(sentence_reps.cpu(), tensor_dir / f"{split}_sents.pt")
    torch.save(torch.tensor(labels, dtype=torch.long), tensor_dir / f"{split}_labels.pt")
    with open(tensor_dir / f"{split}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(
        f"saved {encoder} {task_name}: "
        f"{tuple(sentence_reps.shape)} labels={len(labels)}"
    )


def extract_with_encoder(encoder, texts, device, args):
    if encoder == "bert":
        return extract_bert(texts, device, args.bert_batch_size)
    if encoder == "roberta":
        return extract_roberta(texts, device, args.roberta_batch_size)
    if encoder == "llama2":
        return extract_llama2(texts, device, args.llama2_batch_size)
    raise ValueError(f"unsupported encoder: {encoder}")


def main():
    args = parse_args()
    unknown_rounds = [field for field in args.rounds if field not in ROUND_SLUGS]
    if unknown_rounds:
        raise ValueError(f"unsupported rounds: {unknown_rounds}")

    device = resolve_device(args)
    records = load_rewrite_records(args.rewrite_json, args.allow_invalid, args.max_items)
    binary_texts, binary_labels = load_binary_split(args.binary_dataset_path, args.split)
    model_slug = args.rewrite_model_slug or infer_model_slug(args.rewrite_json, records)
    encoders = list(ENCODER_ROOTS) if args.encoder == "all" else [args.encoder]

    print(f"rewrite_json: {args.rewrite_json}")
    print(f"rewrite_model_slug: {model_slug}")
    print(f"selected records: {len(records)}")
    print(f"binary split records: {len(binary_texts)}")
    print(f"binary_positive_only: {args.binary_positive_only}")
    print(f"rounds: {', '.join(args.rounds)}")
    print(f"encoders: {', '.join(encoders)}")

    for round_field in args.rounds:
        task_inputs = build_round_task_inputs(
            records,
            round_field,
            binary_texts,
            binary_labels,
            args.binary_positive_only,
        )
        round_slug = ROUND_SLUGS[round_field]
        print(
            f"round {round_field}: "
            f"binary_texts={len(task_inputs['binary_texts'])} "
            f"multi_texts={len(task_inputs['multi_texts'])}"
        )

        for encoder in encoders:
            sentence_reps = extract_with_encoder(
                encoder, task_inputs["binary_texts"], device, args
            )
            save_tensor_set(
                args.output_root,
                encoder,
                "fraud_binary",
                args.split,
                round_slug,
                model_slug,
                sentence_reps,
                task_inputs["binary_labels"],
                task_inputs["binary_metadata"],
            )
            multi_reps = sentence_reps[task_inputs["multi_rep_indices"]]
            save_tensor_set(
                args.output_root,
                encoder,
                "fraud_multi",
                args.split,
                round_slug,
                model_slug,
                multi_reps,
                task_inputs["multi_labels"],
                task_inputs["multi_metadata"],
            )
            del sentence_reps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
