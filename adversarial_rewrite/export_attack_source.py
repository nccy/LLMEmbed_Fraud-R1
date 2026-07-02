import argparse
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


DEFAULT_BINARY_DATASET = os.environ.get("FRAUD_BINARY_DATASET", "dataset/fraud_binary")
DEFAULT_MULTI_DATASET = os.environ.get("FRAUD_MULTI_DATASET", "dataset/fraud_multi")
DEFAULT_OUTPUT = "adversarial_data/source/fraud_test_attack_source.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export fraud samples from fraud_binary/test and align them with "
            "fraud_multi/test labels for adversarial rewrite experiments."
        )
    )
    parser.add_argument("--binary_dataset", default=DEFAULT_BINARY_DATASET)
    parser.add_argument("--multi_dataset", default=DEFAULT_MULTI_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fraud_label", type=int, default=1)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow_text_mismatch",
        action="store_true",
        help="Export even if aligned binary/multi texts are not exactly identical.",
    )
    return parser.parse_args()


def load_split(dataset_path, split):
    dataset = load_from_disk(dataset_path)
    if split not in dataset:
        raise ValueError(f"Split {split!r} not found in {dataset_path}")
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataset[split].column_names)
    if missing_columns:
        raise ValueError(
            f"Split {split!r} in {dataset_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )
    return dataset[split]


def build_records(binary_split, multi_split, split_name, fraud_label, allow_text_mismatch):
    binary_fraud_rows = [
        (idx, row)
        for idx, row in enumerate(binary_split)
        if int(row["label"]) == fraud_label
    ]

    if len(binary_fraud_rows) != len(multi_split):
        raise ValueError(
            "Cannot align attack source: "
            f"binary fraud rows={len(binary_fraud_rows)}, multi rows={len(multi_split)}"
        )

    records = []
    mismatches = []
    for attack_idx, ((binary_idx, binary_row), multi_row) in enumerate(
        zip(binary_fraud_rows, multi_split)
    ):
        binary_text = binary_row["text"]
        multi_text = multi_row["text"]
        if binary_text != multi_text:
            mismatches.append(
                {
                    "index": attack_idx,
                    "binary_index": binary_idx,
                    "multi_index": attack_idx,
                    "binary_text_prefix": binary_text[:120],
                    "multi_text_prefix": multi_text[:120],
                }
            )

        records.append(
            {
                "index": attack_idx,
                "split": split_name,
                "binary_index": binary_idx,
                "multi_index": attack_idx,
                "binary_label": int(binary_row["label"]),
                "multi_label": int(multi_row["label"]),
                "round_0_original": binary_text,
            }
        )

    if mismatches and not allow_text_mismatch:
        preview = json.dumps(mismatches[:3], ensure_ascii=False, indent=2)
        raise ValueError(
            f"Found {len(mismatches)} text mismatches between binary and multi data. "
            f"First mismatches:\n{preview}"
        )

    return records, mismatches


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    binary_split = load_split(args.binary_dataset, args.split)
    multi_split = load_split(args.multi_dataset, args.split)

    records, mismatches = build_records(
        binary_split=binary_split,
        multi_split=multi_split,
        split_name=args.split,
        fraud_label=args.fraud_label,
        allow_text_mismatch=args.allow_text_mismatch,
    )

    output_path = Path(args.output)
    write_json(output_path, records)

    label_counts = Counter(record["multi_label"] for record in records)
    summary = {
        "binary_dataset": args.binary_dataset,
        "multi_dataset": args.multi_dataset,
        "split": args.split,
        "fraud_label": args.fraud_label,
        "num_binary_rows": len(binary_split),
        "num_multi_rows": len(multi_split),
        "num_exported_records": len(records),
        "num_text_mismatches": len(mismatches),
        "multi_label_counts": dict(sorted(label_counts.items())),
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print(f"exported records: {len(records)}")
    print(f"text mismatches: {len(mismatches)}")
    print(f"multi label counts: {dict(sorted(label_counts.items()))}")
    print(f"output: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
