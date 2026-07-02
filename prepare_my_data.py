import argparse
import csv
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


DEFAULT_TRAIN_CSV = "raw_data/训练集结果.csv"
DEFAULT_TEST_CSV = "raw_data/测试集结果.csv"
DEFAULT_OUTPUT_ROOT = "dataset"
TEXT_COLUMN = "specific_dialogue_content"
FRAUD_COLUMN = "is_fraud"
TYPE_COLUMN = "fraud_type"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build fraud_binary and fraud_multi Hugging Face datasets from the "
            "course fraud-call CSV files."
        )
    )
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mapping_output",
        default=None,
        help="Optional path for the fraud_type to label-id mapping JSON.",
    )
    return parser.parse_args()


def is_true(value):
    return str(value).strip().upper() == "TRUE" or value is True


def read_csv_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        required = {TEXT_COLUMN, FRAUD_COLUMN, TYPE_COLUMN}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = []
        for row in reader:
            text = (row.get(TEXT_COLUMN) or "").strip()
            if not text:
                continue
            rows.append(row)
        return rows


def build_binary_rows(rows):
    return [
        {
            "text": row[TEXT_COLUMN],
            "label": 1 if is_true(row.get(FRAUD_COLUMN)) else 0,
        }
        for row in rows
    ]


def build_multi_source(rows):
    multi_rows = []
    for row in rows:
        fraud_type = (row.get(TYPE_COLUMN) or "").strip()
        if is_true(row.get(FRAUD_COLUMN)) and fraud_type:
            multi_rows.append(
                {
                    "text": row[TEXT_COLUMN],
                    TYPE_COLUMN: fraud_type,
                }
            )
    return multi_rows


def build_label_mapping(train_multi, test_multi):
    type_to_id = {}
    for row in train_multi + test_multi:
        fraud_type = row[TYPE_COLUMN]
        if fraud_type not in type_to_id:
            type_to_id[fraud_type] = len(type_to_id)
    return type_to_id


def apply_multi_labels(rows, type_to_id):
    return [
        {
            "text": row["text"],
            "label": type_to_id[row[TYPE_COLUMN]],
        }
        for row in rows
    ]


def save_dataset(dataset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def main():
    args = parse_args()
    output_root = Path(args.output_root)

    print("Reading source CSV files...")
    train_rows = read_csv_rows(args.train_csv)
    test_rows = read_csv_rows(args.test_csv)

    print("Building fraud_binary dataset...")
    binary_dataset = DatasetDict(
        {
            "train": Dataset.from_list(build_binary_rows(train_rows)),
            "test": Dataset.from_list(build_binary_rows(test_rows)),
        }
    )
    binary_path = output_root / "fraud_binary"
    save_dataset(binary_dataset, binary_path)

    print("Building fraud_multi dataset...")
    train_multi_source = build_multi_source(train_rows)
    test_multi_source = build_multi_source(test_rows)
    type_to_id = build_label_mapping(train_multi_source, test_multi_source)

    multi_dataset = DatasetDict(
        {
            "train": Dataset.from_list(apply_multi_labels(train_multi_source, type_to_id)),
            "test": Dataset.from_list(apply_multi_labels(test_multi_source, type_to_id)),
        }
    )
    multi_path = output_root / "fraud_multi"
    save_dataset(multi_dataset, multi_path)

    mapping_output = Path(args.mapping_output) if args.mapping_output else output_root / "fraud_multi_label_mapping.json"
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with mapping_output.open("w", encoding="utf-8") as f:
        json.dump(type_to_id, f, ensure_ascii=False, indent=2)

    binary_test_positive = sum(int(label) == 1 for label in binary_dataset["test"]["label"])
    if binary_test_positive != len(multi_dataset["test"]):
        raise ValueError(
            "fraud_multi/test is not aligned with positive fraud_binary/test rows: "
            f"binary positives={binary_test_positive}, multi rows={len(multi_dataset['test'])}"
        )

    print(f"Saved binary dataset: {binary_path}")
    print(f"Saved multi dataset: {multi_path}")
    print(f"Saved label mapping: {mapping_output}")
    print(f"fraud_binary train/test: {len(binary_dataset['train'])}/{len(binary_dataset['test'])}")
    print(f"fraud_multi train/test: {len(multi_dataset['train'])}/{len(multi_dataset['test'])}")
    print(f"fraud_multi classes: {len(type_to_id)}")


if __name__ == "__main__":
    main()