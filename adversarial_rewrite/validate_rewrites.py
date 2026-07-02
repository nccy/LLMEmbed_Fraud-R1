# -*- coding: utf-8 -*-
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rewrite_with_llm import ROUND_FIELDS, basic_check, read_json, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate adversarial rewrite JSON files and write a summary."
    )
    parser.add_argument("rewrite_json", nargs="+")
    parser.add_argument(
        "--summary_json",
        default="adversarial_data/validated/rewrite_validation_summary.json",
    )
    parser.add_argument(
        "--summary_md",
        default="adversarial_data/validated/rewrite_validation_summary.md",
    )
    return parser.parse_args()


def dialogue_line_count(text):
    return sum(
        1
        for line in str(text).splitlines()
        if line.strip().startswith(("left:", "right:"))
    )


def validate_file(path):
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"rewrite file must contain a list: {path}")

    seen_indices = set()
    reason_counter = Counter()
    round_lengths = defaultdict(list)
    round_line_counts = defaultdict(list)
    valid_count = 0
    recomputed_valid_count = 0
    binary_label_counter = Counter()
    multi_label_counter = Counter()

    for record in records:
        index = record.get("index")
        if index in seen_indices:
            reason_counter["duplicate_index"] += 1
        seen_indices.add(index)

        binary_label_counter[int(record.get("binary_label", -1))] += 1
        if "multi_label" in record:
            multi_label_counter[int(record["multi_label"])] += 1

        if record.get("is_valid") is True:
            valid_count += 1

        rewrites = {field: str(record.get(field, "")).strip() for field in ROUND_FIELDS}
        reasons = basic_check(str(record.get("round_0_original", "")), rewrites)
        if int(record.get("binary_label", -1)) != 1:
            reasons.append("binary_label_not_1")
        if not reasons:
            recomputed_valid_count += 1
        for reason in reasons:
            reason_counter[reason] += 1

        for field in ROUND_FIELDS:
            value = rewrites[field]
            round_lengths[field].append(len(value))
            round_line_counts[field].append(dialogue_line_count(value))

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "path": str(path),
        "total": len(records),
        "valid_flag_count": valid_count,
        "recomputed_valid_count": recomputed_valid_count,
        "invalid_flag_count": len(records) - valid_count,
        "recomputed_invalid_count": len(records) - recomputed_valid_count,
        "unique_indices": len(seen_indices),
        "binary_label_counts": dict(sorted(binary_label_counter.items())),
        "multi_label_counts": dict(sorted(multi_label_counter.items())),
        "reason_counts": dict(reason_counter.most_common()),
        "round_stats": {
            field: {
                "avg_chars": avg(round_lengths[field]),
                "min_chars": min(round_lengths[field]) if round_lengths[field] else 0,
                "max_chars": max(round_lengths[field]) if round_lengths[field] else 0,
                "avg_dialogue_lines": avg(round_line_counts[field]),
            }
            for field in ROUND_FIELDS
        },
    }


def write_markdown(path, summaries):
    lines = [
        "# Rewrite Validation Summary",
        "",
        "| File | Total | Valid flag | Recomputed valid | Unique indices | Invalid reasons |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        reasons = ", ".join(
            f"{key}={value}" for key, value in item["reason_counts"].items()
        )
        lines.append(
            f"| `{item['path']}` | {item['total']} | "
            f"{item['valid_flag_count']} | {item['recomputed_valid_count']} | "
            f"{item['unique_indices']} | {reasons or '-'} |"
        )

    lines.extend(["", "## Round Lengths", ""])
    for item in summaries:
        lines.append(f"### `{item['path']}`")
        lines.append("")
        lines.append("| Round | Avg chars | Min | Max | Avg dialogue lines |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for field, stats in item["round_stats"].items():
            lines.append(
                f"| `{field}` | {stats['avg_chars']:.1f} | "
                f"{stats['min_chars']} | {stats['max_chars']} | "
                f"{stats['avg_dialogue_lines']:.1f} |"
            )
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    summaries = [validate_file(path) for path in args.rewrite_json]
    write_json(args.summary_json, summaries)
    write_markdown(args.summary_md, summaries)
    for item in summaries:
        print(
            f"{item['path']}: total={item['total']} "
            f"valid={item['recomputed_valid_count']} "
            f"invalid={item['recomputed_invalid_count']}"
        )
    print(f"summary_json: {args.summary_json}")
    print(f"summary_md: {args.summary_md}")


if __name__ == "__main__":
    main()
