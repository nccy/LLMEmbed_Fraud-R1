# -*- coding: utf-8 -*-
"""Generate figures and compact tables for the course paper.

The script only reads finished experiment summaries. It does not train models
or recompute predictions.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROUND_LABELS = {
    "r1_trust": "R1 trust",
    "r2_urgency": "R2 urgency",
    "r3_emotion": "R3 emotion",
}


def pct(value):
    return value * 100


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_binary_false_negatives(summary, out_dir):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    rows = summary["binary_adv"]
    labels = [f"{r['rewrite_model']}\n{ROUND_LABELS[r['round']]}" for r in rows]
    values = [r["attack"]["attack_success"] for r in rows]
    colors = ["#2f6f9f" if v > 0 else "#9fb3c8" for v in values]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_ylabel("False negatives among rewritten fraud samples")
    ax.set_title("Binary fraud detection: missed fraud counts after rewrites")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, max(values + [2]) + 0.8)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    for i, (v, r) in enumerate(zip(values, rows)):
        text = f"{v}/{r['attack']['asr_denominator']}"
        ax.text(i, v + 0.05, text, ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "binary_false_negative_count.png", dpi=220)
    plt.close(fig)


def save_multi_confusion_matrix(summary, out_dir):
    matrix = np.array(summary["multi_baseline"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Baseline multi-class confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    labels = [str(i) for i in range(matrix.shape[0])]
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    max_value = matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > max_value * 0.55 else "#1f2933"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_baseline_confusion_matrix.png", dpi=220)
    plt.close(fig)


def save_multi_macro(summary, out_dir):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    baseline = summary["multi_baseline"]["macro_f1"]
    rows = summary["multi_adv"]
    labels = [f"{r['rewrite_model']}\n{ROUND_LABELS[r['round']]}" for r in rows]
    values = [r["metrics"]["macro_f1"] for r in rows]
    ax.plot(range(len(values)), values, marker="o", linewidth=1.8, color="#8a4f2a")
    ax.axhline(baseline, color="#2f6f9f", linestyle="--", linewidth=1.6, label="Baseline")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Multi-class fraud type recognition is more sensitive")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.43, 0.55)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_macro_f1_under_rewrite.png", dpi=220)
    plt.close(fig)


def save_quality_vs_drop(summary, quality, out_dir):
    by_model = {}
    for row in summary["multi_adv"]:
        by_model.setdefault(row["rewrite_model"], []).append(row["metrics"]["macro_f1"])
    baseline = summary["multi_baseline"]["macro_f1"]
    drop_by_model = {
        model: pct(baseline - sum(values) / len(values))
        for model, values in by_model.items()
    }
    review_rates = {
        row["Model"].strip("`"): float(row["Needs review"].split("(")[-1].rstrip("%)"))
        for row in quality
    }

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    models = list(drop_by_model)
    x = [review_rates[m] for m in models]
    y = [drop_by_model[m] for m in models]
    ax.scatter(x, y, s=90, color="#426b49")
    for model, xi, yi in zip(models, x, y):
        ax.annotate(model, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Needs-review record rate (%)")
    ax.set_ylabel("Average macro-F1 drop (percentage points)")
    ax.set_title("Rewrite quality affects robustness interpretation")
    ax.grid(linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_macro_f1_drop.png", dpi=220)
    plt.close(fig)


def extract_quality_overall(markdown_path):
    lines = Path(markdown_path).read_text(encoding="utf-8").splitlines()
    start = lines.index("| Model | Items | Pass | Warning | Needs review | Top flags |")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(
            {
                "Model": cells[0],
                "Items": cells[1],
                "Pass": cells[2],
                "Warning": cells[3],
                "Needs review": cells[4],
                "Top flags": cells[5],
            }
        )
    return rows


def write_tables(summary, quality_rows, out_dir):
    baseline_b = summary["binary_baseline"]
    baseline_m = summary["multi_baseline"]
    rows = []
    rows.append("# Derived Experiment Tables\n")
    rows.append("## Baseline\n")
    rows.append("| Task | Samples | Accuracy | Precision | Recall | F1/micro-F1 | macro-F1 |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows.append(
        f"| fraud_binary | {baseline_b['sample_count']} | {baseline_b['accuracy']:.4f} | "
        f"{baseline_b['precision']:.4f} | {baseline_b['recall']:.4f} | {baseline_b['f1']:.4f} | - |"
    )
    rows.append(
        f"| fraud_multi | {baseline_m['sample_count']} | {baseline_m['accuracy']:.4f} | "
        f"{baseline_m['macro_precision']:.4f} | {baseline_m['macro_recall']:.4f} | "
        f"{baseline_m['micro_f1']:.4f} | {baseline_m['macro_f1']:.4f} |"
    )
    rows.append("\n## Binary Adversarial False Negatives\n")
    rows.append("| Rewrite model | Round | Recall | F1 | False negatives | ASR |")
    rows.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in summary["binary_adv"]:
        rows.append(
            f"| {row['rewrite_model']} | {row['round']} | {row['metrics']['recall']:.4f} | "
            f"{row['metrics']['f1']:.4f} | "
            f"{row['attack']['attack_success']}/{row['attack']['asr_denominator']} | "
            f"{row['attack']['asr']:.4f} |"
        )
    rows.append("\n## Multi-Class Macro-F1 Drop\n")
    rows.append("| Rewrite model | Round | Accuracy | macro-F1 | Drop vs baseline |")
    rows.append("| --- | --- | ---: | ---: | ---: |")
    for row in summary["multi_adv"]:
        drop = baseline_m["macro_f1"] - row["metrics"]["macro_f1"]
        rows.append(
            f"| {row['rewrite_model']} | {row['round']} | {row['metrics']['accuracy']:.4f} | "
            f"{row['metrics']['macro_f1']:.4f} | {drop:.4f} |"
        )
    rows.append("\n## Rewrite Quality Overall\n")
    rows.append("| Model | Pass | Warning | Needs review |")
    rows.append("| --- | ---: | ---: | ---: |")
    for row in quality_rows:
        rows.append(
            f"| {row['Model'].strip('`')} | {row['Pass']} | {row['Warning']} | {row['Needs review']} |"
        )
    (out_dir / "derived_experiment_tables.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="adversarial_data/evaluation/adv_evaluation_summary.json",
    )
    parser.add_argument(
        "--quality",
        default="adversarial_data/validated/rewrite_quality_analysis.md",
    )
    parser.add_argument(
        "--out_dir",
        default="../goal/course_paper_result",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(args.summary)
    quality_rows = extract_quality_overall(args.quality)
    save_binary_false_negatives(summary, figures_dir)
    save_multi_confusion_matrix(summary, figures_dir)
    save_multi_macro(summary, figures_dir)
    save_quality_vs_drop(summary, quality_rows, figures_dir)
    write_tables(summary, quality_rows, out_dir)
    print(f"Wrote figures and tables to {out_dir}")


if __name__ == "__main__":
    main()
