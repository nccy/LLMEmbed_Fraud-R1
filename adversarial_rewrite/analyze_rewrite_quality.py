# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROUND_FIELDS = ("round_1_trust", "round_2_urgency", "round_3_emotion")

KEYWORD_GROUPS = {
    "contact_link_app": [
        "链接",
        "网址",
        "http",
        "短信",
        "二维码",
        "扫码",
        "扫描",
        "APP",
        "app",
        "下载",
        "注册",
        "验证",
        "填写",
        "点击",
        "订单号",
    ],
    "money_finance": [
        "钱",
        "转账",
        "汇款",
        "付款",
        "支付",
        "账户",
        "账号",
        "银行卡",
        "收款",
        "保证金",
        "手续费",
        "押金",
        "充值",
        "提现",
        "退款",
        "贷款",
        "利率",
        "投资",
        "收益",
        "年化",
        "基金",
        "数字货币",
    ],
    "urgency_threat": [
        "立即",
        "马上",
        "尽快",
        "不然",
        "否则",
        "逾期",
        "风险",
        "扣押",
        "伤害",
        "紧急",
        "名额",
        "机会",
        "满额",
    ],
    "identity_org": [
        "客服",
        "银行",
        "经理",
        "顾问",
        "中心",
        "公司",
        "平台",
        "公安",
        "快递",
        "投资",
        "金融",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze semantic and structural quality of rewrite JSON files."
    )
    parser.add_argument(
        "--rewrite_json",
        nargs="+",
        required=True,
        help="Rewrite JSON files to analyze.",
    )
    parser.add_argument(
        "--output_json",
        default="adversarial_data/validated/rewrite_quality_analysis.json",
    )
    parser.add_argument(
        "--output_md",
        default="adversarial_data/validated/rewrite_quality_analysis.md",
    )
    parser.add_argument("--max_examples", type=int, default=5)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dialogue_lines(text):
    return [
        line.strip()
        for line in str(text).splitlines()
        if line.strip().startswith(("left:", "right:"))
    ]


def non_role_line_count(text):
    return sum(
        1
        for line in str(text).splitlines()
        if line.strip() and not line.strip().startswith(("left:", "right:"))
    )


def extract_keywords(text):
    found = {}
    for group, keywords in KEYWORD_GROUPS.items():
        present = [kw for kw in keywords if kw in text]
        if present:
            found[group] = sorted(set(present))
    return found


def flatten_keywords(groups):
    return {kw for items in groups.values() for kw in items}


def safe_ratio(num, den):
    return num / den if den else 1.0


def similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()


def has_cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def quality_flags(original, rewrite):
    original = str(original)
    rewrite = str(rewrite)
    original_len = len(original.strip())
    rewrite_len = len(rewrite.strip())
    original_lines = dialogue_lines(original)
    rewrite_lines = dialogue_lines(rewrite)
    original_keywords = extract_keywords(original)
    rewrite_keywords = extract_keywords(rewrite)
    original_kw_set = flatten_keywords(original_keywords)
    rewrite_kw_set = flatten_keywords(rewrite_keywords)
    retained_kw = original_kw_set & rewrite_kw_set
    keyword_retention = safe_ratio(len(retained_kw), len(original_kw_set))
    length_ratio = safe_ratio(rewrite_len, original_len)
    line_ratio = safe_ratio(len(rewrite_lines), len(original_lines))
    sim = similarity(original, rewrite)

    flags = []
    if rewrite_len < 40:
        flags.append("too_short_absolute")
    if length_ratio < 0.60:
        flags.append("too_short_vs_original")
    if length_ratio > 1.80:
        flags.append("too_long_vs_original")
    if len(original_lines) >= 4 and len(rewrite_lines) <= len(original_lines) - 3:
        flags.append("dialogue_line_drop_ge3")
    if line_ratio < 0.70:
        flags.append("dialogue_line_ratio_lt_0.70")
    if non_role_line_count(rewrite) > 0:
        flags.append("non_role_lines")
    if original_kw_set and keyword_retention < 0.50:
        flags.append("keyword_retention_lt_0.50")
    if len(original_kw_set) >= 4 and len(original_kw_set - rewrite_kw_set) >= 4:
        flags.append("major_keyword_loss")
    if sim > 0.95:
        flags.append("near_copy")
    if not has_cjk(rewrite):
        flags.append("no_cjk_text")

    severe = {
        "too_short_absolute",
        "too_short_vs_original",
        "dialogue_line_drop_ge3",
        "dialogue_line_ratio_lt_0.70",
        "keyword_retention_lt_0.50",
        "major_keyword_loss",
        "no_cjk_text",
    }
    warning = {"too_long_vs_original", "non_role_lines", "near_copy"}
    if any(flag in severe for flag in flags):
        grade = "needs_review"
    elif any(flag in warning for flag in flags):
        grade = "warning"
    else:
        grade = "pass"

    return {
        "grade": grade,
        "flags": flags,
        "original_chars": original_len,
        "rewrite_chars": rewrite_len,
        "length_ratio": length_ratio,
        "original_lines": len(original_lines),
        "rewrite_lines": len(rewrite_lines),
        "line_ratio": line_ratio,
        "similarity": sim,
        "original_keyword_count": len(original_kw_set),
        "retained_keyword_count": len(retained_kw),
        "keyword_retention": keyword_retention,
        "lost_keywords": sorted(original_kw_set - rewrite_kw_set),
        "retained_keywords": sorted(retained_kw),
    }


def avg(values):
    return sum(values) / len(values) if values else 0.0


def pct(count, total):
    return 100.0 * count / total if total else 0.0


def analyze_file(path, max_examples):
    rows = read_json(path)
    model = None
    if rows:
        model = rows[0].get("rewrite_model")
    if not model:
        model = Path(path).name.replace("_fraud_test_rewrites.json", "")
    model_slug = model.replace(":", "-")

    per_round = {}
    examples = []
    all_grades = Counter()
    all_flags = Counter()

    for field in ROUND_FIELDS:
        grades = Counter()
        flags = Counter()
        length_ratios = []
        line_ratios = []
        similarities = []
        keyword_retentions = []
        quality_rows = []

        for record in rows:
            item = quality_flags(record.get("round_0_original", ""), record.get(field, ""))
            item.update(
                {
                    "index": record.get("index"),
                    "binary_index": record.get("binary_index"),
                    "multi_index": record.get("multi_index"),
                    "multi_label": record.get("multi_label"),
                    "round": field,
                    "model": model_slug,
                }
            )
            quality_rows.append(item)
            grades[item["grade"]] += 1
            all_grades[item["grade"]] += 1
            for flag in item["flags"]:
                flags[flag] += 1
                all_flags[flag] += 1
            length_ratios.append(item["length_ratio"])
            line_ratios.append(item["line_ratio"])
            similarities.append(item["similarity"])
            if item["original_keyword_count"]:
                keyword_retentions.append(item["keyword_retention"])

        review_candidates = [
            item
            for item in quality_rows
            if item["grade"] == "needs_review"
        ]
        review_candidates.sort(
            key=lambda item: (
                item["keyword_retention"],
                item["line_ratio"],
                item["length_ratio"],
            )
        )
        examples.extend(review_candidates[:max_examples])

        per_round[field] = {
            "total": len(rows),
            "grades": dict(grades),
            "flags": dict(flags.most_common()),
            "avg_length_ratio": avg(length_ratios),
            "min_length_ratio": min(length_ratios) if length_ratios else 0.0,
            "avg_line_ratio": avg(line_ratios),
            "min_line_ratio": min(line_ratios) if line_ratios else 0.0,
            "avg_similarity": avg(similarities),
            "avg_keyword_retention": avg(keyword_retentions),
            "needs_review_rate": pct(grades["needs_review"], len(rows)),
            "warning_rate": pct(grades["warning"], len(rows)),
            "pass_rate": pct(grades["pass"], len(rows)),
        }

    examples.sort(
        key=lambda item: (
            item["keyword_retention"],
            item["line_ratio"],
            item["length_ratio"],
        )
    )

    record_grades = Counter()
    by_multi_label = defaultdict(Counter)
    for record in rows:
        grades = [
            quality_flags(record.get("round_0_original", ""), record.get(field, ""))[
                "grade"
            ]
            for field in ROUND_FIELDS
        ]
        if "needs_review" in grades:
            grade = "needs_review_any_round"
        elif "warning" in grades:
            grade = "warning_any_round"
        else:
            grade = "pass_all_rounds"
        record_grades[grade] += 1
        by_multi_label[str(record.get("multi_label"))][grade] += 1

    return {
        "path": str(path),
        "model": model_slug,
        "total_records": len(rows),
        "total_round_items": len(rows) * len(ROUND_FIELDS),
        "overall_grades": dict(all_grades),
        "overall_flags": dict(all_flags.most_common()),
        "record_level_grades": dict(record_grades),
        "record_level_by_multi_label": {
            label: dict(counter) for label, counter in sorted(by_multi_label.items())
        },
        "per_round": per_round,
        "examples": examples[:max_examples],
    }


def write_markdown(path, report):
    lines = [
        "# Rewrite Quality Analysis",
        "",
        "说明：本报告是在 `is_valid` 基础上增加的内容质量校验。`needs_review` 不等于 JSON 无效，表示改写可能存在截断、对话轮次丢失、诈骗关键词/核心线索丢失、或改写过度等问题，建议人工抽查后再用于攻击有效性解释。",
        "",
        "## Overall",
        "",
        "| Model | Items | Pass | Warning | Needs review | Top flags |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report["files"]:
        total = item["total_round_items"]
        grades = Counter(item["overall_grades"])
        top_flags = ", ".join(
            f"{flag}={count}" for flag, count in list(item["overall_flags"].items())[:5]
        )
        lines.append(
            f"| `{item['model']}` | {total} | "
            f"{grades['pass']} ({pct(grades['pass'], total):.1f}%) | "
            f"{grades['warning']} ({pct(grades['warning'], total):.1f}%) | "
            f"{grades['needs_review']} ({pct(grades['needs_review'], total):.1f}%) | "
            f"{top_flags or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Record-Level Usability",
            "",
            "一条源样本包含三轮改写。这里的 `Pass all rounds` 表示三轮都未触发质量风险；`Needs review any round` 表示至少一轮需要人工复核。",
            "",
            "| Model | Records | Pass all rounds | Warning any round | Needs review any round |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["files"]:
        total = item["total_records"]
        grades = Counter(item["record_level_grades"])
        lines.append(
            f"| `{item['model']}` | {total} | "
            f"{grades['pass_all_rounds']} ({pct(grades['pass_all_rounds'], total):.1f}%) | "
            f"{grades['warning_any_round']} ({pct(grades['warning_any_round'], total):.1f}%) | "
            f"{grades['needs_review_any_round']} ({pct(grades['needs_review_any_round'], total):.1f}%) |"
        )

    lines.extend(
        [
            "",
            "## Needs-Review Rate By Multi Label",
            "",
            "| Model | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Label 6 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["files"]:
        cells = []
        for label in range(7):
            counter = Counter(item["record_level_by_multi_label"].get(str(label), {}))
            total = sum(counter.values())
            cells.append(f"{pct(counter['needs_review_any_round'], total):.1f}%")
        lines.append(f"| `{item['model']}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Round-Level Metrics",
            "",
            "| Model | Round | Pass % | Needs review % | Avg len ratio | Min len ratio | Avg line ratio | Min line ratio | Avg similarity | Avg keyword retention |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["files"]:
        for field, stats in item["per_round"].items():
            lines.append(
                f"| `{item['model']}` | `{field}` | "
                f"{stats['pass_rate']:.1f} | {stats['needs_review_rate']:.1f} | "
                f"{stats['avg_length_ratio']:.2f} | {stats['min_length_ratio']:.2f} | "
                f"{stats['avg_line_ratio']:.2f} | {stats['min_line_ratio']:.2f} | "
                f"{stats['avg_similarity']:.2f} | {stats['avg_keyword_retention']:.2f} |"
            )

    lines.extend(["", "## Review Examples", ""])
    for item in report["files"]:
        lines.append(f"### `{item['model']}`")
        lines.append("")
        if not item["examples"]:
            lines.append("No high-priority review examples found.")
            lines.append("")
            continue
        lines.append(
            "| Round | Index | Multi label | Len ratio | Line ratio | Keyword retention | Flags | Lost keywords |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for example in item["examples"]:
            lines.append(
                f"| `{example['round']}` | {example['index']} | {example['multi_label']} | "
                f"{example['length_ratio']:.2f} | {example['line_ratio']:.2f} | "
                f"{example['keyword_retention']:.2f} | "
                f"{', '.join(example['flags']) or '-'} | "
                f"{', '.join(example['lost_keywords'][:12]) or '-'} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `pass`：格式、长度、轮次和关键词保留均未触发主要风险规则，可以作为主评估数据使用。",
            "- `warning`：主要是近似复制、轻微过长或非角色行等软问题，通常不影响二分类鲁棒性评估，但会影响改写多样性。",
            "- `needs_review`：可能删除了诈骗动作、身份、金额、链接、威胁或收益等核心线索。若这类样本造成 attack success，应优先解释为改写质量问题，而不是模型被有效语义保持攻击绕过。",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    files = [analyze_file(path, args.max_examples) for path in args.rewrite_json]
    report = {
        "rewrite_json": args.rewrite_json,
        "files": files,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_md, report)
    for item in files:
        grades = Counter(item["overall_grades"])
        print(
            f"{item['model']}: pass={grades['pass']} "
            f"warning={grades['warning']} needs_review={grades['needs_review']}"
        )
    print(f"output_json: {args.output_json}")
    print(f"output_md: {args.output_md}")


if __name__ == "__main__":
    main()
