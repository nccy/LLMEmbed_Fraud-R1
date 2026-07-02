import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rewrite_with_llm import (  # noqa: E402
    DEFAULT_INPUT,
    ROUND_FIELDS,
    build_messages,
    build_output_record,
    default_output_path,
    load_existing,
    normalize_dialogue_roles,
    parse_rewrite_json,
    read_json,
    request_completion,
    strip_json_fence,
    write_json,
)


def starts_role(text):
    return text.strip().startswith(("left:", "right:"))


def dialogue_line_count(text):
    return sum(
        1
        for line in text.splitlines()
        if line.strip().startswith(("left:", "right:"))
    )


def build_strict_retry_messages(source_record, previous_record):
    messages = build_messages(source_record)
    original = source_record["round_0_original"].strip()
    min_chars = max(40, int(len(original) * 0.35))
    line_count = dialogue_line_count(original)
    previous_reason = ""
    if previous_record:
        previous_reason = previous_record.get("invalid_reason", "")

    strict_prompt = f"""上一轮输出没有通过本地校验，失败原因：{previous_reason or "未知"}。

请只重新输出严格 JSON，不要解释。

额外硬性要求：
- JSON 顶层只能有三个字段：round_1_trust、round_2_urgency、round_3_emotion。
- 三个字段的值必须是字符串，不能是数组、对象、role/content 列表，也不能输出多个 JSON 对象。
- 三个字段都必须是完整对话，不能只输出 `left:` 或 `right:` 空标签。
- 每个字段都必须至少 {min_chars} 个中文字符或等量内容，且必须明显接近原文长度。
- 每个字段长度不要超过原文的 1.3 倍，避免输出被截断。
- 原始对话有 {line_count} 行 left/right 对话；每个字段也必须尽量保持 {line_count} 行。
- 每一行都必须以 `left:` 或 `right:` 开头，并保留原始对话的先后顺序。
- 对话内部换行必须写成 JSON 字符串里的 `\\n`。
- 不要输出“抱歉”“无法”“不能帮助”“cannot assist”“I can't”等拒绝或安全声明。
- 不要摘要，不要省略，不要只写占位符。

原始对话再次提供如下：
{original}
"""
    messages.append({"role": "user", "content": strict_prompt})
    return messages


def avoid_refusal_false_positives(rewrites):
    replacements = {
        "不能提供": "暂不提供",
    }
    cleaned = {}
    for field, value in rewrites.items():
        for old, new in replacements.items():
            value = value.replace(old, new)
        cleaned[field] = value
    return cleaned


def recover_dialogue_lines_from_keys(raw_text, rewrites):
    """Repair JSON where a model emitted dialogue lines as object keys."""
    text = strip_json_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return rewrites
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return rewrites

    if not isinstance(data, dict):
        return rewrites

    repaired = dict(rewrites)
    for field in ROUND_FIELDS:
        lines = []
        collecting = False
        for key, value in data.items():
            if key == field:
                collecting = True
                if isinstance(value, str) and value.strip():
                    lines.append(value.strip())
                continue
            if key in ROUND_FIELDS:
                collecting = False
                continue
            if not collecting:
                continue
            key_text = key.strip() if isinstance(key, str) else ""
            value_text = value.strip() if isinstance(value, str) else ""
            if key_text.startswith(("left:", "right:")):
                lines.append(key_text)
            elif value_text.startswith(("left:", "right:")):
                lines.append(value_text)

        if len(lines) > 1:
            repaired[field] = normalize_dialogue_roles("\n".join(lines))

    return repaired


def json_prefix_decode(text):
    stripped = strip_json_fence(text)
    start = stripped.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(stripped[start:])
        return data
    except json.JSONDecodeError:
        return None


def append_dialogue_value(lines, value, next_role):
    if isinstance(value, str):
        text = normalize_dialogue_roles(value.strip())
        if not text:
            return next_role
        if starts_role(text):
            lines.extend(text.splitlines())
        else:
            lines.append(f"{next_role}: {text}")
            next_role = "right" if next_role == "left" else "left"
        return next_role

    if isinstance(value, list):
        for item in value:
            next_role = append_dialogue_value(lines, item, next_role)
        return next_role

    if isinstance(value, dict):
        if "left" in value or "right" in value:
            if value.get("left"):
                lines.append(f"left: {str(value['left']).strip()}")
            if value.get("right"):
                lines.append(f"right: {str(value['right']).strip()}")
            return next_role
        if "content" in value and value.get("content"):
            role = value.get("role")
            mapped_role = "left"
            if role in {"assistant", "right"}:
                mapped_role = "right"
            elif role in {"user", "left", "system"}:
                mapped_role = "left"
            lines.append(f"{mapped_role}: {str(value['content']).strip()}")
            return "right" if mapped_role == "left" else "left"
        for key, item in value.items():
            key_text = key.strip() if isinstance(key, str) else ""
            if key_text.startswith(("left:", "right:")):
                lines.append(key_text)
                continue
            if key in {"left", "right"} and item:
                lines.append(f"{key}: {str(item).strip()}")
                continue
            next_role = append_dialogue_value(lines, item, next_role)
        return next_role

    return next_role


def object_to_rewrites(data):
    if not isinstance(data, dict):
        raise ValueError("rewrite response is not a JSON object")

    aliases = {
        "round_1_trust": ["round_1_trust", "round1_trust", "round_1"],
        "round_2_urgency": [
            "round_2_urgency",
            "round_2_emergency",
            "round2_urgency",
            "round_2",
        ],
        "round_3_emotion": [
            "round_3_emotion",
            "round_3_emotional",
            "round3_emotion",
            "round_3",
        ],
    }

    output = {}
    for field, names in aliases.items():
        value = None
        for name in names:
            if name in data:
                value = data[name]
                break
        lines = []
        append_dialogue_value(lines, value, "left")
        output[field] = normalize_dialogue_roles("\n".join(lines))
    return output


def regex_extract_rewrites(raw_text):
    text = strip_json_fence(raw_text)
    output = {}
    for pos, field in enumerate(ROUND_FIELDS):
        start_match = re.search(rf'"{field}"\s*:\s*"', text)
        if not start_match:
            output[field] = ""
            continue
        start = start_match.end()
        if pos + 1 < len(ROUND_FIELDS):
            next_field = ROUND_FIELDS[pos + 1]
            end_match = re.search(rf'"\s*,\s*"{next_field}"\s*:', text[start:], re.S)
            end = start + end_match.start() if end_match else len(text)
        else:
            end_match = re.search(r'"\s*}\s*$', text[start:], re.S)
            end = start + end_match.start() if end_match else len(text)
        value = text[start:end]
        value = value.replace("\\n", "\n").replace('\\"', '"')
        output[field] = normalize_dialogue_roles(value.strip())
    return output


def parse_rewrite_json_flexible(raw_text):
    try:
        rewrites = parse_rewrite_json(raw_text)
        return recover_dialogue_lines_from_keys(raw_text, rewrites)
    except Exception:
        pass

    data = json_prefix_decode(raw_text)
    if data is not None:
        return object_to_rewrites(data)

    return regex_extract_rewrites(raw_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retry invalid rewrite records in existing output files until all "
            "records pass the same validation used by rewrite_with_llm.py."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3.5:9b", "llama3.1:8b"],
        help="Models to repair. Output paths are derived from each model name.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "ollama-native", "openai-compatible"],
        default="ollama-native",
    )
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_ctx", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=5)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Allow thinking-mode models to return reasoning. Default is disabled.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=10,
        help="Maximum retry passes per model.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save after this many rewritten records.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print how many records would be retried.",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create a timestamped backup before modifying an output file.",
    )
    parser.add_argument(
        "--strict_retry_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use an additional strict repair prompt for invalid records.",
    )
    parser.add_argument(
        "--recover_existing_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to repair an invalid record from its saved raw_response before making a new request.",
    )
    return parser.parse_args()


def summarize(records, source_count):
    valid = sum(1 for record in records if record.get("is_valid") is True)
    invalid = len(records) - valid
    unique_indices = {record.get("index") for record in records}
    missing = max(0, source_count - len(unique_indices))
    return valid, invalid, missing


def records_needing_retry(records, source_records):
    by_index = {record.get("index"): record for record in records}
    retry_indices = []
    for source in source_records:
        existing = by_index.get(source["index"])
        if existing is None or existing.get("is_valid") is not True:
            retry_indices.append(source["index"])
    return retry_indices


def upsert_by_index(records, new_record):
    target_index = new_record.get("index")
    for pos, record in enumerate(records):
        if record.get("index") == target_index:
            records[pos] = new_record
            return
    records.append(new_record)


def backup_output(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
    shutil.copy2(path, backup_path)
    print(f"backup: {backup_path}")


def retry_model(args, model, source_records):
    output_path = default_output_path(model)
    if not output_path.exists():
        raise FileNotFoundError(f"missing output file for {model}: {output_path}")

    source_by_index = {record["index"]: record for record in source_records}
    records = load_existing(output_path)
    valid, invalid, missing = summarize(records, len(source_records))
    print(
        f"{model}: start total={len(records)} valid={valid} "
        f"invalid={invalid} missing={missing} output={output_path}"
    )

    if args.dry_run:
        retry_indices = records_needing_retry(records, source_records)
        print(f"{model}: would retry {len(retry_indices)} records")
        return

    if not args.no_backup:
        backup_output(output_path)

    request_args = argparse.Namespace(**vars(args))
    request_args.model = model

    for round_no in range(1, args.max_rounds + 1):
        retry_indices = records_needing_retry(records, source_records)
        if not retry_indices:
            print(f"{model}: reached 100% valid")
            write_json(output_path, sorted(records, key=lambda item: item["index"]))
            return

        print(f"{model}: round {round_no}/{args.max_rounds}, retry={len(retry_indices)}")
        changed_since_save = 0
        existing_by_index = {record.get("index"): record for record in records}
        for pos, index in enumerate(retry_indices, start=1):
            source_record = source_by_index[index]
            previous_record = existing_by_index.get(index)
            previous_raw = ""
            if previous_record:
                previous_raw = previous_record.get("raw_response") or ""

            if args.recover_existing_raw and previous_raw:
                recovered_rewrites = avoid_refusal_false_positives(
                    parse_rewrite_json_flexible(previous_raw)
                )
                recovered_record = build_output_record(
                    source_record=source_record,
                    rewrites=recovered_rewrites,
                    raw_response=previous_raw,
                    model=model,
                    error=None,
                )
                if recovered_record["is_valid"]:
                    upsert_by_index(records, recovered_record)
                    changed_since_save += 1
                    print(
                        f"{model}: round={round_no} item={pos}/{len(retry_indices)} "
                        f"index={index} valid=True reason= recovered_from_saved_raw"
                    )
                    if changed_since_save >= args.save_every:
                        write_json(
                            output_path, sorted(records, key=lambda item: item["index"])
                        )
                        changed_since_save = 0
                    continue

            raw_response = ""
            rewrites = {field: "" for field in ROUND_FIELDS}
            error = None
            try:
                if args.strict_retry_prompt:
                    messages = build_strict_retry_messages(source_record, previous_record)
                else:
                    messages = build_messages(source_record)
                raw_response = request_completion(
                    request_args, messages
                )
                rewrites = parse_rewrite_json_flexible(raw_response)
                rewrites = avoid_refusal_false_positives(rewrites)
            except Exception as exc:
                error = str(exc)

            new_record = build_output_record(
                source_record=source_record,
                rewrites=rewrites,
                raw_response=raw_response,
                model=model,
                error=error,
            )
            upsert_by_index(records, new_record)
            changed_since_save += 1

            print(
                f"{model}: round={round_no} item={pos}/{len(retry_indices)} "
                f"index={index} valid={new_record['is_valid']} "
                f"reason={new_record['invalid_reason']}"
            )

            if changed_since_save >= args.save_every:
                write_json(output_path, sorted(records, key=lambda item: item["index"]))
                changed_since_save = 0

        write_json(output_path, sorted(records, key=lambda item: item["index"]))
        records = load_existing(output_path)
        valid, invalid, missing = summarize(records, len(source_records))
        print(
            f"{model}: after round {round_no} total={len(records)} "
            f"valid={valid} invalid={invalid} missing={missing}"
        )

    retry_indices = records_needing_retry(records, source_records)
    raise RuntimeError(
        f"{model}: still has {len(retry_indices)} invalid/missing records "
        f"after {args.max_rounds} rounds"
    )


def main():
    args = parse_args()
    source_records = read_json(args.input)
    if not isinstance(source_records, list):
        raise ValueError(f"input must be a list: {args.input}")

    for model in args.models:
        retry_model(args, model, source_records)


if __name__ == "__main__":
    main()
