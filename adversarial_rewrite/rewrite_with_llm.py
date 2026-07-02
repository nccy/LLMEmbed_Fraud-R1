import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from prompts import ROUND_FIELDS, build_messages, preview_messages


DEFAULT_INPUT = "adversarial_data/source/fraud_test_attack_source.json"
DEFAULT_OUTPUT_DIR = "adversarial_data/rewrites"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
REFUSAL_MARKERS = [
    "无法处理",
    "抱歉，我无法",
    "抱歉，无法",
    "抱歉，我不能",
    "抱歉，不能",
    "不能协助",
    "不能帮助",
    "无法帮助",
    "不能提供",
    "sorry",
    "i can't",
    "i cannot",
    "cannot assist",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite exported fraud dialogue samples for LLMEmbed robustness "
            "experiments using local Ollama or an online OpenAI-compatible API."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--provider",
        choices=["ollama", "ollama-native", "openai-compatible"],
        default="ollama",
        help=(
            "ollama uses the OpenAI-compatible /v1 endpoint; ollama-native uses "
            "Ollama /api/chat; openai-compatible is for online compatible APIs."
        ),
    )
    parser.add_argument("--model", default=os.getenv("REWRITE_MODEL", "llama3.1:8b"))
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Allow thinking-mode models to return reasoning. Default is disabled.",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=5)
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def default_output_path(model):
    safe_model = model.replace("/", "_").replace(":", "-")
    return Path(DEFAULT_OUTPUT_DIR) / f"{safe_model}_fraud_test_rewrites.json"


def resolve_base_url(args):
    if args.base_url:
        return args.base_url.rstrip("/")
    if args.provider in {"ollama", "ollama-native"}:
        return os.getenv("OLLAMA_URL", DEFAULT_LOCAL_BASE_URL).rstrip("/")
    return os.getenv("REWRITE_BASE_URL", os.getenv("OPENAI_BASE_URL", "")).rstrip("/")


def resolve_api_key(args):
    if args.api_key:
        return args.api_key
    if args.provider in {"ollama", "ollama-native"}:
        return os.getenv("OLLAMA_API_KEY", "ollama")
    return os.getenv("REWRITE_API_KEY", os.getenv("OPENAI_API_KEY", ""))


def chat_endpoint(base_url):
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def ollama_native_endpoint(base_url):
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if base_url.endswith("/api/chat"):
        return base_url
    return f"{base_url}/api/chat"


def request_completion(args, messages):
    base_url = resolve_base_url(args)
    if not base_url:
        raise ValueError(
            "Missing base URL. Use --base_url or set REWRITE_BASE_URL/OPENAI_BASE_URL."
        )
    headers = {"Content-Type": "application/json"}
    endpoint = chat_endpoint(base_url)
    if args.provider == "ollama-native":
        endpoint = ollama_native_endpoint(base_url)
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": False,
            "think": bool(args.enable_thinking),
            "options": {
                "temperature": args.temperature,
                "num_predict": args.max_tokens,
                "num_ctx": args.num_ctx,
            },
        }
        if args.json_mode:
            payload["format"] = "json"
    else:
        api_key = resolve_api_key(args)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.json_mode:
            payload["response_format"] = {"type": "json_object"}

    session = requests.Session()
    if args.provider in {"ollama", "ollama-native"}:
        session.trust_env = False

    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            response = session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=args.timeout,
            )
            response.raise_for_status()
            data = response.json()
            if args.provider == "ollama-native":
                return extract_ollama_native_content(data)
            return extract_content(data)
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                sleep_seconds = args.retry_sleep * attempt
                print(
                    f"request failed on attempt {attempt}/{args.retries}: {exc}; "
                    f"sleep {sleep_seconds:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_seconds)
    raise RuntimeError(f"request failed after {args.retries} attempts: {last_error}")


def extract_content(data):
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"response has no choices: {data}")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if content is None:
        content = choice.get("text")
    if not content:
        raise ValueError(f"response content is empty: {data}")
    return content


def extract_ollama_native_content(data):
    message = data.get("message") or {}
    content = message.get("content")
    if not content:
        raise ValueError(f"native Ollama response content is empty: {data}")
    return content


def strip_json_fence(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_rewrite_json(raw_text):
    text = strip_json_fence(raw_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("rewrite response is not a JSON object")
    return {
        field: normalize_dialogue_roles(str(data.get(field, "")).strip())
        for field in ROUND_FIELDS
    }


def normalize_dialogue_roles(text):
    replacements = [
        ("左边：", "\nleft: "),
        ("右边：", "\nright: "),
        ("左边:", "\nleft: "),
        ("右边:", "\nright: "),
        ("left：", "\nleft: "),
        ("right：", "\nright: "),
    ]
    normalized = text
    for old, new in replacements:
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"(?<!^)(?<!\n)(left:\s*)", r"\n\1", normalized)
    normalized = re.sub(r"(?<!^)(?<!\n)(right:\s*)", r"\n\1", normalized)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def has_refusal(text):
    lower = text.lower()
    return any(marker in lower for marker in REFUSAL_MARKERS)


def basic_check(original, rewrites):
    reasons = []
    original_len = len(original.strip())
    for field in ROUND_FIELDS:
        value = rewrites.get(field, "").strip()
        if not value:
            reasons.append(f"{field}:empty")
            continue
        if has_refusal(value):
            reasons.append(f"{field}:refusal")
        if len(value) < max(40, original_len * 0.25):
            reasons.append(f"{field}:too_short")
        if "left:" not in value or "right:" not in value:
            reasons.append(f"{field}:missing_dialogue_roles")
    return reasons


def build_output_record(source_record, rewrites, raw_response, model, error=None):
    output = dict(source_record)
    output.update(rewrites)
    output["rewrite_model"] = model
    output["rewrite_time"] = datetime.now().isoformat(timespec="seconds")
    output["raw_response"] = raw_response
    if error:
        output["is_valid"] = False
        output["invalid_reason"] = error
    else:
        reasons = basic_check(output["round_0_original"], rewrites)
        output["is_valid"] = not reasons
        output["invalid_reason"] = ";".join(reasons)
    return output


def load_existing(output_path):
    path = Path(output_path)
    if not path.exists():
        return []
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Existing output is not a list: {output_path}")
    return data


def select_records(records, start, max_items):
    selected = records[start:]
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def main():
    args = parse_args()
    source_records = read_json(args.input)
    selected = select_records(source_records, args.start, args.max_items)
    output_path = Path(args.output) if args.output else default_output_path(args.model)

    if args.dry_run:
        if not selected:
            print("no records selected")
            return
        print(preview_messages(selected[0]))
        print(f"selected records: {len(selected)}", file=sys.stderr)
        print(f"would write: {output_path}", file=sys.stderr)
        return

    existing = load_existing(output_path) if args.resume else []
    existing_by_index = {record.get("index"): record for record in existing}
    results = list(existing)

    completed = 0
    failed = 0
    skipped = 0
    for pos, record in enumerate(selected, start=1):
        if args.resume and record["index"] in existing_by_index:
            skipped += 1
            continue

        raw_response = ""
        rewrites = {field: "" for field in ROUND_FIELDS}
        error = None
        try:
            raw_response = request_completion(args, build_messages(record))
            rewrites = parse_rewrite_json(raw_response)
        except Exception as exc:
            error = str(exc)
            failed += 1

        results.append(
            build_output_record(
                source_record=record,
                rewrites=rewrites,
                raw_response=raw_response,
                model=args.model,
                error=error,
            )
        )
        completed += 1
        if completed % args.save_every == 0:
            write_json(output_path, results)

        print(
            f"[{pos}/{len(selected)}] index={record['index']} "
            f"valid={results[-1]['is_valid']} reason={results[-1]['invalid_reason']}"
        )

    write_json(output_path, results)
    print(f"output: {output_path}")
    print(f"new records: {completed}")
    print(f"skipped existing: {skipped}")
    print(f"request/parse failures: {failed}")


if __name__ == "__main__":
    main()
