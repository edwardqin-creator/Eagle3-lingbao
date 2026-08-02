#!/usr/bin/env python3
"""Parse production request logs, apply conservative filters and split data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


PAYLOAD_MARKER = "payload: "
CONTROL_RE = re.compile(r"[\u200b\u200c\u2060\u2061\u2063\ufeff]")
REQUIRED_SAMPLING = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生产请求解析、保守清洗、精确去重和固定随机切分"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-format",
        choices=("auto", "msg-log", "request-jsonl"),
        default="auto",
    )
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--min-prompt-chars", type=int, default=120)
    parser.add_argument("--max-prompt-chars", type=int, default=200000)
    parser.add_argument("--default-model", default="kaiwu-llm-model")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1:
        parser.error("train-size 和 val-size 必须大于 0")
    return args


def rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            yield line_number, value


def extract_payload(message: str) -> dict[str, Any]:
    marker_at = message.find(PAYLOAD_MARKER)
    if marker_at < 0:
        raise ValueError("msg 中没有 payload 标记")
    payload_text = message[marker_at + len(PAYLOAD_MARKER) :].lstrip()
    value, _ = json.JSONDecoder().raw_decode(payload_text)
    if not isinstance(value, dict):
        raise ValueError("payload 不是 JSON 对象")
    return value


def normalize_prompt(text: str) -> tuple[str, int]:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned, removed = CONTROL_RE.subn("", normalized)
    return cleaned.replace("\r\n", "\n").replace("\r", "\n"), removed


def request_record(
    source: dict[str, Any], line_number: int, input_format: str, default_model: str
) -> tuple[dict[str, Any], int]:
    if input_format == "auto":
        input_format = "msg-log" if "msg" in source else "request-jsonl"
    payload = extract_payload(source["msg"]) if input_format == "msg-log" else source

    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("prompt 缺失或不是字符串")
    prompt, removed_controls = normalize_prompt(prompt)

    if input_format == "request-jsonl" and isinstance(payload.get("sampling_params"), dict):
        sampling_source = payload["sampling_params"]
    else:
        sampling_source = payload
    missing = [key for key in REQUIRED_SAMPLING if key not in sampling_source]
    if missing:
        raise ValueError(f"缺少采样字段 {missing}")

    sampling = {key: sampling_source[key] for key in REQUIRED_SAMPLING}
    sampling["stop"] = sampling_source.get("stop", ["<|im_end|>"])
    if sampling_source.get("seed") is not None:
        sampling["seed"] = sampling_source["seed"]

    record_id = source.get("id")
    if not isinstance(record_id, str) or not record_id:
        record_id = f"lingbao-{line_number:06d}"
    model = payload.get("model") or default_model
    return {
        "id": record_id,
        "prompt": prompt,
        "model": model,
        "sampling_params": sampling,
        "source_line": line_number,
    }, removed_controls


def prompt_fingerprint(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "train": args.output_dir / "train_requests.jsonl",
        "val": args.output_dir / "val_requests.jsonl",
        "test": args.output_dir / "test_requests.jsonl",
        "rejects": args.output_dir / "prepare_rejects.jsonl",
        "report": args.output_dir / "prepare_report.json",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("输出已存在；确认后加 --overwrite: " + ", ".join(existing))
    if args.overwrite:
        for path in destinations.values():
            path.unlink(missing_ok=True)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    reasons: Counter[str] = Counter()
    removed_controls = 0
    total = 0

    for line_number, source in rows(args.input):
        total += 1
        try:
            record, removed = request_record(
                source, line_number, args.input_format, args.default_model
            )
            removed_controls += removed
            length = len(record["prompt"])
            if length < args.min_prompt_chars:
                raise ValueError("prompt_too_short")
            if length > args.max_prompt_chars:
                raise ValueError("prompt_too_long")
            if record["id"] in seen_ids:
                raise ValueError("duplicate_id")
            fingerprint = prompt_fingerprint(record["prompt"])
            if fingerprint in seen_prompts:
                raise ValueError("exact_prompt_duplicate")
            seen_ids.add(record["id"])
            seen_prompts.add(fingerprint)
            kept.append(record)
        except Exception as error:
            reason = str(error)
            reasons[reason] += 1
            rejected.append({"source_line": line_number, "reason": reason})

    required = args.train_size + args.val_size + 1
    if len(kept) < required:
        raise ValueError(f"清洗后仅 {len(kept)} 条，至少需要 {required} 条才能切分")
    random.Random(args.seed).shuffle(kept)
    train = kept[: args.train_size]
    val = kept[args.train_size : args.train_size + args.val_size]
    test = kept[args.train_size + args.val_size :]

    for name, records in (("train", train), ("val", val), ("test", test)):
        write_jsonl(destinations[name], records)
    write_jsonl(destinations["rejects"], rejected)
    report = {
        "input": str(args.input),
        "total": total,
        "kept": len(kept),
        "rejected": len(rejected),
        "removed_invisible_characters": removed_controls,
        "reject_reasons": dict(reasons.most_common()),
        "split_seed": args.seed,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "sampling_policy": "原参数原样保留；seed 缺失时不补 seed",
    }
    destinations["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"输出目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
