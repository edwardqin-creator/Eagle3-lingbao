#!/usr/bin/env python3
"""Build pretokenized SpecForge records and the reduced-vocabulary mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=10240)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--mask-policy", choices=("target-faithful", "historical"), default="target-faithful"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            yield line_number, value


def load_prompts(path: Path) -> dict[str, str]:
    result = {}
    for line_number, record in read_jsonl(path):
        record_id, prompt = record.get("id"), record.get("prompt")
        if not isinstance(record_id, str) or not isinstance(prompt, str):
            raise ValueError(f"{path}:{line_number} id/prompt 无效")
        if record_id in result:
            raise ValueError(f"请求文件存在重复 ID: {record_id}")
        result[record_id] = prompt
    return result


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * ratio)] if ordered else 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids(tokenizer: Any, text: str, max_length: int | None = None) -> list[int]:
    kwargs: dict[str, Any] = {"add_special_tokens": False}
    if max_length is not None:
        kwargs.update(truncation=True, max_length=max_length)
    encoded = tokenizer(text, **kwargs)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def faithful_mask(
    tokenizer: Any, text: str, prompt: str, input_ids: list[int], max_length: int, context: str
) -> list[int]:
    if not text.startswith(prompt):
        raise ValueError(f"{context}: text 不是原 prompt 的前缀扩展")
    eos = tokenizer.eos_token
    if not isinstance(eos, str) or not eos:
        raise ValueError("Tokenizer 没有 eos_token")
    eos_at = text.find(eos, len(prompt))
    if eos_at < 0:
        raise ValueError(f"{context}: completion 后缺少 EOS {eos!r}")
    supervised_end_text = text[: eos_at + len(eos)]
    prompt_ids = token_ids(tokenizer, prompt)
    supervised_end_ids = token_ids(tokenizer, supervised_end_text)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{context}: prompt token 不是完整序列的严格前缀")
    if input_ids[: len(supervised_end_ids)] != supervised_end_ids:
        raise ValueError(f"{context}: completion/EOS token 边界不一致")
    if len(supervised_end_ids) > max_length:
        raise ValueError(f"{context}: completion 或 EOS 被 max_length 截断")
    supervised = len(supervised_end_ids) - len(prompt_ids)
    if supervised < 1:
        raise ValueError(f"{context}: 没有监督 token")
    return [0] * len(prompt_ids) + [1] * supervised + [0] * (
        len(input_ids) - len(supervised_end_ids)
    )


def main() -> int:
    args = arguments()
    for path in (args.input, args.requests, args.draft_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.output, args.mapping_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            raise FileExistsError(f"拒绝覆盖 {path}；确认后加 --force")

    # Lazy imports let --help and static checks work off the training server.
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from specforge.data.preprocessing import (
        preprocess_conversations,
        process_token_dict_to_mappings,
    )
    from specforge.data.template import TEMPLATE_REGISTRY

    target_config = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
    text_config = getattr(target_config, "text_config", target_config)
    target_vocab_size = int(text_config.vocab_size)
    draft_json = json.loads(args.draft_config.read_text(encoding="utf-8"))
    draft_vocab_size = int(draft_json["draft_vocab_size"])
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    prompts = load_prompts(args.requests)
    template = TEMPLATE_REGISTRY.get("qwen3.5")

    temp_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    token_counts: Counter[int] = Counter()
    sequence_lengths: list[int] = []
    supervised_lengths: list[int] = []
    seen: set[str] = set()
    started = time.monotonic()

    with temp_output.open("w", encoding="utf-8") as output:
        for line_number, record in read_jsonl(args.input):
            record_id, text = record.get("id"), record.get("text")
            if not isinstance(record_id, str) or not isinstance(text, str):
                raise ValueError(f"{args.input}:{line_number} id/text 无效")
            if record_id in seen:
                raise ValueError(f"重放数据存在重复 ID: {record_id}")
            if record_id not in prompts:
                raise ValueError(f"请求文件缺少 ID: {record_id}")
            seen.add(record_id)
            context = f"id={record_id} line={line_number}"

            processed = preprocess_conversations(
                tokenizer=tokenizer,
                conversations=[text],
                chat_template=template,
                max_length=args.max_length,
                is_preformatted=True,
                train_only_last_turn=True,
                tools=[[]],
            )
            # Process each record independently. Passing all 30k as one batch was the
            # historical bug that collapsed the processed dataset to only 32 records.
            input_ids = [int(value) for value in processed["input_ids"][0].reshape(-1).tolist()]
            historical = [int(value) for value in processed["loss_mask"][0].reshape(-1).tolist()]
            if args.mask_policy == "target-faithful":
                mask = faithful_mask(
                    tokenizer, text, prompts[record_id], input_ids, args.max_length, context
                )
            else:
                mask = historical
            if len(input_ids) != len(mask) or sum(mask) < 1:
                raise ValueError(f"{context}: input_ids/loss_mask 无效")
            if min(input_ids) < 0 or max(input_ids) >= target_vocab_size:
                raise ValueError(f"{context}: token id 超出 target vocab")
            token_counts.update(token for token, selected in zip(input_ids, mask) if selected)
            output.write(
                json.dumps(
                    {"id": record_id, "input_ids": input_ids, "loss_mask": mask},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            sequence_lengths.append(len(input_ids))
            supervised_lengths.append(sum(mask))
            if len(seen) % 100 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"进度={len(seen)} 速度={len(seen) / elapsed:.1f}条/秒 "
                    f"序列={len(input_ids)} 监督={sum(mask)}",
                    flush=True,
                )

    if args.expected_records is not None and len(seen) != args.expected_records:
        temp_output.unlink(missing_ok=True)
        raise ValueError(f"记录数 {len(seen)} != expected {args.expected_records}")
    if set(prompts) != seen:
        temp_output.unlink(missing_ok=True)
        raise ValueError(f"重放与请求 ID 集不一致，差异={len(set(prompts) ^ seen)}")

    d2t, t2d = process_token_dict_to_mappings(
        token_counts, draft_vocab_size, target_vocab_size
    )
    temp_mapping = args.mapping_output.with_name(
        f".{args.mapping_output.name}.tmp-{os.getpid()}"
    )
    torch.save({"d2t": d2t, "t2d": t2d}, temp_mapping)
    os.replace(temp_output, args.output)
    os.replace(temp_mapping, args.mapping_output)

    report = {
        "records": len(seen),
        "mask_policy": args.mask_policy,
        "max_length": args.max_length,
        "sequence_tokens": {
            "p50": percentile(sequence_lengths, 0.50),
            "p90": percentile(sequence_lengths, 0.90),
            "p99": percentile(sequence_lengths, 0.99),
            "max": max(sequence_lengths),
        },
        "supervised_tokens": {
            "total": sum(supervised_lengths),
            "unique": len(token_counts),
            "p50": percentile(supervised_lengths, 0.50),
            "p90": percentile(supervised_lengths, 0.90),
            "min": min(supervised_lengths),
            "max": max(supervised_lengths),
        },
        "mapping": {"d2t": list(d2t.shape), "t2d": list(t2d.shape)},
        "sha256": {
            "input": sha256(args.input),
            "requests": sha256(args.requests),
            "output": sha256(args.output),
            "mapping": sha256(args.mapping_output),
        },
    }
    report_path = Path(str(args.output) + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"预分词: {args.output}\n词表映射: {args.mapping_output}\n报告: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
