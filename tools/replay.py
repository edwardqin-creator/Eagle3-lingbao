#!/usr/bin/env python3
"""Concurrent, resumable Target-model replay for SpecForge training data."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

EOS_TEXT = "<|im_end|>"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("run")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--error-output", type=Path, required=True)
    replay.add_argument("--servers", nargs="+", required=True)
    replay.add_argument("--concurrency-per-server", type=int, default=8)
    replay.add_argument("--max-retries", type=int, default=5)
    replay.add_argument("--limit", type=int)
    replay.add_argument("--progress-interval", type=int, default=50)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--requests", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            records.append(value)
    return records


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record["id"]) for record in load_jsonl(path) if record.get("id")}


def request_body(record: dict[str, Any]) -> dict[str, Any]:
    sampling = record.get("sampling_params")
    if not isinstance(sampling, dict):
        raise ValueError(f"id={record.get('id')} 缺少 sampling_params")
    required = ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty")
    missing = [key for key in required if key not in sampling]
    if missing:
        raise ValueError(f"id={record.get('id')} 缺少采样参数 {missing}")
    body = {
        "model": record["model"],
        "prompt": record["prompt"],
        "max_tokens": sampling["max_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "repetition_penalty": sampling["repetition_penalty"],
        "stop": sampling.get("stop", [EOS_TEXT]),
        "stream": False,
    }
    # Missing seed means "let the server choose". Never silently replace it with 0.
    if sampling.get("seed") is not None:
        body["seed"] = sampling["seed"]
    return body


def training_record(
    source: dict[str, Any], completion: str, finish_reason: str | None
) -> dict[str, Any]:
    if EOS_TEXT in completion:
        completion = completion.split(EOS_TEXT, 1)[0]
    if not completion.strip():
        raise ValueError("Target 返回空 completion")
    return {
        "id": source["id"],
        "text": source["prompt"] + completion + EOS_TEXT + "\n",
        "completion": completion,
        "finish_reason": finish_reason,
    }


async def generate(
    session: aiohttp.ClientSession, server: str, record: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    async with session.post(
        server.rstrip("/") + "/v1/completions", json=request_body(record)
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
        value = json.loads(body)
        choices = value.get("choices") or []
        if not choices or not isinstance(choices[0].get("text"), str):
            raise RuntimeError(f"响应没有 choices[0].text: {body[:500]}")
        reason = choices[0].get("finish_reason")
        return training_record(record, choices[0]["text"], reason), reason


def duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


async def run_replay(args: argparse.Namespace) -> int:
    import aiohttp

    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    done = completed_ids(args.output)
    pending = [record for record in records if str(record.get("id")) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.error_output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"总数={len(records)} 已完成={len(done)} 待处理={len(pending)} "
        f"服务={len(args.servers)} 总并发={len(args.servers) * args.concurrency_per_server}",
        flush=True,
    )
    if not pending:
        return 0

    timeout = aiohttp.ClientTimeout(total=900, connect=30, sock_read=900)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    for record in pending:
        queue.put_nowait(record)
    worker_count = len(args.servers) * args.concurrency_per_server
    for _ in range(worker_count):
        queue.put_nowait(None)

    output_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    stats: Counter[str] = Counter()
    started = time.monotonic()

    with args.output.open("a", encoding="utf-8", buffering=1) as output_file, args.error_output.open(
        "w", encoding="utf-8", buffering=1
    ) as error_file:
        async with aiohttp.ClientSession(timeout=timeout) as session:

            async def worker(server: str) -> None:
                while True:
                    record = await queue.get()
                    if record is None:
                        queue.task_done()
                        return
                    last_error = "unknown"
                    for attempt in range(1, args.max_retries + 1):
                        try:
                            result, reason = await generate(session, server, record)
                            async with output_lock:
                                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            async with state_lock:
                                stats["success"] += 1
                                stats[f"finish_{reason or 'other'}"] += 1
                            last_error = ""
                            break
                        except Exception as error:
                            last_error = f"{type(error).__name__}: {error}"
                            if attempt < args.max_retries:
                                await asyncio.sleep(min(2 ** (attempt - 1), 16))
                    if last_error:
                        async with output_lock:
                            error_file.write(
                                json.dumps(
                                    {"id": record.get("id"), "server": server, "error": last_error},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        async with state_lock:
                            stats["error"] += 1
                    async with state_lock:
                        finished = stats["success"] + stats["error"]
                        if finished % args.progress_interval == 0 or finished == len(pending):
                            elapsed = max(time.monotonic() - started, 1e-9)
                            speed = finished / elapsed
                            eta = (len(pending) - finished) / speed
                            print(
                                f"进度 {finished}/{len(pending)} 成功={stats['success']} "
                                f"失败={stats['error']} length={stats['finish_length']} "
                                f"速度={speed:.2f}条/秒 已用={duration(elapsed)} ETA={duration(eta)}",
                                flush=True,
                            )
                    queue.task_done()

            tasks = []
            for server in args.servers:
                tasks.extend(
                    asyncio.create_task(worker(server))
                    for _ in range(args.concurrency_per_server)
                )
            await queue.join()
            await asyncio.gather(*tasks)

    run_report = {
        "input": str(args.input),
        "output": str(args.output),
        "servers": args.servers,
        "input_records": len(records),
        "already_completed": len(done),
        "attempted_this_run": len(pending),
        "statistics": dict(stats),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = Path(str(args.output) + ".run_report.json")
    report_path.write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_report, ensure_ascii=False, indent=2))
    print(f"运行报告: {report_path}")
    return 1 if stats["error"] else 0


def validate(args: argparse.Namespace) -> int:
    requests = load_jsonl(args.requests)
    outputs = load_jsonl(args.output)
    request_ids = [str(record.get("id")) for record in requests]
    output_ids = [str(record.get("id")) for record in outputs]
    duplicates = len(output_ids) - len(set(output_ids))
    missing = set(request_ids) - set(output_ids)
    extra = set(output_ids) - set(request_ids)
    empty = sum(not str(record.get("completion", "")).strip() for record in outputs)
    truncated = sum(record.get("finish_reason") == "length" for record in outputs)
    summary = {
        "requests": len(requests),
        "outputs": len(outputs),
        "duplicate_ids": duplicates,
        "missing_ids": len(missing),
        "extra_ids": len(extra),
        "empty_completions": empty,
        "finish_length": truncated,
    }
    report_path = Path(str(args.output) + ".validation.json")
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    passed = not any((duplicates, missing, extra, empty, truncated))
    print("重放校验通过" if passed else "重放校验失败")
    print(f"校验报告: {report_path}")
    return 0 if passed else 1


def main() -> int:
    args = arguments()
    if args.command == "validate":
        return validate(args)
    return asyncio.run(run_replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
