#!/usr/bin/env python3
"""Fair streaming benchmark and Baseline/EAGLE3 comparison."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

ACCEPT_RE = re.compile(
    r"#running-req:\s*(\d+).*?accept len:\s*([0-9.]+),\s*accept rate:\s*([0-9.]+)"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--mode", choices=("baseline", "eagle3"), required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--server-log", type=Path)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--limit", type=int, default=500)
    run.add_argument("--warmup", type=int, default=8)
    run.add_argument("--sample-seed", type=int, default=20260723)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--fixed-output-tokens", type=int)
    run.add_argument("--ignore-eos", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--eagle3", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("prompt"), str):
                raise ValueError(f"{path}:{line_number} 缺少 prompt")
            records.append(record)
    return records


def stable_seed(record_id: str) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def payload(record: dict[str, Any], args: argparse.Namespace, warmup: bool) -> dict[str, Any]:
    sampling = dict(record.get("sampling_params") or {})
    body = {
        "model": args.model,
        "prompt": record["prompt"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": sampling.get("max_tokens", 200),
        "temperature": sampling.get("temperature", 0.9),
        "top_p": sampling.get("top_p", 0.8),
        "top_k": sampling.get("top_k", 20),
        "repetition_penalty": sampling.get("repetition_penalty", 1.05),
        # A/B 必须使用同一个随机数流。原数据缺 seed 时，用 ID 生成稳定 seed。
        "seed": sampling.get("seed", stable_seed(str(record.get("id", "missing")))),
    }
    if sampling.get("stop") is not None:
        body["stop"] = sampling["stop"]
    if args.fixed_output_tokens is not None:
        body["max_tokens"] = args.fixed_output_tokens
    if args.ignore_eos:
        body["ignore_eos"] = True
        body.pop("stop", None)
    if warmup:
        body["max_tokens"] = min(int(body["max_tokens"]), 32)
    return body


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("mean", "p50", "p90", "p95", "p99")} | {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def display(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


async def request_once(
    session: aiohttp.ClientSession,
    endpoint: str,
    record: dict[str, Any],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    warmup: bool,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(args.max_retries + 1):
        async with semaphore:
            started = time.perf_counter()
            first_token_at: float | None = None
            pieces: list[str] = []
            usage: dict[str, Any] = {}
            finish_reason = None
            try:
                async with session.post(endpoint, json=payload(record, args, warmup)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
                    async for raw_line in response.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        content = line[5:].strip()
                        if not content or content == "[DONE]":
                            continue
                        event = json.loads(content)
                        if event.get("usage"):
                            usage = event["usage"]
                        choices = event.get("choices") or []
                        if choices:
                            text = choices[0].get("text") or ""
                            if text:
                                first_token_at = first_token_at or time.perf_counter()
                                pieces.append(text)
                            finish_reason = choices[0].get("finish_reason") or finish_reason
                finished = time.perf_counter()
                first_token_at = first_token_at or finished
                return {
                    "id": str(record.get("id")),
                    "success": True,
                    "error": None,
                    "e2e_s": finished - started,
                    "ttft_s": first_token_at - started,
                    "decode_s": finished - first_token_at,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "finish_reason": finish_reason,
                    "_prompt": record["prompt"],
                    "_text": "".join(pieces),
                }
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
        if attempt < args.max_retries:
            await asyncio.sleep(min(2**attempt, 5))
    return {"id": str(record.get("id")), "success": False, "error": last_error}


async def request_many(
    session: aiohttp.ClientSession,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    warmup: bool,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrency)
    endpoint = args.url.rstrip("/") + "/v1/completions"
    tasks = [
        asyncio.create_task(request_once(session, endpoint, record, args, semaphore, warmup))
        for record in records
    ]
    if warmup:
        return await asyncio.gather(*tasks)
    results = []
    for future in asyncio.as_completed(tasks):
        results.append(await future)
        if len(results) % 10 == 0 or len(results) == len(tasks):
            passed = sum(record["success"] for record in results)
            print(f"进度 {len(results)}/{len(tasks)} 成功={passed} 失败={len(results)-passed}", flush=True)
    return results


def log_offset(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def acceptance(path: Path | None, offset: int, mode: str) -> dict[str, Any]:
    if mode == "baseline":
        return {"decode_log_samples": 0, "avg_accept_length": 1.0, "avg_accept_rate": 0.0}
    if path is None or not path.exists():
        return {"decode_log_samples": 0, "avg_accept_length": None, "avg_accept_rate": None}
    with path.open("rb") as source:
        source.seek(offset)
        text = source.read().decode("utf-8", errors="replace")
    matches = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in ACCEPT_RE.finditer(text)]
    matches = [item for item in matches if item[0] > 0]
    if not matches:
        return {"decode_log_samples": 0, "avg_accept_length": None, "avg_accept_rate": None}
    weight = sum(item[0] for item in matches)
    avg_length = sum(item[0] * item[1] for item in matches) / weight
    avg_rate = sum(item[0] * item[2] for item in matches) / weight
    return {
        "decode_log_samples": len(matches),
        "avg_accept_length": avg_length,
        "p50_accept_length": percentile([item[1] for item in matches], 0.5),
        "p90_accept_length": percentile([item[1] for item in matches], 0.9),
        "avg_accept_rate": avg_rate,
        "estimated_target_step_reduction": 1 - 1 / avg_length,
    }


async def run_benchmark(args: argparse.Namespace) -> int:
    import aiohttp
    from transformers import AutoTokenizer

    records = read_jsonl(args.input)
    indexes = list(range(len(records)))
    random.Random(args.sample_seed).shuffle(indexes)
    selected = [records[index] for index in indexes[: args.limit]]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=30, sock_read=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(args.url.rstrip("/") + "/health") as response:
            response.raise_for_status()
        warmup = selected[: min(args.warmup, len(selected))]
        print(f"预热={len(warmup)} 正式样本={len(selected)} 并发={args.concurrency}")
        await request_many(session, warmup, args, True)
        async with session.post(args.url.rstrip("/") + "/flush_cache") as response:
            response.raise_for_status()
        offset = log_offset(args.server_log)
        started = time.perf_counter()
        results = await request_many(session, selected, args, False)
        duration_s = time.perf_counter() - started

    for result in results:
        if not result["success"]:
            continue
        if result["input_tokens"] is None:
            result["input_tokens"] = len(tokenizer.encode(result["_prompt"], add_special_tokens=False))
        if result["output_tokens"] is None:
            result["output_tokens"] = len(tokenizer.encode(result["_text"], add_special_tokens=False))
        output_tokens = int(result["output_tokens"])
        result["tpot_ms"] = (
            result["decode_s"] / (output_tokens - 1) * 1000 if output_tokens > 1 else None
        )

    successful = [result for result in results if result["success"]]
    total_output = sum(int(result["output_tokens"]) for result in successful)
    total_input = sum(int(result["input_tokens"]) for result in successful)
    result_summary = {
        "name": args.name,
        "mode": args.mode,
        "input": str(args.input),
        "sample_seed": args.sample_seed,
        "concurrency": args.concurrency,
        "requested": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": duration_s,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "throughput": {
            "requests_per_s": len(successful) / duration_s,
            "input_tokens_per_s": total_input / duration_s,
            "output_tokens_per_s": total_output / duration_s,
        },
        "e2e_ms": summary([result["e2e_s"] * 1000 for result in successful]),
        "ttft_ms": summary([result["ttft_s"] * 1000 for result in successful]),
        "tpot_ms": summary(
            [result["tpot_ms"] for result in successful if result["tpot_ms"] is not None]
        ),
        "acceptance": acceptance(args.server_log, offset, args.mode),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n")
    details = args.output.with_suffix(".details.jsonl")
    with details.open("w", encoding="utf-8") as output:
        for result in sorted(results, key=lambda item: item["id"]):
            result.pop("_prompt", None)
            result.pop("_text", None)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    a = result_summary["acceptance"]
    print("\n========== Benchmark ==========")
    print(f"{args.name}: {len(successful)}/{len(results)} 成功")
    print(f"吞吐: {result_summary['throughput']['requests_per_s']:.3f} req/s, {result_summary['throughput']['output_tokens_per_s']:.2f} tok/s")
    print(
        "E2E p50/p90/p99: "
        f"{display(result_summary['e2e_ms']['p50'])}/"
        f"{display(result_summary['e2e_ms']['p90'])}/"
        f"{display(result_summary['e2e_ms']['p99'])} ms"
    )
    print(
        f"TTFT p50: {display(result_summary['ttft_ms']['p50'])} ms; "
        f"TPOT p50: {display(result_summary['tpot_ms']['p50'])} ms/token"
    )
    print(f"接受长度/接受率: {a.get('avg_accept_length')} / {a.get('avg_accept_rate')}")
    print(f"结果: {args.output}")
    return 0 if len(successful) == len(results) else 1


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100 if baseline else float("nan")


def compare(args: argparse.Namespace) -> int:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    eagle = json.loads(args.eagle3.read_text(encoding="utf-8"))
    if baseline["concurrency"] != eagle["concurrency"] or baseline["sample_seed"] != eagle["sample_seed"]:
        raise ValueError("A/B 的 concurrency 或 sample_seed 不一致")
    output = {
        "concurrency": baseline["concurrency"],
        "requests": eagle["requested"],
        "success": {"baseline": baseline["successful"], "eagle3": eagle["successful"]},
        "acceptance": eagle["acceptance"],
        "speedup": {
            "requests_per_s": ratio(eagle["throughput"]["requests_per_s"], baseline["throughput"]["requests_per_s"]),
            "output_tokens_per_s": ratio(eagle["throughput"]["output_tokens_per_s"], baseline["throughput"]["output_tokens_per_s"]),
            "tpot_mean": ratio(baseline["tpot_ms"]["mean"], eagle["tpot_ms"]["mean"]),
        },
        "reduction_pct": {
            metric: {
                quantile: reduction(baseline[f"{metric}_ms"][quantile], eagle[f"{metric}_ms"][quantile])
                for quantile in ("mean", "p50", "p90", "p99")
            }
            for metric in ("e2e", "ttft", "tpot")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("\n========== Baseline vs EAGLE3 ==========")
    print(f"并发/请求数: {output['concurrency']} / {output['requests']}")
    print(f"平均接受长度/接受率: {output['acceptance'].get('avg_accept_length')} / {output['acceptance'].get('avg_accept_rate')}")
    print(f"E2E mean降低: {output['reduction_pct']['e2e']['mean']:+.2f}%")
    print(f"TTFT mean降低: {output['reduction_pct']['ttft']['mean']:+.2f}%")
    print(f"TPOT mean加速: {output['speedup']['tpot_mean']:.3f}x")
    print(f"请求/输出吞吐加速: {output['speedup']['requests_per_s']:.3f}x / {output['speedup']['output_tokens_per_s']:.3f}x")
    print(f"结果: {args.output}")
    return 0


def main() -> int:
    args = arguments()
    if args.command == "compare":
        return compare(args)
    if args.ignore_eos and args.fixed_output_tokens is None:
        raise ValueError("--ignore-eos 必须配合 --fixed-output-tokens")
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    raise SystemExit(main())
