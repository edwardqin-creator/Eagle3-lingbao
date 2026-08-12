#!/usr/bin/env python3
"""Join benchmark details with Data V2 metadata and produce filterable reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--eagle-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val", "validation", "test"),
        help="Dataset split; val and validation are treated as aliases.",
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=4,
        help="Serving speculative-num-draft-tokens; used for diagnostic estimates.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def by_id(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("id"))
        if not record_id or record_id == "None":
            raise ValueError(f"{name} contains a record without id")
        if record_id in result:
            raise ValueError(f"{name} contains duplicate id={record_id}")
        result[record_id] = record
    return result


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = ratio * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def reduction(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def last_user_text(prompt: str) -> str:
    matches = re.findall(r"用户：([^\n]*)", prompt)
    if matches:
        return matches[-1].strip()[:120]
    matches = re.findall(
        r"<\|im_start\|>user\s*(.*?)<\|im_end\|>",
        prompt,
        flags=re.DOTALL,
    )
    if matches:
        return re.sub(r"\s+", " ", matches[-1]).strip()[:120]
    return ""


def request_acceptance(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("request_acceptance") or {}
    return value if isinstance(value, dict) else {}


def normalize_split(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "validation": "val",
        "valid": "val",
        "dev": "val",
    }.get(normalized, normalized)


def weighted_tpot_ms(rows: list[dict[str, Any]], prefix: str) -> float | None:
    decode_seconds = 0.0
    decode_tokens = 0
    for row in rows:
        tokens = int(row.get(f"{prefix}_output_tokens") or 0) - 1
        seconds = row.get(f"{prefix}_decode_s")
        if tokens > 0 and seconds is not None:
            decode_seconds += float(seconds)
            decode_tokens += tokens
    return decode_seconds * 1000 / decode_tokens if decode_tokens else None


def accept_bucket(value: Any) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if number < 1.75:
        return "[1.00,1.75)"
    if number < 2.00:
        return "[1.75,2.00)"
    if number < 2.25:
        return "[2.00,2.25)"
    if number < 2.50:
        return "[2.25,2.50)"
    return "[2.50,3.00]"


def output_bucket(value: Any) -> str:
    number = int(value or 0)
    if number <= 16:
        return "<=16"
    if number <= 24:
        return "17-24"
    if number <= 32:
        return "25-32"
    return ">=33"


def aggregate(rows: list[dict[str, Any]], scope: str, task_type: str) -> dict[str, Any]:
    baseline_tpot = weighted_tpot_ms(rows, "baseline")
    eagle_tpot = weighted_tpot_ms(rows, "eagle")
    baseline_e2e = [float(row["baseline_e2e_ms"]) for row in rows]
    eagle_e2e = [float(row["eagle_e2e_ms"]) for row in rows]
    baseline_ttft = [float(row["baseline_ttft_ms"]) for row in rows]
    eagle_ttft = [float(row["eagle_ttft_ms"]) for row in rows]
    accept_lengths = [
        float(row["accept_length"])
        for row in rows
        if row.get("accept_length") is not None
    ]
    weighted_accept_numerator = sum(
        float(row["accept_length"]) * int(row.get("accept_log_samples") or 0)
        for row in rows
        if row.get("accept_length") is not None
        and int(row.get("accept_log_samples") or 0) > 0
    )
    weighted_accept_denominator = sum(
        int(row.get("accept_log_samples") or 0)
        for row in rows
        if row.get("accept_length") is not None
    )
    baseline_output = sum(int(row["baseline_output_tokens"]) for row in rows)
    eagle_output = sum(int(row["eagle_output_tokens"]) for row in rows)
    baseline_e2e_s = sum(value / 1000 for value in baseline_e2e)
    eagle_e2e_s = sum(value / 1000 for value in eagle_e2e)
    return {
        "scope": scope,
        "task_type": task_type,
        "count": len(rows),
        "acceptance_count": len(accept_lengths),
        "accept_length_mean": statistics.fmean(accept_lengths) if accept_lengths else None,
        "accept_length_decode_weighted_mean": (
            weighted_accept_numerator / weighted_accept_denominator
            if weighted_accept_denominator
            else None
        ),
        "accept_length_p50": percentile(accept_lengths, 0.50),
        "accept_length_p90": percentile(accept_lengths, 0.90),
        "accept_length_p95": percentile(accept_lengths, 0.95),
        "accept_length_p99": percentile(accept_lengths, 0.99),
        "baseline_e2e_mean_ms": statistics.fmean(baseline_e2e),
        "eagle_e2e_mean_ms": statistics.fmean(eagle_e2e),
        "e2e_reduction_pct": reduction(
            statistics.fmean(baseline_e2e), statistics.fmean(eagle_e2e)
        ),
        "baseline_e2e_p50_ms": percentile(baseline_e2e, 0.50),
        "eagle_e2e_p50_ms": percentile(eagle_e2e, 0.50),
        "baseline_ttft_mean_ms": statistics.fmean(baseline_ttft),
        "eagle_ttft_mean_ms": statistics.fmean(eagle_ttft),
        "ttft_reduction_pct": reduction(
            statistics.fmean(baseline_ttft), statistics.fmean(eagle_ttft)
        ),
        "baseline_weighted_tpot_ms": baseline_tpot,
        "eagle_weighted_tpot_ms": eagle_tpot,
        "tpot_speedup": ratio(baseline_tpot, eagle_tpot),
        "baseline_output_tokens_per_s_c1": ratio(baseline_output, baseline_e2e_s),
        "eagle_output_tokens_per_s_c1": ratio(eagle_output, eagle_e2e_s),
        "output_speedup_c1": ratio(
            ratio(eagle_output, eagle_e2e_s), ratio(baseline_output, baseline_e2e_s)
        ),
        "baseline_output_tokens_mean": baseline_output / len(rows),
        "eagle_output_tokens_mean": eagle_output / len(rows),
    }


def main() -> int:
    args = arguments()
    manifest = by_id(read_jsonl(args.manifest), "manifest")
    requests = by_id(read_jsonl(args.requests), "requests")
    baseline = by_id(read_jsonl(args.baseline_details), "baseline details")
    eagle = by_id(read_jsonl(args.eagle_details), "eagle details")
    common_ids = sorted(set(baseline) & set(eagle) & set(requests) & set(manifest))
    if not common_ids:
        raise SystemExit("No common request IDs were found")

    rows: list[dict[str, Any]] = []
    for request_id in common_ids:
        meta = manifest[request_id]
        if args.split and normalize_split(meta.get("split")) != normalize_split(args.split):
            continue
        b = baseline[request_id]
        e = eagle[request_id]
        if not b.get("success") or not e.get("success"):
            continue
        acceptance = request_acceptance(e)
        accept_samples = acceptance.get("decode_log_samples")
        accept_length = acceptance.get("avg_accept_length")
        spec_verify_ct = int(accept_samples) if accept_samples else None
        estimated_accepted_tokens = (
            float(accept_length) * spec_verify_ct
            if accept_length is not None and spec_verify_ct is not None
            else None
        )
        estimated_correct_drafts = (
            (float(accept_length) - 1.0) * spec_verify_ct
            if accept_length is not None and spec_verify_ct is not None
            else None
        )
        estimated_proposed_drafts = (
            (args.draft_tokens - 1) * spec_verify_ct
            if spec_verify_ct is not None
            else None
        )
        b_tpot = b.get("tpot_ms")
        e_tpot = e.get("tpot_ms")
        b_e2e_ms = float(b["e2e_s"]) * 1000
        e_e2e_ms = float(e["e2e_s"]) * 1000
        b_ttft_ms = float(b["ttft_s"]) * 1000
        e_ttft_ms = float(e["ttft_s"]) * 1000
        b_output = int(b.get("output_tokens") or 0)
        e_output = int(e.get("output_tokens") or 0)
        baseline_output_tps = ratio(b_output * 1000, b_e2e_ms)
        eagle_output_tps = ratio(e_output * 1000, e_e2e_ms)
        same_hash = (
            bool(b.get("output_sha256"))
            and b.get("output_sha256") == e.get("output_sha256")
        )
        rows.append(
            {
                "request_id": request_id,
                "split": meta.get("split"),
                "task_type": meta.get("task_type", "unknown"),
                "chat_only": meta.get("chat_only"),
                "source_line": meta.get("source_line"),
                "prompt_sha256": meta.get("prompt_sha256"),
                "last_user_text": last_user_text(str(requests[request_id].get("prompt", ""))),
                "input_tokens": b.get("input_tokens"),
                "baseline_output_tokens": b_output,
                "eagle_output_tokens": e_output,
                "same_output_tokens": b_output == e_output,
                "same_output_sha256": same_hash,
                "baseline_ttft_ms": b_ttft_ms,
                "eagle_ttft_ms": e_ttft_ms,
                "ttft_delta_ms": e_ttft_ms - b_ttft_ms,
                "ttft_speedup": ratio(b_ttft_ms, e_ttft_ms),
                "ttft_reduction_pct": reduction(b_ttft_ms, e_ttft_ms),
                "baseline_tpot_ms": b_tpot,
                "eagle_tpot_ms": e_tpot,
                "tpot_speedup": ratio(b_tpot, e_tpot),
                "baseline_decode_tokens_per_s": ratio(1000, b_tpot),
                "eagle_decode_tokens_per_s": ratio(1000, e_tpot),
                "baseline_e2e_ms": b_e2e_ms,
                "eagle_e2e_ms": e_e2e_ms,
                "e2e_speedup": ratio(b_e2e_ms, e_e2e_ms),
                "e2e_reduction_pct": reduction(b_e2e_ms, e_e2e_ms),
                "baseline_request_output_tokens_per_s": baseline_output_tps,
                "eagle_request_output_tokens_per_s": eagle_output_tps,
                "request_output_speedup": ratio(eagle_output_tps, baseline_output_tps),
                "baseline_decode_s": b.get("decode_s"),
                "eagle_decode_s": e.get("decode_s"),
                "accept_log_samples": accept_samples,
                "spec_verify_ct": spec_verify_ct,
                "estimated_proposed_drafts": estimated_proposed_drafts,
                "estimated_correct_drafts": estimated_correct_drafts,
                "estimated_accepted_tokens": estimated_accepted_tokens,
                "accept_length": accept_length,
                "accept_length_p50": acceptance.get("p50_accept_length"),
                "accept_length_p90": acceptance.get("p90_accept_length"),
                "accept_length_p95": acceptance.get("p95_accept_length"),
                "accept_length_p99": acceptance.get("p99_accept_length"),
                "accept_rate": acceptance.get("avg_accept_rate"),
                "accept_bucket": accept_bucket(accept_length),
                "output_length_bucket": output_bucket(e_output),
                "output_preview": e.get("output_preview"),
            }
        )

    if not rows:
        manifest_split_counts: dict[str, int] = defaultdict(int)
        for item in manifest.values():
            manifest_split_counts[str(item.get("split"))] += 1
        raise SystemExit(
            "No successful paired requests remained after split filtering. "
            f"requested_split={args.split!r}; "
            f"manifest_split_counts={dict(manifest_split_counts)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_jsonl = args.output_dir / "request_analysis.jsonl"
    request_csv = args.output_dir / "request_analysis.csv"
    with request_jsonl.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with request_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for scope, scoped_rows in (
        ("all_pairs", rows),
        ("same_output_tokens", [row for row in rows if row["same_output_tokens"]]),
        ("same_output_sha256", [row for row in rows if row["same_output_sha256"]]),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scoped_rows:
            grouped[str(row["task_type"])].append(row)
        for task_type, task_rows in sorted(grouped.items()):
            summaries.append(aggregate(task_rows, scope, task_type))
        for dimension, field in (
            ("overall", None),
            ("task_type", "task_type"),
            ("accept_bucket", "accept_bucket"),
            ("output_length_bucket", "output_length_bucket"),
        ):
            dimension_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in scoped_rows:
                value = "ALL" if field is None else str(row[field])
                dimension_groups[value].append(row)
            for value, group_rows in sorted(dimension_groups.items()):
                item = aggregate(group_rows, scope, value)
                item["dimension"] = dimension
                item["group"] = item.pop("task_type")
                group_summaries.append(item)

    summary_json = args.output_dir / "task_family_summary.json"
    summary_csv = args.output_dir / "task_family_summary.csv"
    summary_json.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    group_json = args.output_dir / "request_group_summary.json"
    group_csv = args.output_dir / "request_group_summary.csv"
    group_json.write_text(
        json.dumps(group_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with group_csv.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(group_summaries[0]))
        writer.writeheader()
        writer.writerows(group_summaries)

    print(f"Common successful requests: {len(rows)}")
    print(f"Request report: {request_csv}")
    print(f"Task-family report: {summary_csv}")
    print(f"Grouped request report: {group_csv}")
    if not any(row.get("accept_length") is not None for row in rows):
        print(
            "Note: current details have no per-request acceptance. "
            "Run a request-diagnostics benchmark to populate it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
