#!/usr/bin/env python3
"""Render the checked-in EAGLE3 experiment registry as CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    "/data/home/leonardoqin/datasets/lingbao_eagle3_data"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT)),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a registered result directory is missing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, data_root: Path) -> Path:
    return Path(value.replace("{DATA_ROOT}", str(data_root)))


def extract(entry: dict[str, Any], data_root: Path) -> dict[str, Any]:
    result_dir = resolve_path(entry["result_dir"], data_root)
    baseline_path = result_dir / "baseline_c1.json"
    eagle_path = result_dir / "eagle3_c1.json"
    comparison_path = result_dir / "comparison_c1.json"
    observed = dict(entry.get("observed") or {})
    source = "registry"

    if baseline_path.is_file() and eagle_path.is_file() and comparison_path.is_file():
        baseline = read_json(baseline_path)
        eagle = read_json(eagle_path)
        comparison = read_json(comparison_path)
        acceptance = eagle.get("acceptance") or {}
        speedup = comparison.get("speedup") or {}
        reduction = comparison.get("reduction_pct") or {}
        observed.update(
            {
                "samples": eagle.get("successful"),
                "accept_length": acceptance.get("avg_accept_length"),
                "accept_rate": acceptance.get("avg_accept_rate"),
                "tpot_speedup": speedup.get("tpot_mean"),
                "output_speedup": speedup.get("output_tokens_per_s"),
                "request_speedup": speedup.get("requests_per_s"),
                "e2e_reduction_pct": (reduction.get("e2e") or {}).get("mean"),
                "ttft_reduction_pct": (reduction.get("ttft") or {}).get("mean"),
                "baseline_output_tps": (
                    baseline.get("throughput") or {}
                ).get("output_tokens_per_s"),
                "eagle_output_tps": (
                    eagle.get("throughput") or {}
                ).get("output_tokens_per_s"),
            }
        )
        source = "result_files"

    return {
        "id": entry["id"],
        "date": entry.get("date"),
        "phase": entry.get("phase"),
        "model": entry.get("model"),
        "data": entry.get("data"),
        "split": entry.get("split"),
        "temperature": entry.get("temperature"),
        "tree": entry.get("tree"),
        "diagnostic": bool(entry.get("diagnostic", False)),
        "samples": observed.get("samples"),
        "accept_length": observed.get("accept_length"),
        "accept_rate": observed.get("accept_rate"),
        "tpot_speedup": observed.get("tpot_speedup"),
        "output_speedup": observed.get("output_speedup"),
        "request_speedup": observed.get("request_speedup"),
        "e2e_reduction_pct": observed.get("e2e_reduction_pct"),
        "ttft_reduction_pct": observed.get("ttft_reduction_pct"),
        "result_dir": str(result_dir),
        "draft_model": entry.get("draft_model"),
        "source": source,
        "conclusion": entry.get("conclusion"),
    }


def display(value: Any, spec: str) -> str:
    if value is None:
        return "—"
    return format(float(value), spec)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# EAGLE-3 实验注册表",
        "",
        "> 自动生成。`source=result_files` 表示指标来自当前机器上的 JSON；"
        "`source=registry` 表示使用已登记的历史观测值。诊断轮的墙钟吞吐不用于验收。",
        "",
        "| 日期 | 实验ID | 模型/数据 | Split/T | Tree | N | ACC | 接受率 | TPOT | 输出 | E2E | 来源 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        diag = "（诊断）" if row["diagnostic"] else ""
        lines.append(
            "| {date} | `{id}` | {model} / {data} | {split} / {temp}{diag} | "
            "{tree} | {samples} | {accept} | {rate} | {tpot} | {output} | "
            "{e2e} | {source} |".format(
                date=row["date"] or "—",
                id=row["id"],
                model=row["model"] or "—",
                data=row["data"] or "—",
                split=row["split"] or "—",
                temp=row["temperature"] if row["temperature"] is not None else "—",
                diag=diag,
                tree=row["tree"] or "—",
                samples=row["samples"] or "—",
                accept=display(row["accept_length"], ".3f"),
                rate=display(row["accept_rate"], ".2%"),
                tpot=display(row["tpot_speedup"], ".3f") + "x",
                output=display(row["output_speedup"], ".3f") + "x",
                e2e=display(row["e2e_reduction_pct"], "+.2f") + "%",
                source=row["source"],
            )
        )
    lines.extend(["", "## 结果目录与结论", ""])
    for row in rows:
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"- 结果：`{row['result_dir']}`",
                f"- Draft：`{row['draft_model'] or '—'}`",
                f"- 结论：{row['conclusion'] or '—'}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = arguments()
    registry = read_json(args.registry)
    entries = registry.get("experiments") or []
    if not entries:
        raise SystemExit("Experiment registry is empty")

    rows = [extract(entry, args.data_root) for entry in entries]
    missing = [row for row in rows if row["source"] != "result_files"]
    if args.strict and missing:
        raise SystemExit(
            "Missing registered results: "
            + ", ".join(row["id"] for row in missing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "experiment_ledger.csv"
    md_path = args.output_dir / "experiment_ledger.md"
    json_path = args.output_dir / "experiment_ledger.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(rows) + "\n", encoding="utf-8")

    print(f"Experiments: {len(rows)}")
    print(f"Resolved from result files: {len(rows) - len(missing)}")
    print(f"Historical registry fallback: {len(missing)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
