#!/usr/bin/env python3
"""Compare task-family reports from two frozen benchmark splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--val-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default="all_pairs",
        choices=("all_pairs", "same_output_tokens", "same_output_sha256"),
    )
    return parser.parse_args()


def read(path: Path, scope: str) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = {
        str(record["task_type"]): record
        for record in records
        if record.get("scope") == scope
    }
    if not selected:
        raise ValueError(f"{path} has no rows for scope={scope}")
    return selected


def delta(train: Any, val: Any) -> float | None:
    if train is None or val is None:
        return None
    return float(train) - float(val)


def display(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "—"
    return format(float(value), spec)


def markdown(rows: list[dict[str, Any]], scope: str) -> str:
    lines = [
        "# Train / Validation 任务族对比",
        "",
        f"口径：`{scope}`。正的 ACC 差表示 Train 高于 Validation；"
        "二者接近时不支持明显记忆/过拟合。",
        "",
        "| 任务族 | Train N | Val N | Train ACC | Val ACC | ACC差 | "
        "Train TPOT | Val TPOT | Train E2E | Val E2E |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {train_n} | {val_n} | {train_acc} | {val_acc} | "
            "{acc_delta} | {train_tpot}x | {val_tpot}x | {train_e2e}% | "
            "{val_e2e}% |".format(
                task=row["task_type"],
                train_n=row["train_count"],
                val_n=row["val_count"],
                train_acc=display(row["train_accept_length"]),
                val_acc=display(row["val_accept_length"]),
                acc_delta=display(row["accept_length_delta"], "+.3f"),
                train_tpot=display(row["train_tpot_speedup"]),
                val_tpot=display(row["val_tpot_speedup"]),
                train_e2e=display(row["train_e2e_reduction_pct"], "+.2f"),
                val_e2e=display(row["val_e2e_reduction_pct"], "+.2f"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = arguments()
    train = read(args.train_summary, args.scope)
    val = read(args.val_summary, args.scope)
    task_types = sorted(set(train) | set(val))
    rows: list[dict[str, Any]] = []
    for task_type in task_types:
        train_row = train.get(task_type, {})
        val_row = val.get(task_type, {})
        train_acc = train_row.get("accept_length_mean")
        val_acc = val_row.get("accept_length_mean")
        rows.append(
            {
                "scope": args.scope,
                "task_type": task_type,
                "train_count": train_row.get("count", 0),
                "val_count": val_row.get("count", 0),
                "train_accept_length": train_acc,
                "val_accept_length": val_acc,
                "accept_length_delta": delta(train_acc, val_acc),
                "train_accept_length_p50": train_row.get("accept_length_p50"),
                "val_accept_length_p50": val_row.get("accept_length_p50"),
                "train_accept_length_p90": train_row.get("accept_length_p90"),
                "val_accept_length_p90": val_row.get("accept_length_p90"),
                "train_tpot_speedup": train_row.get("tpot_speedup"),
                "val_tpot_speedup": val_row.get("tpot_speedup"),
                "train_output_speedup_c1": train_row.get("output_speedup_c1"),
                "val_output_speedup_c1": val_row.get("output_speedup_c1"),
                "train_e2e_reduction_pct": train_row.get("e2e_reduction_pct"),
                "val_e2e_reduction_pct": val_row.get("e2e_reduction_pct"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"train_val_{args.scope}.json"
    csv_path = args.output_dir / f"train_val_{args.scope}.csv"
    md_path = args.output_dir / f"train_val_{args.scope}.md"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown(rows, args.scope), encoding="utf-8")
    print(f"Task families: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
