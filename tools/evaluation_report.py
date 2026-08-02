#!/usr/bin/env python3
"""Aggregate comparison files and apply a transparent acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    parser.add_argument("--min-accept-length", type=float, default=2.0)
    parser.add_argument("--min-output-speedup", type=float, default=1.0)
    parser.add_argument("--max-e2e-regression-pct", type=float, default=5.0)
    args = parser.parse_args()

    rows = []
    for concurrency in args.concurrencies:
        path = args.result_dir / f"comparison_c{concurrency}.json"
        item = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "concurrency": concurrency,
                "accept_length": item["acceptance"].get("avg_accept_length"),
                "accept_rate": item["acceptance"].get("avg_accept_rate"),
                "e2e_reduction_pct": item["reduction_pct"]["e2e"]["mean"],
                "ttft_reduction_pct": item["reduction_pct"]["ttft"]["mean"],
                "tpot_speedup": item["speedup"]["tpot_mean"],
                "request_speedup": item["speedup"]["requests_per_s"],
                "output_speedup": item["speedup"]["output_tokens_per_s"],
                "all_success": item["success"]["baseline"] == item["requests"]
                and item["success"]["eagle3"] == item["requests"],
            }
        )

    failures = []
    for row in rows:
        prefix = f"c{row['concurrency']}"
        if not row["all_success"]:
            failures.append(f"{prefix}: 存在失败请求")
        if row["accept_length"] is None or row["accept_length"] < args.min_accept_length:
            failures.append(
                f"{prefix}: 接受长度 {row['accept_length']} < {args.min_accept_length}"
            )
        if row["output_speedup"] < args.min_output_speedup:
            failures.append(
                f"{prefix}: 输出吞吐加速 {row['output_speedup']:.3f}x < {args.min_output_speedup:.3f}x"
            )
        if row["e2e_reduction_pct"] < -args.max_e2e_regression_pct:
            failures.append(
                f"{prefix}: E2E 回退 {-row['e2e_reduction_pct']:.2f}% > {args.max_e2e_regression_pct:.2f}%"
            )

    report = {
        "result_dir": str(args.result_dir),
        "thresholds": {
            "min_accept_length": args.min_accept_length,
            "min_output_speedup": args.min_output_speedup,
            "max_e2e_regression_pct": args.max_e2e_regression_pct,
        },
        "rows": rows,
        "passed": not failures,
        "failures": failures,
        "note": "接受长度是必要诊断指标，不是性能收益；最终门槛同时检查吞吐和 E2E。",
    }
    output = args.result_dir / "ACCEPTANCE_REPORT.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("\n并发  接受长度  接受率  TPOT加速  输出吞吐加速  E2E降低")
    for row in rows:
        accept_length = (
            "N/A" if row["accept_length"] is None else f"{row['accept_length']:.3f}"
        )
        accept_rate = (
            "N/A" if row["accept_rate"] is None else f"{row['accept_rate']:.2%}"
        )
        print(
            f"{row['concurrency']:<5} {accept_length:<9} "
            f"{accept_rate:<7} {row['tpot_speedup']:<9.3f} "
            f"{row['output_speedup']:<13.3f} {row['e2e_reduction_pct']:+.2f}%"
        )
    print("\n验收结论:", "PASS" if not failures else "FAIL")
    for failure in failures:
        print("-", failure)
    print("报告:", output)
    # A failed quality gate is a valid benchmark outcome, not a script crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
