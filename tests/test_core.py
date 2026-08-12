from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_data", ROOT / "scripts" / "01_prepare_data.py")
benchmark = load("benchmark", ROOT / "tools" / "benchmark.py")
analysis = load("analysis", ROOT / "tools" / "analyze_task_requests.py")
ledger = load("ledger", ROOT / "tools" / "experiment_ledger.py")
split_compare = load("split_compare", ROOT / "tools" / "compare_task_splits.py")


class PrepareDataTests(unittest.TestCase):
    def test_extract_payload_and_optional_seed(self):
        payload = {
            "prompt": "<|im_start|>user\n你好<|im_end|>",
            "model": "kaiwu-llm-model",
            "max_tokens": 200,
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "stop": ["<|im_end|>"],
        }
        source = {"msg": "prefix payload: " + json.dumps(payload) + ", options: ignored"}
        record, _ = prepare.request_record(source, 7, "msg-log", "fallback")
        self.assertEqual(record["id"], "lingbao-000007")
        self.assertNotIn("seed", record["sampling_params"])
        self.assertEqual(record["sampling_params"]["temperature"], 0.9)

    def test_normalization_removes_invisible_characters(self):
        text, removed = prepare.normalize_prompt("Ａ\u200bB\r\n")
        self.assertEqual(text, "AB\n")
        self.assertEqual(removed, 1)


class CompareTests(unittest.TestCase):
    def test_compare_writes_speedup(self):
        base = {
            "concurrency": 1,
            "sample_seed": 7,
            "requested": 2,
            "successful": 2,
            "throughput": {"requests_per_s": 2.0, "output_tokens_per_s": 20.0},
            "e2e_ms": {key: 100.0 for key in ("mean", "p50", "p90", "p99")},
            "ttft_ms": {key: 40.0 for key in ("mean", "p50", "p90", "p99")},
            "tpot_ms": {key: 5.0 for key in ("mean", "p50", "p90", "p99")},
        }
        eagle = json.loads(json.dumps(base))
        eagle["mode"] = "eagle3"
        eagle["throughput"] = {"requests_per_s": 2.4, "output_tokens_per_s": 24.0}
        eagle["e2e_ms"] = {key: 90.0 for key in ("mean", "p50", "p90", "p99")}
        eagle["tpot_ms"] = {key: 4.0 for key in ("mean", "p50", "p90", "p99")}
        eagle["acceptance"] = {"avg_accept_length": 2.1, "avg_accept_rate": 0.3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_path, eagle_path, output = root / "b.json", root / "e.json", root / "o.json"
            base_path.write_text(json.dumps(base))
            eagle_path.write_text(json.dumps(eagle))
            status = benchmark.compare(
                argparse.Namespace(baseline=base_path, eagle3=eagle_path, output=output)
            )
            result = json.loads(output.read_text())
        self.assertEqual(status, 0)
        self.assertAlmostEqual(result["speedup"]["output_tokens_per_s"], 1.2)
        self.assertAlmostEqual(result["reduction_pct"]["e2e"]["mean"], 10.0)

    def test_request_acceptance_has_tail_percentiles(self):
        lines = [
            "#running-req: 1, accept len: 1.0, accept rate: 0.0\n",
            "#running-req: 1, accept len: 2.0, accept rate: 0.3\n",
            "#running-req: 1, accept len: 3.0, accept rate: 0.6\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("".join(lines), encoding="utf-8")
            result = benchmark.acceptance(path, 0, "eagle3")
        self.assertAlmostEqual(result["avg_accept_length"], 2.0)
        self.assertIsNotNone(result["p95_accept_length"])
        self.assertIsNotNone(result["p99_accept_length"])


class AnalysisTests(unittest.TestCase):
    def test_request_and_task_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            requests = root / "requests.jsonl"
            baseline = root / "baseline.jsonl"
            eagle = root / "eagle.jsonl"
            output = root / "out"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "r1",
                        "split": "val",
                        "task_type": "决策辅助",
                        "chat_only": False,
                        "source_line": 1,
                        "prompt_sha256": "p",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            requests.write_text(
                json.dumps(
                    {"id": "r1", "prompt": "用户：应该出什么装备。"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            common = {
                "id": "r1",
                "success": True,
                "e2e_s": 0.2,
                "ttft_s": 0.1,
                "decode_s": 0.1,
                "tpot_ms": 5.0,
                "output_tokens": 20,
                "input_tokens": 100,
                "output_sha256": "same",
            }
            baseline.write_text(json.dumps(common) + "\n", encoding="utf-8")
            candidate = dict(common)
            candidate.update(
                {
                    "e2e_s": 0.18,
                    "ttft_s": 0.105,
                    "decode_s": 0.075,
                    "tpot_ms": 3.75,
                    "request_acceptance": {
                        "decode_log_samples": 8,
                        "avg_accept_length": 2.5,
                        "p50_accept_length": 2.5,
                        "p90_accept_length": 3.0,
                        "p95_accept_length": 3.0,
                        "p99_accept_length": 3.0,
                        "avg_accept_rate": 0.5,
                    },
                }
            )
            eagle.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            argv = [
                "analyze_task_requests.py",
                "--manifest",
                str(manifest),
                "--requests",
                str(requests),
                "--baseline-details",
                str(baseline),
                "--eagle-details",
                str(eagle),
                "--output-dir",
                str(output),
                "--split",
                "validation",
                "--draft-tokens",
                "4",
            ]
            with patch("sys.argv", argv):
                status = analysis.main()
            rows = [
                json.loads(line)
                for line in (output / "request_analysis.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            families = json.loads(
                (output / "task_family_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(status, 0)
        self.assertEqual(rows[0]["last_user_text"], "应该出什么装备。")
        self.assertEqual(rows[0]["accept_length"], 2.5)
        self.assertAlmostEqual(rows[0]["tpot_speedup"], 4 / 3)
        self.assertEqual(families[0]["acceptance_count"], 1)

    def test_split_compare(self):
        train = {"accept_length_mean": 2.2}
        val = {"accept_length_mean": 2.1}
        self.assertAlmostEqual(
            split_compare.delta(
                train["accept_length_mean"], val["accept_length_mean"]
            ),
            0.1,
        )


class LedgerTests(unittest.TestCase):
    def test_registry_fallback(self):
        entry = {
            "id": "history",
            "result_dir": "{DATA_ROOT}/missing",
            "observed": {"accept_length": 2.1, "output_speedup": 1.01},
        }
        row = ledger.extract(entry, Path("/tmp/data-root"))
        self.assertEqual(row["source"], "registry")
        self.assertEqual(row["accept_length"], 2.1)
        self.assertEqual(row["output_speedup"], 1.01)


if __name__ == "__main__":
    unittest.main()
