from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_data", ROOT / "scripts" / "01_prepare_data.py")
benchmark = load("benchmark", ROOT / "tools" / "benchmark.py")


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


if __name__ == "__main__":
    unittest.main()
