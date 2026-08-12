#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTION="${1:-help}"
PYTHON="${POSTPROCESS_PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-/data/home/leonardoqin/datasets/lingbao_eagle3_data}"
ARTIFACT_ROOT="${POSTPROCESS_OUTPUT_ROOT:-${REPO_ROOT}/artifacts/postprocess}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/07_postprocess.sh ledger
  MANIFEST=... REQUESTS=... RESULT_DIR=... SPLIT=val \
    bash scripts/07_postprocess.sh analyze
  TRAIN_ANALYSIS_DIR=... VAL_ANALYSIS_DIR=... \
    bash scripts/07_postprocess.sh compare-splits
  EAGLE_DETAILS=... bash scripts/07_postprocess.sh coverage

关键环境变量：
  ANALYSIS_NAME       本轮分析名，决定默认输出目录
  DRAFT_TOKENS        服务端 speculative-num-draft-tokens，默认4
  SUMMARY_SCOPE       all_pairs/same_output_tokens/same_output_sha256
  POSTPROCESS_PYTHON  Python解释器，默认python3
EOF
}

case "${ACTION}" in
  ledger)
    OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/ledger}"
    "${PYTHON}" "${REPO_ROOT}/tools/experiment_ledger.py" \
      --registry "${REGISTRY:-${REPO_ROOT}/experiments/registry.json}" \
      --data-root "${DATA_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      ${LEDGER_STRICT:+--strict}
    ;;
  analyze)
    : "${MANIFEST:?Set MANIFEST to split_manifest.jsonl}"
    : "${REQUESTS:?Set REQUESTS to the exact benchmark request JSONL}"
    : "${RESULT_DIR:?Set RESULT_DIR containing baseline/eagle details}"
    ANALYSIS_NAME="${ANALYSIS_NAME:-$(basename "$(dirname "${RESULT_DIR}")")-$(basename "${RESULT_DIR}")}"
    OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/${ANALYSIS_NAME}}"
    "${PYTHON}" "${REPO_ROOT}/tools/analyze_task_requests.py" \
      --manifest "${MANIFEST}" \
      --requests "${REQUESTS}" \
      --baseline-details "${RESULT_DIR}/baseline_c1.details.jsonl" \
      --eagle-details "${RESULT_DIR}/eagle3_c1.details.jsonl" \
      --output-dir "${OUTPUT_DIR}" \
      --split "${SPLIT:-val}" \
      --draft-tokens "${DRAFT_TOKENS:-4}"
    ;;
  compare-splits)
    : "${TRAIN_ANALYSIS_DIR:?Set TRAIN_ANALYSIS_DIR}"
    : "${VAL_ANALYSIS_DIR:?Set VAL_ANALYSIS_DIR}"
    OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/train-vs-val}"
    "${PYTHON}" "${REPO_ROOT}/tools/compare_task_splits.py" \
      --train-summary "${TRAIN_ANALYSIS_DIR}/task_family_summary.json" \
      --val-summary "${VAL_ANALYSIS_DIR}/task_family_summary.json" \
      --scope "${SUMMARY_SCOPE:-all_pairs}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  coverage)
    : "${EAGLE_DETAILS:?Set EAGLE_DETAILS to eagle3_c1.details.jsonl}"
    "${PYTHON}" -c '
import json, sys
total = captured = 0
decode_samples = 0
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    total += 1
    acceptance = row.get("request_acceptance") or {}
    if acceptance.get("avg_accept_length") is not None:
        captured += 1
        decode_samples += int(acceptance.get("decode_log_samples") or 0)
print(f"requests={total}")
print(f"acceptance_captured={captured}")
print(f"coverage={captured / total if total else 0:.2%}")
print(f"decode_log_samples={decode_samples}")
' "${EAGLE_DETAILS}"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "未知动作: ${ACTION}" >&2
    usage >&2
    exit 2
    ;;
esac
