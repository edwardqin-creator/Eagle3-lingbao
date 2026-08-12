#!/usr/bin/env bash

set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ACTION="${1:-help}"
STAGE="${2:-validation}"
MODE="${3:-}"
EVAL_ROOT="${DATA_ROOT}/benchmark_results/${EXPERIMENT_NAME}"
RUNTIME_DIR="${DATA_ROOT}/eval_runtime/${EXPERIMENT_NAME}"
mkdir -p "${EVAL_ROOT}" "${RUNTIME_DIR}"

server_pid_file() { echo "${RUNTIME_DIR}/server.pid"; }
server_log() { echo "${RUNTIME_DIR}/${1}_server.log"; }

stop_server() {
  local pid_file pid state
  pid_file="$(server_pid_file)"
  if [[ ! -f "${pid_file}" ]]; then return 0; fi
  pid="$(tr -dc '0-9' < "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ')"
    if [[ "${state}" != Z* ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 60); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
    fi
  fi
  : > "${pid_file}"
}

start_server() {
  local mode="$1"
  local log pid_file
  [[ "${mode}" == baseline || "${mode}" == eagle3 ]] || { echo "mode 必须是 baseline/eagle3" >&2; exit 2; }
  require_dir "${TARGET_MODEL}"
  if [[ "${mode}" == eagle3 ]]; then require_dir "${EVAL_DRAFT_MODEL}"; fi
  if ! port_is_free "${EVAL_PORT}"; then
    echo "评测端口 ${EVAL_PORT} 已占用" >&2
    ss -ltnp "sport = :${EVAL_PORT}" >&2 || true
    exit 1
  fi
  stop_server
  log="$(server_log "${mode}")"
  pid_file="$(server_pid_file)"
  local spec_args=""
  if [[ "${mode}" == eagle3 ]]; then
    spec_args="--speculative-algorithm EAGLE3 --speculative-draft-model-path '${EVAL_DRAFT_MODEL}' --speculative-num-steps '${SPEC_NUM_STEPS}' --speculative-eagle-topk '${SPEC_TOPK}' --speculative-num-draft-tokens '${SPEC_DRAFT_TOKENS}'"
  fi
  echo "启动 ${mode}: GPU=${EVAL_GPU_DEVICES} port=${EVAL_PORT} log=${log}"
  nohup setsid bash -lc "
    cd '${SPEC_FORGE_ROOT}'
    source '${SPEC_FORGE_ROOT}/.venv/bin/activate'
    export CUDA_VISIBLE_DEVICES='${EVAL_GPU_DEVICES}' PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    exec python3 -m sglang.launch_server \\
      --model-path '${TARGET_MODEL}' \\
      --served-model-name '${SERVED_MODEL_NAME}' \\
      --tp-size '${EVAL_TP_SIZE}' \\
      --dtype bfloat16 \\
      --context-length '${EVAL_CONTEXT_LENGTH}' \\
      --mem-fraction-static '${EVAL_MEM_FRACTION}' \\
      --max-running-requests 8 \\
      --decode-log-interval '${EVAL_DECODE_LOG_INTERVAL:-40}' \\
      --host 127.0.0.1 --port '${EVAL_PORT}' --trust-remote-code \\
      ${spec_args}
  " >"${log}" 2>&1 &
  echo "$!" > "${pid_file}"
  if ! wait_for_health "http://127.0.0.1:${EVAL_PORT}" 1800; then
    tail -n 150 "${log}" >&2
    exit 1
  fi
}

stage_input() {
  case "$1" in
    smoke|validation) echo "${VAL_DATA}" ;;
    test) echo "${TEST_DATA}" ;;
    *) echo "stage 必须是 smoke/validation/test" >&2; exit 2 ;;
  esac
}

run_mode() {
  local stage="$1" mode="$2" input limit result_dir log
  local benchmark_extra=()
  input="$(stage_input "${stage}")"
  require_file "${input}"
  limit="${EVAL_LIMIT}"
  [[ "${stage}" == smoke ]] && limit=8
  result_dir="${EVAL_ROOT}/${stage}"
  log="$(server_log "${mode}")"
  mkdir -p "${result_dir}"
  read -r -a concurrencies <<< "${EVAL_CONCURRENCIES}"
  [[ "${stage}" == smoke ]] && concurrencies=(1)
  if [[ -n "${EVAL_FIXED_OUTPUT_TOKENS:-}" ]]; then
    benchmark_extra+=(--fixed-output-tokens "${EVAL_FIXED_OUTPUT_TOKENS}")
  fi
  [[ "${EVAL_IGNORE_EOS:-0}" == 1 ]] && benchmark_extra+=(--ignore-eos)
  [[ "${EVAL_REQUEST_DIAGNOSTICS:-0}" == 1 ]] && benchmark_extra+=(--request-diagnostics)
  for concurrency in "${concurrencies[@]}"; do
    "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/benchmark.py" run \
      --name "${EXPERIMENT_NAME}_${stage}_${mode}_c${concurrency}" \
      --mode "${mode}" \
      --input "${input}" \
      --output "${result_dir}/${mode}_c${concurrency}.json" \
      --url "http://127.0.0.1:${EVAL_PORT}" \
      --model "${SERVED_MODEL_NAME}" \
      --tokenizer "${TARGET_MODEL}" \
      --server-log "${log}" \
      --concurrency "${concurrency}" \
      --limit "${limit}" \
      --warmup "${EVAL_WARMUP}" \
      --sample-seed "${EVAL_SAMPLE_SEED}" \
      "${benchmark_extra[@]}"
  done
}

compare_stage() {
  local stage="$1" result_dir
  result_dir="${EVAL_ROOT}/${stage}"
  read -r -a concurrencies <<< "${EVAL_CONCURRENCIES}"
  [[ "${stage}" == smoke ]] && concurrencies=(1)
  for concurrency in "${concurrencies[@]}"; do
    "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/benchmark.py" compare \
      --baseline "${result_dir}/baseline_c${concurrency}.json" \
      --eagle3 "${result_dir}/eagle3_c${concurrency}.json" \
      --output "${result_dir}/comparison_c${concurrency}.json"
  done
  "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/evaluation_report.py" \
    --result-dir "${result_dir}" --concurrencies "${concurrencies[@]}" \
    --min-accept-length "${GATE_MIN_ACCEPT_LENGTH:-2.0}" \
    --min-output-speedup "${GATE_MIN_OUTPUT_SPEEDUP:-1.0}" \
    --max-e2e-regression-pct "${GATE_MAX_E2E_REGRESSION_PCT:-5.0}"
}

full_stage() {
  local stage="$1"
  if [[ "${stage}" == test && "${CONFIRM_FINAL_TEST:-0}" != 1 ]]; then
    echo "Test 是最终盲测，不用于调参。确认 checkpoint 和 serving 参数已冻结后执行:" >&2
    echo "CONFIRM_FINAL_TEST=1 $0 all test" >&2
    exit 2
  fi
  if [[ -e "${EVAL_ROOT}/${stage}/ACCEPTANCE_REPORT.json" && "${EVAL_OVERWRITE:-0}" != 1 ]]; then
    echo "本轮 ${stage} 已有验收报告，拒绝静默覆盖。请换 EXPERIMENT_NAME；确认重跑可加 EVAL_OVERWRITE=1。" >&2
    exit 2
  fi
  trap stop_server EXIT INT TERM
  start_server baseline
  run_mode "${stage}" baseline
  stop_server
  start_server eagle3
  run_mode "${stage}" eagle3
  stop_server
  compare_stage "${stage}"
}

case "${ACTION}" in
  start) start_server "${MODE:-eagle3}" ;;
  run) run_mode "${STAGE}" "${MODE:-eagle3}" ;;
  compare) compare_stage "${STAGE}" ;;
  all) full_stage "${STAGE}" ;;
  stop) stop_server ;;
  status)
    curl --fail --silent "http://127.0.0.1:${EVAL_PORT}/health" && echo " healthy" || echo " unavailable"
    [[ -f "$(server_pid_file)" ]] && echo "pid=$(cat "$(server_pid_file)")"
    ;;
  *)
    echo "用法:"
    echo "  $0 all smoke"
    echo "  $0 all validation"
    echo "  CONFIRM_FINAL_TEST=1 $0 all test"
    echo "  $0 {start|run} validation {baseline|eagle3}"
    echo "  $0 compare validation"
    echo "  $0 {status|stop}"
    exit 2
    ;;
esac
