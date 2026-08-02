#!/usr/bin/env bash

set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ACTION="${1:-help}"
REQUEST_FILE="${2:-${REQUEST_DIR}/train_requests.jsonl}"
RUNTIME_DIR="${DATA_ROOT}/replay_runtime"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "$(dirname "${REPLAY_OUTPUT}")"

read -r -a GPU_GROUPS <<< "${REPLAY_GPU_GROUPS}"
read -r -a PORTS <<< "${REPLAY_PORTS}"
if [[ "${#GPU_GROUPS[@]}" -ne "${#PORTS[@]}" ]]; then
  echo "REPLAY_GPU_GROUPS 与 REPLAY_PORTS 数量不一致" >&2
  exit 2
fi

start_servers() {
  require_dir "${TARGET_MODEL}"
  for index in "${!PORTS[@]}"; do
    local port="${PORTS[$index]}"
    local gpus="${GPU_GROUPS[$index]}"
    local log="${LOG_DIR}/sglang_${port}.log"
    local pid_file="${PID_DIR}/sglang_${port}.pid"
    if ! port_is_free "${port}"; then
      echo "端口 ${port} 已被占用；先用 ss -ltnp 'sport = :${port}' 确认进程" >&2
      exit 1
    fi
    echo "启动 Target: GPU=${gpus} port=${port} log=${log}"
    nohup bash -lc "
      cd '${SPEC_FORGE_ROOT}'
      source '${SPEC_FORGE_ROOT}/.venv/bin/activate'
      export CUDA_VISIBLE_DEVICES='${gpus}' PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
      exec python3 -m sglang.launch_server \\
        --model-path '${TARGET_MODEL}' \\
        --served-model-name '${SERVED_MODEL_NAME}' \\
        --tp-size '${REPLAY_TP_SIZE}' \\
        --dtype bfloat16 \\
        --context-length '${REPLAY_CONTEXT_LENGTH}' \\
        --mem-fraction-static '${REPLAY_MEM_FRACTION}' \\
        --max-running-requests '${REPLAY_MAX_RUNNING_REQUESTS}' \\
        --host 127.0.0.1 --port '${port}' --trust-remote-code
    " >"${log}" 2>&1 &
    echo "$!" > "${pid_file}"
    if ! wait_for_health "http://127.0.0.1:${port}" 1800; then
      tail -n 100 "${log}" >&2
      exit 1
    fi
  done
}

run_replay() {
  require_file "${REQUEST_FILE}"
  local servers=()
  for port in "${PORTS[@]}"; do
    servers+=("http://127.0.0.1:${port}")
  done
  "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/replay.py" run \
    --input "${REQUEST_FILE}" \
    --output "${REPLAY_OUTPUT}" \
    --error-output "${REPLAY_ERROR_OUTPUT}" \
    --servers "${servers[@]}" \
    --concurrency-per-server "${REPLAY_CONCURRENCY_PER_SERVER}" \
    2>&1 | tee "${LOG_DIR}/replay_client.log"
}

stop_servers() {
  for port in "${PORTS[@]}"; do
    stop_pid_file "${PID_DIR}/sglang_${port}.pid"
  done
}

case "${ACTION}" in
  start) start_servers ;;
  run) run_replay ;;
  all)
    trap stop_servers EXIT INT TERM
    start_servers
    run_replay
    "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/replay.py" validate \
      --requests "${REQUEST_FILE}" --output "${REPLAY_OUTPUT}"
    ;;
  validate)
    "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/replay.py" validate \
      --requests "${REQUEST_FILE}" --output "${REPLAY_OUTPUT}"
    ;;
  status)
    for port in "${PORTS[@]}"; do
      echo "===== port ${port} ====="
      curl --fail --silent "http://127.0.0.1:${port}/health" && echo " healthy" || echo " unavailable"
      [[ -f "${PID_DIR}/sglang_${port}.pid" ]] && echo "pid=$(cat "${PID_DIR}/sglang_${port}.pid")"
    done
    ;;
  stop) stop_servers ;;
  *)
    echo "用法: $0 {start|run|all|validate|status|stop} [train_requests.jsonl]"
    exit 2
    ;;
esac
