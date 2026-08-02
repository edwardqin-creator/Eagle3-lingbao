#!/usr/bin/env bash

set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ACTION="${1:-help}"
TEMPLATE="${TRAIN_TEMPLATE:-${REPO_ROOT}/configs/eagle3-online.yaml.template}"
mkdir -p "${TRAIN_LOG_DIR}"

render() {
  require_file "${TEMPLATE}"
  "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/render_config.py" \
    --template "${TEMPLATE}" --output "${TRAIN_CONFIG}"
}

doctor() {
  require_dir "${SPEC_FORGE_ROOT}"
  require_file "${SPEC_FORGE_BIN}"
  require_file "${PRETOKENIZED_DATA}"
  require_file "${DRAFT_CONFIG}"
  require_file "${VOCAB_MAPPING}"
  echo "检查 Python/SpecForge/SGLang/Mooncake 与源码优先级..."
  "${SPEC_FORGE_PYTHON}" -c '
import importlib.util, shutil, sglang, specforge, sys
print("Python:", sys.executable)
print("SpecForge:", specforge.__file__)
print("SGLang:", sglang.__version__, sglang.__file__)
print("Capture Patch:", importlib.util.find_spec("sglang.srt.spec_capture_sink"))
try: mooncake = importlib.util.find_spec("mooncake.store")
except ModuleNotFoundError: mooncake = None
print("Mooncake Python:", mooncake)
print("Mooncake Master:", shutil.which("mooncake_master"))
assert specforge.__file__.startswith(sys.argv[1]), "正在导入 site-packages SpecForge，不是工作区源码"
assert sglang.__version__.startswith("0.5.14")
assert importlib.util.find_spec("sglang.srt.spec_capture_sink") is not None
assert mooncake is not None and shutil.which("mooncake_master")
' "${SPEC_FORGE_ROOT}"
  echo "检查 GPU 与训练端口..."
  nvidia-smi -L
  for port in "${CAPTURE_PORT}" "${MOONCAKE_RPC_PORT}" "${MOONCAKE_METADATA_PORT}" "${MOONCAKE_METRICS_PORT}"; do
    if ! port_is_free "${port}"; then
      echo "端口 ${port} 被占用:" >&2
      ss -ltnp "sport = :${port}" >&2 || true
      exit 1
    fi
  done
  echo "预检通过"
}

plan() {
  render
  local log="${TRAIN_LOG_DIR}/${EXPERIMENT_NAME}.plan.log"
  echo "Plan 日志: ${log}"
  "${SPEC_FORGE_BIN}" train -c "${TRAIN_CONFIG}" --plan 2>&1 | tee "${log}"
}

run_training() {
  doctor
  plan
  local stamp log pid_file
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${TRAIN_LOG_DIR}/${EXPERIMENT_NAME}_${stamp}.log"
  pid_file="${TRAIN_LOG_DIR}/${EXPERIMENT_NAME}.pid"
  nohup env \
    PYTHONPATH="${SPEC_FORGE_ROOT}:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    NCCL_DEBUG=WARN \
    "${SPEC_FORGE_BIN}" train -c "${TRAIN_CONFIG}" \
    >"${log}" 2>&1 &
  echo "$!" > "${pid_file}"
  echo "训练 PID=$!"
  echo "日志=${log}"
  echo "查看: tail -F '${log}'"
}

latest_log() {
  find "${TRAIN_LOG_DIR}" -maxdepth 1 -type f -name "${EXPERIMENT_NAME}_*.log" \
    -print | sort | tail -n 1
}

case "${ACTION}" in
  doctor) doctor ;;
  render) render; sed -n '1,240p' "${TRAIN_CONFIG}" ;;
  plan) plan ;;
  run) run_training ;;
  status)
    pid_file="${TRAIN_LOG_DIR}/${EXPERIMENT_NAME}.pid"
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}")"
      ps -fp "${pid}" || echo "训练主进程已退出"
    else
      echo "没有 PID 文件"
    fi
    nvidia-smi
    ;;
  tail)
    log="$(latest_log)"
    [[ -n "${log}" ]] || { echo "没有训练日志" >&2; exit 1; }
    tail -F "${log}"
    ;;
  checkpoints)
    require_dir "${TRAIN_OUTPUT}"
    find "${TRAIN_OUTPUT}" -maxdepth 3 -type f -name training_state.pt -ls
    latest="${TRAIN_OUTPUT}/${EXPERIMENT_NAME}-latest"
    [[ -e "${latest}" ]] && echo "latest -> $(readlink -f "${latest}")"
    ;;
  export)
    require_file "${EXPORT_CHECKPOINT}/training_state.pt"
    require_file "${DRAFT_CONFIG}"
    require_file "${VOCAB_MAPPING}"
    if [[ -e "${EXPORT_DIR}" ]]; then
      echo "导出目录已存在，拒绝覆盖: ${EXPORT_DIR}" >&2
      exit 1
    fi
    "${SPEC_FORGE_BIN}" export --to sglang \
      --checkpoint "${EXPORT_CHECKPOINT}" \
      --draft-config "${DRAFT_CONFIG}" \
      --vocab-mapping "${VOCAB_MAPPING}" \
      --output-dir "${EXPORT_DIR}"
    echo "已导出: ${EXPORT_DIR}"
    find "${EXPORT_DIR}" -maxdepth 1 -type f -ls
    ;;
  *)
    echo "用法: $0 {doctor|render|plan|run|status|tail|checkpoints|export}"
    exit 2
    ;;
esac
