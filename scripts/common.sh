#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${LINGBAO_CONFIG:-${REPO_ROOT}/configs/lingbao.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "配置文件不存在: ${CONFIG_FILE}" >&2
  echo "先执行: cp ${REPO_ROOT}/configs/lingbao.env.example ${REPO_ROOT}/configs/lingbao.env" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

export PYTHONPATH="${SPEC_FORGE_ROOT}:${PYTHONPATH:-}"

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || {
    echo "文件不存在: ${path}" >&2
    exit 2
  }
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || {
    echo "目录不存在: ${path}" >&2
    exit 2
  }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "缺少命令: $1" >&2
    exit 2
  }
}

port_is_free() {
  local port="$1"
  ! ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .
}

wait_for_health() {
  local url="$1"
  local timeout_s="${2:-1800}"
  local started
  started="$(date +%s)"
  while true; do
    if curl --fail --silent --show-error "${url%/}/health" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) - started >= timeout_s )); then
      echo "服务健康检查超时: ${url}" >&2
      return 1
    fi
    sleep 5
  done
}

stop_pid_file() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  local pid
  pid="$(tr -dc '0-9' < "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    for _ in $(seq 1 30); do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
  fi
  : > "${pid_file}"
}
