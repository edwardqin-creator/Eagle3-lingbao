#!/usr/bin/env bash

set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

ACTION="${1:-build}"
TRAIN_REQUESTS="${2:-${REQUEST_DIR}/train_requests.jsonl}"
EXTRA_ARGS=()
[[ "${PREPARE_FORCE:-0}" == 1 ]] && EXTRA_ARGS+=(--force)

install_default_draft_config() {
  if [[ ! -f "${DRAFT_CONFIG}" ]]; then
    mkdir -p "$(dirname "${DRAFT_CONFIG}")"
    cp "${REPO_ROOT}/configs/qwen3.5-35b-a3b-eagle3-i8k.json" "${DRAFT_CONFIG}"
    echo "已安装默认 I8K Draft Config: ${DRAFT_CONFIG}"
  fi
}

case "${ACTION}" in
  build)
    require_file "${REPLAY_OUTPUT}"
    require_file "${TRAIN_REQUESTS}"
    install_default_draft_config
    "${SPEC_FORGE_PYTHON}" "${REPO_ROOT}/tools/prepare_training.py" \
      --input "${REPLAY_OUTPUT}" \
      --requests "${TRAIN_REQUESTS}" \
      --output "${PRETOKENIZED_DATA}" \
      --mapping-output "${VOCAB_MAPPING}" \
      --target-model "${TARGET_MODEL}" \
      --draft-config "${DRAFT_CONFIG}" \
      --max-length "${MAX_LENGTH}" \
      --expected-records "${EXPECTED_TRAIN_RECORDS}" \
      --mask-policy "${MASK_POLICY}" \
      "${EXTRA_ARGS[@]}"
    ;;
  inspect)
    require_file "${PRETOKENIZED_DATA}"
    "${SPEC_FORGE_PYTHON}" -c '
import json, sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], trust_remote_code=True)
with open(sys.argv[1], encoding="utf-8") as source:
    for _ in range(3):
        row = json.loads(next(source)); ids = row["input_ids"]; mask = row["loss_mask"]
        supervised = [token for token, selected in zip(ids, mask) if selected]
        print("=" * 72); print("ID:", row["id"]); print("监督Token数:", len(supervised))
        print("监督文本:", repr(tokenizer.decode(supervised, skip_special_tokens=False)))
' "${PRETOKENIZED_DATA}" "${TARGET_MODEL}"
    ;;
  *)
    echo "用法: $0 {build|inspect} [train_requests.jsonl]"
    echo "确认覆盖: PREPARE_FORCE=1 $0 build [train_requests.jsonl]"
    exit 2
    ;;
esac
