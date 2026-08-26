#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# 사용법: scripts/demo_method4.sh [fast|balanced|premium]
# 최종 ARM64 이미지를 공식과 같은 핵심 격리 조건에서 toy 입력으로 실행합니다.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

TIER=${1:-balanced}
case "$TIER" in
  fast|balanced|premium) ;;
  *)
    printf '등급은 fast, balanced, premium 중 하나여야 합니다.\n' >&2
    exit 2
    ;;
esac

# 태그는 다시 가리킬 수 있으므로 기본값도 제출에 쓰는 불변 digest로 고정합니다.
IMAGE=${OSSP_METHOD4_IMAGE:-ghcr.io/diwoni/ossp-2026-llm-router-challenge@sha256:1913d548e33fe9fcad60eac4e33c73142c73e0b3b1754061080c941154b63829}
INPUT=${OSSP_DEMO_INPUT:-data/toy/inputs.json}
OUTPUT_DIR="build/demo/$TIER"
OUTPUT="$OUTPUT_DIR/submission.json"

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker 호환 CLI를 찾을 수 없습니다.\n' >&2
  exit 2
fi
if [ ! -f "$INPUT" ]; then
  printf '시연 입력을 찾을 수 없습니다: %s\n' "$INPUT" >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  printf '로컬에 최종 이미지가 없습니다: %s\n' "$IMAGE" >&2
  printf 'GHCR 공개 전환 뒤 전체 digest로 먼저 pull해 주세요.\n' >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
# 컨테이너의 고정 비특권 사용자(65532)가 bind mount 출력에 쓸 수 있게 합니다.
chmod 0777 "$OUTPUT_DIR"

printf '\n[1/3] 기술 제출 JSON 스키마 확인\n'
python3 tools/validate_technical_submission.py

printf '\n[2/3] 공식형 격리 조건에서 %s 등급 실행\n' "$TIER"
START=$(date +%s)
docker run --rm \
  --pull never \
  --platform linux/arm64 \
  --network none \
  --ipc none \
  --cgroupns private \
  --ulimit core=0:0 \
  --read-only \
  --user 65532:65532 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --cpus 2 \
  --memory 2g \
  --memory-swap 2g \
  --pids-limit 32 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --mount "type=bind,src=$(pwd)/$INPUT,dst=/challenge/input/inputs.json,readonly" \
  --mount "type=bind,src=$(pwd)/$OUTPUT_DIR,dst=/challenge/output" \
  "$IMAGE" \
  --input /challenge/input/inputs.json \
  --tier "$TIER" \
  --output /challenge/output/submission.json
END=$(date +%s)

printf '\n[3/3] 출력 범위와 선택 결과 요약\n'
PYTHONPATH=src:. python3 - "$INPUT" "$OUTPUT" "$TIER" <<'PY'
from collections import Counter
from pathlib import Path
import sys

from ossp_router.protocol import load_bundled_policy, load_input, load_submission
from ossp_router.runtime import validate_runtime_submission

input_path, output_path, tier = sys.argv[1:]
inputs = load_input(Path(input_path))
submission = load_submission(Path(output_path))
validate_runtime_submission(inputs, submission, load_bundled_policy(), tier)
counts = Counter(decision.model_id for decision in submission.decisions)

print(f"스키마·문항 범위 검증: 통과")
print(f"선택 문항 수: {len(submission.decisions)}")
for model_id in ("ax31-light", "ax31", "axk1-think"):
    print(f"  {model_id}: {counts[model_id]}")
print("첫 세 결정:")
for decision in submission.decisions[:3]:
    print(f"  {decision.episode_id} -> {decision.model_id}")
PY

printf '컨테이너 실행 시간: %s초\n' "$((END - START))"
printf '시연 출력: %s\n' "$OUTPUT"
