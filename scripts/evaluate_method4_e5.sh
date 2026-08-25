#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# E5 기준선을 새 선택 파일로 다시 생성합니다. 기존 선택 결과 JSON은 읽지 않습니다.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

PROFILE=${1:-train-only}
COMMON_ASSETS=${METHOD4_E5_COMMON_ASSETS:-build/reproduced-assets/e5}
case "$PROFILE" in
  train-only)
    HEAD_ASSETS=${METHOD4_E5_HEAD_ASSETS:-build/reproduced-assets/e5-train-only}
    GATE=configs/method4-tier-gate.e5-train-only.json
    ;;
  public-all-replay)
    HEAD_ASSETS=${METHOD4_E5_HEAD_ASSETS:-build/reproduced-assets/e5}
    GATE=configs/method4-tier-gate.e5-baseline.json
    ;;
  *)
    printf '알 수 없는 프로필입니다: %s\n' "$PROFILE" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="build/baseline/method4-e5-$PROFILE"
mkdir -p "$OUTPUT_DIR"
for TIER in fast balanced premium; do
  PYTHONPATH=src:. .venv/bin/python -m \
    experiments.method4_finetuned_encoder.final_route \
    --input data/materialized/dev/inputs.json \
    --tier "$TIER" \
    --output "$OUTPUT_DIR/$TIER.json" \
    --onnx-artifact-dir "$COMMON_ASSETS/onnx" \
    --embedding-onnx-artifact-dir "$COMMON_ASSETS/embedding-onnx" \
    --hash-artifact baselines/hash-regex-public.v1.json \
    --stacking-artifact "$HEAD_ASSETS/stacking-head.json" \
    --residual-artifact "$HEAD_ASSETS/residual-head.json" \
    --similarity-artifact "$HEAD_ASSETS/similarity-head.npz" \
    --tier-gate "$GATE" \
    --threads 2
done

PYTHONPATH=src:. .venv/bin/python -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --submissions "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/report.json"

printf 'E5 %s 기준선 재현 완료: %s/report.json\n' "$PROFILE" "$OUTPUT_DIR"
