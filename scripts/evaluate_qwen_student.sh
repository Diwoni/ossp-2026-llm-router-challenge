#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# 공개 전체로 만든 최종 자산을 공개 Dev에 재대입해 패키징 오류를 검사합니다.
# Dev 정답이 자산 생성에 포함됐으므로 이 점수는 숨은 성능 추정치가 아닙니다.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

ASSETS=${METHOD4_QWEN_STUDENT_ASSETS:-build/reproduced-assets/qwen-student}
OUTPUT_DIR=${METHOD4_QWEN_STUDENT_OUTPUT:-build/candidates/qwen-student}
mkdir -p "$OUTPUT_DIR"

for TIER in fast balanced premium; do
  PYTHONPATH=src:. .venv/bin/python -m \
    experiments.method4_finetuned_encoder.final_route \
    --input data/materialized/dev/inputs.json \
    --tier "$TIER" \
    --output "$OUTPUT_DIR/$TIER.json" \
    --onnx-artifact-dir "$ASSETS/onnx" \
    --embedding-onnx-artifact-dir "$ASSETS/embedding-onnx" \
    --hash-artifact baselines/hash-regex-public.v1.json \
    --stacking-artifact "$ASSETS/stacking-head.json" \
    --residual-artifact "$ASSETS/residual-head.json" \
    --similarity-artifact "$ASSETS/similarity-head.npz" \
    --teacher-student-artifact "$ASSETS/teacher-student-head.json" \
    --tier-gate configs/method4-tier-gate.qwen-student.json \
    --threads 2
done

PYTHONPATH=src:. .venv/bin/python -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json \
  --outcomes data/dev/outcomes.json \
  --submissions "$OUTPUT_DIR" \
  --report "$OUTPUT_DIR/report.json"

printf 'Qwen 학생 공개 Dev 재대입 검사 완료: %s/report.json\n' "$OUTPUT_DIR"
