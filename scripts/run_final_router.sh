#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# 사용법: scripts/run_final_router.sh 입력.json fast|balanced|premium 출력.json
if [ "$#" -ne 3 ]; then
  printf '사용법: %s 입력.json fast|balanced|premium 출력.json\n' "$0" >&2
  exit 2
fi

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
ASSETS=${METHOD4_FINAL_ASSETS:-build/reproduced-assets/qwen-student}
GATE=${METHOD4_FINAL_GATE:-configs/method4-tier-gate.qwen-student.json}

PYTHONPATH=src:. .venv/bin/python -m \
  experiments.method4_finetuned_encoder.final_route \
  --input "$1" \
  --tier "$2" \
  --output "$3" \
  --onnx-artifact-dir "$ASSETS/onnx" \
  --embedding-onnx-artifact-dir "$ASSETS/embedding-onnx" \
  --hash-artifact baselines/hash-regex-public.v1.json \
  --stacking-artifact "$ASSETS/stacking-head.json" \
  --residual-artifact "$ASSETS/residual-head.json" \
  --similarity-artifact "$ASSETS/similarity-head.npz" \
  --teacher-student-artifact "$ASSETS/teacher-student-head.json" \
  --tier-gate "$GATE" \
  --threads 2
