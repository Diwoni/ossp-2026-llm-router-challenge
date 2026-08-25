#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# Release 다운로드 또는 로컬 재현 자산을 검증한 뒤 Docker build 전용 위치에 복사합니다.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
SOURCE=${METHOD4_ASSET_SOURCE:-build/reproduced-assets/qwen-student}
TARGET=${METHOD4_ASSET_TARGET:-build/final-assets/method4}

PYTHONPATH=src:. .venv/bin/python tools/verify_method4_assets.py \
  --assets-dir "$SOURCE"

mkdir -p "$TARGET/onnx" "$TARGET/embedding-onnx"
for FILE in \
  onnx/router-int8.onnx \
  onnx/tokenizer.json \
  onnx/metadata.json \
  embedding-onnx/embedding-fp32.onnx \
  embedding-onnx/tokenizer.json \
  embedding-onnx/metadata.json \
  stacking-head.json \
  residual-head.json \
  similarity-head.npz \
  teacher-student-head.json; do
  install -m 0644 "$SOURCE/$FILE" "$TARGET/$FILE"
done

PYTHONPATH=src:. .venv/bin/python tools/verify_method4_assets.py \
  --assets-dir "$TARGET"
printf '최종 Docker 자산 준비 완료: %s\n' "$TARGET"
