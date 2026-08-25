#!/bin/sh
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0
set -eu

# 새 환경에서 Release 압축 파일과 내부 10개 자산을 차례로 검증합니다.
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
TARGET=${METHOD4_ASSET_TARGET:-build/final-assets/method4}
URL=${METHOD4_ASSET_URL:-https://github.com/Diwoni/ossp-2026-llm-router-challenge/releases/download/method4-assets-v1/method4-assets-v1.tar.gz}
EXPECTED_SHA256=d320179bd0c1f22cfcfb2711964d1bed21fa67fcab95af69a8587f9c49044b3a

if [ -d "$TARGET" ]; then
  PYTHONPATH=src:. python3 tools/verify_method4_assets.py --assets-dir "$TARGET"
  printf '이미 검증된 자산을 사용합니다: %s\n' "$TARGET"
  exit 0
fi

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM
curl --fail --location --retry 3 --output "$TEMP_DIR/method4-assets-v1.tar.gz" "$URL"
ACTUAL_SHA256=$(shasum -a 256 "$TEMP_DIR/method4-assets-v1.tar.gz" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  printf '압축 자산 SHA-256이 다릅니다: 기대 %s, 실제 %s\n' \
    "$EXPECTED_SHA256" "$ACTUAL_SHA256" >&2
  exit 2
fi
tar -xzf "$TEMP_DIR/method4-assets-v1.tar.gz" -C "$TEMP_DIR"
PYTHONPATH=src:. python3 tools/verify_method4_assets.py \
  --assets-dir "$TEMP_DIR/method4"
mkdir -p "$(dirname -- "$TARGET")"
mv "$TEMP_DIR/method4" "$TARGET"
printf 'Release 자산 다운로드와 이중 검증 완료: %s\n' "$TARGET"
