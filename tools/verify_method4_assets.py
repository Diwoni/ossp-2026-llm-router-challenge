# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""Method 4 최종 자산의 크기와 SHA-256을 검증합니다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(manifest: Mapping[str, Any], assets_dir: Path) -> list[str]:
    """누락·크기·해시 오류를 반환하고 올바른 자산이면 빈 목록을 반환합니다."""

    errors: list[str] = []
    declared_total = int(manifest.get("total_size", -1))
    observed_total = 0
    seen: set[str] = set()
    for row in manifest.get("files", []):
        relative = Path(str(row.get("path", "")))
        relative_text = relative.as_posix()
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            errors.append(f"안전하지 않은 자산 경로입니다: {relative_text!r}")
            continue
        if relative_text in seen:
            errors.append(f"중복 자산 경로입니다: {relative_text}")
            continue
        seen.add(relative_text)
        path = assets_dir / relative
        if not path.is_file():
            errors.append(f"자산 파일이 없습니다: {relative_text}")
            continue
        expected_size = int(row["size"])
        actual_size = path.stat().st_size
        observed_total += actual_size
        if actual_size != expected_size:
            errors.append(
                f"자산 크기가 다릅니다: {relative_text} "
                f"(기대 {expected_size}, 실제 {actual_size})"
            )
            continue
        expected_sha = str(row["sha256"])
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            errors.append(
                f"자산 SHA-256이 다릅니다: {relative_text} "
                f"(기대 {expected_sha}, 실제 {actual_sha})"
            )
    if not errors and observed_total != declared_total:
        errors.append(
            f"전체 자산 크기가 다릅니다: 기대 {declared_total}, 실제 {observed_total}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/method4-assets.manifest.v1.json"),
    )
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = verify_assets(manifest, args.assets_dir)
    if errors:
        for error in errors:
            print(f"오류: {error}")
        return 2
    print(
        f"Method 4 자산 {len(manifest['files'])}개와 "
        f"{manifest['total_size']}바이트를 확인했습니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
