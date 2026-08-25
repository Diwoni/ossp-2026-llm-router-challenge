#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""정상·역순 실행의 episode별 모델 선택이 같은지 비교합니다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from experiments.common.data import TIERS, load_json, write_json_atomic


def compare_decisions(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict:
    """JSON 배열 순서와 split 이름을 제외하고 ID별 선택만 비교합니다."""

    reference_index = {
        str(row["episode_id"]): str(row["model_id"])
        for row in reference["decisions"]
    }
    candidate_index = {
        str(row["episode_id"]): str(row["model_id"])
        for row in candidate["decisions"]
    }
    missing = sorted(set(reference_index) - set(candidate_index))
    extra = sorted(set(candidate_index) - set(reference_index))
    mismatched = sorted(
        episode_id
        for episode_id in set(reference_index) & set(candidate_index)
        if reference_index[episode_id] != candidate_index[episode_id]
    )
    return {
        "episodes": len(reference_index),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "mismatched_count": len(mismatched),
        "passed": not missing and not extra and not mismatched,
        "missing_sample": missing[:10],
        "extra_sample": extra[:10],
        "mismatched_sample": mismatched[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tiers = {}
    for tier in TIERS:
        tiers[tier] = compare_decisions(
            load_json(args.reference / tier / "submission.json"),
            load_json(args.candidate / tier / "submission.json"),
        )
    report = {
        "artifact_type": "os-skt-method4-order-audit-v1",
        "passed": all(row["passed"] for row in tiers.values()),
        "tiers": tiers,
    }
    write_json_atomic(args.output, report)
    for tier, row in tiers.items():
        print(
            f"{tier}: 누락 {row['missing_count']}, 추가 {row['extra_count']}, "
            f"선택 불일치 {row['mismatched_count']}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
