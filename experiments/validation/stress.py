# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""문제군 비율이 달라지는 숨은 평가 상황을 모의합니다."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def family_multiplier_scenarios(
    families: Sequence[str], *, multiplier: float = 3.0
) -> Mapping[str, list[float]]:
    """각 문제군이 늘거나 줄어드는 결정적 스트레스 가중치를 만듭니다."""

    if multiplier <= 1.0:
        raise ValueError("문제군 배수는 1보다 커야 합니다")
    unique = sorted(set(families))
    if not unique:
        raise ValueError("문제군이 비어 있습니다")
    scenarios: dict[str, list[float]] = {"기준": [1.0] * len(families)}
    for family in unique:
        scenarios[f"{family}-증가"] = [
            multiplier if value == family else 1.0 for value in families
        ]
        scenarios[f"{family}-감소"] = [
            1.0 / multiplier if value == family else 1.0 for value in families
        ]
    return scenarios


def family_counts(families: Sequence[str]) -> Mapping[str, int]:
    return dict(sorted(Counter(families).items()))
