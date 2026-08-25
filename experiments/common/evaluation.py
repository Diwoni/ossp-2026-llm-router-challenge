# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""후보 라우터의 실제 공개 품질과 비용을 같은 방식으로 계산합니다."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from experiments.common.data import MODEL_IDS, TIER_LIMITS, TIER_WEIGHTS


@dataclass(frozen=True)
class TierEvaluation:
    """등급 하나의 품질·비용 결과입니다."""

    tier: str
    quality_score: float
    cost_ratio: float
    budget_passed: bool
    effective_score: float
    model_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_choices(
    examples: Sequence[Mapping[str, object]],
    choices: Sequence[str],
    tier: str,
    *,
    sample_weights: Optional[Sequence[float]] = None,
) -> TierEvaluation:
    """선택 결과를 공식 점수 의미와 같은 형태로 평가합니다.

    `sample_weights`는 문제군 비율을 바꾸는 오프라인 스트레스 검사에만
    사용합니다. 실제 제출 결과에는 항상 동일한 가중치가 적용됩니다.
    """

    if tier not in TIER_LIMITS:
        raise ValueError(f"알 수 없는 등급입니다: {tier}")
    if len(examples) != len(choices) or not examples:
        raise ValueError("문항과 선택은 비어 있지 않고 길이가 같아야 합니다")
    weights = list(sample_weights) if sample_weights is not None else [1.0] * len(examples)
    if len(weights) != len(examples) or any(weight < 0 for weight in weights):
        raise ValueError("표본 가중치는 문항 수와 같고 음수가 아니어야 합니다")
    total_weight = math.fsum(weights)
    if total_weight <= 0:
        raise ValueError("표본 가중치 합은 양수여야 합니다")

    model_index = {model_id: index for index, model_id in enumerate(MODEL_IDS)}
    quality_total = 0.0
    selected_cost = 0.0
    light_cost = 0.0
    counts = {model_id: 0 for model_id in MODEL_IDS}
    for example, model_id, weight in zip(examples, choices, weights):
        if model_id not in model_index:
            raise ValueError(f"허용되지 않은 모델입니다: {model_id}")
        index = model_index[model_id]
        scores = example["scores"]
        costs = example["costs"]
        quality_total += float(scores[index]) * weight  # type: ignore[index]
        selected_cost += float(costs[index]) * weight  # type: ignore[index]
        light_cost += float(costs[0]) * weight  # type: ignore[index]
        counts[model_id] += 1
    ratio = selected_cost / light_cost
    quality = quality_total / total_weight
    passed = ratio <= TIER_LIMITS[tier]
    return TierEvaluation(
        tier=tier,
        quality_score=quality,
        cost_ratio=ratio,
        budget_passed=passed,
        effective_score=quality if passed else 0.0,
        model_counts=counts,
    )


def weighted_final_score(results: Mapping[str, TierEvaluation]) -> float:
    """세 등급 결과를 공식 가중치로 합칩니다."""

    missing = set(TIER_WEIGHTS) - set(results)
    if missing:
        raise ValueError(f"등급 결과가 누락됐습니다: {sorted(missing)}")
    return math.fsum(
        results[tier].effective_score * weight
        for tier, weight in TIER_WEIGHTS.items()
    )
