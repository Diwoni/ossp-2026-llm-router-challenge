# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""입력 전체의 비용 위험을 반영해 모델을 배분합니다."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from experiments.common.data import MODEL_IDS, TIER_LIMITS


def select_models(
    score_rows: Sequence[Mapping[str, float]],
    cost_rows: Sequence[Mapping[str, float]],
    tier: str,
    safety_ratio: float,
    allowed_model_ids: Optional[Sequence[str]] = None,
) -> tuple[tuple[str, ...], float]:
    if tier not in TIER_LIMITS:
        raise ValueError(f"unknown tier: {tier}")
    if len(score_rows) != len(cost_rows) or not score_rows:
        raise ValueError("score and cost rows must be non-empty and aligned")
    if not 0 < safety_ratio <= 1:
        raise ValueError("safety_ratio must be in (0, 1]")
    allowed = tuple(allowed_model_ids) if allowed_model_ids is not None else MODEL_IDS
    if not allowed or any(model_id not in MODEL_IDS for model_id in allowed):
        raise ValueError("allowed_model_ids must be a non-empty subset of model IDs")
    light_total = math.fsum(row[MODEL_IDS[0]] for row in cost_rows)
    cap = light_total * max(1.0, TIER_LIMITS[tier] * safety_ratio)

    def choose(penalty: float) -> tuple[tuple[str, ...], float]:
        chosen = tuple(
            max(
                allowed,
                key=lambda model_id: (
                    scores[model_id] - penalty * costs[model_id] / light_total,
                    -MODEL_IDS.index(model_id),
                ),
            )
            for scores, costs in zip(score_rows, cost_rows)
        )
        total = math.fsum(
            costs[model_id] for costs, model_id in zip(cost_rows, chosen)
        )
        return chosen, total

    selected, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            selected, total = choose(high)
        for _ in range(80):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, selected, total = middle, candidate, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(MODEL_IDS[0] for _ in score_rows)
        total = light_total
    return selected, total / light_total
