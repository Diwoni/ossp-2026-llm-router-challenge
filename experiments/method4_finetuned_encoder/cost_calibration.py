# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""프롬프트 문제군별로 예측 모델 비용을 보정합니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.common.data import MODEL_IDS
from experiments.method4_finetuned_encoder.similarity import runtime_source_name


def calibrate_cost_rows(
    cost_rows: Sequence[Mapping[str, float]],
    examples: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    tier: str,
) -> list[dict[str, float]]:
    """내용으로 추정한 문제군 배수를 상대 모델 비용에 적용합니다."""

    if len(cost_rows) != len(examples):
        raise ValueError("cost calibration rows and examples must be aligned")
    if artifact.get("model_ids") != list(MODEL_IDS):
        raise ValueError("cost calibration artifact model IDs do not match policy")
    if tier not in artifact.get("tiers", {}):
        raise ValueError(f"cost calibration artifact has no {tier} tier")
    policy = artifact["tiers"][tier]
    global_factors = policy["global_factors"]
    source_factors = policy["source_factors"]
    calibrated: list[dict[str, float]] = []
    for costs, example in zip(cost_rows, examples):
        factors = source_factors.get(runtime_source_name(example), global_factors)
        row = {
            model_id: float(costs[model_id]) * float(factors[model_id])
            for model_id in MODEL_IDS
        }
        row[MODEL_IDS[1]] = max(row[MODEL_IDS[1]], row[MODEL_IDS[0]] * (1.0 + 1e-12))
        row[MODEL_IDS[2]] = max(row[MODEL_IDS[2]], row[MODEL_IDS[1]] * (1.0 + 1e-12))
        calibrated.append(row)
    return calibrated
