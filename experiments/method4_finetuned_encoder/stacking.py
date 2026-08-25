# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""hash와 인코더 예측을 결합하는 재배포 가능한 소형 stacking head입니다."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from experiments.common.data import MODEL_IDS
from experiments.common.dense_features import matrix as dense_matrix


def prediction_rows(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = value.get("rows", value.get("episodes"))
    if not isinstance(rows, list):
        raise ValueError("prediction artifact must contain rows or episodes")
    return rows


def feature_matrix(
    examples: Sequence[Mapping[str, Any]],
    hash_rows: Sequence[Mapping[str, Any]],
    encoder_rows: Sequence[Mapping[str, Any]],
) -> Any:
    import numpy as np

    if len(examples) != len(hash_rows) or len(examples) != len(encoder_rows):
        raise ValueError("stacking rows must be aligned")
    rows = []
    for left, right in zip(hash_rows, encoder_rows):
        hash_scores = [float(left["scores"][model_id]) for model_id in MODEL_IDS]
        encoder_scores = [float(right["scores"][model_id]) for model_id in MODEL_IDS]
        hash_costs = [math.log(float(left["costs"][model_id])) for model_id in MODEL_IDS]
        encoder_costs = [
            math.log(float(right["costs"][model_id])) for model_id in MODEL_IDS
        ]
        rows.append(
            hash_scores
            + encoder_scores
            + hash_costs
            + encoder_costs
            + [hash_scores[1] - hash_scores[0], hash_scores[2] - hash_scores[1]]
            + [
                encoder_scores[1] - encoder_scores[0],
                encoder_scores[2] - encoder_scores[1],
            ]
        )
    return np.hstack(
        [np.asarray(rows, dtype=np.float64), dense_matrix(examples).astype(np.float64)]
    )


def fit_ridge(matrix: Any, targets: Any, alpha: float) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    system = standardized.T @ standardized + alpha * np.eye(
        standardized.shape[1], dtype=np.float64
    )
    coefficients = np.linalg.solve(system, standardized.T @ centered)
    return mean, scale, intercept, coefficients


def predict_ridge(
    matrix: Any, mean: Any, scale: Any, intercept: Any, coefficients: Any
) -> Any:
    return (matrix - mean) / scale @ coefficients + intercept


def arrays_to_rows(predictions: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    import numpy as np

    scores = np.clip(predictions[:, :3], 0.0, 1.0)
    costs = np.exp(np.clip(predictions[:, 3:], -50.0, 50.0))
    costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1.0 + 1e-12))
    costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1.0 + 1e-12))
    return (
        [
            {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
            for row in scores
        ],
        [
            {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
            for row in costs
        ],
    )
