# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""작은 ExtraTrees 잔차 head를 순수 NumPy로 추론합니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.common.data import MODEL_IDS
from experiments.method4_finetuned_encoder.stacking import predict_ridge


def export_forest(estimators: Sequence[Any]) -> dict[str, Any]:
    """학습한 sklearn 트리를 런타임 독립 JSON 구조로 변환합니다."""

    trees = []
    for estimator in estimators:
        tree = estimator.tree_
        values = tree.value.reshape(tree.node_count, -1)
        if values.shape[1] != 1:
            raise ValueError("residual forests must have one scalar output")
        trees.append(
            {
                "children_left": [int(value) for value in tree.children_left],
                "children_right": [int(value) for value in tree.children_right],
                "feature": [int(value) for value in tree.feature],
                "threshold": [float(value) for value in tree.threshold],
                "value": [float(value) for value in values[:, 0]],
            }
        )
    if not trees:
        raise ValueError("cannot export an empty forest")
    return {"tree_count": len(trees), "trees": trees}


def predict_forest(matrix: Any, forest: Mapping[str, Any]) -> Any:
    """내보낸 트리를 sklearn 없이 batch 단위로 계산합니다."""

    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64)
    predictions = np.zeros(matrix.shape[0], dtype=np.float64)
    trees = forest["trees"]
    if int(forest["tree_count"]) != len(trees) or not trees:
        raise ValueError("invalid residual forest tree count")
    row_indices = np.arange(matrix.shape[0])
    for tree in trees:
        left = np.asarray(tree["children_left"], dtype=np.int64)
        right = np.asarray(tree["children_right"], dtype=np.int64)
        feature = np.asarray(tree["feature"], dtype=np.int64)
        threshold = np.asarray(tree["threshold"], dtype=np.float64)
        value = np.asarray(tree["value"], dtype=np.float64)
        nodes = np.zeros(matrix.shape[0], dtype=np.int64)
        while True:
            active = left[nodes] >= 0
            if not np.any(active):
                break
            active_rows = row_indices[active]
            active_nodes = nodes[active]
            go_left = (
                matrix[active_rows, feature[active_nodes]]
                <= threshold[active_nodes]
            )
            nodes[active_rows] = np.where(
                go_left, left[active_nodes], right[active_nodes]
            )
        predictions += value[nodes]
    return predictions / len(trees)


def predict_residual(
    matrix: Any,
    artifact: Mapping[str, Any],
    tier: str,
    *,
    decimals: int = 4,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """한 등급의 모델 간 품질 향상과 세 모델 비용을 예측합니다."""

    import numpy as np

    if int(artifact["feature_dimension"]) != int(matrix.shape[1]):
        raise ValueError("residual feature dimension does not match artifact")
    if artifact["model_ids"] != list(MODEL_IDS):
        raise ValueError("residual artifact model IDs do not match policy")
    policy = artifact["tiers"][tier]
    forests = artifact["forests"]
    ax31_delta = predict_forest(matrix, forests[policy["ax31_forest"]])
    if policy.get("think_forest") is None:
        think_delta = np.full(matrix.shape[0], -1.0, dtype=np.float64)
    else:
        think_delta = predict_forest(matrix, forests[policy["think_forest"]])
    score_predictions = np.column_stack(
        [np.zeros(matrix.shape[0]), ax31_delta, think_delta]
    )
    cost_head = artifact["cost_head"]
    log_costs = predict_ridge(
        matrix,
        np.asarray(cost_head["feature_mean"], dtype=np.float64),
        np.asarray(cost_head["feature_scale"], dtype=np.float64),
        np.asarray(cost_head["intercept"], dtype=np.float64),
        np.asarray(cost_head["coefficients"], dtype=np.float64),
    )
    score_predictions = np.round(score_predictions, decimals=decimals)
    log_costs = np.round(log_costs, decimals=decimals)
    costs = np.exp(np.clip(log_costs, -50.0, 50.0))
    costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1.0 + 1e-12))
    costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1.0 + 1e-12))
    return (
        [
            {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
            for row in score_predictions
        ],
        [
            {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
            for row in costs
        ],
    )
