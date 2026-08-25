# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""오프라인 의미 교사의 지식을 증류한 소형 런타임 학생입니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.method4_finetuned_encoder.similarity import runtime_task_profile
from experiments.method4_finetuned_encoder.stacking import predict_ridge


def student_feature_matrix(router_embeddings: Any, examples: Sequence[Mapping[str, Any]]):
    """기존 E5 라우터 표현과 내용 기반 문제 특징을 결합합니다."""

    import numpy as np

    embeddings = np.asarray(router_embeddings, dtype=np.float64).copy()
    if embeddings.ndim != 2 or len(embeddings) != len(examples):
        raise ValueError("E5 임베딩 수와 실행 입력 수가 일치해야 합니다")
    embeddings /= np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    profiles = np.asarray(
        [runtime_task_profile(row) for row in examples], dtype=np.float64
    )
    return np.hstack([embeddings, profiles])


def _semantic_mask(baseline: Any, student: Any, *, mode: str, fraction: float):
    import numpy as np

    baseline = np.asarray(baseline, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if baseline.shape != student.shape or baseline.ndim != 2:
        raise ValueError("기준 점수와 학생 점수는 같은 2차원 모양이어야 합니다")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("교사-학생 적용 비율은 0 이상 1 이하여야 합니다")
    if mode == "all":
        return np.ones(len(baseline), dtype=bool)
    baseline_values = np.column_stack([np.zeros(len(baseline)), baseline])
    student_values = np.column_stack([np.zeros(len(student)), student])
    baseline_choice = np.argmax(baseline_values, axis=1)
    student_choice = np.argmax(student_values, axis=1)
    disagreement = baseline_choice != student_choice
    if mode == "disagreement":
        eligible = disagreement
        priority = np.max(np.abs(student - baseline), axis=1)
    elif mode in ("ambiguous", "ambiguous-disagreement"):
        ordered = np.sort(baseline_values, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
        eligible = (
            disagreement if mode == "ambiguous-disagreement" else np.ones(len(baseline), dtype=bool)
        )
        priority = -margin
    else:
        raise ValueError(f"알 수 없는 교사-학생 적용 모드입니다: {mode}")
    eligible_indices = np.where(eligible)[0]
    if not len(eligible_indices) or fraction <= 0.0:
        return np.zeros(len(baseline), dtype=bool)
    count = min(max(1, int(round(len(baseline) * fraction))), len(eligible_indices))
    selected = eligible_indices[
        np.argsort(priority[eligible_indices], kind="stable")[-count:]
    ]
    mask = np.zeros(len(baseline), dtype=bool)
    mask[selected] = True
    return mask


def apply_absolute_student(
    semantic_delta: Any,
    router_embeddings: Any,
    examples: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    weight: float,
    mode: str,
    fraction: float,
) -> tuple[Any, Any]:
    """의미 신호의 제한된 부분만 학생 출력과 혼합합니다."""

    import numpy as np

    if not 0.0 <= weight <= 1.0:
        raise ValueError("교사-학생 혼합 가중치는 0 이상 1 이하여야 합니다")
    matrix = student_feature_matrix(router_embeddings, examples)
    student = predict_ridge(
        matrix,
        np.asarray(artifact["feature_mean"], dtype=np.float64),
        np.asarray(artifact["feature_scale"], dtype=np.float64),
        np.asarray(artifact["intercept"], dtype=np.float64),
        np.asarray(artifact["coefficients"], dtype=np.float64),
    )
    semantic = np.asarray(semantic_delta, dtype=np.float64).copy()
    mask = _semantic_mask(semantic, student, mode=mode, fraction=fraction)
    semantic[mask] = (1.0 - weight) * semantic[mask] + weight * student[mask]
    return semantic, mask
