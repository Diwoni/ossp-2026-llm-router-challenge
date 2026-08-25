# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""여러 학습형 라우터가 공유하는 품질 향상·위험 비용 회귀 head입니다."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


QUALITY_HEADS = ("score_light", "uplift_ax31", "uplift_think")
COST_HEADS = ("log_cost_light_q90", "log_cost_ax31_q90", "log_cost_think_q90")
ALL_HEADS = QUALITY_HEADS + COST_HEADS


def targets(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    scores = np.asarray([example["scores"] for example in examples], dtype=np.float64)
    costs = np.asarray([example["costs"] for example in examples], dtype=np.float64)
    return {
        "score_light": scores[:, 0],
        "uplift_ax31": scores[:, 1] - scores[:, 0],
        "uplift_think": scores[:, 2] - scores[:, 1],
        "log_cost_light_q90": np.log(costs[:, 0]),
        "log_cost_ax31_q90": np.log(costs[:, 1]),
        "log_cost_think_q90": np.log(costs[:, 2]),
    }


def _fit_lgbm_heads(x: Any, y: Mapping[str, Any], seed: int) -> dict[str, Any]:
    from lightgbm import LGBMRegressor

    models: dict[str, Any] = {}
    for index, name in enumerate(ALL_HEADS):
        is_cost = name in COST_HEADS
        model = LGBMRegressor(
            objective="quantile" if is_cost else "regression_l1",
            alpha=0.90 if is_cost else 0.90,
            n_estimators=320,
            learning_rate=0.025,
            num_leaves=15,
            min_child_samples=24,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.70,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed + index,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(x, y[name])
        models[name] = model
    return models


def fit_ridge_heads(
    x: Any, y: Mapping[str, Any], *, alpha: float = 100.0
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    models: dict[str, Any] = {}
    for name in ALL_HEADS:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
        model.fit(x, y[name])
        models[name] = model
    return models


def fit_ensemble_heads(x: Any, y: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """희소 ridge 기준점과 비선형 LightGBM 잔차 모델을 적합합니다."""

    return {
        "artifact_version": 2,
        "lgbm": _fit_lgbm_heads(x, y, seed),
        "ridge": fit_ridge_heads(x, y),
    }


def predict(models: Mapping[str, Any], x: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    import numpy as np

    if "lgbm" in models and "ridge" in models:
        lgbm = {
            name: np.asarray(models["lgbm"][name].predict(x), dtype=np.float64)
            for name in ALL_HEADS
        }
        ridge = {
            name: np.asarray(models["ridge"][name].predict(x), dtype=np.float64)
            for name in ALL_HEADS
        }
        raw = {
            name: (
                0.35 * lgbm[name] + 0.65 * ridge[name]
                if name in QUALITY_HEADS
                else np.maximum(lgbm[name], ridge[name])
            )
            for name in ALL_HEADS
        }
    else:
        raw = {name: np.asarray(models[name].predict(x), dtype=np.float64) for name in ALL_HEADS}
    light = np.clip(raw["score_light"], 0.0, 1.0)
    ax31 = np.clip(light + raw["uplift_ax31"], 0.0, 1.0)
    think = np.clip(ax31 + raw["uplift_think"], 0.0, 1.0)
    cost_light = np.exp(np.clip(raw["log_cost_light_q90"], -50.0, 50.0))
    cost_ax31 = np.maximum(
        np.exp(np.clip(raw["log_cost_ax31_q90"], -50.0, 50.0)),
        cost_light * (1.0 + 1e-12),
    )
    cost_think = np.maximum(
        np.exp(np.clip(raw["log_cost_think_q90"], -50.0, 50.0)),
        cost_ax31 * (1.0 + 1e-12),
    )
    score_rows = [
        {"ax31-light": float(a), "ax31": float(b), "axk1-think": float(c)}
        for a, b, c in zip(light, ax31, think)
    ]
    cost_rows = [
        {"ax31-light": float(a), "ax31": float(b), "axk1-think": float(c)}
        for a, b, c in zip(cost_light, cost_ax31, cost_think)
    ]
    return score_rows, cost_rows


def training_metrics(
    models: Mapping[str, Any], x: Any, y: Mapping[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    families = ("lgbm", "ridge") if "lgbm" in models else ("single",)
    for family in families:
        head_models = models[family] if family != "single" else models
        for name in ALL_HEADS:
            predicted = head_models[name].predict(x)
            absolute = [abs(float(a) - float(b)) for a, b in zip(predicted, y[name])]
            result[f"train_mae_{family}_{name}"] = math.fsum(absolute) / len(absolute)
    return result
