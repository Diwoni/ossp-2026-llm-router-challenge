# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""방법 4의 학습·예측·자산 저장을 담당합니다."""

from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from experiments.common.data import MODEL_IDS, TIER_LIMITS, TIERS, write_json_atomic
from experiments.method4_finetuned_encoder.model import RouterModel, multitask_loss


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_name(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def target_arrays(examples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any]:
    import numpy as np

    scores = np.asarray([row["scores"] for row in examples], dtype=np.float32)
    log_costs = np.log(
        np.asarray([row["costs"] for row in examples], dtype=np.float32)
    )
    generations = np.asarray(
        [sum(row.get("generations", (2, 2, 2))) / 3.0 for row in examples],
        dtype=np.float32,
    )
    # 4회 생성 outcome은 잡음이 적지만 2회 생성 문항을 압도하면 안 됩니다.
    # 제곱근 가중치로 2배가 아닌 약 1.41배 영향만 줍니다.
    weights = np.sqrt(generations / max(1.0, float(generations.min()))).astype(
        np.float32
    )
    return scores, log_costs, weights


def fit_scaler(values: Any, indices: Any) -> dict[str, Any]:
    import numpy as np

    selected = values[indices]
    mean = selected.mean(axis=0).astype(np.float32)
    scale = selected.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return {"mean": mean, "scale": scale}


def apply_scaler(values: Any, scaler: Mapping[str, Any]) -> Any:
    return (values - scaler["mean"]) / scaler["scale"]


def _loader(
    tokenized: Mapping[str, Any],
    dense: Any,
    scores: Any,
    normalized_costs: Any,
    weights: Any,
    indices: Any,
    batch_size: int,
    shuffle: bool,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(tokenized["input_ids"][indices]),
        torch.from_numpy(tokenized["attention_mask"][indices]),
        torch.from_numpy(dense[indices]),
        torch.from_numpy(scores[indices]),
        torch.from_numpy(normalized_costs[indices].astype("float32")),
        torch.from_numpy(weights[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def initialize_biases(model: Any, score_targets: Any) -> None:
    import numpy as np
    import torch

    means = np.clip(score_targets.mean(axis=0), 1e-4, 1.0 - 1e-4)
    logits = np.log(means / (1.0 - means)).astype("float32")
    with torch.no_grad():
        model.score_head.bias.copy_(torch.from_numpy(logits))
        model.cost_head.bias.zero_()
        best = np.argmax(score_targets, axis=1)
        counts = np.bincount(best, minlength=3).astype("float32") + 1.0
        model.preference_bias.copy_(torch.log(torch.from_numpy(counts / counts.sum())))


def _run_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: str,
    *,
    objective: str = "baseline",
) -> dict[str, float]:
    import torch

    training = optimizer is not None
    model.train(training)
    if not any(parameter.requires_grad for parameter in model.encoder.parameters()):
        model.encoder.eval()
    totals: dict[str, float] = {}
    rows = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids, attention_mask, dense, scores, costs, weights = [
                item.to(device) for item in batch
            ]
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(input_ids, attention_mask, dense)
            loss, metrics = multitask_loss(
                output, scores, costs, weights, objective=objective
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
            count = input_ids.shape[0]
            rows += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / rows for key, value in totals.items()}


def train_one(
    *,
    encoder: str,
    tokenized: Mapping[str, Any],
    dense: Any,
    scores: Any,
    log_costs: Any,
    weights: Any,
    train_indices: Any,
    validation_indices: Optional[Any],
    cost_scaler: Mapping[str, Any],
    projection_dimension: int,
    dropout: float,
    unfreeze_last_n: int,
    epochs: int,
    batch_size: int,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    device: str,
    seed: int,
    local_files_only: bool,
    objective: str = "baseline",
) -> tuple[Any, list[dict[str, Any]]]:
    import numpy as np
    import torch

    set_seed(seed)
    model = RouterModel.create(
        encoder,
        dense_dimension=int(dense.shape[1]),
        projection_dimension=projection_dimension,
        dropout=dropout,
        unfreeze_last_n=unfreeze_last_n,
        local_files_only=local_files_only,
    )
    initialize_biases(model, scores[train_indices])
    model.to(device)
    normalized_costs = apply_scaler(log_costs, cost_scaler).astype(np.float32)
    train_loader = _loader(
        tokenized,
        dense,
        scores,
        normalized_costs,
        weights,
        train_indices,
        batch_size,
        True,
    )
    validation_loader = None
    if validation_indices is not None and len(validation_indices):
        validation_loader = _loader(
            tokenized,
            dense,
            scores,
            normalized_costs,
            weights,
            validation_indices,
            batch_size,
            False,
        )
    backbone = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    backbone_ids = {id(parameter) for parameter in backbone}
    heads = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups = [{"params": heads, "lr": head_learning_rate}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, optimizer, device, objective=objective
        )
        row: dict[str, Any] = {"epoch": epoch, "train": train_metrics}
        if validation_loader is not None:
            row["validation"] = _run_epoch(
                model,
                validation_loader,
                None,
                device,
                objective=objective,
            )
        row["elapsed_seconds"] = time.perf_counter() - started
        history.append(row)
        validation_text = (
            f" validation_loss={row['validation']['loss']:.6f}"
            if "validation" in row
            else ""
        )
        print(
            f"epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.6f}"
            f"{validation_text} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return model, history


def predict_model(
    model: Any,
    tokenized: Mapping[str, Any],
    dense: Any,
    indices: Any,
    cost_scaler: Mapping[str, Any],
    batch_size: int,
    device: str,
) -> tuple[Any, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(tokenized["input_ids"][indices]),
        torch.from_numpy(tokenized["attention_mask"][indices]),
        torch.from_numpy(dense[indices]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    score_rows = []
    cost_rows = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, dense_batch in loader:
            output = model(
                input_ids.to(device),
                attention_mask.to(device),
                dense_batch.to(device),
            )
            score_rows.append(output["scores"].cpu().numpy())
            cost_rows.append(output["costs"].cpu().numpy())
    scores = np.concatenate(score_rows, axis=0).astype(np.float64)
    normalized_costs = np.concatenate(cost_rows, axis=0).astype(np.float64)
    log_costs = normalized_costs * cost_scaler["scale"] + cost_scaler["mean"]
    return np.clip(scores, 0.0, 1.0), log_costs


def prediction_diagnostics(
    predicted_scores: Any,
    predicted_log_costs: Any,
    actual_scores: Any,
    actual_log_costs: Any,
) -> dict[str, Any]:
    import numpy as np

    score_error = predicted_scores - actual_scores
    cost_error = predicted_log_costs - actual_log_costs
    return {
        "score_mae": float(np.abs(score_error).mean()),
        "score_rmse": float(np.sqrt(np.square(score_error).mean())),
        "log_cost_mae": float(np.abs(cost_error).mean()),
        "best_model_accuracy": float(
            (predicted_scores.argmax(axis=1) == actual_scores.argmax(axis=1)).mean()
        ),
        "per_model_score_mae": {
            model_id: float(np.abs(score_error[:, index]).mean())
            for index, model_id in enumerate(MODEL_IDS)
        },
    }


def rows_from_arrays(scores: Any, log_costs: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    import numpy as np

    costs = np.exp(np.clip(log_costs, -50.0, 50.0))
    costs[:, 1] = np.maximum(costs[:, 1], costs[:, 0] * (1.0 + 1e-12))
    costs[:, 2] = np.maximum(costs[:, 2], costs[:, 1] * (1.0 + 1e-12))
    score_rows = [
        {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
        for row in scores
    ]
    cost_rows = [
        {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
        for row in costs
    ]
    return score_rows, cost_rows


def calibrate_safety_ratios(
    score_rows: Sequence[Mapping[str, float]],
    cost_rows: Sequence[Mapping[str, float]],
    actual_scores: Any,
    actual_costs: Any,
    *,
    reserve_ratio: float,
    grid_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    import numpy as np

    from experiments.common.budget import select_models

    if not 0 < reserve_ratio <= 1:
        raise ValueError("reserve_ratio must be in (0, 1]")
    actual_costs = np.asarray(actual_costs, dtype=np.float64)
    actual_scores = np.asarray(actual_scores, dtype=np.float64)
    light_total = float(actual_costs[:, 0].sum())
    safety_ratios: dict[str, float] = {}
    reports: dict[str, Any] = {}
    for tier in TIERS:
        lower = 1.0 / TIER_LIMITS[tier]
        best = None
        for safety in np.linspace(lower, 1.0, grid_size):
            selected, predicted_ratio = select_models(
                score_rows, cost_rows, tier, float(safety)
            )
            selected_index = np.asarray([MODEL_IDS.index(value) for value in selected])
            rows = np.arange(len(selected_index))
            actual_ratio = float(actual_costs[rows, selected_index].sum() / light_total)
            quality = float(actual_scores[rows, selected_index].mean())
            if actual_ratio > TIER_LIMITS[tier] * reserve_ratio + 1e-12:
                continue
            rank = (quality, -actual_ratio, -float(safety))
            if best is None or rank > best[0]:
                best = (rank, float(safety), predicted_ratio, actual_ratio, quality, selected)
        if best is None:
            selected = tuple(MODEL_IDS[0] for _ in score_rows)
            best = ((0.0, -1.0, -lower), lower, 1.0, 1.0, float(actual_scores[:, 0].mean()), selected)
        safety_ratios[tier] = best[1]
        reports[tier] = {
            "safety_ratio": best[1],
            "predicted_budget_ratio": best[2],
            "actual_budget_ratio": best[3],
            "quality_score": best[4],
            "selection_counts": {
                model_id: int(best[5].count(model_id)) for model_id in MODEL_IDS
            },
        }
    return safety_ratios, reports


def save_artifact(
    model: Any,
    tokenizer: Any,
    artifact_dir: Path,
    metadata: Mapping[str, Any],
) -> None:
    import torch
    from safetensors.torch import save_file

    artifact_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = artifact_dir / "encoder"
    model.encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(encoder_dir)
    head_state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if not key.startswith("encoder.")
    }
    save_file(head_state, str(artifact_dir / "router-heads.safetensors"))
    write_json_atomic(artifact_dir / "metadata.json", dict(metadata))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_artifact(artifact_dir: Path, device: str) -> tuple[Any, Any, Mapping[str, Any]]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from experiments.common.data import load_json

    metadata = load_json(artifact_dir / "metadata.json")
    encoder_dir = artifact_dir / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    model = RouterModel.create(
        str(encoder_dir),
        dense_dimension=int(metadata["dense_dimension"]),
        projection_dimension=int(metadata["projection_dimension"]),
        dropout=float(metadata["dropout"]),
        unfreeze_last_n=0,
        local_files_only=True,
    )
    state = load_file(str(artifact_dir / "router-heads.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise RuntimeError(f"invalid router head state: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model, tokenizer, metadata


def release_model(model: Any) -> None:
    del model
    gc.collect()
