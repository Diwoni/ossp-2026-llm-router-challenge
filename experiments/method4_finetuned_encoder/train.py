# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""그룹 OOF 검증 뒤 공개 Train 전체로 방법 4를 최종 학습합니다."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from experiments.common.budget import select_models
from experiments.common.data import MODEL_IDS, TIERS, load_examples, write_json_atomic
from experiments.common.dense_features import FEATURE_NAMES, matrix as dense_matrix
from experiments.common.submission import evaluate_official, write_submission
from experiments.method4_finetuned_encoder.features import encode_examples
from experiments.method4_finetuned_encoder.splits import (
    content_cluster_folds,
    stratified_folds,
)
from experiments.method4_finetuned_encoder.training import (
    calibrate_safety_ratios,
    device_name,
    fit_scaler,
    predict_model,
    prediction_diagnostics,
    release_model,
    rows_from_arrays,
    save_artifact,
    set_seed,
    target_arrays,
    train_one,
)


def _serializable_scaler(scaler: dict[str, object]) -> dict[str, list[float]]:
    return {
        "mean": [float(value) for value in scaler["mean"]],  # type: ignore[index]
        "scale": [float(value) for value in scaler["scale"]],  # type: ignore[index]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--input-prefix", default="")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--projection-dimension", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--unfreeze-last-n", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified", "content-cluster"),
        default="stratified",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--budget-reserve", type=float, default=0.985)
    parser.add_argument("--safety-grid-size", type=int, default=151)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="never contact the model hub; recommended after the encoder is downloaded",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("epochs/batch-size/max-length are outside the valid range")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = device_name(args.device)
    started = time.perf_counter()
    examples = load_examples(args.input, args.outcomes)
    print(
        f"method4 load examples={len(examples)} encoder={args.encoder} device={device}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder, local_files_only=args.local_files_only
    )
    tokenized = encode_examples(
        examples, tokenizer, args.max_length, input_prefix=args.input_prefix
    )
    raw_dense = dense_matrix(examples).astype(np.float32)
    scores, log_costs, weights = target_arrays(examples)
    if args.split_strategy == "content-cluster":
        folds, groups = content_cluster_folds(examples, args.cv_folds, args.seed)
    else:
        folds, groups = stratified_folds(examples, args.cv_folds, args.seed)
    oof_scores = np.zeros_like(scores, dtype=np.float64)
    oof_log_costs = np.zeros_like(log_costs, dtype=np.float64)
    fold_reports = []
    history_rows = []

    for fold_index, (train_indices, validation_indices) in enumerate(folds):
        print(
            f"OOF fold={fold_index + 1}/{len(folds)} "
            f"train={len(train_indices)} validation={len(validation_indices)}",
            flush=True,
        )
        dense_scaler = fit_scaler(raw_dense, train_indices)
        fold_dense = ((raw_dense - dense_scaler["mean"]) / dense_scaler["scale"]).astype(
            np.float32
        )
        cost_scaler = fit_scaler(log_costs, train_indices)
        model, history = train_one(
            encoder=args.encoder,
            tokenized=tokenized,
            dense=fold_dense,
            scores=scores,
            log_costs=log_costs,
            weights=weights,
            train_indices=train_indices,
            validation_indices=validation_indices,
            cost_scaler=cost_scaler,
            projection_dimension=args.projection_dimension,
            dropout=args.dropout,
            unfreeze_last_n=args.unfreeze_last_n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_learning_rate=args.backbone_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed + fold_index,
            local_files_only=args.local_files_only,
        )
        predicted_scores, predicted_log_costs = predict_model(
            model,
            tokenized,
            fold_dense,
            validation_indices,
            cost_scaler,
            args.batch_size * 2,
            device,
        )
        oof_scores[validation_indices] = predicted_scores
        oof_log_costs[validation_indices] = predicted_log_costs
        diagnostics = prediction_diagnostics(
            predicted_scores,
            predicted_log_costs,
            scores[validation_indices],
            log_costs[validation_indices],
        )
        fold_reports.append(
            {
                "fold": fold_index + 1,
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                **diagnostics,
            }
        )
        history_rows.append({"fold": fold_index + 1, "epochs": history})
        model.to("cpu")
        del model
        gc.collect()

    score_rows, cost_rows = rows_from_arrays(oof_scores, oof_log_costs)
    actual_costs = np.asarray([row["costs"] for row in examples], dtype=np.float64)
    safety_ratios, safety_report = calibrate_safety_ratios(
        score_rows,
        cost_rows,
        scores,
        actual_costs,
        reserve_ratio=args.budget_reserve,
        grid_size=args.safety_grid_size,
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    oof_submission_dir = args.work_dir / "oof-submissions"
    for tier in TIERS:
        selected, _predicted_ratio = select_models(
            score_rows, cost_rows, tier, safety_ratios[tier]
        )
        write_submission(args.input, tier, selected, oof_submission_dir / f"{tier}.json")
    official_oof = evaluate_official(
        args.input,
        args.outcomes,
        oof_submission_dir,
        args.work_dir / "oof-official-report.json",
    )
    oof_diagnostics = prediction_diagnostics(
        oof_scores, oof_log_costs, scores, log_costs
    )
    write_json_atomic(
        args.work_dir / "oof-predictions.json",
        {
            "artifact_type": "os-skt-method4-oof-predictions-v1",
            "episodes": [
                {
                    "episode_id": example["episode_id"],
                    "scores": score_row,
                    "costs": cost_row,
                    "validation_group": groups[index],
                    "content_cluster": groups[index],
                }
                for index, (example, score_row, cost_row) in enumerate(
                    zip(examples, score_rows, cost_rows)
                )
            ],
            "model_ids": list(MODEL_IDS),
        },
    )

    # 공개 Train 전체로 한 번 재학습하며 Dev는 gradient 갱신에 사용하지 않습니다.
    all_indices = np.arange(len(examples), dtype=np.int64)
    dense_scaler = fit_scaler(raw_dense, all_indices)
    full_dense = ((raw_dense - dense_scaler["mean"]) / dense_scaler["scale"]).astype(
        np.float32
    )
    cost_scaler = fit_scaler(log_costs, all_indices)
    print("final full-data fit", flush=True)
    final_model, final_history = train_one(
        encoder=args.encoder,
        tokenized=tokenized,
        dense=full_dense,
        scores=scores,
        log_costs=log_costs,
        weights=weights,
        train_indices=all_indices,
        validation_indices=None,
        cost_scaler=cost_scaler,
        projection_dimension=args.projection_dimension,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_learning_rate=args.backbone_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed + len(folds),
        local_files_only=args.local_files_only,
    )
    metadata = {
        "artifact_type": "os-skt-method4-finetuned-encoder-v1",
        "backbone_learning_rate": args.backbone_learning_rate,
        "budget_reserve": args.budget_reserve,
        "cost_scaler": _serializable_scaler(cost_scaler),
        "cost_target": "standardized_log_cost_smooth_l1_batch_mean",
        "cv_folds": args.cv_folds,
        "split_strategy": args.split_strategy,
        "dense_dimension": len(FEATURE_NAMES),
        "dense_feature_names": list(FEATURE_NAMES),
        "dense_scaler": _serializable_scaler(dense_scaler),
        "dropout": args.dropout,
        "encoder_source": args.encoder,
        "epochs": args.epochs,
        "head_learning_rate": args.head_learning_rate,
        "input_strategy": "content-only-token-head44-tail56",
        "input_prefix": args.input_prefix,
        "max_length": args.max_length,
        "model_ids": list(MODEL_IDS),
        "projection_dimension": args.projection_dimension,
        "safety_ratios": safety_ratios,
        "seed": args.seed,
        "train_examples": len(examples),
        "unfreeze_last_n": args.unfreeze_last_n,
        "weight_decay": args.weight_decay,
    }
    save_artifact(final_model, tokenizer, args.artifact_dir, metadata)
    elapsed = time.perf_counter() - started
    report = {
        **metadata,
        "elapsed_seconds": elapsed,
        "fold_reports": fold_reports,
        "oof_diagnostics": oof_diagnostics,
        "oof_official_score": official_oof,
        "oof_safety_calibration": safety_report,
    }
    write_json_atomic(args.work_dir / "train-report.json", report)
    write_json_atomic(
        args.work_dir / "training-history.json",
        {"cv": history_rows, "final": final_history},
    )
    print(
        f"method4 OOF final_score={official_oof['final_score']} "
        f"elapsed={elapsed:.1f}s artifact={args.artifact_dir}",
        flush=True,
    )
    release_model(final_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
