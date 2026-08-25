# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""등급별로 검증된 전문가를 선택하는 최종 하이브리드 라우터입니다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from baselines.hash_regex import load_artifact as load_hash_artifact
from baselines.hash_regex import make_hash_regex_submission, predict_episode
from experiments.common.budget import select_models
from experiments.common.data import MODEL_IDS, load_json
from experiments.common.submission import write_submission
from experiments.method4_finetuned_encoder.cost_calibration import calibrate_cost_rows
from experiments.method4_finetuned_encoder.embedding_inference import (
    predict_embeddings_onnx,
)
from experiments.method4_finetuned_encoder.features import encode_examples, transform_dense
from experiments.method4_finetuned_encoder.onnx_inference import predict_onnx
from experiments.method4_finetuned_encoder.residual import predict_residual
from experiments.method4_finetuned_encoder.route import _runtime_examples
from experiments.method4_finetuned_encoder.similarity import (
    load_similarity_artifact,
    predict_similarity_delta,
)
from experiments.method4_finetuned_encoder.stacking import (
    arrays_to_rows,
    feature_matrix,
    predict_ridge,
)
from experiments.method4_finetuned_encoder.teacher_student import (
    apply_absolute_student,
)
from experiments.method4_finetuned_encoder.training import (
    device_name,
    load_artifact,
    predict_model,
    rows_from_arrays,
)
from ossp_router.protocol import load_bundled_policy, load_input


def canonical_content_indices(examples):
    return sorted(
        range(len(examples)),
        key=lambda index: (
            str(examples[index]["text"]),
            int(examples[index]["message_count"]),
        ),
    )


def _canonical_examples(input_path):
    original_examples = _runtime_examples(input_path)
    canonical_indices = canonical_content_indices(original_examples)
    examples = [original_examples[index] for index in canonical_indices]
    return original_examples, canonical_indices, examples


def _hybrid_feature_matrix(
    args, inputs, hash_artifact, *, return_router_embeddings=False
):
    import numpy as np

    _original_examples, canonical_indices, examples = _canonical_examples(args.input)
    # 양자화 행렬 연산은 같은 batch에 어떤 문항이 있느냐에 따라 마지막 비트가
    # 달라질 수 있습니다. 내용 기준 정렬로 ID·입력 순서 감사에서도 batch와
    # 예산 합계를 동일하게 유지합니다.
    if args.onnx_artifact_dir is not None:
        encoder_output = predict_onnx(
            examples,
            args.onnx_artifact_dir,
            batch_size=args.batch_size,
            threads=args.threads,
            return_router_embeddings=return_router_embeddings,
        )
        encoder_scores, encoder_log_costs = encoder_output[:2]
        router_embeddings = (
            encoder_output[2] if return_router_embeddings else None
        )
    else:
        if return_router_embeddings:
            raise ValueError(
                "router-embedding similarity requires --onnx-artifact-dir"
            )
        if args.encoder_artifact_dir is None:
            raise ValueError("hybrid routing needs an ONNX or PyTorch encoder artifact")
        import torch

        torch.set_num_threads(args.threads)
        device = device_name(args.device)
        model, tokenizer, encoder_metadata = load_artifact(
            args.encoder_artifact_dir, device
        )
        tokenized = encode_examples(
            examples,
            tokenizer,
            int(encoder_metadata["max_length"]),
            input_prefix=str(encoder_metadata.get("input_prefix", "")),
        )
        dense_scaler = {
            "mean": np.asarray(
                encoder_metadata["dense_scaler"]["mean"], dtype=np.float32
            ),
            "scale": np.asarray(
                encoder_metadata["dense_scaler"]["scale"], dtype=np.float32
            ),
        }
        dense = transform_dense(examples, dense_scaler)
        cost_scaler = {
            "mean": np.asarray(
                encoder_metadata["cost_scaler"]["mean"], dtype=np.float32
            ),
            "scale": np.asarray(
                encoder_metadata["cost_scaler"]["scale"], dtype=np.float32
            ),
        }
        indices = np.arange(len(examples), dtype=np.int64)
        encoder_scores, encoder_log_costs = predict_model(
            model,
            tokenized,
            dense,
            indices,
            cost_scaler,
            args.batch_size,
            device,
        )
        router_embeddings = None
    encoder_score_rows, encoder_cost_rows = rows_from_arrays(
        encoder_scores, encoder_log_costs
    )
    encoder_rows = [
        {"scores": scores, "costs": costs}
        for scores, costs in zip(encoder_score_rows, encoder_cost_rows)
    ]
    hash_predictions = [
        predict_episode(inputs.episodes[index], hash_artifact)
        for index in canonical_indices
    ]
    hash_rows = [
        {"scores": scores, "costs": costs}
        for scores, costs in hash_predictions
    ]
    return (
        canonical_indices,
        examples,
        feature_matrix(examples, hash_rows, encoder_rows),
        router_embeddings,
    )


def _restore_order(selected, canonical_indices):
    restored = [""] * len(selected)
    for canonical_index, original_index in enumerate(canonical_indices):
        restored[original_index] = selected[canonical_index]
    return tuple(restored)


def _stacking_decisions(args, inputs, hash_artifact, gate):
    import numpy as np

    canonical_indices, _examples, matrix, _router_embeddings = _hybrid_feature_matrix(
        args, inputs, hash_artifact
    )
    stacking = load_json(args.stacking_artifact)
    predictions = predict_ridge(
        matrix,
        np.asarray(stacking["feature_mean"], dtype=np.float64),
        np.asarray(stacking["feature_scale"], dtype=np.float64),
        np.asarray(stacking["intercept"], dtype=np.float64),
        np.asarray(stacking["coefficients"], dtype=np.float64),
    )
    predictions = np.round(
        predictions, decimals=int(gate.get("decision_round_decimals", 4))
    )
    score_rows, cost_rows = arrays_to_rows(predictions)
    selected, predicted_ratio = select_models(
        score_rows,
        cost_rows,
        "fast",
        float(gate["safety_ratio"]),
        allowed_model_ids=gate["allowed_model_ids"],
    )
    return _restore_order(selected, canonical_indices), predicted_ratio


def _residual_decisions(args, inputs, hash_artifact, gate):
    canonical_indices, _examples, matrix, _router_embeddings = _hybrid_feature_matrix(
        args, inputs, hash_artifact
    )
    artifact = load_json(args.residual_artifact)
    score_rows, cost_rows = predict_residual(
        matrix,
        artifact,
        args.tier,
        decimals=int(gate.get("decision_round_decimals", 4)),
    )
    if getattr(args, "cost_calibration_artifact", None) is not None:
        cost_rows = calibrate_cost_rows(
            cost_rows,
            _examples,
            load_json(args.cost_calibration_artifact),
            args.tier,
        )
    selected, predicted_ratio = select_models(
        score_rows,
        cost_rows,
        args.tier,
        float(gate["safety_ratio"]),
        allowed_model_ids=gate["allowed_model_ids"],
    )
    return _restore_order(selected, canonical_indices), predicted_ratio


def _similarity_delta(args, examples, gate, router_embeddings=None):
    secondary_weight = float(gate.get("secondary_embedding_weight", 0.0))
    if secondary_weight < 1.0 and args.embedding_onnx_artifact_dir is None:
        raise ValueError("similarity routing needs --embedding-onnx-artifact-dir")
    if args.similarity_artifact is None:
        raise ValueError("similarity routing needs --similarity-artifact")
    embeddings = None
    if secondary_weight < 1.0:
        embeddings = predict_embeddings_onnx(
            examples,
            args.embedding_onnx_artifact_dir,
            batch_size=args.batch_size,
            threads=args.threads,
            model_filename=str(
                gate.get("embedding_model_filename", "embedding-fp32.onnx")
            ),
        )
    return predict_similarity_delta(
        embeddings,
        examples,
        load_similarity_artifact(args.similarity_artifact),
        neighbors=int(gate["similarity_neighbors"]),
        temperature=float(gate["similarity_temperature"]),
        source_bonus=float(gate.get("similarity_source_bonus", 0.0)),
        secondary_embeddings=router_embeddings,
        secondary_weight=secondary_weight,
    )


def _rows_from_delta(delta, decimals):
    import numpy as np

    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] not in (1, 2):
        raise ValueError("delta must contain ax31 and optional think columns")
    if delta.shape[1] == 1:
        values = np.column_stack(
            [
                np.zeros(len(delta), dtype=np.float64),
                delta[:, 0],
                np.full(len(delta), -1.0, dtype=np.float64),
            ]
        )
    else:
        values = np.column_stack([np.zeros(len(delta), dtype=np.float64), delta])
    values = np.round(values, decimals=decimals)
    return [
        {model_id: float(row[index]) for index, model_id in enumerate(MODEL_IDS)}
        for row in values
    ]


def _similarity_residual_decisions(args, inputs, hash_artifact, gate):
    import numpy as np

    if args.residual_artifact is None:
        raise ValueError("similarity-residual tier needs --residual-artifact")
    secondary_weight = float(gate.get("secondary_embedding_weight", 0.0))
    canonical_indices, examples, matrix, router_embeddings = _hybrid_feature_matrix(
        args,
        inputs,
        hash_artifact,
        return_router_embeddings=secondary_weight > 0.0,
    )
    residual_scores, cost_rows = predict_residual(
        matrix,
        load_json(args.residual_artifact),
        args.tier,
        decimals=int(gate.get("decision_round_decimals", 4)),
    )
    if getattr(args, "cost_calibration_artifact", None) is not None:
        cost_rows = calibrate_cost_rows(
            cost_rows,
            examples,
            load_json(args.cost_calibration_artifact),
            args.tier,
        )
    residual_delta = np.asarray(
        [
            [row[MODEL_IDS[1]], row[MODEL_IDS[2]]]
            for row in residual_scores
        ],
        dtype=np.float64,
    )
    similarity_delta = _similarity_delta(
        args, examples, gate, router_embeddings=router_embeddings
    )
    teacher_student_weight = float(gate.get("teacher_student_weight", 0.0))
    if teacher_student_weight > 0.0:
        if args.teacher_student_artifact is None:
            raise ValueError("교사-학생 라우팅에는 학생 head 자산이 필요합니다")
        if router_embeddings is None:
            raise ValueError("교사-학생 라우팅에는 E5 라우터 임베딩이 필요합니다")
        teacher_student = load_json(args.teacher_student_artifact)
        if str(teacher_student.get("target_kind", "absolute")) != "absolute":
            raise ValueError("런타임 보정기는 absolute 학생 출력만 지원합니다")
        similarity_delta, _student_mask = apply_absolute_student(
            similarity_delta,
            router_embeddings,
            examples,
            teacher_student,
            weight=teacher_student_weight,
            mode=str(gate.get("teacher_student_mode", "disagreement")),
            fraction=float(gate.get("teacher_student_fraction", 0.05)),
        )
    residual_weight = float(gate.get("residual_weight", 0.0))
    delta = residual_weight * residual_delta + (1.0 - residual_weight) * similarity_delta
    if args.tier == "fast":
        delta = delta[:, :1]
    score_rows = _rows_from_delta(
        delta, int(gate.get("decision_round_decimals", 4))
    )
    selected, predicted_ratio = select_models(
        score_rows,
        cost_rows,
        args.tier,
        float(gate["safety_ratio"]),
        allowed_model_ids=gate["allowed_model_ids"],
    )
    return _restore_order(selected, canonical_indices), predicted_ratio


def _similarity_hash_decisions(args, inputs, hash_artifact, gate):
    import numpy as np

    _original_examples, canonical_indices, examples = _canonical_examples(args.input)
    hash_predictions = [
        predict_episode(inputs.episodes[index], hash_artifact)
        for index in canonical_indices
    ]
    hash_delta = np.asarray(
        [
            [scores[MODEL_IDS[1]] - scores[MODEL_IDS[0]], scores[MODEL_IDS[2]] - scores[MODEL_IDS[0]]]
            for scores, _costs in hash_predictions
        ],
        dtype=np.float64,
    )
    similarity_delta = _similarity_delta(args, examples, gate)
    similarity_weight = float(gate["similarity_weight"])
    delta = (1.0 - similarity_weight) * hash_delta + similarity_weight * similarity_delta
    score_rows = _rows_from_delta(
        delta, int(gate.get("decision_round_decimals", 4))
    )
    cost_rows = [costs for _scores, costs in hash_predictions]
    if getattr(args, "cost_calibration_artifact", None) is not None:
        cost_rows = calibrate_cost_rows(
            cost_rows,
            examples,
            load_json(args.cost_calibration_artifact),
            args.tier,
        )
    selected, predicted_ratio = select_models(
        score_rows,
        cost_rows,
        args.tier,
        float(gate["safety_ratio"]),
        allowed_model_ids=gate["allowed_model_ids"],
    )
    return _restore_order(selected, canonical_indices), predicted_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=("fast", "balanced", "premium"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-artifact-dir", type=Path)
    parser.add_argument("--onnx-artifact-dir", type=Path)
    parser.add_argument("--embedding-onnx-artifact-dir", type=Path)
    parser.add_argument("--hash-artifact", type=Path, required=True)
    parser.add_argument("--stacking-artifact", type=Path, required=True)
    parser.add_argument("--residual-artifact", type=Path)
    parser.add_argument("--similarity-artifact", type=Path)
    parser.add_argument("--cost-calibration-artifact", type=Path)
    parser.add_argument("--teacher-student-artifact", type=Path)
    parser.add_argument("--tier-gate", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    inputs = load_input(args.input)
    policy = load_bundled_policy()
    hash_artifact = load_hash_artifact(args.hash_artifact)
    gate = load_json(args.tier_gate)[args.tier]
    if gate["router"] == "stacking":
        selected, predicted_ratio = _stacking_decisions(
            args, inputs, hash_artifact, gate
        )
    elif gate["router"] == "residual-forest":
        if args.residual_artifact is None:
            raise ValueError("residual tier needs --residual-artifact")
        selected, predicted_ratio = _residual_decisions(
            args, inputs, hash_artifact, gate
        )
    elif gate["router"] == "similarity-residual":
        selected, predicted_ratio = _similarity_residual_decisions(
            args, inputs, hash_artifact, gate
        )
    elif gate["router"] == "similarity-hash":
        selected, predicted_ratio = _similarity_hash_decisions(
            args, inputs, hash_artifact, gate
        )
    elif gate["router"] == "hash-regex-public":
        plan = make_hash_regex_submission(inputs, policy, hash_artifact, args.tier)
        selected = tuple(decision.model_id for decision in plan.submission.decisions)
        predicted_ratio = plan.predicted_budget_ratio
    else:
        raise ValueError(f"unsupported tier router: {gate['router']}")
    write_submission(args.input, args.tier, selected, args.output)
    print(
        f"tier={args.tier} router={gate['router']} "
        f"predicted_budget_ratio={predicted_ratio:.6f} "
        f"elapsed_seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
