#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""비용 스트레스에서 재배분할 최종 품질·비용 예측 행을 한 번 추출합니다."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from baselines.hash_regex import load_artifact as load_hash_artifact
from baselines.hash_regex import predict_episode
from experiments.common.data import MODEL_IDS, TIERS, load_json, write_json_atomic
from experiments.method4_finetuned_encoder.final_route import (
    _canonical_examples,
    _hybrid_feature_matrix,
    _rows_from_delta,
    _similarity_delta,
)
from experiments.method4_finetuned_encoder.residual import predict_residual
from experiments.method4_finetuned_encoder.similarity import runtime_source_name
from experiments.method4_finetuned_encoder.teacher_student import apply_absolute_student
from ossp_router.protocol import load_input


def _restore(rows, canonical_indices):
    restored = [None] * len(rows)
    for canonical_index, original_index in enumerate(canonical_indices):
        restored[original_index] = rows[canonical_index]
    return restored


def _similarity_residual_rows(args, inputs, hash_artifact, gate, tier):
    import numpy as np

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
        tier,
        decimals=int(gate.get("decision_round_decimals", 4)),
    )
    residual_delta = np.asarray(
        [[row[MODEL_IDS[1]], row[MODEL_IDS[2]]] for row in residual_scores],
        dtype=np.float64,
    )
    similarity_delta = _similarity_delta(
        args, examples, gate, router_embeddings=router_embeddings
    )
    student_weight = float(gate.get("teacher_student_weight", 0.0))
    if student_weight > 0.0:
        similarity_delta, _mask = apply_absolute_student(
            similarity_delta,
            router_embeddings,
            examples,
            load_json(args.teacher_student_artifact),
            weight=student_weight,
            mode=str(gate.get("teacher_student_mode", "disagreement")),
            fraction=float(gate.get("teacher_student_fraction", 0.05)),
        )
    residual_weight = float(gate.get("residual_weight", 0.0))
    delta = residual_weight * residual_delta + (1.0 - residual_weight) * similarity_delta
    if tier == "fast":
        delta = delta[:, :1]
    score_rows = _rows_from_delta(
        delta, int(gate.get("decision_round_decimals", 4))
    )
    return (
        _restore(score_rows, canonical_indices),
        _restore(cost_rows, canonical_indices),
        _restore([runtime_source_name(row) for row in examples], canonical_indices),
    )


def _similarity_hash_rows(args, inputs, hash_artifact, gate):
    import numpy as np

    _original, canonical_indices, examples = _canonical_examples(args.input)
    hash_predictions = [
        predict_episode(inputs.episodes[index], hash_artifact)
        for index in canonical_indices
    ]
    hash_delta = np.asarray(
        [
            [
                scores[MODEL_IDS[1]] - scores[MODEL_IDS[0]],
                scores[MODEL_IDS[2]] - scores[MODEL_IDS[0]],
            ]
            for scores, _costs in hash_predictions
        ],
        dtype=np.float64,
    )
    similarity_delta = _similarity_delta(args, examples, gate)
    weight = float(gate["similarity_weight"])
    delta = (1.0 - weight) * hash_delta + weight * similarity_delta
    score_rows = _rows_from_delta(
        delta, int(gate.get("decision_round_decimals", 4))
    )
    cost_rows = [costs for _scores, costs in hash_predictions]
    return (
        _restore(score_rows, canonical_indices),
        _restore(cost_rows, canonical_indices),
        _restore([runtime_source_name(row) for row in examples], canonical_indices),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    runtime_args = SimpleNamespace(
        input=args.input,
        onnx_artifact_dir=args.assets / "onnx",
        encoder_artifact_dir=None,
        embedding_onnx_artifact_dir=args.assets / "embedding-onnx",
        stacking_artifact=args.assets / "stacking-head.json",
        residual_artifact=args.assets / "residual-head.json",
        similarity_artifact=args.assets / "similarity-head.npz",
        teacher_student_artifact=args.assets / "teacher-student-head.json",
        cost_calibration_artifact=None,
        batch_size=32,
        threads=args.threads,
        device="auto",
    )
    inputs = load_input(args.input)
    hash_artifact = load_hash_artifact(Path("baselines/hash-regex-public.v1.json"))
    gates = load_json(args.gate)
    tiers = {}
    for tier in TIERS:
        gate = gates[tier]
        if gate["router"] == "similarity-residual":
            score_rows, cost_rows, families = _similarity_residual_rows(
                runtime_args, inputs, hash_artifact, gate, tier
            )
        elif gate["router"] == "similarity-hash":
            score_rows, cost_rows, families = _similarity_hash_rows(
                runtime_args, inputs, hash_artifact, gate
            )
        else:
            raise ValueError(f"추출기가 지원하지 않는 라우터입니다: {gate['router']}")
        tiers[tier] = {
            "score_rows": score_rows,
            "cost_rows": cost_rows,
            "families": families,
            "safety_ratio": gate["safety_ratio"],
            "allowed_model_ids": gate["allowed_model_ids"],
        }
        print(f"최종 예측 행 추출 완료: {tier} {len(score_rows)}개")
    write_json_atomic(
        args.output,
        {
            "artifact_type": "os-skt-final-prediction-rows-v1",
            "episode_ids": [row.episode_id for row in inputs.episodes],
            "tiers": tiers,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
