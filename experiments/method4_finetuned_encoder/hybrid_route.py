# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""OOF hash-regex와 파인튜닝 인코더를 결합해 라우팅합니다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from baselines.hash_regex import load_artifact as load_hash_artifact
from baselines.hash_regex import predict_episode
from experiments.common.budget import select_models
from experiments.common.data import MODEL_IDS, TIERS, load_examples, load_json
from experiments.common.submission import evaluate_official, write_submission
from experiments.method4_finetuned_encoder.features import encode_examples, transform_dense
from experiments.method4_finetuned_encoder.route import _runtime_examples
from experiments.method4_finetuned_encoder.training import (
    device_name,
    load_artifact,
    predict_model,
    rows_from_arrays,
)
from ossp_router.protocol import load_input


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--encoder-artifact-dir", type=Path, required=True)
    parser.add_argument("--hash-artifact", type=Path, required=True)
    parser.add_argument("--hybrid-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    examples = (
        load_examples(args.input, args.outcomes)
        if args.outcomes is not None
        else _runtime_examples(args.input)
    )
    device = device_name(args.device)
    model, tokenizer, metadata = load_artifact(args.encoder_artifact_dir, device)
    tokenized = encode_examples(
        examples,
        tokenizer,
        int(metadata["max_length"]),
        input_prefix=str(metadata.get("input_prefix", "")),
    )
    dense_scaler = {
        "mean": np.asarray(metadata["dense_scaler"]["mean"], dtype=np.float32),
        "scale": np.asarray(metadata["dense_scaler"]["scale"], dtype=np.float32),
    }
    dense = transform_dense(examples, dense_scaler)
    cost_scaler = {
        "mean": np.asarray(metadata["cost_scaler"]["mean"], dtype=np.float32),
        "scale": np.asarray(metadata["cost_scaler"]["scale"], dtype=np.float32),
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
    encoder_score_rows, _encoder_cost_rows = rows_from_arrays(
        encoder_scores, encoder_log_costs
    )
    inputs = load_input(args.input)
    hash_artifact = load_hash_artifact(args.hash_artifact)
    hash_predictions = [
        predict_episode(episode, hash_artifact) for episode in inputs.episodes
    ]
    hash_scores = [row[0] for row in hash_predictions]
    hash_costs = [row[1] for row in hash_predictions]
    policy = load_json(args.hybrid_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tier in TIERS:
        tier_policy = policy["tiers"][tier]
        hash_weight = float(tier_policy["hash_score_weight"])
        blended = [
            {
                model_id: (
                    hash_weight * left[model_id]
                    + (1.0 - hash_weight) * right[model_id]
                )
                for model_id in MODEL_IDS
            }
            for left, right in zip(hash_scores, encoder_score_rows)
        ]
        selected, predicted_ratio = select_models(
            blended,
            hash_costs,
            tier,
            float(tier_policy["safety_ratio"]),
        )
        write_submission(args.input, tier, selected, args.output_dir / f"{tier}.json")
        print(
            f"{tier}: predicted_budget_ratio={predicted_ratio:.6f} "
            f"hash_weight={hash_weight:.2f} counts="
            f"{dict((model_id, selected.count(model_id)) for model_id in MODEL_IDS)}"
        )
    if args.outcomes is not None:
        report = evaluate_official(
            args.input,
            args.outcomes,
            args.output_dir,
            args.output_dir / "report.json",
        )
        print(f"method4-hybrid final_score={report['final_score']}")
    print(f"hybrid route elapsed_seconds={time.perf_counter() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
