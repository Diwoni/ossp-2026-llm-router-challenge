# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""학습된 방법 4 자산을 실행하고 선택적으로 공개 outcome을 채점합니다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.common.budget import select_models
from experiments.common.data import TIERS, load_examples, load_json
from experiments.common.data import write_json_atomic
from experiments.common.submission import evaluate_official, write_submission
from experiments.method4_finetuned_encoder.features import encode_examples, transform_dense
from experiments.method4_finetuned_encoder.training import (
    device_name,
    load_artifact,
    predict_model,
    rows_from_arrays,
)


def _runtime_examples(input_path: Path) -> list[dict[str, object]]:
    root = load_json(input_path)
    rows = []
    for episode in root["episodes"]:
        prompt = episode.get("prompt")
        text = prompt if isinstance(prompt, str) else "\n".join(
            str(message["content"]) for message in episode["messages"]
        )
        messages = episode.get("messages")
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "text": text,
                "message_count": len(messages) if isinstance(messages, list) else 1,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    started = time.perf_counter()
    examples = (
        load_examples(args.input, args.outcomes)
        if args.outcomes is not None
        else _runtime_examples(args.input)
    )
    model, tokenizer, metadata = load_artifact(args.artifact_dir, device)
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
    predicted_scores, predicted_log_costs = predict_model(
        model,
        tokenized,
        dense,
        indices,
        cost_scaler,
        args.batch_size,
        device,
    )
    score_rows, cost_rows = rows_from_arrays(predicted_scores, predicted_log_costs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "predictions.json",
        {
            "method": "method4",
            "rows": [
                {
                    "episode_id": example["episode_id"],
                    "scores": scores,
                    "costs": costs,
                }
                for example, scores, costs in zip(examples, score_rows, cost_rows)
            ],
        },
    )
    for tier in TIERS:
        selected, predicted_ratio = select_models(
            score_rows,
            cost_rows,
            tier,
            float(metadata["safety_ratios"][tier]),
        )
        write_submission(args.input, tier, selected, args.output_dir / f"{tier}.json")
        print(
            f"{tier}: predicted_budget_ratio={predicted_ratio:.6f} "
            f"counts={dict((model_id, selected.count(model_id)) for model_id in metadata['model_ids'])}",
            flush=True,
        )
    if args.outcomes is not None:
        report = evaluate_official(
            args.input,
            args.outcomes,
            args.output_dir,
            args.output_dir / "report.json",
        )
        print(f"method4 final_score={report['final_score']}", flush=True)
    print(f"route elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
