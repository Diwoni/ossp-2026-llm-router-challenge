# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""동결한 INT8 ONNX 인코더로 Torch 없이 정규화 임베딩을 계산합니다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.common.data import load_json
from experiments.method4_finetuned_encoder.features import encode_examples
from experiments.method4_finetuned_encoder.onnx_inference import TokenizerAdapter


def predict_embeddings_onnx(
    examples: Sequence[Mapping[str, Any]],
    artifact_dir: Path,
    *,
    batch_size: int,
    threads: int,
    model_filename: str = "embedding-int8.onnx",
) -> Any:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

    import numpy as np
    import onnxruntime as ort

    metadata = load_json(artifact_dir / "metadata.json")
    tokenizer = TokenizerAdapter(artifact_dir / "tokenizer.json")
    tokenized = encode_examples(
        examples,
        tokenizer,
        int(metadata["max_length"]),
        input_prefix=str(metadata.get("input_prefix", "")),
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(artifact_dir / model_filename),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    batches = []
    for start in range(0, len(examples), batch_size):
        stop = min(len(examples), start + batch_size)
        batches.append(
            session.run(
                None,
                {
                    "input_ids": tokenized["input_ids"][start:stop],
                    "attention_mask": tokenized["attention_mask"][start:stop],
                },
            )[0]
        )
    embeddings = np.concatenate(batches, axis=0).astype(np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)
