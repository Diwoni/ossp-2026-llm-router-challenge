# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""최종 컨테이너에서 Torch 없이 INT8 ONNX 추론을 실행합니다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.common.data import load_json
from experiments.method4_finetuned_encoder.features import encode_examples, transform_dense


class TokenizerAdapter:
    """`head_tail_ids`에 필요한 최소 토크나이저 인터페이스를 제공합니다."""

    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.cls_token_id = self._tokenizer.token_to_id("<s>")
        self.bos_token_id = self.cls_token_id
        self.sep_token_id = self._tokenizer.token_to_id("</s>")
        self.eos_token_id = self.sep_token_id
        self.pad_token_id = self._tokenizer.token_to_id("<pad>")

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        # 이 SentencePiece 모델의 AutoTokenizer는 끝 공백을 제거하지만 기본
        # tokenizers.Tokenizer는 제거하지 않습니다. 긴 입력의 보존 토큰이 달라져
        # 학습·ONNX 결과가 어긋나지 않도록 학습 토크나이저와 맞춥니다.
        encoding = self._tokenizer.encode(text.rstrip(), add_special_tokens=False)
        return {"input_ids": [int(value) for value in encoding.ids]}


def predict_onnx(
    examples: Sequence[Mapping[str, Any]],
    artifact_dir: Path,
    *,
    batch_size: int,
    threads: int,
    return_router_embeddings: bool = False,
) -> tuple[Any, Any] | tuple[Any, Any, Any]:
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
    dense_scaler = {
        "mean": np.asarray(metadata["dense_scaler"]["mean"], dtype=np.float32),
        "scale": np.asarray(metadata["dense_scaler"]["scale"], dtype=np.float32),
    }
    dense = transform_dense(examples, dense_scaler)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(artifact_dir / "router-int8.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    score_batches = []
    cost_batches = []
    router_embedding_batches = []
    output_names = ["scores", "normalized_log_costs"]
    if return_router_embeddings:
        available = {value.name for value in session.get_outputs()}
        if "router_embeddings" not in available:
            raise ValueError(
                "ONNX artifact does not expose router_embeddings; re-export it"
            )
        output_names.append("router_embeddings")
    for start in range(0, len(examples), batch_size):
        stop = min(len(examples), start + batch_size)
        outputs = session.run(
            output_names,
            {
                "input_ids": tokenized["input_ids"][start:stop],
                "attention_mask": tokenized["attention_mask"][start:stop],
                "dense": dense[start:stop],
            },
        )
        scores, normalized_costs = outputs[:2]
        score_batches.append(scores)
        cost_batches.append(normalized_costs)
        if return_router_embeddings:
            router_embedding_batches.append(outputs[2])
    scores = np.concatenate(score_batches, axis=0).astype(np.float64)
    normalized_costs = np.concatenate(cost_batches, axis=0).astype(np.float64)
    mean = np.asarray(metadata["cost_scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(metadata["cost_scaler"]["scale"], dtype=np.float64)
    result = (np.clip(scores, 0.0, 1.0), normalized_costs * scale + mean)
    if not return_router_embeddings:
        return result
    router_embeddings = np.concatenate(router_embedding_batches, axis=0).astype(
        np.float64
    )
    router_embeddings /= np.maximum(
        np.linalg.norm(router_embeddings, axis=1, keepdims=True), 1e-12
    )
    return result + (router_embeddings,)
