# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""방법 4의 프롬프트 토큰과 수작업 특징을 준비합니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.common.dense_features import matrix as dense_matrix


def head_tail_ids(tokenizer: Any, text: str, max_length: int) -> tuple[list[int], list[int]]:
    """질문의 시작과 끝에 있는 답·선택지를 함께 보존합니다.

    단순 오른쪽 자르기는 긴 독해·코드 프롬프트의 끝을 잃습니다. 이 변환은
    프롬프트 내용만 사용하며 문항 ID나 입력 순서 신호를 사용하지 않습니다.
    """

    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    raw = tokenizer(
        text, add_special_tokens=False, truncation=False, verbose=False
    )["input_ids"]
    prefix = []
    suffix = []
    if tokenizer.cls_token_id is not None:
        prefix = [int(tokenizer.cls_token_id)]
    elif tokenizer.bos_token_id is not None:
        prefix = [int(tokenizer.bos_token_id)]
    if tokenizer.sep_token_id is not None:
        suffix = [int(tokenizer.sep_token_id)]
    elif tokenizer.eos_token_id is not None:
        suffix = [int(tokenizer.eos_token_id)]
    budget = max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError("max_length is smaller than tokenizer special tokens")
    if len(raw) > budget:
        # Belebele·BABILong 형식은 선택지와 최종 질문이 끝에 자주 나오므로
        # 앞부분보다 뒷부분 토큰을 조금 더 많이 보존합니다.
        head = max(1, int(budget * 0.44))
        raw = raw[:head] + raw[-(budget - head) :]
    ids = prefix + [int(value) for value in raw] + suffix
    mask = [1] * len(ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    padding = max_length - len(ids)
    ids.extend([int(pad_id)] * padding)
    mask.extend([0] * padding)
    return ids, mask


def encode_examples(
    examples: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
    input_prefix: str = "",
) -> dict[str, Any]:
    import numpy as np

    rows = [
        head_tail_ids(tokenizer, input_prefix + str(row["text"]), max_length)
        for row in examples
    ]
    return {
        "input_ids": np.asarray([row[0] for row in rows], dtype=np.int64),
        "attention_mask": np.asarray([row[1] for row in rows], dtype=np.int64),
    }


def fit_dense_scaler(examples: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    values = dense_matrix(examples).astype(np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return ((values - mean) / scale).astype(np.float32), {"mean": mean, "scale": scale}


def transform_dense(
    examples: Sequence[Mapping[str, Any]], scaler: Mapping[str, Any]
) -> Any:
    import numpy as np

    values = dense_matrix(examples).astype(np.float32)
    return ((values - scaler["mean"]) / scaler["scale"]).astype(np.float32)
