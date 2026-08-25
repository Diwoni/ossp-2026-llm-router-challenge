# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""공개 baseline의 내용 기반 signed hash·정규식 특징을 계산합니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_HASH_BINS = 256


def matrix(
    examples: Sequence[Mapping[str, Any]], hash_bins: int = DEFAULT_HASH_BINS
) -> Any:
    """문항 메타데이터 없이 공개 baseline의 원시 특징을 반환합니다."""

    import numpy as np

    from baselines.hash_regex import raw_feature_vector
    from ossp_router.protocol import Episode

    if hash_bins <= 0:
        return np.empty((len(examples), 0), dtype=np.float32)
    rows = [
        raw_feature_vector(
            Episode(episode_id="content-only", prompt=str(example["text"])),
            hash_bins,
        )
        for example in examples
    ]
    return np.asarray(rows, dtype=np.float32)
