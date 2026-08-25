# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""유사 프롬프트 누수를 줄이는 내용 군집 fold를 만듭니다."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def stratified_folds(
    examples: Sequence[Mapping[str, Any]], folds: int, seed: int
) -> tuple[list[tuple[Any, Any]], list[int]]:
    """outcome 신뢰도와 난이도로 층화한 기본 OOF 분할을 만듭니다."""

    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    if folds < 2 or len(examples) < folds:
        raise ValueError("folds must be >= 2 and no larger than the example count")
    lengths = np.asarray([len(str(row["text"])) for row in examples])
    boundaries = np.quantile(lengths, [0.25, 0.50, 0.75])
    labels = []
    for row, length in zip(examples, lengths):
        best = int(np.argmax(np.asarray(row["scores"], dtype=np.float64)))
        generations = int(round(sum(row.get("generations", (2, 2, 2))) / 3.0))
        bucket = int(np.searchsorted(boundaries, length, side="right"))
        labels.append(f"{best}:{generations}:{bucket}")
    indices = np.arange(len(examples))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    fold_ids = np.empty(len(examples), dtype=np.int64)
    for fold, (train, validation) in enumerate(splitter.split(indices, labels)):
        result.append((train.astype(np.int64), validation.astype(np.int64)))
        fold_ids[validation] = fold
    return result, [int(value) for value in fold_ids]


def content_cluster_folds(
    examples: Sequence[Mapping[str, Any]], folds: int, seed: int
) -> tuple[list[tuple[Any, Any]], list[int]]:
    """비슷한 프롬프트를 군집화하고 각 군집을 하나의 fold에만 둡니다.

    무작위 행 분할은 거의 같은 생성 틀을 양쪽에 넣을 수 있습니다. 이 분할은
    의도적으로 더 엄격하며 공개 Dev를 미지 자료처럼 주장하지 않도록 함께
    보고합니다.
    """

    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold

    if folds < 2 or len(examples) < folds:
        raise ValueError("folds must be >= 2 and no larger than the example count")
    texts = []
    for row in examples:
        text = str(row["text"])
        # 인코더의 내용 기반 앞·뒤 보존 원칙을 문자 수준에도 적용합니다.
        texts.append(text if len(text) <= 8000 else text[:3500] + "\n" + text[-4500:])
    matrix = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=6000,
        sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(texts)
    cluster_count = min(max(folds * 8, 24), max(folds, len(examples) // 20))
    groups = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=256,
        n_init=5,
        random_state=seed,
    ).fit_predict(matrix)
    indices = np.arange(len(examples))
    result = [
        (train.astype(np.int64), validation.astype(np.int64))
        for train, validation in GroupKFold(n_splits=folds).split(indices, groups=groups)
    ]
    return result, [int(value) for value in groups]
