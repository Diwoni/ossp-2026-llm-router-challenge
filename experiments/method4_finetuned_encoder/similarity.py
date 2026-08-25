# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""RouteLLM과 DecoR에서 착안한 유사 문항 검색 head입니다."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_NAMES = (
    "aime",
    "babilong",
    "belebele",
    "cruxeval",
    "deepmind-mathematics",
    "gsm8k",
    "hrmcr",
    "ruletaker",
    "truthfulqa",
)


def runtime_task_profile(example: Mapping[str, Any]) -> list[float]:
    """내용 기반 문제 특징만 반환하며 ID와 입력 순서는 읽지 않습니다."""

    text = str(example["text"])
    lower = text.lower()
    word_count = len(text.split())
    hangul_ratio = sum("\uac00" <= char <= "\ud7a3" for char in text) / max(
        1, len(text)
    )
    dollar_math = len(re.findall(r"\$[^\d\s]", text)) + len(
        re.findall(r"\$\d[^,. ]", text)
    )
    currency = len(re.findall(r"\$\s?\d[\d,.]*", text))
    choice_count = sum(marker in text for marker in ("A.", "B.", "C.", "D."))
    question_inside = "Question:" in text and not text.startswith("Question:")
    rule_language = bool(
        re.search(
            r"\b(if (?:someone|something|the)|all [a-z, ]+ (?:things|people) are|"
            r"young people are|[a-z]+ things are)\b",
            lower,
        )
    )
    story_math = bool(
        re.search(r"\b(how many|how much|how long|how far|how old|in total)\b", lower)
        and word_count >= 18
    )
    synthetic_math = bool(
        re.search(
            r"\b(calculate|simplify|expand|factorise|factorize|differentiate|"
            r"integrate|derivative|divided by|prime factors|convert|round|wrt)\b",
            lower,
        )
        or "**" in text
    )
    competition_math = bool(
        "\\[" in text
        or "\\frac" in text
        or "\\begin" in text
        or ("$" in text and currency == 0 and word_count >= 25)
        or re.search(
            r"\b(regular (?:polygon|octagon|dodecagon)|positive integers|"
            r"distinct roots|modulo|circumcircle|tangent to|collinear)\b",
            lower,
        )
    )
    code = "def f(" in text and "assert f(" in text
    long_context = len(text) >= 2000
    korean = hangul_ratio > 0.10
    return [
        float(code),
        float(long_context),
        float(korean),
        float(korean and "Question:" in text),
        float(korean and "Question:" not in text),
        float(text.startswith("Question:") and choice_count >= 2),
        float(question_inside and choice_count == 0),
        float(rule_language),
        float(story_math),
        float(synthetic_math),
        float(competition_math),
        math.log1p(choice_count),
        math.log1p(currency),
        math.log1p(dollar_math),
        math.log1p(text.count("\\")),
        math.log1p(text.count("?")),
        math.log1p(text.count("Question:")),
        math.log1p(text.count("If ") + text.count("All ")),
        math.log1p(text.count("def ") + text.count("assert ")),
        math.log1p(max(0, word_count - 20)),
    ]


def runtime_source_name(example: Mapping[str, Any]) -> str:
    profile = runtime_task_profile(example)
    if profile[0]:
        return "cruxeval"
    if profile[1]:
        return "babilong"
    if profile[4]:
        return "hrmcr"
    if profile[3]:
        return "belebele"
    if profile[5]:
        return "truthfulqa"
    if profile[6] or profile[7]:
        return "ruletaker"
    if profile[10]:
        return "aime"
    if profile[8] and not profile[9]:
        return "gsm8k"
    return "deepmind-mathematics"


def runtime_source_indices(examples: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    lookup = {name: index for index, name in enumerate(SOURCE_NAMES)}
    return np.asarray(
        [lookup[runtime_source_name(example)] for example in examples],
        dtype=np.uint8,
    )


def load_similarity_artifact(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path) as value:
        artifact = {name: value[name] for name in value.files}
    required = {"reference_embeddings", "target_delta", "source_indices"}
    if not required.issubset(artifact):
        raise ValueError(f"similarity artifact missing: {sorted(required - artifact.keys())}")
    if artifact["reference_embeddings"].shape[0] != artifact["target_delta"].shape[0]:
        raise ValueError("similarity artifact rows are not aligned")
    if "secondary_reference_embeddings" in artifact and (
        artifact["secondary_reference_embeddings"].shape[0]
        != artifact["target_delta"].shape[0]
    ):
        raise ValueError("secondary similarity artifact rows are not aligned")
    return artifact


def predict_similarity_delta(
    embeddings: Any,
    examples: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    neighbors: int,
    temperature: float,
    source_bonus: float,
    secondary_embeddings: Any | None = None,
    secondary_weight: float = 0.0,
) -> Any:
    import numpy as np

    reference = np.asarray(artifact["reference_embeddings"], dtype=np.float64)
    targets = np.asarray(artifact["target_delta"], dtype=np.float64)
    if not 0.0 <= secondary_weight <= 1.0:
        raise ValueError("secondary_weight must be in [0, 1]")
    similarities = None
    if secondary_weight < 1.0:
        if embeddings is None:
            raise ValueError("primary embeddings are required by the similarity gate")
        similarities = np.asarray(embeddings, dtype=np.float64) @ reference.T
    if secondary_weight > 0.0:
        if secondary_embeddings is None:
            raise ValueError("secondary embeddings are required by the similarity gate")
        if "secondary_reference_embeddings" not in artifact:
            raise ValueError("similarity artifact has no secondary reference embeddings")
        secondary_reference = np.asarray(
            artifact["secondary_reference_embeddings"], dtype=np.float64
        )
        secondary_query = np.asarray(secondary_embeddings, dtype=np.float64)
        secondary_similarity = secondary_query @ secondary_reference.T
        similarities = (
            secondary_similarity
            if similarities is None
            else (1.0 - secondary_weight) * similarities
            + secondary_weight * secondary_similarity
        )
    if source_bonus:
        query_source = runtime_source_indices(examples)
        reference_source = np.asarray(artifact["source_indices"], dtype=np.uint8)
        similarities += source_bonus * (query_source[:, None] == reference_source[None, :])
    count = min(int(neighbors), reference.shape[0])
    indices = np.argpartition(similarities, -count, axis=1)[:, -count:]
    selected_similarity = np.take_along_axis(similarities, indices, axis=1)
    logits = (selected_similarity - selected_similarity.max(axis=1, keepdims=True)) / float(
        temperature
    )
    weights = np.exp(np.clip(logits, -50.0, 0.0))
    weights /= weights.sum(axis=1, keepdims=True)
    return (targets[indices] * weights[:, :, None]).sum(axis=1)
