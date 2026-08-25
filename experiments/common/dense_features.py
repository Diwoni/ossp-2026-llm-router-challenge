# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""학습형 라우터가 공유하는 프롬프트 내용 기반 수작업 특징입니다."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


_WORD = re.compile(r"[A-Za-z가-힣]+")
_SENTENCE = re.compile(r"[.!?。！？]")
_CODE = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include|assert)\b|[{};]",
    re.IGNORECASE | re.MULTILINE,
)
_MATH = re.compile(r"[=+\-*/^∑∫√≈≠≤≥<>]|\\(?:frac|sum|int|sqrt|begin)\b")
_REASON = re.compile(
    r"\b(?:prove|derive|reason|analyze|theorem|counterexample|algorithm|complexity|"
    r"증명|유도|추론|분석|정리|반례|알고리즘|복잡도)\b",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)
_CHOICE = re.compile(r"(?:^|\n)\s*(?:[A-Da-d][.)]|[①-⑤]|\([1-5]\))", re.MULTILINE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "latin_ratio",
    "numeric_density",
    "whitespace_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "log_reasoning_marker_count",
    "log_constraint_marker_count",
    "simple_transform",
    "multiple_choice",
    "log_list_line_count",
    "log_url_count",
    "long_2k",
    "long_8k",
    "question_mark_count_log",
    "newline_count_log",
)


def one(example: Mapping[str, Any]) -> list[float]:
    text = str(example["text"])
    characters = len(text)
    nonspace = max(1, sum(not character.isspace() for character in text))
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    latin = sum(character.isascii() and character.isalpha() for character in text)
    numbers = sum(character.isdigit() for character in text)
    list_lines = sum(
        bool(re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line))
        for line in text.splitlines()
    )
    return [
        math.log1p(characters),
        math.log1p(len(_WORD.findall(text))),
        math.log1p(max(1, len(_SENTENCE.findall(text)))),
        math.log1p(int(example.get("message_count", 1))),
        hangul / nonspace,
        latin / nonspace,
        numbers / nonspace,
        sum(character.isspace() for character in text) / max(1, characters),
        math.log1p(len(_CODE.findall(text))),
        math.log1p(len(_MATH.findall(text))),
        math.log1p(len(_REASON.findall(text))),
        math.log1p(len(_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE.search(text))),
        float(bool(_CHOICE.search(text))),
        math.log1p(list_lines),
        math.log1p(len(_URL.findall(text))),
        float(characters >= 2_000),
        float(characters >= 8_000),
        math.log1p(text.count("?")),
        math.log1p(text.count("\n")),
    ]


def matrix(examples: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    return np.asarray([one(example) for example in examples], dtype=np.float32)
