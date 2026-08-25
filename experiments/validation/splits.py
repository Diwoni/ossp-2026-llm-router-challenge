# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""비슷한 프롬프트가 학습·검증에 함께 들어가지 않도록 fold를 만듭니다."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


_CAPITALIZED_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
_QUOTED_LITERAL = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1")
_HEX_OR_UUID = re.compile(
    r"\b(?:0x[0-9a-f]+|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?", re.I)
_WHITESPACE = re.compile(r"\s+")


def normalize_template(text: str) -> str:
    """이름·숫자·문자열 값만 다른 같은 문제 틀을 하나로 묶습니다."""

    text = unicodedata.normalize("NFKC", text)
    text = _CAPITALIZED_NAME.sub("<name>", text)
    text = text.casefold()
    text = _QUOTED_LITERAL.sub("<string>", text)
    text = _HEX_OR_UUID.sub("<identifier>", text)
    text = _NUMBER.sub("<number>", text)
    return _WHITESPACE.sub(" ", text).strip()


def infer_prompt_family(text: str) -> tuple[str, str, str]:
    """프롬프트 내용만으로 공개 문제군과 같은 형식군을 추정합니다."""

    lower = text.lower()
    has_hangul = any("가" <= char <= "힣" for char in text)
    if "assert f(" in text:
        material = text.split("\n\nassert f(", 1)[0]
        subtype = "input" if "assert f(??" in text else "output"
        return "cruxeval", subtype, material
    if len(text) >= 4_000:
        subtype = "16k" if len(text) >= 30_000 else "4k"
        return "babilong", subtype, unicodedata.normalize("NFKC", text)
    if has_hangul:
        if "\nQuestion:" in text and "A." in text and "B." in text:
            return "belebele", "korean", text.split("\n\nQuestion:", 1)[0]
        return "hrmcr", "korean", normalize_template(text)
    if "\nQuestion:" in text and not text.startswith("Question:"):
        return "ruletaker", "logic", text.rsplit("\nQuestion:", 1)[0]
    if text.startswith("Question:") and "A." in text and "B." in text:
        return "truthfulqa", "multiple-choice", unicodedata.normalize("NFKC", text)
    story_math = bool(
        re.search(r"\b(how many|how much|how long|how far|how old|in total)\b", lower)
    )
    return "gsm8k", "story" if story_math else "math", normalize_template(text)


def content_hash_folds(
    examples: Sequence[Mapping[str, Any]], folds: int
) -> tuple[list[tuple[Any, Any]], list[int]]:
    """정규화한 프롬프트 틀의 SHA-256으로 결정적 fold를 만듭니다."""

    import numpy as np

    if folds < 2 or len(examples) < folds:
        raise ValueError("fold 수는 2 이상이고 문항 수 이하여야 합니다")
    fold_ids = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(normalize_template(str(row["text"])).encode("utf-8")).digest()[:8],
                "big",
            )
            % folds
            for row in examples
        ],
        dtype=np.int64,
    )
    indices = np.arange(len(examples), dtype=np.int64)
    result = [
        (indices[fold_ids != fold], indices[fold_ids == fold])
        for fold in range(folds)
    ]
    return result, [int(value) for value in fold_ids]


def source_aware_folds(
    examples: Sequence[Mapping[str, Any]],
    folds: int,
    *,
    split: str,
    deepmind_selection_path: Path,
    aime_selection_path: Path,
) -> tuple[list[tuple[Any, Any]], list[str], list[str]]:
    """같은 원천 모듈·문제 변형을 하나의 fold에 넣습니다.

    공개 provenance와 ID는 이 오프라인 그룹 구성에서만 사용합니다. 반환한
    그룹은 런타임 특징이나 최종 선택 조회표에 포함하지 않습니다.
    """

    import numpy as np

    deepmind = json.loads(deepmind_selection_path.read_text(encoding="utf-8"))
    aime = json.loads(aime_selection_path.read_text(encoding="utf-8"))
    overrides: dict[str, tuple[str, str, str]] = {}
    for row in deepmind["splits"][split]:
        pieces = str(row["sample_id"]).split(":")
        if len(pieces) < 2:
            raise ValueError("DeepMind Mathematics sample_id가 올바르지 않습니다")
        module = pieces[1]
        overrides[str(row["episode_id"])] = ("deepmind-mathematics", module, module)
    if aime.get("split") != split:
        raise ValueError("AIME 선택 파일의 split이 다릅니다")
    for row in aime["episodes"]:
        overrides[str(row["episode_id"])] = (
            "aime",
            str(row["source_key"]["year"]),
            str(row["prompt_sha256"]),
        )

    group_rows: dict[str, list[int]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    row_groups: list[str] = []
    row_families: list[str] = []
    for index, example in enumerate(examples):
        source, subtype, material = overrides.get(
            str(example["episode_id"]), infer_prompt_family(str(example["text"]))
        )
        if source == "deepmind-mathematics":
            material = subtype + "\0" + normalize_template(str(example["text"]))
        digest = hashlib.sha256(
            ("source-aware-v3\0" + source + "\0" + material).encode("utf-8")
        ).hexdigest()
        group_rows.setdefault(digest, []).append(index)
        metadata[digest] = (source, subtype)
        row_groups.append(digest)
        row_families.append(source)
    if len(group_rows) < folds:
        raise ValueError("source-aware 그룹 수가 fold 수보다 작습니다")

    total_load = [0] * folds
    source_load: dict[str, list[int]] = {}
    assignment: dict[str, int] = {}
    for source in sorted(set(row_families)):
        source_load[source] = [0] * folds
        keys = sorted(
            (key for key, value in metadata.items() if value[0] == source),
            key=lambda key: (-len(group_rows[key]), key),
        )
        for key in keys:
            fold = min(
                range(folds),
                key=lambda candidate: (
                    source_load[source][candidate],
                    total_load[candidate],
                    candidate,
                ),
            )
            assignment[key] = fold
            size = len(group_rows[key])
            source_load[source][fold] += size
            total_load[fold] += size
    fold_ids = np.asarray([assignment[group] for group in row_groups], dtype=np.int64)
    indices = np.arange(len(examples), dtype=np.int64)
    result = [
        (indices[fold_ids != fold], indices[fold_ids == fold])
        for fold in range(folds)
    ]
    return result, row_groups, row_families
