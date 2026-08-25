# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""프롬프트 입력과 공개 outcome을 학습용 행으로 안전하게 연결합니다."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
TIERS = ("fast", "balanced", "premium")
TIER_LIMITS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
TIER_WEIGHTS = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
POLICY_ID = "ossp-2026-prompt-router-v1"
TOKEN_UNIT = 1_000_000.0
RATES = {
    "ax31-light": (1.0, 4.0),
    "ax31": (2.127, 8.509),
    "axk1-think": (6.565, 26.260),
}


def load_json(path: Path) -> Any:
    """UTF-8 JSON 파일을 읽습니다."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    """중간 파일을 거쳐 JSON을 원자적으로 교체합니다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def episode_text(episode: Mapping[str, Any]) -> str:
    """prompt와 messages 입력을 하나의 내용 문자열로 정규화합니다."""

    prompt = episode.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = episode.get("messages")
    if not isinstance(messages, list):
        raise ValueError("문항에는 prompt 또는 messages가 필요합니다")
    return "\n".join(str(message["content"]) for message in messages)


def message_count(episode: Mapping[str, Any]) -> int:
    messages = episode.get("messages")
    return len(messages) if isinstance(messages, list) else 1


def outcome_cost(model_id: str, row: Mapping[str, Any]) -> float:
    """공식 비용 계수로 문항 하나의 모델 비용을 계산합니다."""

    input_rate, output_rate = RATES[model_id]
    cost = (
        float(row["input_tokens"]) * input_rate
        + float(row["output_tokens"]) * output_rate
    ) / TOKEN_UNIT
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError(f"{model_id}의 비용이 올바르지 않습니다: {cost}")
    return cost


def load_examples(input_path: Path, outcomes_path: Path) -> list[dict[str, Any]]:
    """입력 순서를 유지하면서 공개 결과를 결합합니다.

    `episode_id`는 오프라인 자료를 정확히 연결하는 키로만 사용합니다. 반환된
    ID는 런타임 특징이나 모델 선택 입력으로 사용하면 안 됩니다.
    """

    inputs = load_json(input_path)
    outcomes = load_json(outcomes_path)
    outcome_index = {
        row["episode_id"]: row["models"] for row in outcomes["episodes"]
    }
    examples: list[dict[str, Any]] = []
    for episode in inputs["episodes"]:
        episode_id = episode["episode_id"]
        models = outcome_index.get(episode_id)
        if models is None:
            raise ValueError(f"공개 outcome이 없는 문항입니다: {episode_id}")
        examples.append(
            {
                "episode_id": episode_id,
                "text": episode_text(episode),
                "message_count": message_count(episode),
                "scores": [float(models[model_id]["score"]) for model_id in MODEL_IDS],
                "costs": [outcome_cost(model_id, models[model_id]) for model_id in MODEL_IDS],
                "generations": [
                    int(models[model_id]["num_generations"])
                    for model_id in MODEL_IDS
                ],
            }
        )
    return examples


def choose_existing_input(root: Path, split: str) -> Path:
    """재현한 전체 입력이 있으면 사용하고, 없으면 base 입력을 반환합니다."""

    materialized = root / "data" / "materialized" / split / "inputs.json"
    if materialized.is_file():
        return materialized
    return root / "data" / split / "inputs-base.json"


def texts(examples: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(example["text"]) for example in examples]
