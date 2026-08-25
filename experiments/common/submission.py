# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""v1 선택 결과를 만들고 공식 채점기로 평가합니다."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from experiments.common.data import (
    MODEL_IDS,
    POLICY_ID,
    TIERS,
    aligned_outcomes,
    load_json,
    write_json_atomic,
)


def write_submission(
    input_path: Path,
    tier: str,
    model_ids: Sequence[str],
    output_path: Path,
) -> None:
    inputs = load_json(input_path)
    episodes = inputs["episodes"]
    if tier not in TIERS or len(model_ids) != len(episodes):
        raise ValueError("invalid tier or decision count")
    if any(model_id not in MODEL_IDS for model_id in model_ids):
        raise ValueError("unknown model id")
    write_json_atomic(
        output_path,
        {
            "challenge_id": inputs["challenge_id"],
            "decisions": [
                {"episode_id": episode["episode_id"], "model_id": model_id}
                for episode, model_id in zip(episodes, model_ids)
            ],
            "policy_id": POLICY_ID,
            "schema_version": inputs["schema_version"],
            "split": inputs["split"],
            "tier": tier,
        },
    )


def evaluate_official(
    input_path: Path,
    outcomes_path: Path,
    submissions_dir: Path,
    report_path: Path,
) -> Mapping[str, object]:
    from ossp_router.protocol import (
        load_bundled_policy,
        load_input,
        load_outcomes,
        load_submission,
        write_json,
    )
    from ossp_router.scoring import score_submissions

    aligned_path = report_path.with_name("aligned-outcomes.json")
    write_json_atomic(aligned_path, aligned_outcomes(input_path, outcomes_path))
    inputs = load_input(input_path)
    outcomes = load_outcomes(aligned_path)
    submissions = [load_submission(submissions_dir / f"{tier}.json") for tier in TIERS]
    report = score_submissions(inputs, outcomes, submissions, load_bundled_policy())
    write_json(report_path, report)
    return report
