#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""최종 선택을 문제군 비율·모델 비용·생성 횟수 변화에 노출합니다."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.common.data import (
    MODEL_IDS,
    TIERS,
    TIER_LIMITS,
    choose_existing_input,
    load_examples,
    load_json,
    write_json_atomic,
)
from experiments.common.evaluation import evaluate_choices
from experiments.validation.splits import source_aware_folds
from experiments.validation.stress import family_counts, family_multiplier_scenarios


ROOT = Path(__file__).resolve().parents[2]


def cost_profiles() -> Mapping[str, tuple[tuple[float, float, float], float]]:
    """비싼 모델과 4회 생성 비용이 함께 틀리는 보수적 배수를 반환합니다."""

    return {
        "기준비용": ((1.0, 1.0, 1.0), 1.0),
        "모델상관오차": ((1.0, 1.05, 1.10), 1.0),
        "4회생성오차": ((1.0, 1.0, 1.0), 1.10),
        "결합최악오차": ((1.0, 1.05, 1.10), 1.10),
    }


def apply_cost_profile(
    examples: Sequence[Mapping[str, Any]],
    model_multipliers: Sequence[float],
    four_generation_multiplier: float,
) -> list[dict[str, Any]]:
    """light 기준 비용은 유지하고 비싼 모델 비용만 보수적으로 키웁니다."""

    result: list[dict[str, Any]] = []
    for example in examples:
        costs = []
        for index, cost in enumerate(example["costs"]):
            multiplier = float(model_multipliers[index])
            if index > 0 and int(example["generations"][index]) >= 4:
                multiplier *= four_generation_multiplier
            costs.append(float(cost) * multiplier)
        result.append({**example, "costs": costs})
    return result


def _families(split: str, examples: Sequence[Mapping[str, Any]]) -> list[str]:
    _folds, _groups, families = source_aware_folds(
        examples,
        5,
        split=split,
        deepmind_selection_path=(
            ROOT / "data/sources/deepmind-mathematics-selection.v1.json"
        ),
        aime_selection_path=ROOT / f"data/{split}/aime-selection.json",
    )
    return families


def _choices(path: Path, expected_ids: Sequence[str]) -> list[str]:
    submission = load_json(path)
    index = {
        str(row["episode_id"]): str(row["model_id"])
        for row in submission["decisions"]
    }
    if len(index) != len(submission["decisions"]):
        raise ValueError(f"제출 파일에 중복 episode_id가 있습니다: {path}")
    missing = set(expected_ids) - set(index)
    extra = set(index) - set(expected_ids)
    if missing or extra:
        raise ValueError(
            f"제출 범위가 다릅니다: 누락 {len(missing)}개, 추가 {len(extra)}개"
        )
    return [index[episode_id] for episode_id in expected_ids]


def select_models_weighted(
    score_rows: Sequence[Mapping[str, float]],
    cost_rows: Sequence[Mapping[str, float]],
    weights: Sequence[float],
    tier: str,
    safety_ratio: float,
    allowed_model_ids: Sequence[str],
) -> tuple[str, ...]:
    """문제군 비율이 바뀐 새 배치에서 전역 비용 배분을 다시 수행합니다."""

    if len(score_rows) != len(cost_rows) or len(score_rows) != len(weights):
        raise ValueError("예측 행과 스트레스 가중치 수가 같아야 합니다")
    light_total = math.fsum(
        costs[MODEL_IDS[0]] * weight for costs, weight in zip(cost_rows, weights)
    )
    cap = light_total * max(1.0, TIER_LIMITS[tier] * safety_ratio)

    def choose(penalty: float) -> tuple[tuple[str, ...], float]:
        selected = tuple(
            max(
                allowed_model_ids,
                key=lambda model_id: (
                    scores[model_id] - penalty * costs[model_id] / light_total,
                    -MODEL_IDS.index(model_id),
                ),
            )
            for scores, costs in zip(score_rows, cost_rows)
        )
        total = math.fsum(
            costs[model_id] * weight
            for costs, model_id, weight in zip(cost_rows, selected, weights)
        )
        return selected, total

    selected, total = choose(0.0)
    if total <= cap:
        return selected
    low, high = 0.0, 1.0
    selected, total = choose(high)
    while total > cap and high < 2**60:
        low, high = high, high * 2.0
        selected, total = choose(high)
    for _ in range(80):
        middle = (low + high) / 2.0
        candidate, candidate_total = choose(middle)
        if candidate_total <= cap:
            high, selected, total = middle, candidate, candidate_total
        else:
            low = middle
    if total > cap:
        return tuple(MODEL_IDS[0] for _ in score_rows)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--safety-override",
        action="append",
        default=[],
        metavar="등급=값",
        help="예측 행을 다시 만들지 않고 등급별 안전계수를 감사합니다",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.submissions is None) == (args.predictions is None):
        parser.error("--submissions와 --predictions 중 하나만 지정해야 합니다")
    safety_overrides: dict[str, float] = {}
    for value in args.safety_override:
        try:
            tier_name, raw_ratio = value.split("=", 1)
            ratio = float(raw_ratio)
        except ValueError:
            parser.error(f"안전계수 형식이 올바르지 않습니다: {value}")
        if tier_name not in TIERS or not 0.0 < ratio <= 1.0:
            parser.error(f"안전계수 등급 또는 값이 올바르지 않습니다: {value}")
        safety_overrides[tier_name] = ratio

    train = load_examples(
        choose_existing_input(ROOT, "train"), ROOT / "data/train/outcomes.json"
    )
    dev = load_examples(
        choose_existing_input(ROOT, "dev"), ROOT / "data/dev/outcomes.json"
    )
    examples = train + dev
    families = _families("train", train) + _families("dev", dev)
    family_scenarios = family_multiplier_scenarios(families, multiplier=3.0)
    episode_ids = [str(row["episode_id"]) for row in examples]
    prediction_report = load_json(args.predictions) if args.predictions else None
    if prediction_report is not None and prediction_report["episode_ids"] != episode_ids:
        raise ValueError("예측 행과 공개 outcome의 episode 순서가 다릅니다")

    tier_reports: dict[str, Any] = {}
    all_failures: list[dict[str, Any]] = []
    for tier in TIERS:
        fixed_choices = (
            _choices(args.submissions / f"{tier}.json", episode_ids)
            if args.submissions is not None
            else None
        )
        tier_predictions = (
            prediction_report["tiers"][tier]
            if prediction_report is not None
            else None
        )
        scenarios = []
        for profile_name, (model_multipliers, generation_multiplier) in cost_profiles().items():
            stressed = apply_cost_profile(
                examples, model_multipliers, generation_multiplier
            )
            for family_name, weights in family_scenarios.items():
                choices = fixed_choices
                if tier_predictions is not None:
                    choices = select_models_weighted(
                        tier_predictions["score_rows"],
                        tier_predictions["cost_rows"],
                        weights,
                        tier,
                        safety_overrides.get(
                            tier, float(tier_predictions["safety_ratio"])
                        ),
                        tier_predictions["allowed_model_ids"],
                    )
                if choices is None:
                    raise AssertionError("비용 감사 선택 결과가 없습니다")
                evaluation = evaluate_choices(
                    stressed, choices, tier, sample_weights=weights
                )
                row = {
                    "cost_profile": profile_name,
                    "family_profile": family_name,
                    "quality_score": evaluation.quality_score,
                    "cost_ratio": evaluation.cost_ratio,
                    "budget_limit": TIER_LIMITS[tier],
                    "budget_passed": evaluation.budget_passed,
                }
                scenarios.append(row)
                if not evaluation.budget_passed:
                    all_failures.append({"tier": tier, **row})
        worst = max(scenarios, key=lambda row: float(row["cost_ratio"]))
        base = next(
            row
            for row in scenarios
            if row["cost_profile"] == "기준비용"
            and row["family_profile"] == "기준"
        )
        tier_reports[tier] = {
            "base": base,
            "worst": worst,
            "scenario_count": len(scenarios),
            "budget_failure_count": sum(
                not bool(row["budget_passed"]) for row in scenarios
            ),
            "scenarios": scenarios,
        }

    report = {
        "artifact_type": "os-skt-final-cost-robustness-v1",
        "episodes": len(examples),
        "family_counts": family_counts(families),
        "family_multiplier": 3.0,
        "safety_overrides": safety_overrides,
        "allocation_simulation": (
            "문제군 비율별 전역 재배분"
            if prediction_report is not None
            else "고정 공개 선택 재평가"
        ),
        "cost_profiles": {
            name: {
                "model_multipliers": list(model_multipliers),
                "four_generation_multiplier": generation_multiplier,
            }
            for name, (model_multipliers, generation_multiplier) in cost_profiles().items()
        },
        "scenario_count": sum(row["scenario_count"] for row in tier_reports.values()),
        "budget_failure_count": len(all_failures),
        "passed": not all_failures,
        "tiers": tier_reports,
        "failures": all_failures,
    }
    write_json_atomic(args.output, report)
    print(
        f"비용 스트레스 {report['scenario_count']}개 중 "
        f"예산 초과 {report['budget_failure_count']}개를 기록했습니다."
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
