# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""실험 검증 도구가 누수 방지·비용 규칙을 지키는지 확인합니다."""

from __future__ import annotations

import unittest

from experiments.common.data import MODEL_IDS
from experiments.common.evaluation import evaluate_choices, weighted_final_score
from experiments.validation.splits import content_hash_folds, normalize_template
from experiments.validation.stress import family_multiplier_scenarios


class ExperimentValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            {
                "episode_id": f"row-{index}",
                "text": f"Alice has {index + 1} apples. How many apples?",
                "scores": [0.5, 0.75, 1.0],
                "costs": [1.0, 2.0, 6.0],
            }
            for index in range(12)
        ]

    def test_template_normalization_hides_values(self) -> None:
        first = normalize_template("Alice has 3 apples")
        second = normalize_template("Brian has 42 apples")
        self.assertEqual(first, second)

    def test_same_template_stays_in_one_fold(self) -> None:
        _folds, fold_ids = content_hash_folds(self.examples, 3)
        self.assertEqual(1, len(set(fold_ids)))

    def test_budget_failure_zeroes_effective_score(self) -> None:
        choices = [MODEL_IDS[1]] * len(self.examples)
        result = evaluate_choices(self.examples, choices, "fast")
        self.assertFalse(result.budget_passed)
        self.assertEqual(0.0, result.effective_score)

    def test_weighted_score_uses_official_tier_weights(self) -> None:
        choices = [MODEL_IDS[0]] * len(self.examples)
        results = {
            tier: evaluate_choices(self.examples, choices, tier)
            for tier in ("fast", "balanced", "premium")
        }
        self.assertAlmostEqual(0.5, weighted_final_score(results))

    def test_stress_scenarios_are_order_independent(self) -> None:
        first = family_multiplier_scenarios(["수학", "코드", "수학"])
        second = family_multiplier_scenarios(["수학", "코드", "수학"])
        self.assertEqual(first, second)
        self.assertEqual(5, len(first))


if __name__ == "__main__":
    unittest.main()
