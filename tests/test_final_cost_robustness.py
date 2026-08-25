# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""최종 비용 스트레스 변환을 검사합니다."""

from __future__ import annotations

import unittest

from experiments.analysis.audit_final_cost_robustness import (
    apply_cost_profile,
    cost_profiles,
    select_models_weighted,
)
from experiments.analysis.build_public_runtime_input import build_public_input


class FinalCostRobustnessTest(unittest.TestCase):
    def test_combined_cost_profile_only_inflates_expensive_models(self) -> None:
        examples = [
            {
                "costs": [1.0, 2.0, 3.0],
                "generations": [4, 4, 2],
            }
        ]
        multipliers, generation = cost_profiles()["결합최악오차"]
        result = apply_cost_profile(examples, multipliers, generation)
        self.assertAlmostEqual(1.0, result[0]["costs"][0])
        self.assertAlmostEqual(2.0 * 1.05 * 1.10, result[0]["costs"][1])
        self.assertAlmostEqual(3.0 * 1.10, result[0]["costs"][2])

    def test_public_input_rejects_duplicate_ids(self) -> None:
        train = {"challenge_id": "c", "episodes": [{"episode_id": "same"}]}
        dev = {"challenge_id": "c", "episodes": [{"episode_id": "same"}]}
        with self.assertRaisesRegex(ValueError, "중복"):
            build_public_input(train, dev)

    def test_weighted_allocator_reacts_to_new_family_mix(self) -> None:
        scores = [
            {"ax31-light": 0.0, "ax31": 1.0, "axk1-think": -1.0},
            {"ax31-light": 0.0, "ax31": 0.9, "axk1-think": -1.0},
        ]
        costs = [
            {"ax31-light": 1.0, "ax31": 2.0, "axk1-think": 9.0},
            {"ax31-light": 1.0, "ax31": 1.1, "axk1-think": 9.0},
        ]
        selected = select_models_weighted(
            scores,
            costs,
            [3.0, 1.0],
            "fast",
            1.0,
            ("ax31-light", "ax31"),
        )
        self.assertEqual(("ax31-light", "ax31"), selected)


if __name__ == "__main__":
    unittest.main()
