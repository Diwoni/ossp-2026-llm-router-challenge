# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""E5 method4 기준선의 내용 기반 라우팅과 결정성을 검사합니다."""

from __future__ import annotations

import math
import unittest

from experiments.common.budget import select_models
from experiments.method4_finetuned_encoder.features import head_tail_ids
from experiments.method4_finetuned_encoder.final_route import (
    _rows_from_delta,
    canonical_content_indices,
)
from experiments.method4_finetuned_encoder.residual import predict_residual
from experiments.method4_finetuned_encoder.similarity import predict_similarity_delta
from experiments.method4_finetuned_encoder.teacher_student import (
    apply_absolute_student,
)


class _Tokenizer:
    cls_token_id = 101
    bos_token_id = None
    sep_token_id = 102
    eos_token_id = None
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": [int(value) for value in text.split()]}


class Method4RouterTest(unittest.TestCase):
    def test_head_tail_keeps_both_sides(self) -> None:
        ids, mask = head_tail_ids(_Tokenizer(), " ".join(str(i) for i in range(20)), 10)
        self.assertEqual(ids, [101, 0, 1, 2, 15, 16, 17, 18, 19, 102])
        self.assertEqual(mask, [1] * 10)

    def test_canonical_order_depends_on_content_not_id(self) -> None:
        original = [
            {"episode_id": "z", "text": "beta", "message_count": 1},
            {"episode_id": "a", "text": "alpha", "message_count": 1},
        ]
        changed = [
            {"episode_id": "new-1", "text": "alpha", "message_count": 1},
            {"episode_id": "new-2", "text": "beta", "message_count": 1},
        ]
        self.assertEqual([1, 0], canonical_content_indices(original))
        self.assertEqual([0, 1], canonical_content_indices(changed))

    def test_fast_delta_always_disables_think(self) -> None:
        import numpy as np

        rows = _rows_from_delta(np.asarray([[0.25], [-0.1]]), decimals=4)
        self.assertEqual(-1.0, rows[0]["axk1-think"])
        self.assertEqual(-1.0, rows[1]["axk1-think"])

    def test_secondary_embedding_changes_neighbor_space(self) -> None:
        import numpy as np

        artifact = {
            "reference_embeddings": np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            "secondary_reference_embeddings": np.asarray(
                [[0.0, 1.0], [1.0, 0.0]]
            ),
            "target_delta": np.asarray([[0.1, 0.2], [0.8, 0.9]]),
            "source_indices": np.asarray([0, 0]),
        }
        primary = predict_similarity_delta(
            np.asarray([[1.0, 0.0]]),
            [{"text": "plain prompt"}],
            artifact,
            neighbors=1,
            temperature=0.1,
            source_bonus=0.0,
        )
        secondary = predict_similarity_delta(
            np.asarray([[1.0, 0.0]]),
            [{"text": "plain prompt"}],
            artifact,
            neighbors=1,
            temperature=0.1,
            source_bonus=0.0,
            secondary_embeddings=np.asarray([[1.0, 0.0]]),
            secondary_weight=1.0,
        )
        np.testing.assert_allclose(primary, [[0.1, 0.2]])
        np.testing.assert_allclose(secondary, [[0.8, 0.9]])

    def test_residual_costs_keep_model_order(self) -> None:
        import numpy as np

        forest = {
            "tree_count": 1,
            "trees": [
                {
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature": [-2],
                    "threshold": [-2.0],
                    "value": [0.25],
                }
            ],
        }
        artifact = {
            "feature_dimension": 1,
            "model_ids": ["ax31-light", "ax31", "axk1-think"],
            "tiers": {"fast": {"ax31_forest": "gain", "think_forest": None}},
            "forests": {"gain": forest},
            "cost_head": {
                "feature_mean": [0.0],
                "feature_scale": [1.0],
                "intercept": [0.0, math.log(2.0), math.log(3.0)],
                "coefficients": [[0.0, 0.0, 0.0]],
            },
        }
        scores, costs = predict_residual(
            np.asarray([[7.0]]), artifact, "fast", decimals=4
        )
        self.assertEqual(0.25, scores[0]["ax31"])
        self.assertLess(costs[0]["ax31-light"], costs[0]["ax31"])
        self.assertLess(costs[0]["ax31"], costs[0]["axk1-think"])

    def test_allocator_uses_only_allowed_models(self) -> None:
        scores = [
            {"ax31-light": 0.1, "ax31": 0.8, "axk1-think": 1.0}
            for _ in range(10)
        ]
        costs = [
            {"ax31-light": 1.0, "ax31": 1.1, "axk1-think": 1.2}
            for _ in range(10)
        ]
        selected, _ratio = select_models(
            scores,
            costs,
            "fast",
            0.95,
            allowed_model_ids=("ax31-light", "ax31"),
        )
        self.assertNotIn("axk1-think", selected)

    def test_teacher_student_uses_existing_runtime_features_only(self) -> None:
        import numpy as np

        # E5 임베딩 2차원과 내용 기반 profile 20차원을 결합한 22차원 학생입니다.
        artifact = {
            "feature_mean": [0.0] * 22,
            "feature_scale": [1.0] * 22,
            "intercept": [0.5, 0.25],
            "coefficients": [[0.0, 0.0] for _ in range(22)],
        }
        replaced, mask = apply_absolute_student(
            np.asarray([[0.1, 0.2]]),
            np.asarray([[1.0, 0.0]]),
            [{"text": "How many apples are left?", "message_count": 1}],
            artifact,
            weight=1.0,
            mode="all",
            fraction=1.0,
        )
        np.testing.assert_allclose(replaced, [[0.5, 0.25]])
        self.assertEqual([True], mask.tolist())


if __name__ == "__main__":
    unittest.main()
