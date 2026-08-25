# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""품질·비용 위험·모델 선호를 함께 예측하는 다중 과제 인코더입니다."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _encoder_layers(encoder: Any) -> list[Any]:
    candidates = (
        ("encoder", "layer"),
        ("transformer", "layer"),
        ("encoder", "block"),
    )
    for outer, inner in candidates:
        owner = getattr(encoder, outer, None)
        layers = getattr(owner, inner, None) if owner is not None else None
        if layers is not None:
            return list(layers)
    base = getattr(encoder, "base_model", None)
    if base is not None and base is not encoder:
        return _encoder_layers(base)
    return []


def configure_trainable_backbone(encoder: Any, unfreeze_last_n: int) -> int:
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if unfreeze_last_n < 0:
        raise ValueError("unfreeze_last_n cannot be negative")
    layers = _encoder_layers(encoder)
    if unfreeze_last_n > len(layers):
        raise ValueError(
            f"cannot unfreeze {unfreeze_last_n} blocks; encoder exposes {len(layers)}"
        )
    for layer in layers[len(layers) - unfreeze_last_n :] if unfreeze_last_n else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return len(layers)


def masked_mean(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    total = (last_hidden_state * mask).sum(dim=1)
    return total / mask.sum(dim=1).clamp_min(1.0)


class RouterModel:  # 얇은 팩토리로 Torch를 선택적 실험 의존성으로 유지합니다.
    @staticmethod
    def create(
        encoder_name_or_path: str,
        dense_dimension: int,
        projection_dimension: int,
        dropout: float,
        unfreeze_last_n: int,
        *,
        local_files_only: bool = False,
    ) -> Any:
        import torch
        from transformers import AutoModel

        class _RouterModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = AutoModel.from_pretrained(
                    encoder_name_or_path, local_files_only=local_files_only
                )
                self.encoder_layer_count = configure_trainable_backbone(
                    self.encoder, unfreeze_last_n
                )
                hidden = int(self.encoder.config.hidden_size)
                self.projection = torch.nn.Sequential(
                    torch.nn.Linear(hidden + dense_dimension, projection_dimension),
                    torch.nn.LayerNorm(projection_dimension),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                )
                self.score_head = torch.nn.Linear(projection_dimension, 3)
                self.cost_head = torch.nn.Linear(projection_dimension, 3)
                self.model_embeddings = torch.nn.Parameter(
                    torch.empty(3, projection_dimension)
                )
                self.preference_bias = torch.nn.Parameter(torch.zeros(3))
                torch.nn.init.normal_(self.model_embeddings, std=0.02)

            def forward(self, input_ids: Any, attention_mask: Any, dense: Any) -> Mapping[str, Any]:
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                pooled = masked_mean(hidden, attention_mask)
                projected = self.projection(torch.cat([pooled, dense], dim=1))
                scores = torch.sigmoid(self.score_head(projected))
                costs = self.cost_head(projected)
                preference = (
                    projected @ self.model_embeddings.T / math.sqrt(projected.shape[1])
                    + self.preference_bias
                )
                return {
                    "scores": scores,
                    "costs": costs,
                    "preference": preference,
                    # RouterDC·RadialRouter 방식 검색을 위해 두 표현을 제공합니다.
                    # 투영 표현에는 지도학습 head가 쓰는 내용 기반 특징도 담깁니다.
                    "pooled_embedding": torch.nn.functional.normalize(
                        pooled, p=2, dim=1
                    ),
                    "router_embedding": torch.nn.functional.normalize(
                        projected, p=2, dim=1
                    ),
                }

        return _RouterModel()


def multitask_loss(
    output: Mapping[str, Any],
    score_target: Any,
    cost_target: Any,
    sample_weight: Any,
    *,
    preference_temperature: float = 0.12,
    objective: str = "baseline",
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    weight = sample_weight.unsqueeze(1)
    if objective == "binomial":
        # 2회 중 1회 성공한 0.5는 성공 1회·실패 1회의 음의 로그우도와 같습니다.
        # L1 회귀와 달리 공개 점수를 잡음 없는 실수가 아니라 반복 Bernoulli
        # 관측으로 취급합니다.
        score_element = functional.binary_cross_entropy(
            output["scores"].clamp(1e-6, 1.0 - 1e-6),
            score_target,
            reduction="none",
        )
    elif objective in ("baseline", "contrastive"):
        score_element = functional.smooth_l1_loss(
            output["scores"], score_target, reduction="none", beta=0.15
        )
    else:
        raise ValueError(f"unsupported Method-4 objective: {objective}")
    score_loss = (score_element * weight).sum() / (weight.sum() * 3.0)
    # 공식 제약은 수백 문항 비용의 합에 적용됩니다. 각 행을 q90으로 부풀리기보다
    # 평균이 맞는 회귀가 유용하며, batch 전체 여유는 뒤의 안전 보정이 담당합니다.
    cost_element = functional.smooth_l1_loss(
        output["costs"], cost_target, reduction="none", beta=0.25
    )
    cost_loss = (cost_element * weight).sum() / (weight.sum() * 3.0)
    preference_target = torch.softmax(
        score_target / preference_temperature, dim=1
    )
    preference_loss = -(
        preference_target * torch.log_softmax(output["preference"], dim=1)
    ).sum(dim=1)
    preference_loss = (preference_loss * sample_weight).sum() / sample_weight.sum()

    # 절대값 보정뿐 아니라 실제 선택을 결정하는 모델 간 순서를 직접 학습합니다.
    actual_differences = score_target.unsqueeze(2) - score_target.unsqueeze(1)
    predicted_differences = output["scores"].unsqueeze(2) - output["scores"].unsqueeze(1)
    ranking_loss = functional.smooth_l1_loss(
        predicted_differences, actual_differences, beta=0.10
    )
    monotonic_cost = functional.relu(
        output["costs"][:, :-1] - output["costs"][:, 1:]
    ).mean()
    contrastive_loss = output["scores"].new_zeros(())
    if objective == "contrastive" and output["router_embedding"].shape[0] > 1:
        # 라우터 공간의 코사인 유사도가 두 품질 향상 좌표의 유사성을 따르게 합니다.
        # 자기 자신과의 대각 원소는 항상 1이고 라우팅 정보가 없어 제외합니다.
        target_delta = score_target[:, 1:] - score_target[:, [0]]
        target_scale = target_delta.std(dim=0, unbiased=False).clamp_min(0.05)
        normalized_delta = target_delta / target_scale
        target_distance = (
            normalized_delta.unsqueeze(1) - normalized_delta.unsqueeze(0)
        ).square().sum(dim=2)
        target_similarity = torch.exp(-0.5 * target_distance)
        predicted_similarity = (
            output["router_embedding"] @ output["router_embedding"].T + 1.0
        ) / 2.0
        mask = ~torch.eye(
            predicted_similarity.shape[0],
            dtype=torch.bool,
            device=predicted_similarity.device,
        )
        contrastive_loss = functional.smooth_l1_loss(
            predicted_similarity[mask], target_similarity[mask], beta=0.10
        )
    total = (
        score_loss
        + 0.30 * cost_loss
        + 0.12 * preference_loss
        + 0.20 * ranking_loss
        + 0.05 * monotonic_cost
        + 0.25 * contrastive_loss
    )
    metrics = {
        "loss": float(total.detach()),
        "score_loss": float(score_loss.detach()),
        "cost_loss": float(cost_loss.detach()),
        "preference_loss": float(preference_loss.detach()),
        "ranking_loss": float(ranking_loss.detach()),
        "contrastive_loss": float(contrastive_loss.detach()),
    }
    return total, metrics
