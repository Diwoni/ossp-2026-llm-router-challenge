<!--
SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
SPDX-License-Identifier: Apache-2.0
-->

# 최종 비용 강건성 실험

## 결론

Fast 0.95, Balanced 0.90은 그대로 유지하고 Premium 안전계수를 0.84에서
**0.78**로 낮췄다. 이 설정은 문제군 비율 변화와 공개 비용의 상관 오차를 합친
228개 시나리오에서 예산 초과가 0회였다.

공개 Dev 재대입 점수는 0.717613636364에서 0.716420454545로 0.001193182 줄었다.
그러나 Premium이 예산을 한 번이라도 넘으면 그 등급 전체가 0점이므로, 작은 공개
재대입 이득보다 숨은 분포에서 0점이 될 위험을 줄이는 편이 기대값이 높다고
판단했다.

## 무엇을 바꿔 검사했나

공개 Train+Dev 2,640문항을 공식 실행처럼 하나의 배치로 합쳤다. 각 등급의 품질과
예측비용 행을 한 번 추출한 뒤 다음 상황마다 전역 배분기를 다시 실행했다.

### 문제군 비율 19종

9개 공개 문제군 각각을 3배 늘리거나 1/3로 줄인 18종과 기준 1종이다. 고정된
선택을 단순 재채점하지 않고, 새 비율의 입력 배치를 받은 것처럼 비용 배분을 다시
수행한다.

### 실제 비용 오차 4종

| 비용 상황 | ax31 | think | 4회 생성 비싼 모델 |
| --- | ---: | ---: | ---: |
| 기준 | 1.00배 | 1.00배 | 1.00배 |
| 모델 상관 오차 | 1.05배 | 1.10배 | 1.00배 |
| 4회 생성 오차 | 1.00배 | 1.00배 | 1.10배 |
| 결합 최악 오차 | 1.05배 | 1.10배 | 추가 1.10배 |

light 기준 비용은 키우지 않고 선택한 비싼 모델의 실제 비용만 키웠다. 예측비용은
그대로 두므로 라우터가 비용 오차를 미리 안다는 비현실적인 가정도 하지 않는다.

총 시나리오는 `19 문제군 비율 × 4 비용 상황 × 3 등급 = 228`개다.

## 안전계수 sweep

Fast와 Balanced는 기존 설정으로 모든 시나리오를 통과했다. Premium만 예측 행을
다시 만들지 않고 안전계수를 바꿔 비교했다.

| Premium 안전계수 | 전체 실패 | 공개 전체 Premium 품질 | 기준 실제 비용비 | 최악 비용비 |
| ---: | ---: | ---: | ---: | ---: |
| 0.84 | 4 | 0.761931818 | 3.504789 | 4.219381 |
| 0.80 | 2 | 0.757575758 | 3.344576 | 4.057276 |
| 0.79 | 1 | 0.756250000 | 3.302178 | 4.013805 |
| 0.785 | 0 | 0.755871212 | 3.285168 | 3.991572 |
| **0.78** | **0** | **0.755303030** | **3.269492** | **3.973534** |

0.785도 계산상 통과하지만 최악 비용비가 4.0과 0.0084밖에 차이 나지 않는다.
0.78의 공개 전체 품질 손실은 0.000568182이고 최종 가중 점수로는 약
0.000170455다. 이 작은 차이로 추가 여유를 사는 것이 합리적이다.

## 최종 0.78 결과

| 등급 | 공개 전체 품질 | 기준 실제 비용비 | 최악 비용비 | 실패 |
| --- | ---: | ---: | ---: | ---: |
| Fast | 0.693276515 | 1.180709 | 1.208498 | 0/76 |
| Balanced | 0.692708333 | 1.732056 | 1.894680 | 0/76 |
| Premium | 0.755303030 | 3.269492 | 3.973534 | 0/76 |
| 전체 | 0.711714015 | - | - | **0/228** |

이 0.711714015도 숨은 점수 추정치는 아니다. 공개 Train+Dev outcome을 사용한
강건성 비교용 숫자다.

원본 기계 판독 보고서는
[`final-cost-robustness.v1.json`](../reports/final-cost-robustness.v1.json)에
보존했으며 SHA-256은
`30f0ed34f0392df384f756516bd20074dac8674f39decacf9475b9da52de4dd2`다.

## 문제군별 강제 상한 후보는 왜 버렸나

추가로 보수적 상한을 적용해 각 문제군 안에서 Fast, Balanced, Premium 한도를
각각 강제하는 후보도 만들었다. 고정선택 스트레스 실패는
28개에서 2개로 줄었지만 공개 전체 품질이 다음처럼 크게 떨어졌다.

| 등급 | 기존 품질 | 문제군별 상한 | 변화 |
| --- | ---: | ---: | ---: |
| Fast | 0.693276515 | 0.677462121 | -0.015814394 |
| Balanced | 0.692708333 | 0.687973485 | -0.004734848 |
| Premium | 0.761931818 | 0.718087121 | -0.043844697 |

작은 문제군에서 예산이 쪼개지면서 좋은 후보를 배정할 자유를 지나치게 잃었다.
따라서 실험 코드와 런타임 분기는 최종 경로에서 제거하고 집계 보고서만
[`family-allocation-candidate.v1.json`](../reports/family-allocation-candidate.v1.json)에
남겼다.

## 재현 명령

```console
PYTHONPATH=src:. python3 -m experiments.analysis.build_public_runtime_input \
  --output build/final-public/inputs.json

PYTHONPATH=src:. python3 -m experiments.analysis.extract_final_prediction_rows \
  --input build/final-public/inputs.json \
  --assets build/final-assets/method4 \
  --gate configs/method4-tier-gate.qwen-student.json \
  --output build/final-public/prediction-rows.json

PYTHONPATH=src:. python3 -m experiments.analysis.audit_final_cost_robustness \
  --predictions build/final-public/prediction-rows.json \
  --output build/final-public/final-cost-robustness.json
```
