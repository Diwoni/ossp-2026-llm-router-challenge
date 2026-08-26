<!--
SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
SPDX-License-Identifier: Apache-2.0
-->

# 기준선 재현 보고서

## 결론

공식 upstream의 깨끗한 fork에서 공개 데이터와 선택 결과를 새로 생성했습니다.
이전 저장소의 결과 JSON을 복사하지 않고 동일한 ONNX·head 자산의 SHA-256만
확인한 뒤 새 런타임 코드로 다시 라우팅했습니다.

| 후보 | Fast 품질 / 비용 | Balanced 품질 / 비용 | Premium 품질 / 비용 | 종합 |
| --- | ---: | ---: | ---: | ---: |
| 공식 hash-regex | 0.663068 / 1.235989 | 0.693750 / 1.961506 | 0.740057 / 3.985205 | 0.695369 |
| E5 Train-only | 0.673011 / 1.222433 | 0.706250 / 1.846850 | 0.752557 / 3.900549 | **0.706847** |
| E5 public-all 재대입 | 0.683239 / 1.176137 | 0.728977 / 1.878879 | 0.751136 / 3.646345 | 0.717330 |

`E5 Train-only`는 Train으로 학습한 자산을 Dev에서 평가한 값입니다. 다만 이전
개발 과정에서 Dev로 gate와 후보를 여러 번 골랐으므로 완전히 손대지 않은 최종
평가 추정치는 아닙니다. `public-all 재대입`은 Train+Dev outcome을 사용해 만든
head를 같은 Dev에 다시 적용했으므로 일반화 점수가 아닙니다. 이 숫자는 선택
파일 생성과 Docker 회귀 검사에만 사용합니다.

## 기준선 구조

1. `multilingual-e5-small`을 공개 Train outcome으로 2 epoch 파인튜닝합니다.
2. 앞 44%·뒤 56% 토큰과 20개 수작업 특징을 함께 사용합니다.
3. 인코더가 세 모델의 품질과 log 비용을 동시에 예측합니다.
4. Fast는 유사 문항 KNN, Balanced는 KNN과 잔차 forest, Premium은 hash와
   의미 KNN을 결합합니다.
5. 등급별 예측 비용 상한 안에서 Lagrange 이진 탐색으로 모델을 배분합니다.
6. ID·입력 순서가 달라져도 양자화 batch가 같도록 프롬프트 내용으로 먼저
   정렬하고, 마지막에 원래 출력 순서로 되돌립니다.

## 자산 출처와 고정값

- 원본 모델: `intfloat/multilingual-e5-small`
- 원본 리비전: `d1d99a1efae6779390caba937d92c54b5bc70e51`
- 원본 가중치 SHA-256:
  `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`
- 라이선스: MIT
- 원본 모델 카드: <https://huggingface.co/intfloat/multilingual-e5-small>

이번 재현에서 확인한 핵심 파일은 다음과 같습니다.

| 파일 | SHA-256 |
| --- | --- |
| 파인튜닝 router INT8 ONNX | `e86af4846744df7a17cd9d9dced05c4d52735f5876ae3a8d57560da582970774` |
| frozen E5 FP32 ONNX | `fc9fd4cb0bb23a24b9349f38147836204f9798dfb92ad701cb090bbd28ac11a6` |
| Train-only residual | `33e2063e95083c794a3fe2408d17a0a214f6fb907c8930588e2228e34c63f11d` |
| Train-only similarity | `4a83da2af9f06dd6f59f41026dee4567104f23cba4ea714718881f162548005b` |
| public-all residual | `4bb1b9ea3f77f47a2b44938e6fc0336586e333dcbca24e7208cc20fbd723f63d` |
| public-all similarity | `315650e845fc273e369135e40ba7eefb300712160d9a5d9a355b620fd6aaf9af` |

## 환경 차이 발견

처음 새 venv의 공식 baseline 학습 버전인 NumPy 2.0.2로 ONNX 특징을 계산했을 때
macOS 행렬 곱에서 overflow 경고가 발생했습니다. 해당 실행 결과는 폐기하고 최종
런타임과 같은 NumPy 2.5.2로 다시 실행해 경고가 사라지고 이전 선택 결과의 정확한
점수가 재현되는 것을 확인했습니다. 따라서 학습용 baseline 의존성과 제출
런타임 의존성을 같은 환경으로 오해하지 않도록 별도 파일에 고정합니다.

## 재현 명령

자산을 `build/reproduced-assets/e5`와
`build/reproduced-assets/e5-train-only`에 준비한 뒤 실행합니다.

```console
./scripts/evaluate_method4_e5.sh train-only
./scripts/evaluate_method4_e5.sh public-all-replay
```

다음 단계에서는 이 E5 기준선을 대조군으로 사용해 Qwen 교사-학생 보정과 추가
구조를 동일한 평가 절차에서 비교합니다.
