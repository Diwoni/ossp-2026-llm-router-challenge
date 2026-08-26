<!--
SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
SPDX-License-Identifier: Apache-2.0
-->

# 1차 서면평가 30점 증거표

이 문서는 서면평가의 다섯 항목을 저장소 안의 확인 가능한 증거와 연결한다. 단순한
홍보 문구가 아니라 심사자가 파일·이슈·PR·실행 보고서로 역추적할 수 있는 내용만
기록한다.

## 프로젝트 구조 및 코드 완성도 — 6점

| 평가 관점 | 증거 |
| --- | --- |
| 목적에 맞는 기능 | 입력의 모든 `episode_id`에 세 모델 중 하나를 선택하는 최종 라우터와 공식 출력 스키마 구현 |
| 합리적인 구조 | `experiments/` 학습·연구, `src/` 공식 프로토콜, `container/` 실행, `artifacts/` 고정 자산, `reports/` 증거를 분리 |
| 가독성·주석 | 최종 경로와 제출 스크립트에 한국어 docstring·주석, 각 파일 SPDX 헤더 |
| 테스트 | 280개 단위 테스트 통과, 실제 ARM64 컨테이너 격리 검사는 별도 JSON으로 보존 |
| 완성도 | 네트워크·GPU 없이 세 등급 실행, 원자적 JSON 출력, 비특권 사용자·읽기 전용 rootfs 통과 |

핵심 파일:

- [`final_route.py`](../experiments/method4_finetuned_encoder/final_route.py)
- [`method4.Dockerfile`](../container/method4.Dockerfile)
- [`method4-runtime-check.v1.json`](../reports/method4-runtime-check.v1.json)
- [`method4-order-audit.v1.json`](../reports/method4-order-audit.v1.json)

## 오픈소스 프로젝트로의 발전 가능성 — 6점

| 평가 관점 | 증거 |
| --- | --- |
| 공개 라이선스 | 자체 코드·문서 Apache-2.0, E5 MIT, Qwen Apache-2.0 |
| 재현 가능한 자산 | 고정 upstream revision, 파일별 SHA-256 manifest, 공개 GitHub Release |
| 공급망 투명성 | SPDX 2.3 SBOM, 기반 이미지 digest, THIRD_PARTY_NOTICES |
| 확장 가능성 | 인코더·학생·특징·등급 정책·비용 배분기를 분리해 다른 모델 또는 정책으로 교체 가능 |
| 실패 지식 보존 | 채택하지 않은 GLM·GBM·문제군 전문가·상한 후보도 기계 판독 보고서로 공개 |

핵심 파일:

- [`method4-assets.manifest.v1.json`](../artifacts/method4-assets.manifest.v1.json)
- [`method4-sbom.spdx.json`](../reports/method4-sbom.spdx.json)
- [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)
- [`CANDIDATE_EXPERIMENTS_KO.md`](CANDIDATE_EXPERIMENTS_KO.md)

## 개발 문서의 구체성 — 6점

| 필요한 독자 | 문서 |
| --- | --- |
| 처음 보는 심사자·개발자 | 루트 [`README.md`](../README.md)의 60초 요약과 쉬운 용어 설명 |
| 모델 개발자 | [`FINAL_ROUTER_KO.md`](FINAL_ROUTER_KO.md)의 학습 인자·특징·등급별 혼합 |
| 실험 재현자 | [`FINAL_EXPERIMENT_REPORT_KO.md`](FINAL_EXPERIMENT_REPORT_KO.md)의 원본 JSON·명령·기각 근거 |
| 운영자 | [`CONTAINER_VERIFICATION_KO.md`](CONTAINER_VERIFICATION_KO.md)의 자원·결정성·digest |
| 제출 담당자 | [`SUBMISSION_CHECKLIST_KO.md`](SUBMISSION_CHECKLIST_KO.md)와 [`DEMO_VIDEO_KO.md`](DEMO_VIDEO_KO.md) |

점수는 `홀드아웃`, `OOF`, `공개 재대입`, `스트레스`로 이름을 나눠 과대 해석을
막았다. 어려운 용어에는 일상적인 설명을 함께 적었다.

## 프로젝트 혁신성 — 6점

1. “질문의 난이도 한 개”가 아니라 세 모델의 **예상 품질과 예상 비용**을 함께
   학습하고 전체 배치 예산을 최적화한다.
2. 큰 Qwen 임베딩 모델은 오프라인 교사로만 쓰고, 그 의미 지식을 22,894바이트
   Ridge 학생으로 증류해 ARM64 CPU 제한에서 사용한다.
3. Fast·Balanced·Premium에 같은 분류기를 반복하지 않고 각 예산의 실패 비용에
   맞게 신호·허용 모델·안전계수를 다르게 설계한다.
4. 공개 점수 최대 후보를 버리고 228개 비용 스트레스를 모두 통과한 후보를 택해
   실제 서비스 라우팅의 예산 초과 위험을 설계에 포함한다.
5. 내용 정렬·점수 반올림·원순서 복원으로 ONNX 부동소수점과 배치 순서 차이까지
   결정성 위협으로 다룬다.

## 프로젝트 협업 및 관리체계 — 6점

| 평가 관점 | 증거 |
| --- | --- |
| 작업 추적 | 기반·검증·기준선·후보·최종 모델·비용·컨테이너·문서의 8개 한국어 이슈 |
| 단계적 검토 | 각 단계가 별도 `codex/issue-*` 브랜치와 PR #9~#15로 병합 |
| 논리적 커밋 | 기능·검증·문서를 나눈 한국어 커밋 메시지와 이슈 번호 |
| 의사결정 | PR에 가설, 실행 명령, 전후 수치, 채택·기각 이유 기록 |
| 외부 아이디어 | 경쟁 포크 링크와 배운 원칙을 기록하고 독립 실험 뒤 반영 여부 결정 |

GitHub 기록:

- [이슈 목록](https://github.com/Diwoni/ossp-2026-llm-router-challenge/issues)
- [PR 목록](https://github.com/Diwoni/ossp-2026-llm-router-challenge/pulls?q=is%3Apr)
- [프로젝트 계획](PROJECT_PLAN_KO.md)
- [실험 규약](EXPERIMENT_PROTOCOL_KO.md)

## 서면평가에서 먼저 보여 줄 세 장면

1. README의 60초 요약과 최종 아키텍처
2. `0.84` 고득점 후보를 버리고 `0.78` 228/228 통과 후보를 채택한 표
3. ARM64 2회 실행·역순 불일치 0·SBOM·Release를 한 화면에 묶은 컨테이너 증거

이 세 장면은 성능, 신뢰성, 오픈소스 완성도를 동시에 설명한다.
