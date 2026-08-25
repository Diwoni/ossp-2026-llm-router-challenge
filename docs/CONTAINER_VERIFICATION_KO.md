<!--
SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
SPDX-License-Identifier: Apache-2.0
-->

# 최종 Method 4 컨테이너 검증

## 결론

후보 이미지는 공식과 같은 linux/arm64, 2CPU, 2GiB, 90초, PID 32, 네트워크
차단, 읽기 전용 rootfs, 비특권 사용자 조건에서 공개 Train+Dev 2,640문항을
세 등급 모두 통과했다. 등급별 2회 실행의 출력 SHA-256이 같고, 전체 입력을
뒤집어도 episode별 모델 선택 불일치가 0개였다.

## 이미지 구성

| 항목 | 값 |
| --- | --- |
| 후보 소스 커밋 | `7dfd0044282f966171c23338d0357e43632c9750` |
| 후보 로컬 이미지 ID | `sha256:4ffb3ec740da60d4810e907e0a0ddcf07a5b673e64fbc13f06475a14bc074948` |
| 플랫폼 | linux/arm64 |
| 사용자 | `65532:65532` |
| Docker reported 크기 | 468,590,643바이트 |
| 병합 rootfs 겉보기 크기 | 1,012,995,768바이트 |
| rootfs 실제 블록 사용량 | 약 934MB |
| 계층 수 | 16 |
| VOLUME | 없음 |
| GPU·네트워크 | 사용 안 함 |

공식 한도는 OCI 압축 계층 1GiB, 병합 rootfs 2GiB다. 두 값 모두 충분한 여유가
있다. 이미지에는 PyTorch, transformers, scikit-learn, scipy가 없다.

## 공식 자원 제한 반복 결과

| 등급 | 반복 1 | 반복 2 | 출력 SHA 동일 | 90초 통과 |
| --- | ---: | ---: | :---: | :---: |
| Fast | 49.583초 | 49.705초 | 예 | 예 |
| Balanced | 50.183초 | 50.598초 | 예 | 예 |
| Premium | 64.061초 | 63.943초 | 예 | 예 |

등급별 출력 SHA-256은 다음과 같다.

- Fast: `721d63e209e2a836ee7096fde32ff7932331f1fab13898261dfb0acc75ee834b`
- Balanced: `d4a809272a6f023c2e1a6c9216c32aee4bcbc50beb883a52ce3be3d11333eb42`
- Premium: `8bb740c221b252069b5fa848978d5e35fa1fbcb56f40bf1a77eacaa216c95604`

원본 보고서는
[`method4-runtime-check.v1.json`](../reports/method4-runtime-check.v1.json)에
있고 SHA-256은
`9f2a9397153c8951c33a803818f664b687223faf11afdb367803f1c2386e67b4`다.

## 입력 순서 감사

같은 2,640문항을 정상 순서와 완전 역순으로 동일 이미지에 넣었다. 두 실행 모두
공식 격리 조건을 적용했다.

| 등급 | 누락 | 추가 | model_id 불일치 |
| --- | ---: | ---: | ---: |
| Fast | 0 | 0 | 0 |
| Balanced | 0 | 0 | 0 |
| Premium | 0 | 0 | 0 |

원본 보고서는
[`method4-order-audit.v1.json`](../reports/method4-order-audit.v1.json)에 있고
SHA-256은
`7f28c2dc57dba9338b03c5f56274e99bb6ce1f0dab7fbf6a5c929e3a9ffc83f8`다.

## SBOM과 자산

- SPDX 2.3 SBOM: Debian 105개, Python 6개, 모델 2개
- SBOM SHA-256:
  `93b7b41d0011c5b1a2e768742ba7e3eb1cba5f3611c0071c550c718b521482dc`
- Release 압축 크기: 376,520,961바이트
- Release 압축 SHA-256:
  `d320179bd0c1f22cfcfb2711964d1bed21fa67fcab95af69a8587f9c49044b3a`
- 압축 해제 자산: 10개, 636,519,519바이트

자산은 먼저 압축 SHA를 확인하고, 압축 해제 뒤
[`method4-assets.manifest.v1.json`](../artifacts/method4-assets.manifest.v1.json)의
파일별 크기·SHA를 다시 확인한다.

## 재현 명령

```console
scripts/download_method4_assets.sh

docker build --provenance=false --platform linux/arm64 \
  --file container/method4.Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --tag ossp-method4:final .

PYTHONPATH=src:. python3 tools/check_runtime.py \
  --image ossp-method4:final \
  --repetitions 2 \
  --report build/method4-runtime-check.json
```

Docker 29는 기본 build에 provenance attestation manifest를 더한다. 로컬 공식
검사기는 플랫폼 manifest를 불변 로컬 ID로 실행하므로 단일 플랫폼 제출 빌드에는
`--provenance=false`를 사용했다. 공급망 구성은 별도 SPDX SBOM으로 공개한다.
