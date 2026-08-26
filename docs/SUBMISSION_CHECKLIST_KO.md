<!--
SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
SPDX-License-Identifier: Apache-2.0
-->

# 최종 제출 체크리스트

## 현재 고정값

| 항목 | 값 |
| --- | --- |
| 저장소 | `https://github.com/Diwoni/ossp-2026-llm-router-challenge` |
| 이미지 소스 커밋 | `ca829e1f7d8277a4dcc7e8434607e18c3dd4b068` |
| 이미지 digest | `sha256:1913d548e33fe9fcad60eac4e33c73142c73e0b3b1754061080c941154b63829` |
| 전체 이미지 참조 | `ghcr.io/diwoni/ossp-2026-llm-router-challenge@sha256:1913d548e33fe9fcad60eac4e33c73142c73e0b3b1754061080c941154b63829` |
| 주 라이선스 | `Apache-2.0` |
| 마감 | 2026년 8월 27일 18:00 KST |

## 이미 완료한 항목

- [x] 깨끗한 upstream fork에서 단계별 이슈·PR·한국어 커밋으로 구현
- [x] 최종 10개 모델 자산 manifest와 GitHub Release 업로드
- [x] Release 압축 SHA와 압축 해제 파일별 SHA 재다운로드 검증
- [x] linux/arm64 최종 이미지 빌드와 GHCR push
- [x] 최종 이미지 digest 고정
- [x] 공개 Train+Dev 2,640문항, 세 등급, 공식형 자원 제한 2회 실행
- [x] 정상·역순 입력 결정성 감사
- [x] SPDX 2.3 SBOM과 제3자 라이선스·NOTICE 작성
- [x] `submission-ossp-skt.json` 스키마 검증
- [x] 시연 영상 대본과 결과보고서용 원고 작성

## 제출 직전 반드시 할 항목

아래는 이미지·저장소를 공개해야 하므로 제출 버튼을 누르기 전에 수행한다.

- [ ] GHCR 패키지 visibility를 `public`으로 전환
- [ ] `gh api /user/packages/container/ossp-2026-llm-router-challenge`에서
  `visibility: public` 확인
- [ ] 인증 정보가 없는 Docker 설정에서 전체 digest pull 성공 확인
- [ ] 원격 이미지가 `linux/arm64`, UID/GID `65532:65532`, `VOLUME` 없음인지 확인
- [ ] 제출 스냅샷 커밋 URL을 로그아웃 창에서 열 수 있는지 확인
- [ ] `submission-ossp-skt.json`의 여섯 필드와 이미지 소스 커밋 대응 재확인
- [ ] 결과보고서의 프로젝트 등록 URL에 **브랜치가 아닌 전체 커밋 SHA URL** 입력
- [ ] 시연 영상 URL을 비로그인 상태에서 재생 확인
- [ ] 공식 결과보고서 원본과 PDF에서 안내 문구·자리표시자 삭제
- [ ] 접수번호·팀명에 맞게 파일명 변경
- [ ] 원본과 PDF를 마감 전에 업로드하고 접수 완료 시각 캡처

## 익명 pull 검증

GHCR을 공개한 뒤 기존 로그인 정보를 쓰지 않도록 임시 Docker 설정을 만든다.

```console
검증용_도커설정="$(mktemp -d)"
DOCKER_CONFIG="$검증용_도커설정" docker pull \
  ghcr.io/diwoni/ossp-2026-llm-router-challenge@sha256:1913d548e33fe9fcad60eac4e33c73142c73e0b3b1754061080c941154b63829
```

검증 후 임시 디렉터리는 휴지통 또는 안전한 방법으로 정리한다. 제출 이미지에는
토큰·비밀번호·로컬 경로가 포함되지 않아야 한다.

## 기술 제출 JSON 검증

```console
python3 tools/validate_technical_submission.py
```

JSON의 `commit_sha`는 JSON 파일을 추가한 커밋이 아니라 **이미지를 빌드한 앞쪽
코드 커밋**이다. 결과보고서의 프로젝트 등록 URL은 JSON을 포함하는 뒤쪽 커밋의
고정 스냅샷이다.

## 최종 실행 증거 확인

```console
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py'

python3 -m json.tool reports/method4-runtime-check.v1.json >/dev/null
python3 -m json.tool reports/method4-order-audit.v1.json >/dev/null
python3 -m json.tool reports/final-cost-robustness.v1.json >/dev/null
python3 -m json.tool reports/method4-sbom.spdx.json >/dev/null
```

기대 결과는 단위 테스트 280개 통과, 컨테이너 등급 6회 통과, 역순 불일치 0,
비용 스트레스 228/228 통과다.

## 제출 뒤 유지

- 평가가 끝날 때까지 제출 스냅샷과 이미지 digest를 공개 상태로 유지한다.
- 수상 시 수상일로부터 5년 동안 제출 저장소를 공개 상태로 유지한다.
- 태그를 다시 밀더라도 제출한 digest와 소스 커밋은 삭제하지 않는다.
