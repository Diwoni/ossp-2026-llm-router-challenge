# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""제출 이미지에서 한 등급의 최종 Method 4 라우터를 실행합니다."""

from __future__ import annotations

import sys

from experiments.method4_finetuned_encoder.final_route import main


if __name__ == "__main__":
    # 사용자는 input·tier·output만 전달합니다. 검증된 자산 경로와 스레드 수는
    # 이미지 내부에서 마지막 인자로 고정해 임의 변경과 실행 환경 차이를 막습니다.
    sys.argv.extend(
        [
            "--onnx-artifact-dir",
            "/opt/router/assets/method4/onnx",
            "--embedding-onnx-artifact-dir",
            "/opt/router/assets/method4/embedding-onnx",
            "--hash-artifact",
            "/opt/router/baselines/hash-regex-public.v1.json",
            "--stacking-artifact",
            "/opt/router/assets/method4/stacking-head.json",
            "--residual-artifact",
            "/opt/router/assets/method4/residual-head.json",
            "--similarity-artifact",
            "/opt/router/assets/method4/similarity-head.npz",
            "--teacher-student-artifact",
            "/opt/router/assets/method4/teacher-student-head.json",
            "--tier-gate",
            "/opt/router/configs/method4-tier-gate.final.json",
            "--threads",
            "2",
        ]
    )
    raise SystemExit(main())
