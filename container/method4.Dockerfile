# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

COPY container/requirements-method4-runtime.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && python3 -m pip uninstall --yes pip setuptools wheel \
    && rm -f /tmp/requirements.txt

ARG SOURCE_COMMIT=unbound
LABEL org.opencontainers.image.source="https://github.com/Diwoni/ossp-2026-llm-router-challenge" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.sktelecom.ossp.router="method4-e5-qwen-student"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router/src:/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    ORT_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

COPY --chown=65532:65532 src /opt/router/src
COPY --chown=65532:65532 baselines /opt/router/baselines
COPY --chown=65532:65532 experiments/__init__.py /opt/router/experiments/__init__.py
COPY --chown=65532:65532 experiments/common /opt/router/experiments/common
COPY --chown=65532:65532 experiments/method4_finetuned_encoder /opt/router/experiments/method4_finetuned_encoder
COPY --chown=65532:65532 configs/method4-tier-gate.qwen-student.json /opt/router/configs/method4-tier-gate.final.json
COPY --chown=65532:65532 build/final-assets/method4 /opt/router/assets/method4
COPY --chown=65532:65532 artifacts/method4-assets.manifest.v1.json /opt/router/assets/method4/MANIFEST.json
COPY --chown=65532:65532 container/method4_entrypoint.py /opt/router/method4_entrypoint.py

WORKDIR /opt/router
USER 65532:65532
ENTRYPOINT ["python3", "/opt/router/method4_entrypoint.py"]
