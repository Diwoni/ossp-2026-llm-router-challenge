#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""최종 Method 4 이미지의 SPDX 2.3 SBOM을 생성합니다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from experiments.common.data import load_json, write_json_atomic


PYTHON_LICENSES = {
    "flatbuffers": "Apache-2.0",
    "numpy": "BSD-3-Clause",
    "onnxruntime": "MIT",
    "packaging": "Apache-2.0 OR BSD-2-Clause",
    "protobuf": "BSD-3-Clause",
    "tokenizers": "Apache-2.0",
}


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return completed.stdout


def _spdx_id(kind: str, value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")
    return f"SPDXRef-{kind}-{normalized}"


def _package(
    name: str,
    version: str,
    license_id: str,
    *,
    kind: str,
    purl: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "SPDXID": _spdx_id(kind, name),
        "name": name,
        "versionInfo": version,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": False,
        "licenseConcluded": license_id,
        "licenseDeclared": license_id,
        "copyrightText": "NOASSERTION",
    }
    if purl:
        row["externalRefs"] = [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": purl,
            }
        ]
    if comment:
        row["comment"] = comment
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker-command", default="docker")
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("artifacts/method4-assets.manifest.v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(
        _run([args.docker_command, "image", "inspect", args.image])
    )[0]
    image_id = str(metadata["Id"])
    revision = str(metadata["Config"]["Labels"]["org.opencontainers.image.revision"])
    os_packages = []
    for line in _run(
        [
            args.docker_command,
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "dpkg-query",
            args.image,
            "-W",
            "-f=${binary:Package}\\t${Version}\\n",
        ]
    ).splitlines():
        name, version = line.split("\t", 1)
        os_packages.append(
            _package(
                name,
                version,
                "NOASSERTION",
                kind="Debian",
                purl=f"pkg:deb/debian/{name}@{version}",
            )
        )
    python_rows = json.loads(
        _run(
            [
                args.docker_command,
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python3",
                args.image,
                "-c",
                (
                    "import importlib.metadata as m,json;"
                    "print(json.dumps(sorted((d.metadata['Name'],d.version) "
                    "for d in m.distributions())))"
                ),
            ]
        )
    )
    python_packages = [
        _package(
            name,
            version,
            PYTHON_LICENSES.get(name.lower(), "NOASSERTION"),
            kind="Python",
            purl=f"pkg:pypi/{name.lower()}@{version}",
        )
        for name, version in python_rows
    ]
    manifest = load_json(args.asset_manifest)
    model_packages = [
        _package(
            str(row["name"]),
            str(row["revision"]),
            str(row["license"]),
            kind="Model",
            purl=(
                "pkg:huggingface/"
                f"{str(row['name']).replace('/', '%2F')}@{row['revision']}"
            ),
            comment=(
                "제출 런타임 포함"
                if row["runtime_included"]
                else "오프라인 교사 전용이며 제출 런타임에 미포함"
            ),
        )
        for row in manifest["source_models"]
    ]
    project = _package(
        "Diwoni OSSP 2026 LLM Router",
        revision,
        "Apache-2.0",
        kind="Project",
        purl=f"pkg:github/Diwoni/ossp-2026-llm-router-challenge@{revision}",
        comment=f"linux/arm64 이미지 {image_id}",
    )
    base = _package(
        "python-slim-bookworm",
        "3.12.11",
        "Python-2.0 AND NOASSERTION",
        kind="BaseImage",
        purl="pkg:docker/python@3.12.11-slim-bookworm",
        comment=(
            "기반 이미지 다이제스트 "
            "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
        ),
    )
    packages = [project, base, *os_packages, *python_packages, *model_packages]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": project["SPDXID"],
        }
    ] + [
        {
            "spdxElementId": project["SPDXID"],
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": row["SPDXID"],
        }
        for row in packages[1:]
    ]
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "Diwoni OSSP 2026 Method 4 이미지 SBOM",
        "documentNamespace": f"https://github.com/Diwoni/ossp-2026-llm-router-challenge/sbom/{image_id.removeprefix('sha256:')}",
        "creationInfo": {
            "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "creators": ["Tool: tools/generate_method4_sbom.py"],
            "licenseListVersion": "3.27",
        },
        "documentDescribes": [project["SPDXID"]],
        "packages": packages,
        "relationships": relationships,
    }
    write_json_atomic(args.output, document)
    print(
        f"SPDX SBOM 기록 완료: Debian {len(os_packages)}개, "
        f"Python {len(python_packages)}개, 모델 {len(model_packages)}개"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
