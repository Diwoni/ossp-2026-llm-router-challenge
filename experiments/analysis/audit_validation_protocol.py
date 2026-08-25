#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""공개 데이터 SHA와 누수 방지 fold의 무결성을 검사합니다."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from experiments.common.data import (
    choose_existing_input,
    load_examples,
    load_json,
    write_json_atomic,
)
from experiments.validation.splits import (
    content_hash_folds,
    normalize_template,
    source_aware_folds,
)
from experiments.validation.stress import family_counts, family_multiplier_scenarios


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public_manifest = load_json(ROOT / "data" / "public-data.v1.json")
    report: dict[str, object] = {"schema_version": 1, "folds": args.folds, "splits": {}}
    for split in ("train", "dev"):
        input_path = choose_existing_input(ROOT, split)
        outcomes_path = ROOT / "data" / split / "outcomes.json"
        examples = load_examples(input_path, outcomes_path)
        content_folds, _content_fold_ids = content_hash_folds(examples, args.folds)
        source_folds, source_groups, families = source_aware_folds(
            examples,
            args.folds,
            split=split,
            deepmind_selection_path=(
                ROOT / "data" / "sources" / "deepmind-mathematics-selection.v1.json"
            ),
            aime_selection_path=ROOT / "data" / split / "aime-selection.json",
        )
        input_sha256 = _sha256(input_path)
        outcomes_sha256 = _sha256(outcomes_path)
        expected = public_manifest["splits"][split]["sha256"]
        report["splits"][split] = {  # type: ignore[index]
            "episodes": len(examples),
            "input_sha256": input_sha256,
            "expected_input_sha256": expected["materialized_inputs"],
            "input_sha256_passed": input_sha256 == expected["materialized_inputs"],
            "outcomes_sha256": outcomes_sha256,
            "expected_outcomes_sha256": expected["outcomes"],
            "outcomes_sha256_passed": outcomes_sha256 == expected["outcomes"],
            "content_hash_fold_sizes": [len(validation) for _train, validation in content_folds],
            "source_aware_fold_sizes": [len(validation) for _train, validation in source_folds],
            "content_hash_groups": len(
                {normalize_template(str(row["text"])) for row in examples}
            ),
            "source_aware_groups": len(set(source_groups)),
            "family_counts": family_counts(families),
            "stress_scenarios": len(family_multiplier_scenarios(families)),
        }
    write_json_atomic(args.output, report)
    print(f"검증 규약 감사 보고서를 기록했습니다: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
