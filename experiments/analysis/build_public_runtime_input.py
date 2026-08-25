#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""공개 Train과 Dev를 공식 실행과 같은 하나의 prompt-only 입력으로 합칩니다."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.data import choose_existing_input, load_json, write_json_atomic


ROOT = Path(__file__).resolve().parents[2]


def build_public_input(train: dict, dev: dict) -> dict:
    """두 공개 split을 outcomes 없이 입력 순서대로 결합합니다."""

    if train["challenge_id"] != dev["challenge_id"]:
        raise ValueError("Train과 Dev의 challenge_id가 다릅니다")
    episodes = list(train["episodes"]) + list(dev["episodes"])
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Train과 Dev를 합친 episode_id가 중복됩니다")
    return {
        "challenge_id": train["challenge_id"],
        "episodes": episodes,
        "schema_version": 1,
        "split": "public-train-dev",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = load_json(choose_existing_input(ROOT, "train"))
    dev = load_json(choose_existing_input(ROOT, "dev"))
    result = build_public_input(train, dev)
    write_json_atomic(args.output, result)
    print(f"공개 실행 입력 {len(result['episodes'])}문항을 기록했습니다: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
