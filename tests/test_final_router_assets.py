# SPDX-FileCopyrightText: Copyright 2026 Diwoni contributors
# SPDX-License-Identifier: Apache-2.0

"""최종 라우터 정책과 자산 계약을 검사합니다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_method4_assets import verify_assets


ROOT = Path(__file__).resolve().parents[1]


class FinalRouterAssetsTest(unittest.TestCase):
    def test_final_gate_keeps_budget_margin_and_fast_forbids_think(self) -> None:
        gate = json.loads(
            (ROOT / "configs/method4-tier-gate.qwen-student.json").read_text()
        )
        self.assertEqual(["ax31-light", "ax31"], gate["fast"]["allowed_model_ids"])
        self.assertLessEqual(gate["fast"]["safety_ratio"], 0.95)
        self.assertLessEqual(gate["balanced"]["safety_ratio"], 0.90)
        self.assertLessEqual(gate["premium"]["safety_ratio"], 0.78)
        self.assertEqual(0.75, gate["balanced"]["teacher_student_weight"])

    def test_asset_manifest_has_unique_safe_paths_and_exact_total(self) -> None:
        manifest = json.loads(
            (ROOT / "artifacts/method4-assets.manifest.v1.json").read_text()
        )
        paths = [row["path"] for row in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))
        self.assertTrue(all(".." not in Path(path).parts for path in paths))
        self.assertEqual(
            manifest["total_size"], sum(int(row["size"]) for row in manifest["files"])
        )
        self.assertFalse(manifest["runtime_contains_qwen"])

    def test_asset_verifier_detects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = "검증 자산".encode("utf-8")
            (root / "asset.bin").write_bytes(payload)
            manifest = {
                "total_size": len(payload),
                "files": [
                    {
                        "path": "asset.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
            self.assertEqual([], verify_assets(manifest, root))
            manifest["files"][0]["sha256"] = "0" * 64
            self.assertIn("SHA-256", verify_assets(manifest, root)[0])


if __name__ == "__main__":
    unittest.main()
