"""Check desktop SLC export against the previously verified fabrication files."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fiber_metalens_desktop_slc import export_slc, validate_height


class DesktopSlcTest(unittest.TestCase):
    def test_reject_invalid_cylinder_height_without_snapping(self):
        self.assertEqual(validate_height("67.5"), 67.5)
        for invalid in ("40.25", 39.5, 100.5, float("nan"), float("inf"), "hello"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_height(invalid)

    def test_export_matches_verified_500nm_binaries(self):
        references = ROOT / "exports/height_family_cylinder_attached_slc_500nm_v5"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for height, filename, layers in ((40, "C01.slc", 86), (75, "C02.slc", 156), (100, "C03.slc", 206)):
                with self.subTest(height=height):
                    source_path = references / "fabrication_sources" / f"NN_H{height}_500nm_fabrication.json"
                    design = json.loads(source_path.read_text(encoding="utf-8"))
                    chosen = folder / f"User custom name H{height}.slc"
                    result = export_slc(chosen, height, design["heights_um"])
                    raw = chosen.read_bytes()
                    self.assertEqual(raw, (references / f"H{height:03d}" / filename).read_bytes())
                    self.assertEqual(Path(result["slc_path"]), chosen)
                    self.assertEqual(result["audit_status"], "PASS")
                    self.assertEqual(result["audit_check_count"], 19)
                    self.assertEqual(result["layer_count"], layers)
                    self.assertEqual(result["layer_interval_um"], 0.5)
                    self.assertEqual(result["total_height_um"], height + 3)

                    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
                    audit_bytes = Path(result["audit_path"]).read_bytes()
                    audit = json.loads(audit_bytes)
                    source_bytes = manifest["source_design_exact_json_utf8"].encode("utf-8")
                    self.assertEqual(hashlib.sha256(source_bytes).hexdigest(), manifest["source_design_sha256"])
                    self.assertEqual(json.loads(source_bytes), manifest["source_design"])
                    self.assertEqual(hashlib.sha256(audit_bytes).hexdigest(), manifest["independent_audit_sha256"])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), audit["provenance"]["slc_sha256"])
                    self.assertEqual(audit["status"], "PASS")
                    self.assertTrue(all(audit["checks"].values()))

                    with self.assertRaises(FileExistsError):
                        export_slc(chosen, height, design["heights_um"])
                    self.assertEqual(chosen.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
