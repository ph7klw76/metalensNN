"""Memory-gate checks can run before scientific dependencies are installed."""
import tempfile
import unittest
import json
import os
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path

from fiber_metalens_nn.audit import effective_hardware, memory_policy, preflight
from fiber_metalens_nn.guard import owned_child, rss_tree, run_guarded, violations


class SafetyTests(unittest.TestCase):
    def test_desktop_policy_covers_phases_two_through_seven(self):
        policy = memory_policy()
        self.assertIsNone(policy["host_free_floor_bytes"])
        self.assertTrue(set(range(2, 8)).issubset(policy["applies_to_phases"]))
        with patch("fiber_metalens_nn.audit.Path.exists", return_value=False):
            self.assertIsNone(memory_policy()["host_free_floor_bytes"])

    def test_saved_hardware_cannot_restore_removed_windows_gate(self):
        saved = {"windows": {"available_bytes": int(0.56*1024**3)},
                 "budgets": {"project_ram_bytes": 2*1024**3, "project_gpu_bytes": 2*1024**3,
                             "host_free_floor_bytes": 768*1024**2, "wsl_free_floor_bytes": 10,
                             "gpu_free_floor_bytes": 10},
                 "gpu": {"devices": [{"name": "RTX 4060"}]}}
        sample = {"project_rss_bytes": 50, "host_available_bytes": saved["windows"]["available_bytes"],
                  "wsl_available_bytes": 20, "gpu_free_bytes": 20, "gpu_delta_bytes": 0}
        self.assertIn("memory pressure: host_available_bytes", violations(sample, saved["budgets"]))
        effective = effective_hardware(saved)
        self.assertEqual(preflight(effective), [])
        self.assertEqual(violations(sample, effective["budgets"]), [])
        self.assertEqual(saved["budgets"]["host_free_floor_bytes"], 768*1024**2)
        self.assertEqual(effective["budgets"]["project_ram_bytes"], saved["budgets"]["project_ram_bytes"])

    def test_disabled_host_floor_keeps_other_limits(self):
        budgets = {"project_ram_bytes": 100, "project_gpu_bytes": 100,
                   "host_free_floor_bytes": None, "wsl_free_floor_bytes": 10,
                   "gpu_free_floor_bytes": 10}
        sample = {"project_rss_bytes": 50, "host_available_bytes": 0,
                  "wsl_available_bytes": 20, "gpu_free_bytes": 20, "gpu_delta_bytes": 30}
        self.assertEqual(violations(sample, budgets), [])
        sample["host_available_bytes"] = None
        self.assertEqual(violations(sample, budgets), [])
        sample.update(project_rss_bytes=101, wsl_available_bytes=5, gpu_delta_bytes=101)
        errors = violations(sample, budgets)
        self.assertIn("project RAM budget exceeded", errors)
        self.assertIn("memory pressure: wsl_available_bytes", errors)
        self.assertIn("GPU allocation delta budget exceeded", errors)

    def test_memory_pressure_and_missing_readings(self):
        budgets = {"project_ram_bytes": 100, "project_gpu_bytes": 100,
                   "host_free_floor_bytes": 10, "wsl_free_floor_bytes": 10,
                   "gpu_free_floor_bytes": 10}
        sample = {"project_rss_bytes": 50, "host_available_bytes": 20,
                  "wsl_available_bytes": 20, "gpu_free_bytes": 20, "gpu_delta_bytes": 30}
        self.assertEqual(violations(sample, budgets), [])
        sample["host_available_bytes"] = 5
        self.assertIn("memory pressure: host_available_bytes", violations(sample, budgets))
        sample["host_available_bytes"] = None
        self.assertTrue(any("unavailable" in s for s in violations(sample, budgets)))
        sample["project_rss_bytes"] = 101
        self.assertIn("project RAM budget exceeded", violations(sample, budgets))

    def test_only_owned_process_tree_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, ppid, rss in [(100, 1, 10), (101, 100, 20), (102, 101, 30), (200, 1, 9000)]:
                folder = root / str(pid)
                folder.mkdir()
                (folder / "stat").write_text(f"{pid} (name with (parentheses)) S {ppid} 0 0\n")
                (folder / "status").write_text(f"VmRSS:\t{rss} kB\n")
            self.assertEqual(rss_tree(100, root), 60*1024)

    def test_low_ram_preflight_rejected(self):
        data = {"windows": {"available_bytes": 400*1024**2},
                "budgets": {"project_ram_bytes": 0, "project_gpu_bytes": 4*1024**3},
                "gpu": {"devices": [{"name": "RTX 4060"}]}}
        self.assertTrue(any("project RAM" in s for s in preflight(data)))

    def test_preflight_cannot_launch_child_when_ram_too_low(self):
        data = {"windows": {"available_bytes": 400*1024**2},
                "budgets": {"project_ram_bytes": 0, "project_gpu_bytes": 4*1024**3},
                "gpu": {"devices": [{"name": "RTX 4060"}]}}
        with tempfile.TemporaryDirectory() as temp:
            hardware = Path(temp)/"hardware.json"
            hardware.write_text(json.dumps(data))
            with patch("fiber_metalens_nn.guard.subprocess.Popen") as launch:
                code = run_guarded(["must-not-start"], str(hardware), str(Path(temp)/"log.jsonl"))
                launch.assert_not_called()
            self.assertEqual(code, 2)

    @unittest.skipUnless(os.name == "posix", "Linux owned process-group behavior")
    def test_exception_cleans_owned_job_and_preserves_unrelated_job(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with owned_child([sys.executable, "-c", "import time; time.sleep(20)"]) as owned:
                    raise RuntimeError("intentional supervisor exception")
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
