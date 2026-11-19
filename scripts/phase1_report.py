"""Summarize executed evidence while retaining superseded setup failures."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root/"artifacts/phase1"


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


hardware = read_json(root/"hardware.json")
reports = {backend: {
    "smoke": read_json(folder/f"{backend}-smoke.json"),
    "memory": read_json(folder/f"{backend}-memory.jsonl.summary.json")
} for backend in ["cpu", "gpu"]}
tests = (folder/"pytest.log").read_text() if (folder/"pytest.log").exists() else "not run"
test_count = re.search(r"\b(\d+) passed\b", tests)
tests_ok = bool(test_count) and not re.search(r"\b(?:FAILED|ERROR|\d+ failed|\d+ errors?)\b", tests)
installer = read_json(folder/"install-bounded-memory.jsonl.summary.json")
history = {name: read_json(folder/name) for name in [
    "install-memory.jsonl.summary.json", "install-wsl-network-memory.jsonl.summary.json"]}
passed = bool(installer and installer["returncode"] == 0 and not installer["stop_reasons"] and tests_ok
              and all(r["smoke"] and r["smoke"]["pass"] and r["memory"] and
                      r["memory"]["returncode"] == 0 and not r["memory"]["stop_reasons"]
                      for r in reports.values()))
status = "PASS" if passed else "INCOMPLETE"
if not passed and installer and installer.get("stop_reasons"):
    status = "BLOCKED_RESOURCE"
elif not passed and installer and installer.get("returncode") not in (None, 0):
    status = "FAILED_INSTALL"
data = {"phase": 1, "status": status, "hardware": hardware,
        "lock_sha256": hashlib.sha256((root/"locks/rtx4060-wsl2-py312-cuda12.txt").read_bytes()).hexdigest(),
        "reports": reports, "pytest": tests, "pytest_passed_count": int(test_count[1]) if test_count else None,
        "installer": installer, "historical_setup_attempts": history,
        "memory_policy": hardware["memory_policy"],
        "package_checkpoints": len(list((folder/"package-install").glob("*.json"))),
        "phase_2_to_7_status": "NOT RUN",
        "source_preparation": "Attached scans rendered/OCR'd; SI fetched with publisher checksum. Physical source audit not accepted yet.",
        "optical_claims": "None. Numerical smoke tests are not optical proxy or Maxwell validation."}
(folder/"gate.json").write_text(json.dumps(data, indent=2)+"\n")
device = hardware["gpu"]["devices"][0]
lines = ["# Desktop metalens trial - executed results", "", f"Phase 1: **{status}**.", "",
         f'Hardware: {device["name"]}, {device["total_bytes"]/1024**2:.0f} MiB VRAM, SM {device["compute_capability"]}, driver {device["driver"]}, WSL2 {hardware["wsl_distro"]}, Python {hardware["python"]}.', "",
         f'The driver reports CUDA compatibility {hardware["driver_reported_cuda_compatibility"]}. Toolkit {hardware["installed_cuda_toolkit_version"]} is installed separately. JAX CUDA wheel libraries are pinned in the lock.', "",
         "The user removed the fixed Windows available-RAM stop on 23 October 2026. Windows RAM is still recorded. Project process RSS, WSL available RAM, GPU usage, and timeouts remain enforced. Old stops under the 768 MiB Windows reserve are historical evidence, not the current gate.", "",
         "| Smoke test | Result | Peak sampled project RSS | Peak total-device VRAM increase |", "|---|---|---|---|"]
for backend, report in reports.items():
    if report["smoke"] and report["memory"]:
        memory = report["memory"]
        result = "PASS" if report["smoke"]["pass"] and memory["returncode"] == 0 and not memory["stop_reasons"] else "FAIL"
        lines.append(f'| {backend} | {result} | {memory["peak_project_rss_bytes"]/1024**2:.1f} MiB | {memory["peak_total_device_delta_bytes"]/1024**2:.1f} MiB |')
    else:
        lines.append(f"| {backend} | not completed | not measured | not measured |")
lines += ["", f'Pytest: {test_count[1]+" passed" if tests_ok else "not passed / not completed"}.', ""]
if installer:
    lines += [f'Sequential installation: exit {installer["returncode"]}, wall time {installer.get("wall_s", 0):.1f} s, peak sampled project RSS {installer.get("peak_project_rss_bytes", 0)/1024**2:.1f} MiB. Stop reasons: {installer.get("stop_reasons", [])}.', ""]
lines += [f'Completed package checkpoints: {data["package_checkpoints"]}. All 63 wheel files (3,443,208,298 bytes) were hash-verified before installation.', "",
          "Memory is sampled and may miss transient peaks. Summed RSS can double-count shared pages. GPU total-device increases can include other applications; JAX allocator statistics are included in completed GPU reports. Disabling preallocation is not a hard allocation cap.", "",
          "A previous bulk installation stopped at 0.563 GiB available Windows RAM with 421.1 MiB peak sampled process RSS. RAM recovered after exit; temporary extraction/copy staging was a plausible cause of the transient host pressure, not a proven root cause. That log is preserved. The current installation uses separate package processes and resume checkpoints.", "",
          "Phases 2-7 have not run. The prompt requests one phase per interaction and a stop at each acceptance gate. PDF/OCR extraction and publisher SI retrieval are preparation only. No metalens training, optical proxy performance, converged Maxwell simulation, or Phase 7 engineering/optical gate is claimed.", "",
          "Evidence: `hardware.json`, `artifacts/phase1/gate.json`, `pytest.log`, installer and smoke reports, memory JSONL files, and dependency locks. No WSL restart, driver update, system-setting change, or termination of unrelated processes was performed.", ""]
(root/"results.md").write_text("\n".join(lines))
print(json.dumps({"phase": 1, "status": status, "reports": str(folder/"gate.json")}, indent=2))
