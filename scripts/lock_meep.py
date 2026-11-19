"""Export a standard installable conda explicit lock plus SHA-256 manifest."""
import json
from pathlib import Path

records = [json.loads(path.read_text()) for path in sorted(Path(".venv-meep/conda-meta").glob("*.json"))]
assert records and any(record["name"] == "pymeep" for record in records)
lines = ["# platform: linux-64", "# Installed CPU Meep environment; URLs carry MD5; SHA-256 recorded alongside.", "@EXPLICIT"]
packages = []
for record in records:
    assert record["url"].startswith("https://conda.anaconda.org/conda-forge/")
    assert len(record["md5"]) == 32 and len(record["sha256"]) == 64
    lines.append(record["url"]+"#"+record["md5"])
    packages.append({key:record[key] for key in ["name","version","build","url","md5","sha256"]})
Path("locks/meep-linux-64-explicit.txt").write_text("\n".join(lines)+"\n")
Path("locks/meep-linux-64-packages.json").write_text(json.dumps(packages,indent=2)+"\n")
print(f"Locked {len(records)} installed CPU packages")
