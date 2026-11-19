"""Bound uv preparation to one locked wheel at a time, with resume checkpoints."""
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
lock_path = root/"locks/rtx4060-wsl2-py312-cuda12.txt"
text = lock_path.read_text()
blocks = [block for block in re.split(r"(?m)^(?=[A-Za-z0-9][A-Za-z0-9_.-]*==)", text)
          if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==", block)]
staging = root/".cache/one-package"
staging.mkdir(parents=True, exist_ok=True)
folder = root/"artifacts/phase1/package-install"
folder.mkdir(parents=True, exist_ok=True)
python = root/".venv-gpu/bin/python"
for i, block in enumerate(blocks):
    name, version = re.match(r"^([\w.-]+)==([^\s\\]+)", block).groups()
    requirement = staging/f"{name}.txt"
    requirement.write_text(block)
    log = folder/f"{name}.log"
    checkpoint = folder/f"{name}.json"
    if checkpoint.exists():
        recorded = json.loads(checkpoint.read_text())
        if recorded.get("version") == version and recorded.get("complete"):
            verify = subprocess.run([str(python), "-c", "import importlib.metadata,sys; assert importlib.metadata.version(sys.argv[1])==sys.argv[2]", name, version], capture_output=True)
            if verify.returncode == 0:
                print(f"Resume: {name}=={version} ({i+1}/{len(blocks)})", flush=True)
                continue
    command = ["uv", "pip", "install", "--python", str(python), "--reinstall", "--no-deps",
               "--require-hashes", "--no-index", "--find-links", str(root/".cache/locked-wheels"),
               "-r", str(requirement)]
    print(f"Install: {name}=={version} ({i+1}/{len(blocks)})", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-4000:])
        sys.exit(result.returncode)
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"name": name, "version": version, "complete": True})+"\n")
    temporary.replace(checkpoint)
subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
