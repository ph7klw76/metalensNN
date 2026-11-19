"""Record the final source/config/lock identities alongside executed results."""
import hashlib
import json
from pathlib import Path

files = []
for pattern in ["src/**/*.py","tests/*.py","scripts/*.py","scripts/*.sh","configs/*","locks/*","reference.yaml","assumptions.md"]:
    files.extend(Path(".").glob(pattern))
manifest = {str(path):{"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
            for path in sorted(set(files)) if path.is_file()}
folder = Path("artifacts/desktop-report")
folder.mkdir(exist_ok=True)
(folder/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"Recorded {len(manifest)} source/config/lock identities")
