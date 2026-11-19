"""Fetch the official Linux package, verifying its published digest."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
metadata = json.load(urllib.request.urlopen("https://api.anaconda.org/package/conda-forge/micromamba", timeout=30))
file = next(f for f in reversed(metadata["files"]) if f["attrs"].get("subdir") == "linux-64" and f["version"] == "2.9.0")
url = "https:"+file["download_url"] if file["download_url"].startswith("//") else file["download_url"]
archive = root/".cache/micromamba.tar.bz2"
payload = urllib.request.urlopen(url, timeout=60).read(15*1024**2)
assert len(payload) == file["size"]
if file.get("sha256"):
    assert hashlib.sha256(payload).hexdigest() == file["sha256"]
else:
    assert hashlib.md5(payload).hexdigest() == file["md5"]
archive.write_bytes(payload)
with tarfile.open(archive) as tf:
    destination = root/".tools/bin/micromamba"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(tf.extractfile("bin/micromamba").read())
    destination.chmod(0o755)
(root/"artifacts/micromamba-provenance.json").write_text(json.dumps({"url":url,"sha256":hashlib.sha256(payload).hexdigest(),"version":file["version"]}, indent=2))
print("Verified micromamba", file["version"], flush=True)
metadata = json.load(urllib.request.urlopen("https://api.anaconda.org/package/conda-forge/pymeep", timeout=30))
candidates = [f for f in metadata["files"] if f["version"] == metadata["latest_version"] and f["attrs"].get("subdir") == "linux-64"]
print(json.dumps({"meep_latest_version":metadata["latest_version"], "builds":[f["basename"] for f in candidates]}), flush=True)
