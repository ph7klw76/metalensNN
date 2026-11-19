"""Retrieve the publisher's small SI PDF with source hashes and metadata."""
import hashlib
import json
import urllib.request
from pathlib import Path

out = Path(__file__).resolve().parents[1]/"artifacts/source_audit"
url = "https://api.figshare.com/v2/articles/14219470"
metadata = json.load(urllib.request.urlopen(url, timeout=30))
(out/"figshare-metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
source = next(x for x in metadata["files"] if x["name"] == "nl0c04463_si_001.pdf")
payload = urllib.request.urlopen(source["download_url"], timeout=30).read(1024**2)
assert len(payload) == source["size"] < 1024**2
assert hashlib.md5(payload).hexdigest() == source["computed_md5"]
(out/source["name"]).write_bytes(payload)
record = {"metadata_url": url, "download_url": source["download_url"],
          "name": source["name"], "bytes": len(payload), "publisher_md5_verified": True,
          "sha256": hashlib.sha256(payload).hexdigest()}
(out/"supporting-info-provenance.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record, indent=2))
