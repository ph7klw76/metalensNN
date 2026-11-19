"""Bounded 1 MiB diagnostic for the unusually slow CUDA wheel transfer."""
import json
import platform
import time
import urllib.request

meta = json.load(urllib.request.urlopen("https://pypi.org/pypi/nvidia-cudnn-cu12/9.27.0.42/json", timeout=15))
wheel = next(x for x in meta["urls"] if "x86_64" in x["filename"] and "manylinux" in x["filename"])
t = time.monotonic()
request = urllib.request.Request(wheel["url"], headers={"Range": "bytes=0-1048575"})
with urllib.request.urlopen(request, timeout=20) as response:
    payload = response.read(1048576)
    status = response.status
print(json.dumps({"platform": platform.system(), "filename": wheel["filename"],
                  "bytes": len(payload), "elapsed_s": time.monotonic()-t,
                  "http_status": status}))
