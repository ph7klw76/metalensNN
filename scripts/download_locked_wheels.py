"""Download the Linux lock via Windows networking, with hashes and bounded RAM/disk."""
import concurrent.futures
import ctypes
import hashlib
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

root = Path(__file__).resolve().parents[1]
policy_path = root/"configs/memory_policy.json"
host_floor = json.loads(policy_path.read_text()).get("host_free_floor_bytes") if policy_path.exists() else None
target = root/".cache/locked-wheels"
target.mkdir(parents=True, exist_ok=True)
lock = (root/"locks/rtx4060-wsl2-py312-cuda12.txt").read_text()
requirements = []
current = None
for line in lock.splitlines():
    match = re.match(r"^([\w.-]+)==([^\s\\]+)", line)
    if match:
        current = {"name": match[1], "version": match[2], "hashes": []}
        requirements.append(current)
    if current:
        current["hashes"].extend(re.findall(r"--hash=sha256:([a-f0-9]{64})", line))
platforms = []
for minor in range(31, 4, -1):
    platforms.append(f"manylinux_2_{minor}_x86_64")
    if minor in {17, 12, 5}:
        platforms.append({17: "manylinux2014_x86_64", 12: "manylinux2010_x86_64", 5: "manylinux1_x86_64"}[minor])
platforms.append("linux_x86_64")
tags = list(cpython_tags((3,12), abis=["cp312"], platforms=platforms))
tags += list(compatible_tags((3,12), interpreter="cp312", platforms=platforms))
rank = {tag: i for i, tag in enumerate(tags)}


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
        ["total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended"]]


def check_memory():
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError("Cannot observe host memory")
    if host_floor is not None and memory.available < host_floor:
        raise RuntimeError("Host memory below configured reserve; download stopped")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def metadata(requirement):
    url = f'https://pypi.org/pypi/{requirement["name"]}/{requirement["version"]}/json'
    info = json.load(urllib.request.urlopen(url, timeout=30))
    choices = []
    for file in info["urls"]:
        if not file["filename"].endswith(".whl"):
            continue
        _, _, _, wheel_tags = parse_wheel_filename(file["filename"])
        matches = [rank[t] for t in wheel_tags if t in rank]
        if matches and file["digests"]["sha256"] in requirement["hashes"]:
            choices.append((min(matches), file))
    if not choices:
        raise RuntimeError(f'No compatible hash-locked wheel: {requirement["name"]}')
    return min(choices, key=lambda x:x[0])[1]


def download(file):
    check_memory()
    destination = target/file["filename"]
    checksum = file["digests"]["sha256"]
    if destination.exists() and digest(destination) == checksum:
        return {"filename": file["filename"], "status": "already_verified", "sha256": checksum}
    partial = destination.with_suffix(".whl.part")
    offset = partial.stat().st_size if partial.exists() else 0
    # Small HTTP ranges avoid the observed severe throttling after the
    # first MiB of long responses, and give bounded interruption latency.
    def fetch_range(bounds):
        begin, end = bounds
        check_memory()
        request = urllib.request.Request(file["url"], headers={"Range": f"bytes={begin}-{end}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {begin}-{end}/"):
                raise RuntimeError("Server did not honor bounded resume range")
            chunk = response.read(end-begin+2)
        if len(chunk) != end-begin+1:
            raise RuntimeError("Unexpected range response size")
        return chunk

    with partial.open("ab" if offset else "wb") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=4) as segments:
        last_report = time.monotonic()
        while offset < file["size"]:
            check_memory()
            bounds = [(begin, min(file["size"]-1, begin+1024**2-1)) for begin in
                      range(offset, min(file["size"], offset+4*1024**2), 1024**2)]
            # At most four 1 MiB chunks per file are buffered. Four files
            # are active, so payload buffers total at most 16 MiB.
            for chunk in segments.map(fetch_range, bounds):
                stream.write(chunk)
                offset += len(chunk)
            stream.flush()
            if time.monotonic()-last_report > 30:
                print(json.dumps({"filename":file["filename"], "downloaded_bytes":offset, "total_bytes":file["size"]}), flush=True)
                last_report = time.monotonic()
    if partial.stat().st_size != file["size"] or digest(partial) != checksum:
        raise RuntimeError(f'Wheel size/hash mismatch: {file["filename"]}')
    partial.replace(destination)
    result = {"filename": file["filename"], "status": "verified", "sha256": checksum, "bytes": file["size"]}
    print(json.dumps(result), flush=True)
    return result


start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    files = list(pool.map(metadata, requirements))
    total = sum(file["size"] for file in files)
    if total > 4*1024**3:
        raise RuntimeError("Locked wheel cache would exceed 4 GiB")
    print(json.dumps({"selected_wheels":len(files), "total_bytes":total, "file_workers":4, "range_workers_per_file":4, "payload_buffer_cap_bytes":16*1024**2, "cache_cap_bytes":4*1024**3}), flush=True)
    records = list(pool.map(download, files))
(root/"artifacts/phase1/wheel-downloads.json").write_text(json.dumps({"wall_s": time.monotonic()-start, "wheels":records}, indent=2)+"\n")
