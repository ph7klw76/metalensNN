"""Read-only host/WSL/GPU audit. No driver or WSL configuration changes."""
import csv
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

GIB = 1024 ** 3


def memory_policy():
    path = Path(__file__).resolve().parents[2]/"configs/memory_policy.json"
    # The user's desktop-wide override must not revert to the old fixed
    # free-RAM gate when a config is absent in a later phase.
    default = {"host_free_floor_bytes": None, "host_memory_policy": "telemetry_only",
               "applies_to_phases": list(range(1, 8))}
    return {**default, **json.loads(path.read_text())} if path.exists() else default


def effective_hardware(data):
    """Use today's policy with a saved snapshot, preserving its other limits."""
    policy = memory_policy()
    budgets = {**data["budgets"], "host_free_floor_bytes": policy["host_free_floor_bytes"]}
    if data.get("phase") in (8, 9, 10):
        budgets.update(project_ram_bytes=None, host_free_floor_bytes=None, wsl_free_floor_bytes=None)
        policy = {"applies_to_phases": [8, 9, 10], "ram_policy": "telemetry_only",
                  "host_free_floor_bytes": None, "project_ram_bytes": None,
                  "wsl_free_floor_bytes": None, "source": "explicit user override"}
    return {**data, "budgets": budgets, "memory_policy": policy}


def command(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "error": str(exc)}


def windows_memory():
    exe = shutil.which("powershell.exe")
    if not exe:
        return None
    script = "$m=Get-CimInstance Win32_OperatingSystem; @{total_bytes=[long]$m.TotalVisibleMemorySize*1024; available_bytes=[long]$m.FreePhysicalMemory*1024; caption=$m.Caption; version=$m.Version} | ConvertTo-Json -Compress"
    result = command([exe, "-NoProfile", "-NonInteractive", "-Command", script])
    if result.get("returncode") == 0:
        try:
            return json.loads(result["stdout"])
        except ValueError:
            pass
    return {"error": result}


def linux_memory():
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, val = line.split(":", 1)
        mem[key] = int(val.strip().split()[0]) * 1024
    limits = {}
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        if Path(path).exists():
            raw = Path(path).read_text().strip()
            limits[path] = int(raw) if raw != "max" else None
    return {"total_bytes": mem["MemTotal"], "available_bytes": mem["MemAvailable"],
            "swap_total_bytes": mem["SwapTotal"], "swap_free_bytes": mem["SwapFree"],
            "cgroup_limits_bytes": limits}


def gpu_info():
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    query = command([exe, "--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap,driver_version", "--format=csv,nounits,noheader"])
    devices = []
    if query.get("returncode") == 0:
        for row in csv.reader(query["stdout"].splitlines(), skipinitialspace=True):
            devices.append({"index": int(row[0]), "name": row[1],
                            "total_bytes": int(row[2])*1024**2,
                            "used_bytes": int(row[3])*1024**2,
                            "free_bytes": int(row[4])*1024**2,
                            "compute_capability": row[5], "driver": row[6]})
    return {"devices": devices, "query": query,
            "driver_compatibility_report": command([exe]),
            "process_query": command([exe, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])}


def audit(output="hardware.json", phase=None):
    linux = linux_memory()
    windows = windows_memory()
    gpu = gpu_info()
    policy = memory_policy()
    host_floor = policy["host_free_floor_bytes"]
    # A null host floor records Windows available RAM without using a
    # fixed free-memory stop. RSS budgets still account for initial headroom.
    host_available = windows.get("available_bytes") if windows else None
    ram_budget = min(10*GIB, max(0, linux["available_bytes"]-GIB))
    if host_available is not None:
        ram_budget = min(ram_budget, max(0, host_available-(host_floor or 0)))
    limits = [x for x in linux["cgroup_limits_bytes"].values() if x and x < 2**60]
    if limits:
        ram_budget = min(ram_budget, max(0, min(limits)-GIB))
    device = gpu["devices"][0] if gpu["devices"] else None
    gpu_budget = min(6*GIB, max(0, device["free_bytes"]-768*1024**2)) if device else 0
    toolkit = command([shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc", "--version"])
    driver_cuda = re.search(r"CUDA (?:UMD Version|Version):\s*([\d.]+)", gpu["driver_compatibility_report"].get("stdout", ""))
    toolkit_cuda = re.search(r"release ([\d.]+)", toolkit.get("stdout", ""))
    data = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
            "windows": windows, "linux": linux, "gpu": gpu,
            "driver_reported_cuda_compatibility": driver_cuda.group(1) if driver_cuda else None,
            "installed_cuda_toolkit_version": toolkit_cuda.group(1) if toolkit_cuda else None,
            "installed_toolkit": toolkit,
            "memory_policy": policy,
            "budgets": {"project_ram_bytes": int(ram_budget), "project_gpu_bytes": int(gpu_budget),
                        "host_free_floor_bytes": host_floor, "wsl_free_floor_bytes": int(0.75*GIB),
                        "gpu_free_floor_bytes": 512*1024**2,
                        "workers": 1, "batch_size": 1, "field_dtype": "complex64",
                        "real_dtype": "float32", "jax_preallocate": False,
                        "jax_persistent_cache_bytes": 0,
                        "cuda_driver_cache_max_bytes": 256*1024**2,
                        "note": "Monitored budgets, not guaranteed allocator caps. GPU attribution may be unavailable under WDDM."}}
    data["phase"] = phase
    data = effective_hardware(data)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps({"hardware": output, "budgets": data["budgets"], "host": windows,
                      "gpu": gpu["devices"]}, indent=2))
    return data


def preflight(data):
    errors = []
    host_floor = data["budgets"].get("host_free_floor_bytes")
    if host_floor is not None and not (data.get("windows") or {}).get("available_bytes"):
        errors.append("Windows host-memory observation unavailable")
    if data["budgets"].get("project_ram_bytes") is not None and data["budgets"]["project_ram_bytes"] < 512*1024**2:
        errors.append("Less than 512 MiB project RAM within the monitored budget")
    if not data["gpu"]["devices"]:
        errors.append("No NVIDIA GPU observed in WSL")
    if data["budgets"]["project_gpu_bytes"] < 512*1024**2:
        errors.append("Insufficient free GPU memory")
    return errors
