"""Monitor a single owned subprocess tree; never terminate other jobs."""
import json
from contextlib import contextmanager
import os
import signal
import subprocess
import time
from pathlib import Path

from .audit import gpu_info, windows_memory


def stop_owned(process):
    if process.poll() is not None:
        return
    for sig, timeout in [(signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 3)]:
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            process.wait()
            return
        except subprocess.TimeoutExpired:
            continue


@contextmanager
def owned_child(args):
    process = subprocess.Popen(args, start_new_session=True)
    try:
        yield process
    finally:
        stop_owned(process)


def rss_tree(root_pid, proc_root=Path("/proc")):
    """Sum resident memory for an owned Linux process tree using procfs."""
    processes = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm can include spaces and parentheses: split after final ')'.
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            parent = int(fields[1])
            rss = 0
            for line in (path / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])*1024
                    break
            processes[int(path.name)] = (parent, rss)
        except (OSError, ValueError, IndexError):
            continue
    owned = {root_pid}
    while True:
        expanded = owned | {pid for pid, (ppid, _) in processes.items() if ppid in owned}
        if expanded == owned:
            break
        owned = expanded
    return sum(processes.get(pid, (0, 0))[1] for pid in owned)


def violations(sample, budgets):
    errors = []
    if budgets.get("project_ram_bytes") is not None and sample["project_rss_bytes"] > budgets["project_ram_bytes"]:
        errors.append("project RAM budget exceeded")
    for key, floor in [("host_available_bytes", "host_free_floor_bytes"),
                       ("wsl_available_bytes", "wsl_free_floor_bytes"),
                       ("gpu_free_bytes", "gpu_free_floor_bytes")]:
        if budgets[floor] is None:
            continue  # explicitly configured telemetry-only observation
        if sample.get(key) is None:
            errors.append(f"required memory observation unavailable: {key}")
        elif sample[key] < budgets[floor]:
            errors.append(f"memory pressure: {key}")
    # Total-device delta is conservative when per-PID data is unavailable.
    if sample.get("gpu_delta_bytes", 0) > budgets["project_gpu_bytes"]:
        errors.append("GPU allocation delta budget exceeded")
    return errors


def run_guarded(args, hardware, log, interval=1.0, timeout=600):
    from .audit import effective_hardware, linux_memory, preflight
    # Later phases can consume saved hardware snapshots. A historical
    # Windows floor in a snapshot must not restore the removed gate.
    config = effective_hardware(json.loads(Path(hardware).read_text()))
    budgets = config["budgets"]
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    errors = preflight(config)
    if errors:
        summary = {"command": args, "returncode": None, "started": False,
                   "stop_reasons": errors}
        Path(log+".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2))
        return 2
    first_gpu = gpu_info()["devices"]
    baseline_used = first_gpu[0]["used_bytes"] if first_gpu else 0
    start = time.monotonic()
    # Child alone owns this process group. Parent stays outside it.
    peak_rss = 0
    peak_gpu_delta = 0
    stop_reasons = []
    host = None
    next_host = 0
    with owned_child(args) as process, Path(log).open("w") as stream:
        while True:
            # Include the supervisor and its entire owned child tree.
            rss = rss_tree(os.getpid())
            now = time.monotonic()
            if now >= next_host:
                host = windows_memory()
                next_host = now + 3
            devices = gpu_info()["devices"]
            gpu = devices[0] if devices else {}
            sample = {"elapsed_s": now-start, "project_rss_bytes": rss,
                      "host_available_bytes": host.get("available_bytes") if host else None,
                      "wsl_available_bytes": linux_memory()["available_bytes"],
                      "gpu_free_bytes": gpu.get("free_bytes"),
                      "gpu_total_used_bytes": gpu.get("used_bytes"),
                      "gpu_delta_bytes": max(0, gpu.get("used_bytes", baseline_used)-baseline_used)}
            peak_rss = max(peak_rss, rss)
            peak_gpu_delta = max(peak_gpu_delta, sample["gpu_delta_bytes"])
            stop_reasons = violations(sample, budgets)
            if now-start > timeout:
                stop_reasons.append("job timeout")
            sample["violations"] = stop_reasons
            stream.write(json.dumps(sample)+"\n")
            stream.flush()
            if process.poll() is not None:
                break
            if stop_reasons:
                stop_owned(process)
                break
            time.sleep(interval)
    summary = {"command": args, "returncode": process.returncode,
               "budgets": budgets,
               "wall_s": time.monotonic()-start, "peak_project_rss_bytes": peak_rss,
               "peak_total_device_delta_bytes": peak_gpu_delta,
               "stop_reasons": stop_reasons, "sample_interval_s": interval,
               "note": "Sampled RSS sums may double-count shared pages. GPU total delta includes other applications; transient allocations between samples may be missed."}
    Path(log+".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    return 2 if stop_reasons else process.returncode
