"""Approved bounded CPU validation of the frozen unequal-index height family.

This runner does not optimize, retrain, change system settings or enforce RAM
gates. Every attempted Meep subprocess consumes a solve slot, including errors.
The dated approval receipt amends the historical allocation without editing it.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "artifacts/1550nm_height_family_n15263_v2"
FOLDER = CAMPAIGN / "maxwell_validation_v1"
WORKER = ROOT / "scripts/height_family_maxwell_worker_v2.py"
ANALYZER = ROOT / "scripts/height_family_maxwell_analysis_v2.py"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def duration(H):
    return float(max(160, math.ceil(2 * 1.5263 * (H + 3) + 20)))


def make_plan():
    plan = []
    def add(method, H, resolution, kind="nominal", pml=2., padding=10., time_factor=1.):
        plan.append(dict(method=method, H_um=float(H), resolution=float(resolution),
                         job_type=kind, pml_um=float(pml), padding_um=float(padding),
                         duration=duration(H) * time_factor))
    # Coarse measurements first establish the actual source/device controls.
    # Each reference is reused only for exactly matching full settings.
    for resolution in (8, 16):
        for H in (40, 70, 100, 50, 60, 80, 90):
            add("reference", H, resolution)
            add("NN", H, resolution)
            if H in (40, 70, 100):
                add("baseline", H, resolution)
            if H == 70:
                add("flat_cylinder", H, resolution)
    for H in (40, 70, 100, 50, 60, 80, 90):
        add("reference", H, 24)
        add("NN", H, 24)
    for H in (40, 100):
        add("reference", H, 16, "pml", pml=3.)
        add("NN", H, 16, "pml", pml=3.)
    add("reference", 70, 16, "padding", padding=15.)
    add("NN", 70, 16, "padding", padding=15.)
    add("reference", 100, 16, "duration", time_factor=1.5)
    add("NN", 100, 16, "duration", time_factor=1.5)
    assert len(plan) == 58
    return plan


def cache_identity(job):
    return tuple(float(job[key]) for key in ("H_um", "resolution", "pml_um", "padding_um", "duration"))


def check_capacity(attempted, elapsed, *, max_solves=60, max_wall=14400, child_timeout=1800):
    if attempted >= max_solves:
        raise RuntimeError("Approved total solve limit reached")
    remaining = max_wall - elapsed
    if remaining <= 1:
        raise RuntimeError("Approved total wall-time limit reached")
    return min(float(child_timeout), remaining)


def prepare():
    if FOLDER.exists():
        raise FileExistsError("Preserve the existing validation root; use --run/--resume.")
    proposal_path = CAMPAIGN / "maxwell_validation_proposal.json"
    proposal = read(proposal_path)
    assert proposal["new_unique_device_ceiling"] == 11
    assert proposal["proposed_cumulative_unique_ceiling"] == 24
    assert proposal["maximum_total_CPU_solves_including_references_and_retries"] == 60
    assert proposal["maximum_total_wall_time_s"] == 14400
    registry = ROOT / "artifacts/natural_inverse_design_candidate_registry_v1.json"
    if sha(registry) != proposal["historical_registry_sha256"]:
        raise ValueError("Historical registry differs from the approved proposal")
    train = read(CAMPAIGN / "train_source_manifest.json")
    if sha(CAMPAIGN / "config.json") != train["config_sha256"]:
        raise ValueError("Trained configuration changed")
    for name, digest in train["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Trained physics source changed: {name}")
    FOLDER.mkdir()
    for name in ("config.json", "maxwell_validation_proposal.json"):
        shutil.copyfile(CAMPAIGN / name, FOLDER / name)
    receipt = dict(
        status="APPROVED_BY_USER", recorded_utc=utc(),
        user_response="Approve local validation within these limits",
        approved_question="Approve the prepared independent validation stage: up to 11 additional device cases (cumulative ceiling 24), at most 60 local CPU Meep solves and four hours, with one worker and no RAM gates?",
        proposal_sha256=sha(proposal_path), historical_registry_sha256=sha(registry),
        limits=dict(new_unique_devices=11, cumulative_unique_ceiling=24,
                    total_CPU_solves=60, total_wall_s=14400, workers=1, RAM_gates=False),
        interpretation="Authorizes frozen-profile local validation only; no new optical optimization, printing, external jobs or system changes.",
        original_config_unchanged=True,
    )
    write(FOLDER / "approval_receipt.json", receipt)
    cases = []
    for case in proposal["device_cases"]:
        method, H = case["method"], case["H_um"]
        out = FOLDER / "frozen_designs" / f"{method}_H{H:g}.json"
        if method == "flat_cylinder":
            cfg = read(CAMPAIGN / "config.json")
            design = dict(kind="flat_cylinder", H_um=float(H), units="um",
                          cylinder_diameter_um=cfg["cylinder_diameter_um"],
                          heights_um=[0.] * cfg["annuli"],
                          target_diameter_um=read(CAMPAIGN / f"designs/H{H:g}/NN.json")["target_diameter_um"])
            write(out, design)
        else:
            out.parent.mkdir(exist_ok=True)
            shutil.copyfile(CAMPAIGN / f"designs/H{H:g}/{method}.json", out)
            design = read(out)
        identity = dict(H_um=float(H), cylinder_diameter_um=design["cylinder_diameter_um"],
                        heights_um=design["heights_um"], materials=read(CAMPAIGN / "config.json")["materials"])
        cases.append({**case, "design_file":str(out.relative_to(FOLDER)),
                      "design_sha256":sha(out),
                      "geometry_sha256":hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()})
    assert len(cases) == len({item["geometry_sha256"] for item in cases}) == 11
    write(FOLDER / "cumulative_amendment.json", dict(
        status="AUTHORIZED_ALLOCATION_RESERVED", historical_used=13, additional_reserved=11,
        cumulative_reserved=24, approved_ceiling=24, cases=cases,
        historical_registry_unchanged_sha256=sha(registry),
        note="The new allocation is cumulative; the historical 13/13 registry is preserved."))
    sources = [Path(__file__), WORKER, ANALYZER]
    sources += [ROOT / "src/fiber_metalens_nn" / name for name in
                ("geometry.py", "collimation_optics.py", "collimation_metrics.py",
                 "collimation_dft_power.py", "metrics.py")]
    manifest = {str(path.relative_to(ROOT)):sha(path) for path in sources}
    for name in manifest:
        out = FOLDER / "source_snapshots" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, out)
    write(FOLDER / "source_manifest.json", manifest)
    write(FOLDER / "plan.json", dict(jobs=make_plan(), planned_solves=58, reserve_solves=2,
                                    no_RAM_gates=True, one_CPU_worker=True))
    write(FOLDER / "ledger.json", dict(status="PREPARED_APPROVED", entries=[],
                                      total_attempted_solves=0, elapsed_s=0, approval_sha256=sha(FOLDER/"approval_receipt.json")))
    print(json.dumps(dict(status="PREPARED_APPROVED", folder=str(FOLDER), devices=len(cases), solves=58)), flush=True)


class Validation:
    def __init__(self):
        self.ledger = read(FOLDER / "ledger.json")
        self.receipt = read(FOLDER / "approval_receipt.json")
        if self.receipt["status"] != "APPROVED_BY_USER":
            raise ValueError("Explicit approval receipt required")
        if sha(FOLDER / "maxwell_validation_proposal.json") != self.receipt["proposal_sha256"]:
            raise ValueError("Approved proposal changed")
        self.started = time.monotonic()
        self.prior_elapsed = float(self.ledger.get("elapsed_s", 0))
        self.env = os.environ.copy()
        self.env.update(PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.cases = read(FOLDER / "cumulative_amendment.json")["cases"]
        self.active = None
        self.verify()

    def elapsed(self):
        return self.prior_elapsed + time.monotonic() - self.started

    def save(self):
        self.ledger["elapsed_s"] = self.elapsed()
        self.ledger["total_attempted_solves"] = len(self.ledger["entries"])
        self.ledger["updated_utc"] = utc()
        write(FOLDER / "ledger.json", self.ledger)

    def verify(self):
        for name, digest in read(FOLDER / "source_manifest.json").items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Frozen validation source changed: {name}")
        if sha(FOLDER / "config.json") != sha(CAMPAIGN / "config.json"):
            raise ValueError("Frozen training configuration changed")
        protected = read(CAMPAIGN / "provenance.json")["protected_before"]
        for name, digest in protected.items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Protected historical evidence changed: {name}")
        for case in self.cases:
            if sha(FOLDER / case["design_file"]) != case["design_sha256"]:
                raise ValueError("Frozen design changed")

    def terminate(self):
        if self.active and self.active.poll() is None:
            os.killpg(self.active.pid, signal.SIGTERM)
            try:
                self.active.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.active.pid, signal.SIGKILL)
                self.active.wait()

    def analyze_entry(self, entry):
        """Recover saved-field analysis without launching or charging another solve."""
        if entry['method'] == 'reference':
            return
        if entry.get('analysis_status') == 'completed':
            path = FOLDER / entry['analysis_file']
            if not path.is_file():
                raise ValueError('Completed analysis evidence disappeared')
            if sha(path) != entry['analysis_sha256']:
                raise ValueError('Completed analysis evidence changed')
            return
        self.verify()
        remaining = 14400 - self.elapsed()
        if remaining <= 1:
            self.save()
            raise RuntimeError('Wall budget exhausted before analysis')
        case = next(c for c in self.cases if c['method'] == entry['method']
                    and c['H_um'] == entry['H_um'])
        output = FOLDER / entry['record_file']
        label = output.stem
        (FOLDER / 'analyses').mkdir(exist_ok=True)
        (FOLDER / 'logs').mkdir(exist_ok=True)
        number = len(entry.get('analysis_attempts', [])) + 1
        while True:
            stem = f'{label}.analysis-attempt-{number:02d}'
            analysis = FOLDER / 'analyses' / (stem + '.json')
            log = FOLDER / 'logs' / (stem + '.log')
            if not analysis.exists() and not log.exists():
                break
            number += 1
        target = read(FOLDER / case['design_file'])['target_diameter_um']
        command = [str(ROOT/'.venv-meep/bin/python'), '-B', str(ANALYZER), '--record', str(output),
                   '--config', str(FOLDER/'config.json'), '--target-diameter', str(target), '--output', str(analysis)]
        attempt = dict(number=number, status='started', started_utc=utc(), command=command,
                       log_file=str(log.relative_to(FOLDER)), analysis_file=str(analysis.relative_to(FOLDER)))
        entry.setdefault('analysis_attempts', []).append(attempt)
        entry['analysis_status'] = 'started'
        self.save()
        try:
            with log.open('x') as stream:
                proc = subprocess.run(command, cwd=ROOT, env=self.env, stdout=stream,
                                      stderr=subprocess.STDOUT, text=True, timeout=min(120, remaining))
            attempt['exit_code'] = proc.returncode
            entry['analysis_exit_code'] = proc.returncode
            if proc.returncode != 0 or not analysis.is_file():
                attempt['status'] = entry['analysis_status'] = 'failed'
                raise RuntimeError('Field analysis failed; diagnose saved raw records before any new solve')
            value = read(analysis)
            entry.update(analysis_file=str(analysis.relative_to(FOLDER)), analysis_sha256=sha(analysis),
                         nominal_optical_pass=value['synthetic_optical_pass'],
                         reconstruction_pass=value['reconstruction_pass'],
                         C=value['normalized_curvature'], eta=value['conservative_useful_efficiency'])
            attempt['status'] = entry['analysis_status'] = 'completed'
            attempt['analysis_sha256'] = entry['analysis_sha256']
        except subprocess.TimeoutExpired:
            attempt['status'] = entry['analysis_status'] = 'timeout'
            raise
        except BaseException:
            if attempt['status'] == 'started':
                attempt['status'] = entry['analysis_status'] = 'failed'
            raise
        finally:
            attempt['finished_utc'] = utc()
            self.save()

    def execute(self, job):
        existing = [e for e in self.ledger["entries"] if e["method"] == job["method"]
                    and cache_identity(e) == cache_identity(job) and e["status"] == "completed"]
        if existing:
            entry = existing[-1]
            if not (FOLDER / entry["record_file"]).is_file():
                raise ValueError("Cached solver record disappeared")
            if sha(FOLDER / entry["record_file"]) != entry["record_sha256"]:
                raise ValueError("Cached solver record changed")
            record=read(FOLDER / entry["record_file"])
            for key in ("fields_file","power_audit_fields_file"):
                if not Path(record[key]).is_file():
                    raise ValueError("Cached field evidence disappeared")
            self.analyze_entry(entry)
            return entry
        # Failed settings are never retried silently. A source correction or
        # identical-geometry duration extension needs a separately recorded attempt.
        if any(e["method"] == job["method"] and cache_identity(e) == cache_identity(job)
               for e in self.ledger["entries"]):
            raise RuntimeError("Prior attempted settings failed; diagnose before another solve")
        timeout = check_capacity(len(self.ledger["entries"]), self.elapsed())
        self.verify()
        reference = None
        if job["method"] != "reference":
            matches = [e for e in self.ledger["entries"] if e["method"] == "reference"
                       and cache_identity(e) == cache_identity(job) and e["status"] == "completed"]
            if not matches:
                raise ValueError("Exactly matching completed homogeneous reference is required")
            reference = matches[-1]
            if read(FOLDER / reference["record_file"])["power_balance_error"] >= .05:
                raise RuntimeError("Reference power closure failed; preserve and diagnose")
        idx = len(self.ledger["entries"]) + 1
        label = f"{idx:02d}_{job['method']}_H{job['H_um']:g}_r{job['resolution']:g}_{job['job_type']}"
        output = FOLDER / "records" / (label + ".json")
        output.parent.mkdir(exist_ok=True)
        log = FOLDER / "logs" / (label + ".log")
        log.parent.mkdir(exist_ok=True)
        args = [str(ROOT/".venv-meep/bin/python"), "-B", str(WORKER), "--config", str(FOLDER/"config.json"),
                "--output", str(output), "--H", str(job["H_um"]), "--resolution", str(job["resolution"]),
                "--pml", str(job["pml_um"]), "--padding", str(job["padding_um"]), "--duration", str(job["duration"])]
        entry = dict(job, id=idx, status="started", started_utc=utc(),
                     record_file=str(output.relative_to(FOLDER)), log_file=str(log.relative_to(FOLDER)))
        if reference:
            case = next(c for c in self.cases if c["method"] == job["method"] and c["H_um"] == job["H_um"])
            entry["reference_id"] = reference["id"]
            entry["design_sha256"] = case["design_sha256"]
            args += ["--design", str(FOLDER / case["design_file"]), "--reference", str(FOLDER/reference["record_file"])]
        entry["command"] = args
        self.ledger["entries"].append(entry)
        self.save()
        print(json.dumps(dict(event="SOLVE_STARTED", **{k:entry[k] for k in ("id","method","H_um","resolution","job_type")}, timeout_s=timeout)), flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            self.active = subprocess.Popen(args, cwd=ROOT, env=self.env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            entry["owned_pid"] = self.active.pid
            self.save()
            try:
                code = self.active.wait(timeout=timeout)
                entry["solver_exit_code"] = code
                entry["status"] = "completed" if code == 0 and output.exists() else "failed"
            except subprocess.TimeoutExpired:
                self.terminate()
                entry["status"] = "timeout"
                entry["solver_exit_code"] = self.active.returncode
            finally:
                entry["wrapper_wall_s"] = time.monotonic() - start
                entry["finished_utc"] = utc()
                self.save()
        if entry["status"] != "completed":
            raise RuntimeError(f"Solve {idx} {entry['status']}; inspect preserved log")
        record = read(output)
        entry["power_balance_error"] = record["power_balance_error"]
        entry["record_sha256"] = sha(output)
        self.save()
        self.analyze_entry(entry)
        print(json.dumps(dict(event="SOLVE_COMPLETED", id=idx, method=job["method"], H_um=job["H_um"],
                              resolution=job["resolution"], wall_s=entry["wrapper_wall_s"],
                              closure=record["power_balance_error"], C=entry.get("C"),eta=entry.get("eta"))), flush=True)
        return entry

    def run(self):
        self.ledger["status"] = "RUNNING_APPROVED"
        self.ledger.setdefault("first_started_utc", utc())
        self.save()
        try:
            for job in read(FOLDER / "plan.json")["jobs"]:
                self.execute(job)
            # Reserve is a single same-geometry/reference pair, chosen only
            # from an actual device closure failure. No candidate correction.
            failed = [e for e in self.ledger["entries"] if e["method"] == "NN"
                      and e["status"] == "completed" and e.get("power_balance_error",1) >= .05]
            if failed:
                worst = max(failed, key=lambda e:e["power_balance_error"])
                base = {k:worst[k] for k in ("H_um","resolution","pml_um","padding_um","duration")}
                base.update(job_type="reserve_duration",duration=base["duration"]*1.5)
                self.execute(dict(base,method="reference"))
                self.execute(dict(base,method="NN"))
            else:
                unresolved = []
                for H in (40,50,60,70,80,90,100):
                    pair = [next(e for e in self.ledger["entries"] if e["method"] == "NN"
                                 and e["H_um"] == H and e["resolution"] == resolution
                                 and e["job_type"] == "nominal") for resolution in (16,24)]
                    delta = abs(pair[1]["C"] - pair[0]["C"])
                    if delta > .005 or delta >= .1-abs(pair[1]["C"]):
                        unresolved.append((delta, pair[1]))
                if unresolved:
                    worst = max(unresolved,key=lambda item:item[0])[1]
                    base = {k:worst[k] for k in ("H_um","pml_um","padding_um","duration")}
                    base.update(job_type="reserve_mesh",resolution=32.)
                    self.execute(dict(base,method="reference"))
                    self.execute(dict(base,method="NN"))
            self.verify()
            self.ledger["status"] = "PLANNED_VALIDATION_EXECUTED_EVALUATE_ACCEPTANCE"
        except BaseException as exc:
            self.terminate()
            self.ledger["status"] = "STOPPED_REQUIRES_DIAGNOSIS"
            self.ledger.setdefault("failures",[]).append(dict(utc=utc(),type=type(exc).__name__,reason=str(exc)))
            self.save()
            raise
        finally:
            self.save()
        print(json.dumps(dict(status=self.ledger["status"], elapsed_s=self.elapsed(),
                              attempted_solves=len(self.ledger["entries"]))), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        import fcntl
        with (FOLDER / "runner.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            Validation().run()
    if not(args.prepare or args.run):
        parser.error("Choose --prepare or --run; this command launches no work implicitly")


if __name__ == "__main__":
    main()
