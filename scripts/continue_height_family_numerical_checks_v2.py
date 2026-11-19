"""Explicit, bounded continuation of predeclared numerical boundary checks.

Preparing or importing this helper launches nothing. Execution requires --run,
the original stopped runner's exclusive lock, and remaining ORIGINAL approval
time. It cannot retry failed settings, add devices, change physics, or report
optical acceptance. Frozen runner/worker/analyzer sources remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT/'scripts/run_height_family_maxwell_v2.py'
spec = importlib.util.spec_from_file_location('frozen_height_family_runner_for_supervision', RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
TOTAL_WALL_S = 14400.
REPORT_RESERVE_S = 180.
PROCESS_CLEANUP_S = 5.
MAX_SOLVES = 60
FIT_MARGIN = 1.25
ADAPTIVE_NAME = 'reserve_mesh_H40_r32'
ADAPTIVE_MARGIN = 1.15
STOPPED_STATUSES = {'STOPPED_REQUIRES_DIAGNOSIS', 'SUPERVISION_FINISHED_UNRESOLVED',
                    'SUPERVISION_STOPPED_UNRESOLVED'}


def parse_utc(value):
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Timezone-aware original start timestamp required')
    return value.astimezone(timezone.utc)


def absolute_elapsed(ledger, now=None):
    """Downtime never resets the original approval clock."""
    now = datetime.now(timezone.utc) if now is None else parse_utc(now)
    elapsed = (now-parse_utc(ledger['first_started_utc'])).total_seconds()
    recorded = float(ledger.get('elapsed_s', 0))
    if elapsed < 0 or not math.isfinite(recorded) or recorded < 0:
        raise ValueError('Invalid original start or recorded elapsed time')
    return max(elapsed, recorded)


def select_pairs(plan, names):
    """Return exact original jobs, in the explicitly requested pair order."""
    if not names or len(names) != len(set(names)):
        raise ValueError('Explicit unique numerical pair names required')
    groups = {}
    for job in plan:
        if job['job_type'] in ('pml', 'padding', 'duration'):
            name = f"{job['job_type']}_H{job['H_um']:g}"
            groups.setdefault(name, []).append(dict(job))
    result = []
    for name in names:
        if name == ADAPTIVE_NAME:
            base = [dict(job) for job in plan if job['method'] in ('reference', 'NN')
                    and job['job_type']=='nominal' and job['H_um']==40 and job['resolution']==24]
            if (len(base)!=2 or [job['method'] for job in base]!=['reference','NN']
                    or any(job['pml_um']!=2 or job['padding_um']!=10 or job['duration']!=160 for job in base)):
                raise ValueError('Frozen nominal H40 pair differs from the approved adaptive reserve geometry/settings')
            jobs=[dict(job,resolution=32.,job_type='reserve_mesh') for job in base]
            result.append(dict(name=name,jobs=jobs,selection_rule='approved_original_adaptive_mesh_reserve',
                               in_nominal_58_plan=False))
            continue
        jobs = groups.get(name)
        if (not jobs or len(jobs) != 2 or [job['method'] for job in jobs] != ['reference', 'NN']
                or runner.cache_identity(jobs[0]) != runner.cache_identity(jobs[1])):
            raise ValueError('Not an original predeclared reference/NN numerical pair: '+name)
        result.append(dict(name=name, jobs=jobs))
    return result


def matching_entries(ledger, job):
    return [entry for entry in ledger['entries'] if entry['method'] == job['method']
            and runner.cache_identity(entry) == runner.cache_identity(job)]


def require_unfailed_selected_settings(ledger, pairs):
    for pair in pairs:
        for job in pair['jobs']:
            if any(entry['status'] != 'completed' for entry in matching_entries(ledger, job)):
                raise ValueError('Selected settings were already attempted unsuccessfully; retries are forbidden')


def estimated_pair_cost(pair, ledger):
    """Measured costs: nominal16/domain+25%; adaptive H40 nominal24+15%.

    Cell volume in this axisymmetric problem scales with radial*axial extent;
    the temporal/raster factor scales as resolution^3 and after-source time.
    For the predeclared changes, these ratios conservatively overestimate
    source-pulse and setup portions when the duration increases.
    """
    evidence, total = [], 0.
    for job in pair['jobs']:
        prior = matching_entries(ledger, job)
        if prior:
            if any(entry['status'] != 'completed' for entry in prior):
                raise ValueError('A failed setting cannot be reused')
            if job['method'] == 'reference' or prior[-1].get('analysis_status') == 'completed':
                evidence.append(dict(method=job['method'], already_completed=True, estimated_s=0.))
                continue
            total += 120.  # Saved-field recovery consumes time, never a solve.
            evidence.append(dict(method=job['method'], analysis_only=True, estimated_s=120.))
            continue
        adaptive=pair['name']==ADAPTIVE_NAME
        base_resolution=24 if adaptive else 16
        margin=ADAPTIVE_MARGIN if adaptive else FIT_MARGIN
        base = [entry for entry in ledger['entries']
                if entry['method'] == job['method'] and entry['H_um'] == job['H_um']
                and entry['resolution'] == base_resolution and entry['job_type'] == 'nominal'
                and entry['status'] == 'completed']
        if len(base) != 1:
            raise ValueError(f'Exactly one completed matching nominal{base_resolution} timing required')
        base = base[0]
        measured = float(base['wrapper_wall_s'])
        if not math.isfinite(measured) or measured <= 0:
            raise ValueError('Positive finite measured nominal timing required')
        record_path = runner.FOLDER/base['record_file']
        if runner.sha(record_path) != base['record_sha256']:
            raise ValueError('Nominal timing record hash changed')
        settings = runner.read(record_path)['settings']
        nominal_r = float(settings['cell_radial_extent_um'])
        nominal_z = float(settings['cell_axial_extent_um'])
        radial = nominal_r+job['padding_um']-base['padding_um']+job['pml_um']-base['pml_um']
        axial = nominal_z+job['pml_um']-base['pml_um']
        scale = (radial/nominal_r)*(axial/nominal_z)*(job['resolution']/base['resolution'])**3
        scale *= job['duration']/base['duration']
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Positive finite numerical-domain scaling required')
        # An entire120s analysis allowance is conservative relative to the
        # measured few-second diagnostics; the original reporting reserve is
        # additional and remains unavailable to these children.
        analysis_allowance = 120. if job['method'] != 'reference' else 0.
        estimate = margin*measured*scale+analysis_allowance
        total += estimate
        evidence.append(dict(method=job['method'], nominal_entry_id=base['id'],
                             nominal_wrapper_wall_s=measured, domain_time_scale=scale,
                             measured_resolution=base_resolution, margin=margin, analysis_allowance_s=analysis_allowance,
                             estimated_s=estimate))
    return dict(estimated_total_s=total, measurements=evidence)


def adaptive_selection_evidence(pair, ledger, cases):
    """Only the original observed-curvature reserve rule may add this pair."""
    if pair['name']!=ADAPTIVE_NAME:
        return None
    expected=select_pairs(runner.make_plan(),[ADAPTIVE_NAME])[0]
    if pair!=expected:
        raise ValueError('Adaptive reserve settings differ from the one approved H40/res32 pair')
    if any(str(entry.get('job_type','')).startswith('reserve') for entry in ledger['entries']):
        raise ValueError('Adaptive reserve requires no prior reserve solve attempts')
    case=next((case for case in cases if case['method']=='NN' and case['H_um']==40),None)
    if case is None or runner.sha(runner.FOLDER/case['design_file'])!=case['design_sha256']:
        raise ValueError('Frozen H40 design evidence changed')
    config=runner.read(runner.FOLDER/'config.json')
    config_fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    observed=[]
    for resolution in (16,24):
        matches=[entry for entry in ledger['entries'] if entry['method']=='NN' and entry['H_um']==40
                 and entry['resolution']==resolution and entry['job_type']=='nominal' and entry['status']=='completed']
        if len(matches)!=1:
            raise ValueError('Completed actual H40 NN16 and NN24 observations are required')
        entry=matches[0]
        if (entry.get('analysis_status')!='completed' or entry['design_sha256']!=case['design_sha256']
                or entry['pml_um']!=2 or entry['padding_um']!=10 or entry['duration']!=160):
            raise ValueError('Observed H40 mesh pair is not the unchanged frozen device/settings')
        record_path=runner.FOLDER/entry['record_file']
        analysis_path=runner.FOLDER/entry['analysis_file']
        if (runner.sha(record_path)!=entry['record_sha256']
                or runner.sha(analysis_path)!=entry['analysis_sha256']):
            raise ValueError('Observed H40 record/analysis hashes changed')
        record,analysis=runner.read(record_path),runner.read(analysis_path)
        if (record['config_fingerprint']!=config_fingerprint
                or analysis.get('settings')!=record.get('settings')
                or any(record['settings'].get(key)!=entry[key] for key in
                       ('H_um','resolution','pml_um','padding_um','duration'))
                or analysis.get('reconstruction_pass') is not True):
            raise ValueError('Observed H40 native reconstruction or provenance is unresolved')
        current=float(analysis['normalized_curvature'])
        if not math.isfinite(current):
            raise ValueError('Finite observed H40 phase-current curvature is required')
        observed.append(dict(entry_id=entry['id'],resolution=resolution,normalized_curvature=current,
                             record_sha256=entry['record_sha256'],analysis_sha256=entry['analysis_sha256']))
    delta=abs(observed[1]['normalized_curvature']-observed[0]['normalized_curvature'])
    cap=float(runner.read(runner.FOLDER/'maxwell_validation_proposal.json')['numerical_acceptance']['curvature_last_pair_delta_max'])
    if cap!=.005 or delta<=cap:
        raise ValueError('Approved H40 adaptive rule requires actual NN16/24 curvature delta>.005')
    return dict(rule='Original approved adaptive resolution32 reserve; not part of nominal58 solves',
                observed_H40_nonconverged=True,observations=observed,curvature_absolute_delta=delta,
                unchanged_curvature_delta_limit=cap,original_reserve_attempts_available=2,
                frozen_H40_design_sha256=case['design_sha256'],configuration_sha256=runner.sha(runner.FOLDER/'config.json'))


def require_adaptive_attempt_ceiling(ledger):
    attempts=[entry for entry in ledger['entries'] if str(entry.get('job_type','')).startswith('reserve')]
    expected=select_pairs(runner.make_plan(),[ADAPTIVE_NAME])[0]['jobs']
    if len(attempts)>2:
        raise ValueError('The approved adaptive reserve contains at most two attempted solves')
    for entry in attempts:
        if (entry.get('job_type')!='reserve_mesh' or not any(
                entry['method']==job['method'] and runner.cache_identity(entry)==runner.cache_identity(job)
                for job in expected)):
            raise ValueError('Attempt lies outside the one approved H40 adaptive reserve pair')


def pair_fits(pair, ledger, elapsed, estimated_s):
    new_solves = sum(not matching_entries(ledger, job) for job in pair['jobs'])
    remaining = TOTAL_WALL_S-REPORT_RESERVE_S-PROCESS_CLEANUP_S-float(elapsed)
    return (len(ledger['entries'])+new_solves <= MAX_SOLVES
            and math.isfinite(estimated_s) and estimated_s >= 0 and estimated_s < remaining)


def require_no_live_owned_child(ledger, alive=None):
    def process_exists(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    alive = process_exists if alive is None else alive
    for entry in ledger['entries']:
        pid = entry.get('owned_pid')
        if pid is not None and alive(int(pid)):
            raise RuntimeError(f'Prior owned PID {pid} still exists; continuation is refused')


class SupervisedValidation(runner.Validation):
    def __init__(self):
        super().__init__()
        self.absolute_at_start = absolute_elapsed(self.ledger)
        self.supervision_monotonic_start = time.monotonic()

    def elapsed(self):
        monotonic_elapsed = self.absolute_at_start+time.monotonic()-self.supervision_monotonic_start
        return max(monotonic_elapsed, absolute_elapsed(self.ledger))


@contextmanager
def supervision_caps(validation):
    """Recorded runtime scheduling override; original sources stay byte-identical."""
    original_capacity, original_subprocess = runner.check_capacity, runner.subprocess
    def bounded_capacity(attempted, elapsed, **kwargs):
        return original_capacity(attempted, max(elapsed, validation.elapsed()),
                                 max_solves=MAX_SOLVES, max_wall=TOTAL_WALL_S-REPORT_RESERVE_S-PROCESS_CLEANUP_S,
                                 child_timeout=1800.)
    class ProcessProxy:
        def __getattr__(self, name):
            return getattr(original_subprocess, name)
        def run(self, *args, **kwargs):
            remaining = TOTAL_WALL_S-REPORT_RESERVE_S-PROCESS_CLEANUP_S-validation.elapsed()
            if remaining <= 1:
                raise RuntimeError('Original deadline reporting reserve reached before analysis')
            kwargs['timeout'] = min(float(kwargs.get('timeout', 120.)), remaining)
            return original_subprocess.run(*args, **kwargs)
    runner.check_capacity, runner.subprocess = bounded_capacity, ProcessProxy()
    try:
        yield
    finally:
        runner.check_capacity, runner.subprocess = original_capacity, original_subprocess


def verify_allocation(validation):
    receipt = validation.receipt
    if (receipt['limits']['total_CPU_solves'] != MAX_SOLVES
            or receipt['limits']['total_wall_s'] != TOTAL_WALL_S
            or receipt['limits']['workers'] != 1
            or receipt['limits']['new_unique_devices'] != 11
            or receipt['limits']['cumulative_unique_ceiling'] != 24):
        raise ValueError('Original approved limits differ from this supervisor')
    if runner.sha(runner.FOLDER/'approval_receipt.json') != validation.ledger['approval_sha256']:
        raise ValueError('Ledger approval hash differs')
    amendment = runner.read(runner.FOLDER/'cumulative_amendment.json')
    cases = validation.cases
    if (amendment['historical_used'] != 13 or len(cases) > 11
            or len({case['geometry_sha256'] for case in cases}) != len(cases)
            or amendment['historical_used']+len(cases) > 24 or len(validation.ledger['entries']) > MAX_SOLVES):
        raise ValueError('Original cumulative device or solve allocation exceeded')
    allowed = {(case['method'], case['H_um']):case['design_sha256'] for case in cases}
    for entry in validation.ledger['entries']:
        if entry['method'] != 'reference':
            key = (entry['method'], entry['H_um'])
            if key not in allowed or entry['design_sha256'] != allowed[key]:
                raise ValueError('Ledger device is outside frozen approved cases')
    historical = ROOT/'artifacts/natural_inverse_design_candidate_registry_v1.json'
    if runner.sha(historical) != receipt['historical_registry_sha256']:
        raise ValueError('Historical cumulative allocation changed')
    for name, digest in runner.read(runner.FOLDER/'source_manifest.json').items():
        if runner.sha(runner.FOLDER/'source_snapshots'/name) != digest:
            raise ValueError('Original frozen source snapshot changed')


def run_continuation(names):
    if os.name != 'posix':
        raise RuntimeError('Execution requires the original POSIX/WSL environment and flock')
    import fcntl
    with (runner.FOLDER/'runner.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        validation = SupervisedValidation()
        if validation.ledger['status'] not in STOPPED_STATUSES:
            raise RuntimeError('Original runner has not stopped in a diagnosed/unresolved state')
        require_no_live_owned_child(validation.ledger)
        verify_allocation(validation)
        plan = runner.read(runner.FOLDER/'plan.json')['jobs']
        if plan != runner.make_plan():
            raise ValueError('Saved plan differs from the original frozen prospective plan')
        pairs = select_pairs(plan, names)
        require_unfailed_selected_settings(validation.ledger, pairs)
        if validation.elapsed() >= TOTAL_WALL_S-REPORT_RESERVE_S-PROCESS_CLEANUP_S:
            raise RuntimeError('Original reporting reserve already reached; no continuation allowed')
        adaptive_evidence={}
        for pair in pairs:
            if pair['name']==ADAPTIVE_NAME:
                evidence=adaptive_selection_evidence(pair,validation.ledger,validation.cases)
                estimate=estimated_pair_cost(pair,validation.ledger)
                elapsed=validation.elapsed()
                if not pair_fits(pair,validation.ledger,elapsed,estimate['estimated_total_s']):
                    raise RuntimeError('Observed H40 adaptive reserve cannot fit original remaining time/solve budget')
                evidence.update(feasible_under_original_remaining_budget=True,
                                elapsed_including_downtime_at_selection_s=elapsed,
                                numerical_time_remaining_at_selection_s=TOTAL_WALL_S-REPORT_RESERVE_S-PROCESS_CLEANUP_S-elapsed,
                                cost_estimate=estimate)
                adaptive_evidence[pair['name']]=evidence
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        supervision = runner.FOLDER/'supervision'/stamp
        supervision.mkdir(parents=True, exist_ok=False)
        ledger_snapshot = supervision/'prior_ledger.json'
        shutil.copyfile(runner.FOLDER/'ledger.json', ledger_snapshot)
        shutil.copyfile(__file__, supervision/Path(__file__).name)
        receipt = dict(
            status='EXPLICIT_SUPERVISION_OF_ORIGINAL_APPROVAL', created_utc=runner.utc(),
            helper_sha256=runner.sha(__file__), original_runner_sha256=runner.sha(RUNNER_PATH),
            prior_ledger_sha256=runner.sha(ledger_snapshot),
            original_approval_sha256=runner.sha(runner.FOLDER/'approval_receipt.json'),
            original_plan_sha256=runner.sha(runner.FOLDER/'plan.json'),
            original_source_manifest_sha256=runner.sha(runner.FOLDER/'source_manifest.json'),
            selected_pairs=pairs, first_started_utc=validation.ledger['first_started_utc'],
            adaptive_reserve_selection_evidence=adaptive_evidence,
            original_deadline_utc=(parse_utc(validation.ledger['first_started_utc'])+
                                   timedelta(seconds=TOTAL_WALL_S)).isoformat(),
            elapsed_including_downtime_s=validation.elapsed(),
            numerical_children_stop_before_deadline_s=REPORT_RESERVE_S+PROCESS_CLEANUP_S,
            final_reporting_reserve_s=REPORT_RESERVE_S,
            process_cleanup_allowance_s=PROCESS_CLEANUP_S,
            supervision_override='Original Validation.execute; runtime solver and analysis caps at absolute4h minus180s reporting and5s process-cleanup allowances; original1800s child ceiling retained',
            maximum_solves_including_all_prior_attempts=MAX_SOLVES, cumulative_unique_ceiling=24,
            failed_setting_retries=False, new_jobs_or_geometries=False, physics_or_frozen_source_changes=False,
            final_acceptance_status='UNRESOLVED_VALIDATION_STAGE',
        )
        runner.write(supervision/'receipt.json', receipt)
        validation.ledger.setdefault('supervision_runs', []).append(str(supervision.relative_to(runner.FOLDER)))
        validation.ledger.setdefault('supervision_receipt_sha256', {})[str(supervision.relative_to(runner.FOLDER))] = runner.sha(supervision/'receipt.json')
        validation.ledger['status'] = 'SUPERVISION_RUNNING_UNRESOLVED'
        validation.save()
        outcomes = []
        failure = None
        try:
            with supervision_caps(validation):
                for pair in pairs:
                    require_no_live_owned_child(validation.ledger)
                    if pair['name']==ADAPTIVE_NAME:
                        adaptive_selection_evidence(pair,validation.ledger,validation.cases)
                    estimate = estimated_pair_cost(pair, validation.ledger)
                    decision = dict(name=pair['name'], elapsed_including_downtime_s=validation.elapsed(), **estimate)
                    if not pair_fits(pair, validation.ledger, validation.elapsed(), estimate['estimated_total_s']):
                        decision['status'] = 'SKIPPED_CANNOT_FIT_ORIGINAL_REMAINING_BUDGET'
                        outcomes.append(decision)
                        runner.write(supervision/'progress.json', dict(outcomes=outcomes))
                        continue
                    decision['status'] = 'STARTED_PREDECLARED_PAIR'
                    outcomes.append(decision)
                    runner.write(supervision/'progress.json', dict(outcomes=outcomes))
                    for job in pair['jobs']:
                        if pair['name']==ADAPTIVE_NAME:
                            require_adaptive_attempt_ceiling(validation.ledger)
                        validation.execute(job)
                        if pair['name']==ADAPTIVE_NAME:
                            require_adaptive_attempt_ceiling(validation.ledger)
                    decision['status'] = 'EXECUTED_REQUIRES_UNRESOLVED_FINAL_ASSESSMENT'
                    runner.write(supervision/'progress.json', dict(outcomes=outcomes))
            validation.verify()
            validation.ledger['status'] = 'SUPERVISION_FINISHED_UNRESOLVED'
        except BaseException as error:
            validation.terminate()
            failure = error
            validation.ledger['status'] = 'SUPERVISION_STOPPED_UNRESOLVED'
            validation.ledger.setdefault('failures', []).append(dict(
                utc=runner.utc(), type=type(error).__name__, reason=str(error),
                supervision_receipt=str((supervision/'receipt.json').relative_to(runner.FOLDER))))
        finally:
            validation.save()
            runner.write(supervision/'result.json', dict(
                status='UNRESOLVED_VALIDATION_STAGE', ledger_status=validation.ledger['status'],
                outcomes=outcomes, attempted_solves=len(validation.ledger['entries']),
                elapsed_including_downtime_s=validation.elapsed(),
                original_deadline_utc=receipt['original_deadline_utc'],
                failure=None if failure is None else str(failure), optical_acceptance_claimed=False))
        if failure is not None:
            raise failure
        print(json.dumps(dict(status='UNRESOLVED_VALIDATION_STAGE', supervision=str(supervision),
                              attempted_solves=len(validation.ledger['entries']), outcomes=outcomes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', help='Comma-separated original boundary pairs or reserve_mesh_H40_r32')
    parser.add_argument('--run', action='store_true', help='Explicitly execute only after original runner stops')
    args = parser.parse_args()
    if not args.pairs:
        parser.error('Choose explicit --pairs; execution also requires --run')
    names = [name.strip() for name in args.pairs.split(',')]
    selected = select_pairs(runner.make_plan(), names)
    if not args.run:
        print(json.dumps(dict(status='INSPECTION_ONLY_NO_LAUNCH', selected_pairs=selected,
                              optical_solves_launched=0), indent=2))
        return
    run_continuation(names)


if __name__ == '__main__':
    main()
