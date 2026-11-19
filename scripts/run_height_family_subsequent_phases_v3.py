"""Explicit priority selection after the frozen stage2 phaseA completes.

Import/default inspection launches nothing and writes nothing. --run requires
--mode and a recorded decision. repair_only accepts a root reason; a full
priority branch requires the actual user reply. The common2 or selected8 jobs
use the frozen Stage, existing lock, ledger and original48h clock. The final
two adaptive attempts are never launched by this driver.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil


ROOT=Path(__file__).resolve().parents[1]
STAGE_PATH=ROOT/'scripts/run_height_family_continuation_v3.py'
EXPECTED_STAGE_SHA='a287f5ef647d105c413761aa0b9dd7449263d4e7d8e06960c2c2e079d07d2586'
MODES=('repair_only','comparison_first','original_first')
PRE_ADAPTIVE_ATTEMPT_CEILING=18


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_frozen_stage():
    if sha(STAGE_PATH)!=EXPECTED_STAGE_SHA:raise ValueError('Frozen stage2 phaseA runner changed')
    spec=importlib.util.spec_from_file_location('frozen_stage2_for_subsequent_phases',STAGE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def selected_jobs(mode,prepared_plan,original_plan):
    if mode not in MODES:raise ValueError('Choose an explicit approved priority mode')
    if mode=='repair_only':
        jobs=[dict(job) for job in prepared_plan['prospective_phase_C']]
    elif mode=='comparison_first':
        jobs=[dict(job) for key in ('prospective_phase_B','prospective_phase_C','prospective_phase_D')
              for job in prepared_plan[key]]
    else:
        jobs=[]
        for kind,H in [('pml',100.),('padding',70.),('duration',100.)]:
            pair=[dict(job,phase='ORIGINAL_MATRIX') for job in original_plan['jobs']
                  if job['job_type']==kind and job['H_um']==H and job['resolution']==16.
                  and job['method'] in ('reference','NN')]
            if len(pair)!=2 or [job['method'] for job in pair]!=['reference','NN']:
                raise ValueError('Original predeclared boundary pair is incomplete')
            jobs.extend(pair)
        jobs.extend(dict(job) for job in prepared_plan['prospective_phase_C'])
    if len(jobs)!=(2 if mode=='repair_only' else 8):
        raise ValueError('Repair-only requires2 common jobs; each full priority mode requires8')
    return jobs


def require_phase_A_complete(stage,module):
    if not stage.ledger.get('first_started_utc'):
        raise ValueError('Original stage2 clock must already be running; it cannot reset')
    for item in module.prospective_plan()['phase_A']:
        matches=[entry for entry in stage.ledger['entries'] if module.identity(entry)==module.identity(item)]
        if (len(matches)!=1 or matches[0]['status']!='completed'
                or (item['method']!='reference' and matches[0].get('analysis_status')!='completed')):
            raise ValueError('PhaseA must complete all10 raw solves and independent analyses before subsequent workers')
        cached=stage.cache(item)
        if cached is None or cached['stage']!='stage2':raise ValueError('PhaseA evidence must be from this authorized stage')
        if item['method']!='reference':
            entry=matches[0]
            if module.sha(module.FOLDER/entry['analysis_file'])!=entry['analysis_sha256']:
                raise ValueError('PhaseA completed analysis evidence changed')


def budget_plan(stage,module,jobs):
    """Count missing solves, reserving the final2 even after failed attempts."""
    needed=0;cached=[]
    for item in jobs:
        cache=stage.cache(item)
        if cache is not None:
            cached.append(dict(method=item['method'],H_um=item['H_um'],resolution=item['resolution'],
                               record_id=cache['entry']['id'],origin=cache['stage']))
            continue
        module.require_permitted_attempt(item,stage.old_entries,stage.ledger['entries'])
        needed+=1
    attempted=len(stage.ledger['entries'])
    if attempted+needed>PRE_ADAPTIVE_ATTEMPT_CEILING:
        raise RuntimeError('Selected branch cannot fit while preserving the final2 adaptive attempts')
    if stage.elapsed()>=module.MAX_WALL-module.RESERVE-1:
        raise RuntimeError('Original stage2 deadline/reporting reserve reached')
    return dict(already_attempted_new=attempted,additional_solver_attempts_needed=needed,
                projected_new_attempts_after_branch=attempted+needed,
                cumulative_attempts_after_branch=45+attempted+needed,
                final_adaptive_attempts_unlaunched=2,cached_records=cached,
                unspent_additional_attempt_slots=20-attempted-needed,
                elapsed_s=stage.elapsed(),original_first_started_utc=stage.ledger['first_started_utc'],
                original_absolute_deadline_utc=stage.ledger.get('absolute_deadline_utc'))


def verify_stage1_delivery(module):
    delivery=module.read(module.FOLDER/'prior_delivery_manifest.json')
    for name,entry in delivery['files'].items():
        if module.sha(module.PRIOR/name)!=entry['sha256']:
            raise ValueError('Stage1 original delivered bytes changed: '+name)
    return dict(stage1_unchanged=True,verified_delivered_files=len(delivery['files']),
                original_delivery_manifest_sha256=module.sha(module.PRIOR/'delivery_manifest.json'),
                prior_ledger_sha256=module.sha(module.PRIOR/'ledger.json'))


def verify_existing_selection(stage,module,mode):
    previous=stage.ledger.get('subsequent_phase_selection')
    if previous is None:return
    if previous['mode']!=mode and not (previous['mode']=='repair_only' and mode in ('comparison_first','original_first')):
        raise ValueError('Priority mode cannot change after a frozen selection; root diagnosis/new prospective receipt required')
    path=module.FOLDER/previous['receipt_file']
    if module.sha(path)!=previous['receipt_sha256']:
        raise ValueError('Original subsequent-phase selection receipt changed')
    selection=module.read(path)
    if selection['driver_sha256']!=sha(__file__):
        raise ValueError('Selected subsequent-phase driver source changed')


def preflight(stage,module,mode):
    stage.verify()
    verify_existing_selection(stage,module,mode)
    require_phase_A_complete(stage,module)
    module.require_no_live_children(stage.old_entries+stage.ledger['entries'])
    prepared=module.read(module.FOLDER/'plan.json')
    if prepared!=module.prospective_plan():raise ValueError('Frozen prospective stage2 plan changed')
    original=module.read(module.PRIOR/'plan.json')
    module.check_delivered_file(module.PRIOR/'plan.json',module.read(module.FOLDER/'prior_delivery_manifest.json'),module.PRIOR)
    jobs=selected_jobs(mode,prepared,original)
    return jobs,budget_plan(stage,module,jobs)


def execution_scope(mode):
    return ('COMMON_AUTHORIZED_H70_MESH_REPAIR_ONLY_PRIORITY_AND_DIRECT_MEANING_PENDING'
            if mode=='repair_only' else
            'CONFIRMED_SEPARATELY_OPTIMIZED_NON_NEURAL_DIRECT_TABLE_COMPARISON_FIRST'
            if mode=='comparison_first' else 'CONFIRMED_ORIGINAL_BOUNDARY_DURATION_CHECKS_FIRST')


def job_with_receipt(item,module,mode,receipt_path,receipt_sha):
    """Stage.execute copies these fields into a new ledger entry before Popen."""
    selected=dict(item)
    selected.update(selection_receipt_file=str(receipt_path.relative_to(module.FOLDER)),
                    selection_receipt_sha256=receipt_sha,selection_mode=mode,
                    selection_scope=execution_scope(mode))
    return selected


def execute_selected(stage,module,jobs,mode,receipt_path,receipt_sha):
    """Delegates every worker/cache/recovery operation to the frozen Stage."""
    outcomes=[]
    stage.ledger['status']='RUNNING_SELECTED_SUBSEQUENT_PHASES'
    stage.save()
    failure=None
    try:
        for item in jobs:
            if module.sha(receipt_path)!=receipt_sha:raise ValueError('Prospective selection receipt changed')
            receipt=module.read(receipt_path)
            if receipt['driver_sha256']!=sha(__file__):raise ValueError('Active selected driver source changed')
            stage.verify()
            # Recount before each operation. Failed attempts never disappear;
            # completed solver/failed-analysis cache recovery costs no solve.
            budget_plan(stage,module,jobs)
            selected=job_with_receipt(item,module,mode,receipt_path,receipt_sha)
            result=stage.execute(selected)
            outcomes.append(dict(method=item['method'],H_um=item['H_um'],resolution=item['resolution'],
                job_type=item['job_type'],record_id=result['entry']['id'],origin=result['stage'],
                status=result['entry']['status'],analysis_status=result['entry'].get('analysis_status'),
                operation_selection_receipt_file=selected['selection_receipt_file'],
                operation_selection_receipt_sha256=receipt_sha,
                original_attempt_selection_receipt_file=result['entry'].get('selection_receipt_file')))
        stage.verify()
        unchanged=verify_stage1_delivery(module)
        stage.ledger['status']=('COMMON_REPAIR_EXECUTED_PRIORITY_SELECTION_PENDING' if mode=='repair_only'
                               else 'SELECTED_SUBSEQUENT_PHASES_EXECUTED_ADAPTIVE_CHOICE_PENDING')
    except BaseException as error:
        stage.terminate();failure=error;unchanged=None
        stage.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS'
        stage.ledger.setdefault('failures',[]).append(dict(utc=utc(),type=type(error).__name__,reason=str(error),
                                                       selection_receipt=str(receipt_path.relative_to(module.FOLDER))))
    finally:
        stage.save()
        module.write(receipt_path.parent/'outcomes.json',dict(
            status='UNRESOLVED_OPTICAL_ACCEPTANCE_ADAPTIVE_CHOICE_PENDING',selected_mode=mode,
            ledger_status=stage.ledger['status'],outcomes=outcomes,stage1_delivery=unchanged,
            new_attempts=len(stage.ledger['entries']),cumulative_attempts=45+len(stage.ledger['entries']),
            elapsed_s=stage.elapsed(),original_first_started_utc=stage.ledger['first_started_utc'],
            original_absolute_deadline_utc=stage.ledger.get('absolute_deadline_utc'),
            final_adaptive_attempts_automatically_launched=0,
            failure=None if failure is None else repr(failure),full_optical_acceptance_claimed=False))
    if failure is not None:raise failure


def run_mode(module,mode,decision_text,decision_origin):
    if os.name!='posix':raise RuntimeError('Execution requires original WSL/POSIX environment')
    if not decision_text.strip():raise ValueError('Exact user reply or explicit root selection reason required')
    if mode!='repair_only' and decision_origin!='EXPLICIT_USER_REPLY':
        raise ValueError('Priority-dependent branch requires the actual user reply; waiting time is not approval')
    import fcntl
    with (module.PRIOR/'runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage=module.Stage();jobs,budget=preflight(stage,module,mode)
        before=verify_stage1_delivery(module)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder=module.FOLDER/'subsequent_phase_selections'/stamp;folder.mkdir(parents=True,exist_ok=False)
        ledger_bytes=(module.FOLDER/'ledger.json').read_bytes()
        with (folder/'prior_ledger.json').open('xb') as stream:stream.write(ledger_bytes)
        shutil.copyfile(__file__,folder/Path(__file__).name)
        receipt=dict(status='PROSPECTIVE_MODE_SELECTION_BEFORE_SUBSEQUENT_DATA',created_utc=utc(),
            selected_mode=mode,execution_scope=execution_scope(mode),decision_origin=decision_origin,decision_text_exact=decision_text,
            decision_text_sha256=hashlib.sha256(decision_text.encode()).hexdigest(),driver_sha256=sha(__file__),
            frozen_stage_runner_sha256=sha(STAGE_PATH),approval_sha256=module.sha(module.FOLDER/'approval_receipt.json'),
            preparation_manifest_sha256=module.sha(module.FOLDER/'preparation_manifest.json'),
            frozen_plan_sha256=module.sha(module.FOLDER/'plan.json'),
            prior_ledger_snapshot_sha256=hashlib.sha256(ledger_bytes).hexdigest(),selected_jobs=jobs,
            budget_before_selection=budget,stage1_delivery_before=before,
            final_adaptive_attempts_reserved=2,adaptive_launch_authorized_by_this_receipt=False,
            unchanged_deadline=True,unchanged_thresholds=True,physics_source_changes=False,
            mode_tradeoff=('Common authorized H70 refinement only; priority choice pending and H100PML3,H70padding15,H100duration502.5 remain unresolved'
                if mode=='repair_only' else
                'Prioritizes requested independent NN/direct comparison; original H100PML3,H70padding15,H100duration502.5 remain unresolved'
                if mode=='comparison_first' else 'Prioritizes remaining original H100PML3,H70padding15,H100duration502.5 controls; independent direct/H75 comparison remains deferred'))
        path=folder/'selection_receipt.json';write_new(path,receipt);digest=sha(path)
        entry=dict(mode=mode,receipt_file=str(path.relative_to(module.FOLDER)),receipt_sha256=digest,
                   driver_sha256=receipt['driver_sha256'])
        previous=stage.ledger.get('subsequent_phase_selection')
        if previous is None or previous['mode']=='repair_only':
            stage.ledger['subsequent_phase_selection']=entry
        stage.ledger.setdefault('subsequent_phase_runs',[]).append(entry)
        stage.save()
        execute_selected(stage,module,jobs,mode,path,digest)
        print(json.dumps(dict(status=stage.ledger['status'],selection_receipt=str(path),
                              new_attempts=len(stage.ledger['entries']),adaptive_calls_launched=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--run',action='store_true')
    decision=parser.add_mutually_exclusive_group()
    decision.add_argument('--user-reply-file',type=Path)
    decision.add_argument('--selection-reason')
    args=parser.parse_args()
    if args.run and (args.mode is None or (args.user_reply_file is None and args.selection_reason is None)):
        parser.error('--run requires explicit --mode and --user-reply-file or --selection-reason')
    if args.run and args.mode!='repair_only' and args.user_reply_file is None:
        parser.error('Priority-dependent comparison/original branch requires --user-reply-file; only repair_only accepts a root reason')
    module=load_frozen_stage();mode=args.mode or 'comparison_first'
    if args.run:
        if args.user_reply_file is not None:
            text=args.user_reply_file.read_text(encoding='utf-8-sig');origin='EXPLICIT_USER_REPLY'
        else:text=args.selection_reason;origin='EXPLICIT_ROOT_SELECTION_REASON'
        run_mode(module,mode,text,origin)
        return
    try:
        stage=module.Stage();jobs,budget=preflight(stage,module,mode)
        print(json.dumps(dict(status='INSPECTION_ONLY_READY',mode=mode,selected_jobs=jobs,budget=budget,
                              worker_launches=0,receipt_files_written=0),indent=2))
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        print(json.dumps(dict(status='INSPECTION_ONLY_NOT_READY',mode=mode,reason=str(error),worker_launches=0,
                              receipt_files_written=0)))


if __name__=='__main__':main()
