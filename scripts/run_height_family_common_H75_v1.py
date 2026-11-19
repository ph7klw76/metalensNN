"""Advance only the explicitly requested frozen NN75 forward result.

Inspection is read only. Explicit --run/--selection-reason is permitted after
phaseA; H70 common refinement may precede or follow. This chooses neither a
priority branch nor the meaning of direct calculation. Existing drivers,
physics, plan, lock, ledger clock and global20/65 limits remain unchanged.
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
FRAMEWORK_PATH=ROOT/'scripts/run_height_family_subsequent_phases_v3.py'
FRAMEWORK_SHA='cfe4119c0b2e0f0bdda4df99765752c36e7309f2b46f59a4c7602a3a6093d80c'
MODE='common_requested_H75'
SCOPE='COMMON_REQUESTED_FROZEN_NN75_FORWARD_ONLY_PRIORITY_AND_DIRECT_MEANING_PENDING'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_framework():
    if sha(FRAMEWORK_PATH)!=FRAMEWORK_SHA:raise ValueError('Frozen subsequent-phase framework changed')
    spec=importlib.util.spec_from_file_location('frozen_subsequent_helpers_for_common75',FRAMEWORK_PATH)
    framework=importlib.util.module_from_spec(spec);spec.loader.exec_module(framework)
    return framework,framework.load_frozen_stage()


def common_jobs(module):
    jobs=[dict(item,phase='COMMON_REQUESTED_H75') for item in module.prospective_plan()['prospective_phase_B']
          if item['H_um']==75 and item['method'] in ('reference','NN')]
    require_exact_scope(jobs,module)
    return jobs
def require_exact_scope(jobs,module):
    if (len(jobs)!=2 or [item['method'] for item in jobs]!=['reference','NN']
            or any(module.identity(item)!=(item['method'],75.,24.,2.,10.,259.)
                   or item['job_type']!='nominal' or item['phase']!='COMMON_REQUESTED_H75' for item in jobs)):
        raise ValueError('Only frozen NN75/reference24,PML2,padding10,duration259 is authorized by this common scope')
def calls_needed(stage,module,jobs):
    count=0;cached=[]
    for item in jobs:
        result=stage.cache(item)
        if result is None:
            module.require_permitted_attempt(item,stage.old_entries,stage.ledger['entries']);count+=1
        else:cached.append(dict(id=result['entry']['id'],origin=result['stage'],method=item['method']))
    return count,cached
def budget(stage,module,jobs):
    require_exact_scope(jobs,module)
    needed,cached=calls_needed(stage,module,jobs)
    common70_needed,_=calls_needed(stage,module,module.prospective_plan()['prospective_phase_C'])
    projected=len(stage.ledger['entries'])+needed+common70_needed
    if projected>14:
        raise RuntimeError('Common75 would consume slots needed to keep both pending priority paths feasible')
    if stage.elapsed()>=module.MAX_WALL-module.RESERVE-1:raise RuntimeError('Original48h deadline/reporting reserve reached')
    return dict(already_attempted_new=len(stage.ledger['entries']),common75_new_calls=needed,
                common70_remaining_calls=common70_needed,projected_attempts_after_all_common=projected,
                comparison_remaining_calls=4,comparison_adaptive_slots=2,original_controls_remaining_calls=6,
                original_controls_adaptive_slots=0,original_controls_requires_separate20cap_amendment=True,
                cached_records=cached,elapsed_s=stage.elapsed(),unchanged_global_new_cap=20,unchanged_cumulative_cap=65)
def budget_effect(estimate):
    common=estimate['projected_attempts_after_all_common']
    return dict(all_common_projected_attempts=common,comparison_remaining=4,comparison_projected_total=common+4,
                comparison_adaptive_remaining=2,original_controls_remaining=6,original_projected_total=common+6,
                original_adaptive_remaining=0)
def preflight(stage,module,framework):
    stage.verify();framework.require_phase_A_complete(stage,module)
    selection=stage.ledger.get('subsequent_phase_selection')
    if selection is not None:
        if selection['mode']!='repair_only':raise ValueError('A priority branch is already selected; common-pending scope cannot replace it')
        framework.verify_existing_selection(stage,module,'repair_only')
    for prior in stage.ledger.get('subsequent_phase_runs',[]):
        if prior.get('mode')==MODE:
            path=module.FOLDER/prior['receipt_file']
            if module.sha(path)!=prior['receipt_sha256'] or module.read(path)['driver_sha256']!=sha(__file__):
                raise ValueError('Previous common75 receipt/source changed')
    module.require_no_live_children(stage.old_entries+stage.ledger['entries'])
    jobs=common_jobs(module)
    return jobs,budget(stage,module,jobs)
def with_receipt(item,module,path,digest):
    return dict(item,selection_receipt_file=str(path.relative_to(module.FOLDER)),selection_receipt_sha256=digest,
                selection_mode=MODE,selection_scope=SCOPE)
def register_receipt(stage,module,path,digest):
    link=dict(mode=MODE,receipt_file=str(path.relative_to(module.FOLDER)),receipt_sha256=digest,driver_sha256=sha(__file__))
    stage.ledger.setdefault('subsequent_phase_runs',[]).append(link)
    stage.save()
    return link


def execute_common(stage,module,framework,jobs,path,digest):
    require_exact_scope(jobs,module)
    outcomes=[];failure=None;unchanged=None
    stage.ledger['status']='RUNNING_COMMON_REQUESTED_NN75_FORWARD_ONLY';stage.save()
    try:
        for item in jobs:
            if module.sha(path)!=digest or module.read(path)['driver_sha256']!=sha(__file__):
                raise ValueError('Prospective common75 receipt/source changed')
            stage.verify();budget(stage,module,jobs)
            selected=with_receipt(item,module,path,digest)
            result=stage.execute(selected)
            outcomes.append(dict(method=item['method'],record_id=result['entry']['id'],origin=result['stage'],
                status=result['entry']['status'],analysis_status=result['entry'].get('analysis_status'),
                operation_selection_receipt_file=selected['selection_receipt_file'],operation_selection_receipt_sha256=digest))
        stage.verify();unchanged=framework.verify_stage1_delivery(module)
        stage.ledger['status']='COMMON_REQUESTED_NN75_EXECUTED_PRIORITY_AND_DIRECT_MEANING_PENDING'
    except BaseException as error:
        failure=error;stage.terminate();stage.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS'
        stage.ledger.setdefault('failures',[]).append(dict(utc=utc(),type=type(error).__name__,reason=str(error),
            selection_receipt=str(path.relative_to(module.FOLDER))))
    finally:
        stage.save();module.write(path.parent/'outcomes.json',dict(
            status='COMMON_FORWARD_EVIDENCE_ONLY_NO_PRIORITY_OR_DIRECT_MEANING_DECISION',outcomes=outcomes,
            new_attempts=len(stage.ledger['entries']),cumulative_attempts=45+len(stage.ledger['entries']),
            elapsed_s=stage.elapsed(),original_first_started_utc=stage.ledger['first_started_utc'],
            original_absolute_deadline_utc=stage.ledger.get('absolute_deadline_utc'),stage1_delivery=unchanged,
            failure=None if failure is None else repr(failure),full_optical_acceptance_claimed=False,
            unrelated_controls_still_unresolved=['H100PML3','H70padding15','H100duration502.5']))
    if failure is not None:raise failure


def run_common(framework,module,reason):
    if os.name!='posix':raise RuntimeError('Execution requires original WSL/POSIX environment')
    if not reason.strip():raise ValueError('Explicit root common-scope selection reason required')
    import fcntl
    with (module.PRIOR/'runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage=module.Stage();jobs,estimate=preflight(stage,module,framework)
        unchanged=framework.verify_stage1_delivery(module)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder=module.FOLDER/'common_requested_H75_selections'/stamp;folder.mkdir(parents=True,exist_ok=False)
        data=(module.FOLDER/'ledger.json').read_bytes()
        with (folder/'prior_ledger.json').open('xb') as stream:stream.write(data)
        shutil.copyfile(__file__,folder/Path(__file__).name)
        receipt=dict(status='PROSPECTIVE_MODE_SELECTION_BEFORE_SUBSEQUENT_DATA',created_utc=utc(),
            selected_mode=MODE,execution_scope=SCOPE,decision_origin='EXPLICIT_ROOT_SELECTION_REASON',
            decision_text_exact=reason,decision_text_sha256=hashlib.sha256(reason.encode()).hexdigest(),
            driver_sha256=sha(__file__),frozen_framework_sha256=FRAMEWORK_SHA,
            frozen_stage_runner_sha256=module.sha(framework.STAGE_PATH),
            approval_sha256=module.sha(module.FOLDER/'approval_receipt.json'),
            preparation_manifest_sha256=module.sha(module.FOLDER/'preparation_manifest.json'),
            frozen_plan_sha256=module.sha(module.FOLDER/'plan.json'),prior_ledger_snapshot_sha256=hashlib.sha256(data).hexdigest(),
            selected_jobs=jobs,budget_before_selection=estimate,common75_budget_effect=budget_effect(estimate),
            stage1_delivery_before=unchanged,
            scope_authority='User explicitly requested H75 NN comparison; frozen NN75 forward result is common to both pending direct-calculation meanings',
            height75_domain='validation-observed, including seed selection; not heldout',
            priority_selected=False,direct_calculation_meaning_selected=False,separately_optimized_direct_launched=False,
            original_control_deferral_selected=False,no_original_controls_deferred_by_this_selection=True,
            subsequent_phase_selection_unchanged=True,original48h_and20solve_limits_unchanged=True,
            unchanged_deadline=True,unchanged_thresholds=True,physics_source_changes=False,
            adaptive_launch_authorized_by_this_receipt=False,
            branch_tradeoff='After allcommon14: comparison needs4 and retains2adaptive; original controls need6 and retain0adaptive under a separate prospective20cap driver',
            geometry_or_physics_changes=False,full_optical_acceptance_claimed=False)
        path=folder/'selection_receipt.json';write_new(path,receipt);digest=sha(path)
        # Existing priority selection is deliberately untouched. The recognized
        # hash-linked registry carries this independent common-work receipt.
        register_receipt(stage,module,path,digest)
        execute_common(stage,module,framework,jobs,path,digest)
        print(json.dumps(dict(status=stage.ledger['status'],selection_receipt=str(path),
                              new_attempts=len(stage.ledger['entries']),priority_selected=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--selection-reason')
    args=parser.parse_args()
    if args.run and not args.selection_reason:parser.error('--run requires explicit --selection-reason for common requested NN75 only')
    framework,module=load_framework()
    if args.run:run_common(framework,module,args.selection_reason);return
    try:
        stage=module.Stage();jobs,estimate=preflight(stage,module,framework)
        print(json.dumps(dict(status='INSPECTION_ONLY_COMMON_NN75_READY',jobs=jobs,budget=estimate,
                              worker_launches=0,priority_selected=False),indent=2))
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        print(json.dumps(dict(status='INSPECTION_ONLY_COMMON_NN75_NOT_READY',reason=str(error),worker_launches=0)))


if __name__=='__main__':main()
