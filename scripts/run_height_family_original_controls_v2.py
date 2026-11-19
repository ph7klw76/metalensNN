"""Complete six originally requested controls after14 common stage2 results.

Default inspection is read only. --run uses the immutable original request,
with a separately attributed root interpretation; it invents no new reply.
This separate prospective selection uses the frozen Stage and unchanged20/65
attempt and48h limits and no adaptive reserve. Direct is explicitly assumed
to mean Meep forward validation of the same frozen NN lens. Existing drivers
stay intact; no separately optimized non-neural lens is launched.
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
MODE='original_controls_from_request'
SCOPE='ORIGINAL_EXPLICIT_USER_REQUEST_REMAINING_SIX_CONTROLS_SAME_NN_FORWARD_NO_ADAPTIVE_RESERVE'
V1_DRIVER_SHA='cc357b84a04accccb822021805bd87938c88dc8298f1fe53e41f522e04ee9ef0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def request_authority(module,request_file=None):
    approval=module.read(module.FOLDER/'approval_receipt.json')
    immutable=module.FOLDER/'user_authorization_exact.txt'
    data=immutable.read_bytes();message=data.decode('utf-8');digest=hashlib.sha256(data).hexdigest()
    supplied=immutable if request_file is None else Path(request_file)
    if (approval.get('status')!='APPROVED_BY_USER' or not message.strip()
            or approval.get('user_message_exact')!=message or approval.get('user_message_sha256')!=digest
            or supplied.read_bytes()!=data):
        raise ValueError('Original request bytes/text/SHA must match immutable approval and authorization file exactly')
    return dict(message_exact=message,message_sha256=digest,approval_sha256=module.sha(module.FOLDER/'approval_receipt.json'))
def authority_fields(authority,interpretation):
    if not isinstance(interpretation,str) or not interpretation.strip():
        raise ValueError('Explicit root scope interpretation is required separately from the user quote')
    return dict(decision_origin='EXPLICIT_USER_REQUEST',decision_text_exact=authority['message_exact'],
        decision_text_sha256=authority['message_sha256'],original_user_request_sha256=authority['message_sha256'],
        original_user_request_file='original_user_request.txt',root_scope_interpretation=interpretation,
        root_scope_interpretation_file='root_scope_interpretation.txt',
        root_scope_interpretation_sha256=hashlib.sha256(interpretation.encode()).hexdigest(),
        root_scope_interpretation_origin='ROOT_SCOPE_INTERPRETATION_NOT_USER_QUOTE',root_scope_interpretation_is_user_quote=False,
        new_priority_reply_received=False,clarification_pending=True,elapsed_used_as_authorization=False,
        direct_calculation_assumption='INDEPENDENT_MEEP_FORWARD_OF_SAME_FROZEN_NN_GEOMETRY',
        separate_non_neural_comparison_scope='PROXY_ONLY_UNLESS_USER_STEERS',user_direct_calculation_meaning_claimed=False)
def validate_request_receipt(receipt,module,folder=None):
    authority=request_authority(module)
    if (receipt.get('selected_mode')!=MODE or receipt.get('execution_scope')!=SCOPE
            or receipt.get('decision_origin')!='EXPLICIT_USER_REQUEST'
            or receipt.get('decision_text_exact')!=authority['message_exact']
            or receipt.get('decision_text_sha256')!=authority['message_sha256']
            or receipt.get('original_user_request_sha256')!=authority['message_sha256']
            or receipt.get('approval_sha256')!=authority['approval_sha256']):
        raise ValueError('Receipt must cite the original explicit user request, not a new reply or root-only authorization')
    root_text=receipt.get('root_scope_interpretation')
    if (not isinstance(root_text,str) or not root_text.strip()
            or receipt.get('root_scope_interpretation_sha256')!=hashlib.sha256(root_text.encode()).hexdigest()
            or receipt.get('root_scope_interpretation_origin')!='ROOT_SCOPE_INTERPRETATION_NOT_USER_QUOTE'
            or receipt.get('root_scope_interpretation_is_user_quote') is not False
            or receipt.get('new_priority_reply_received') is not False
            or receipt.get('clarification_pending') is not True
            or receipt.get('elapsed_used_as_authorization') is not False
            or receipt.get('user_direct_calculation_meaning_claimed') is not False
            or receipt.get('direct_calculation_assumption')!='INDEPENDENT_MEEP_FORWARD_OF_SAME_FROZEN_NN_GEOMETRY'
            or receipt.get('separate_non_neural_comparison_scope')!='PROXY_ONLY_UNLESS_USER_STEERS'):
        raise ValueError('Root interpretation must remain separate and must not claim a new reply or elapsed-time approval')
    if receipt.get('original_user_request_file')!='original_user_request.txt' or receipt.get('root_scope_interpretation_file')!='root_scope_interpretation.txt':
        raise ValueError('Fixed original-request and root-interpretation sidecar names required')
    if folder is not None:
        if ((Path(folder)/'original_user_request.txt').read_bytes()!=authority['message_exact'].encode()
                or (Path(folder)/'root_scope_interpretation.txt').read_bytes()!=root_text.encode()):
            raise ValueError('User request/root interpretation sidecar bytes changed')
def load_framework():
    if sha(FRAMEWORK_PATH)!=FRAMEWORK_SHA:raise ValueError('Frozen subsequent-phase framework changed')
    spec=importlib.util.spec_from_file_location('frozen_framework_for_original_controls',FRAMEWORK_PATH)
    framework=importlib.util.module_from_spec(spec);spec.loader.exec_module(framework)
    return framework,framework.load_frozen_stage()


def selected_controls(original_plan,module):
    result=[]
    for kind,H,pml,padding,duration in [('pml',100.,3.,10.,335.),('padding',70.,2.,15.,243.),
                                       ('duration',100.,2.,10.,502.5)]:
        pair=[dict(item,phase='ORIGINAL_CONTROLS_AFTER_COMMON') for item in original_plan['jobs']
              if item['job_type']==kind and item['H_um']==H and item['resolution']==16.
              and item['method'] in ('reference','NN')]
        if (len(pair)!=2 or [item['method'] for item in pair]!=['reference','NN']
                or any(module.identity(item)!=(item['method'],H,16.,pml,padding,duration) for item in pair)):
            raise ValueError('Only the exact6 sealed original PML100/padding70/duration100 jobs are permitted')
        result.extend(pair)
    return result
def require_scope(jobs,module):
    canonical=[]
    for kind,H,pml,padding,duration in [('pml',100.,3.,10.,335.),('padding',70.,2.,15.,243.),
                                       ('duration',100.,2.,10.,502.5)]:
        for method in ('reference','NN'):
            canonical.append((method,H,16.,pml,padding,duration,kind))
    actual=[module.identity(item)+(item['job_type'],) for item in jobs]
    if actual!=canonical:raise ValueError('Original-controls scope cannot add, reorder or alter jobs')
def require_common_complete(stage,module,framework):
    framework.require_phase_A_complete(stage,module)
    common=list(module.prospective_plan()['prospective_phase_C'])
    common.extend(item for item in module.prospective_plan()['prospective_phase_B']
                  if item['H_um']==75 and item['method'] in ('reference','NN'))
    evidence=[]
    for item in common:
        matching=[entry for entry in stage.ledger['entries'] if module.identity(entry)==module.identity(item)]
        if (len(matching)!=1 or matching[0]['status']!='completed'
                or (item['method']=='NN' and matching[0].get('analysis_status')!='completed')):
            raise ValueError('All NN75/reference24 and H70/reference32 common results must be complete first')
        entry=matching[0];cached=stage.cache(item)
        if cached is None or cached['stage']!='stage2':raise ValueError('All14 common results require authenticated stage2 evidence')
        if item['method']=='NN' and module.sha(module.FOLDER/entry['analysis_file'])!=entry['analysis_sha256']:
            raise ValueError('Common completed analysis SHA changed')
        evidence.append(dict(method=item['method'],H_um=item['H_um'],resolution=item['resolution'],
            record_id=entry['id'],record_file=entry.get('record_file'),record_sha256=entry.get('record_sha256'),
            analysis_file=entry.get('analysis_file'),analysis_sha256=entry.get('analysis_sha256')))
    return evidence
def budget(stage,module,jobs):
    require_scope(jobs,module);needed=0;cached=[]
    for item in jobs:
        result=stage.cache(item)
        if result is None:
            module.require_permitted_attempt(item,stage.old_entries,stage.ledger['entries']);needed+=1
        else:cached.append(dict(record_id=result['entry']['id'],origin=result['stage'],method=item['method'],
                               H_um=item['H_um'],job_type=item['job_type']))
    total=len(stage.ledger['entries'])+needed
    if total>20 or 45+total>65:raise RuntimeError('Six original controls cannot fit the unchanged20/65 total attempt cap')
    if stage.elapsed()>=module.MAX_WALL-module.RESERVE-1:raise RuntimeError('Original48h deadline/reporting reserve reached')
    return dict(already_attempted_new=len(stage.ledger['entries']),new_calls_needed=needed,
        projected_new_attempts=total,projected_cumulative_attempts=45+total,adaptive_reserve=0,
        cached_records=cached,elapsed_s=stage.elapsed(),original_first_started_utc=stage.ledger['first_started_utc'],
        original_absolute_deadline_utc=stage.ledger.get('absolute_deadline_utc'))
def preflight(stage,module,framework):
    stage.verify();request_authority(module);common=require_common_complete(stage,module,framework)
    choice=stage.ledger.get('subsequent_phase_selection')
    if choice is not None:
        if choice['mode']=='comparison_first':raise ValueError('A conflicting comparison-priority branch was already selected')
        if choice['mode'] not in ('repair_only','original_first'):
            raise ValueError('Current subsequent selection is not an authenticated repair/original scope')
        framework.verify_existing_selection(stage,module,choice['mode'])
    for link in stage.ledger.get('subsequent_phase_runs',[]):
        if link.get('mode') in (MODE,'original_controls_after_common'):
            path=module.FOLDER/link['receipt_file']
            expected_driver=sha(__file__) if link['mode']==MODE else V1_DRIVER_SHA
            if module.sha(path)!=link['receipt_sha256'] or module.read(path)['driver_sha256']!=expected_driver:
                raise ValueError('Previous original-controls receipt/source changed')
    module.require_no_live_children(stage.old_entries+stage.ledger['entries'])
    delivery=module.read(module.FOLDER/'prior_delivery_manifest.json')
    module.check_delivered_file(module.PRIOR/'plan.json',delivery,module.PRIOR)
    jobs=selected_controls(module.read(module.PRIOR/'plan.json'),module)
    return jobs,budget(stage,module,jobs),common
def with_receipt(item,module,path,digest):
    return dict(item,selection_receipt_file=str(path.relative_to(module.FOLDER)),selection_receipt_sha256=digest,
                selection_mode=MODE,selection_scope=SCOPE)


def execute_controls(stage,module,framework,jobs,path,digest):
    require_scope(jobs,module);outcomes=[];failure=None;unchanged=None
    stage.ledger['status']='RUNNING_USER_SELECTED_ORIGINAL_CONTROLS';stage.save()
    try:
        for item in jobs:
            if module.sha(path)!=digest or module.read(path)['driver_sha256']!=sha(__file__):
                raise ValueError('Prospective original-controls receipt/source changed')
            stage.verify();validate_request_receipt(module.read(path),module,path.parent);budget(stage,module,jobs)
            result=stage.execute(with_receipt(item,module,path,digest))
            outcomes.append(dict(method=item['method'],H_um=item['H_um'],job_type=item['job_type'],
                record_id=result['entry']['id'],origin=result['stage'],status=result['entry']['status'],
                analysis_status=result['entry'].get('analysis_status'),operation_selection_receipt_file=str(path.relative_to(module.FOLDER)),
                operation_selection_receipt_sha256=digest))
        stage.verify();unchanged=framework.verify_stage1_delivery(module)
        stage.ledger['status']='USER_SELECTED_ORIGINAL_CONTROLS_EXECUTED_REQUIRES_FINAL_ASSESSMENT'
    except BaseException as error:
        failure=error;stage.terminate();stage.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS'
        stage.ledger.setdefault('failures',[]).append(dict(utc=utc(),type=type(error).__name__,reason=str(error),
            selection_receipt=str(path.relative_to(module.FOLDER))))
    finally:
        stage.save();module.write(path.parent/'outcomes.json',dict(
            status='ORIGINAL_CONTROL_EVIDENCE_ONLY_REQUIRES_FINAL_ASSESSMENT',outcomes=outcomes,
            new_attempts=len(stage.ledger['entries']),cumulative_attempts=45+len(stage.ledger['entries']),
            elapsed_s=stage.elapsed(),original_first_started_utc=stage.ledger['first_started_utc'],
            original_absolute_deadline_utc=stage.ledger.get('absolute_deadline_utc'),stage1_delivery=unchanged,
            failure=None if failure is None else repr(failure),adaptive_reserve=0,
            user_direct_calculation_meaning_claimed=False,root_direct_calculation_assumption_stated=True,
            full_optical_acceptance_claimed=False))
    if failure is not None:raise failure


def run_controls(framework,module,request_file,interpretation):
    authority=request_authority(module,request_file);decision=authority_fields(authority,interpretation)
    if os.name!='posix':raise RuntimeError('Execution requires original WSL/POSIX environment')
    import fcntl
    with (module.PRIOR/'runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage=module.Stage();jobs,estimate,common=preflight(stage,module,framework)
        unchanged=framework.verify_stage1_delivery(module)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder=module.FOLDER/'original_controls_selections'/stamp;folder.mkdir(parents=True,exist_ok=False)
        data=(module.FOLDER/'ledger.json').read_bytes()
        with (folder/'prior_ledger.json').open('xb') as stream:stream.write(data)
        shutil.copyfile(__file__,folder/Path(__file__).name)
        with (folder/'original_user_request.txt').open('xb') as stream:stream.write(authority['message_exact'].encode())
        with (folder/'root_scope_interpretation.txt').open('xb') as stream:stream.write(interpretation.encode())
        receipt=dict(status='PROSPECTIVE_MODE_SELECTION_BEFORE_SUBSEQUENT_DATA',created_utc=utc(),
            selected_mode=MODE,execution_scope=SCOPE,**decision,driver_sha256=sha(__file__),
            frozen_framework_sha256=FRAMEWORK_SHA,frozen_stage_runner_sha256=module.sha(framework.STAGE_PATH),
            approval_sha256=module.sha(module.FOLDER/'approval_receipt.json'),
            preparation_manifest_sha256=module.sha(module.FOLDER/'preparation_manifest.json'),
            frozen_plan_sha256=module.sha(module.FOLDER/'plan.json'),
            frozen_cases_sha256=module.sha(module.FOLDER/'frozen_cases.json'),
            selected_case_ids=sorted({module.case_id(item['method'],item['H_um']) for item in jobs if item['method']=='NN'}),
            prior_ledger_snapshot_sha256=hashlib.sha256(data).hexdigest(),selected_jobs=jobs,
            common_completed_evidence=common,budget_before_selection=estimate,stage1_delivery_before=unchanged,
            original_controls_budget_effect=dict(common_completed_attempts=14,original_control_calls=6,
                projected_new_attempts=20,projected_cumulative_attempts=65,adaptive_remaining=0),
            unchanged_deadline=True,unchanged_thresholds=True,physics_source_changes=False,
            adaptive_launch_authorized_by_this_receipt=False,adaptive_reserve=0,
            separate20cap_selection_after_common=True,frozen18cap_driver_unchanged=True,
            frozen_v1_actual_reply_driver_unchanged=True,separately_optimized_direct_launched=False,
            full_optical_acceptance_claimed=False)
        validate_request_receipt(receipt,module,folder)
        path=folder/'selection_receipt.json';write_new(path,receipt);digest=sha(path)
        link=dict(mode=MODE,receipt_file=str(path.relative_to(module.FOLDER)),receipt_sha256=digest,driver_sha256=receipt['driver_sha256'])
        stage.ledger.setdefault('original_controls_selection',link)
        stage.ledger.setdefault('subsequent_phase_runs',[]).append(link)
        stage.save();execute_controls(stage,module,framework,jobs,path,digest)
        print(json.dumps(dict(status=stage.ledger['status'],selection_receipt=str(path),new_attempts=len(stage.ledger['entries']),adaptive_reserve=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--user-request-file',type=Path)
    parser.add_argument('--root-scope-interpretation')
    args=parser.parse_args()
    if args.run and not args.root_scope_interpretation:
        parser.error('--run requires --root-scope-interpretation; authority is the immutable original user request')
    framework,module=load_framework()
    if args.run:run_controls(framework,module,args.user_request_file,args.root_scope_interpretation);return
    try:
        stage=module.Stage();jobs,estimate,_=preflight(stage,module,framework)
        print(json.dumps(dict(status='INSPECTION_ONLY_ORIGINAL_CONTROLS_READY',jobs=jobs,budget=estimate,worker_launches=0),indent=2))
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        print(json.dumps(dict(status='INSPECTION_ONLY_ORIGINAL_CONTROLS_NOT_READY',reason=str(error),worker_launches=0)))


if __name__=='__main__':main()
