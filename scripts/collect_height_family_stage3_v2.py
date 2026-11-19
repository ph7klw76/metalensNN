"""Authenticate saved stage1/2/3 NN evidence; no execution or ledger writes."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from assess_height_family_stage3_v1 import HEIGHTS,load_science,aggregate
from compare_height_family_nn_direct_v2 import (read,sha,local_path,contained,finite,close,
    fingerprint,core_angles,NATIVE_KEYS,OPTICAL_KEYS,NUMERICS,safe_json)

AUTH_SHA='51249da9f94d4eb066ddf028e55c145f80f01cedd5a2ae714f5eb3dc10e3413c'
PROPOSAL_SHA='49f5301f2d7167d039bab39a086d4866f413e3e3274e49e1aff6e2d3accb90b0'
ORIGINAL_START='2026-11-13T08:59:13.107420+00:00'
DEADLINE='2026-11-15T08:59:13.107420+00:00'
PRIMARY_MODE='stage3_primary_mesh_refinement'
PRIMARY_SCOPE='APPROVED_EIGHTEEN_AMENDMENT_PRIMARY_FROZEN_NN_ONLY_ORIGINAL_DEADLINE'
ADAPTIVE_MODE='stage3_adaptive_isolated_pairs'
ADAPTIVE_SCOPE='APPROVED_FOUR_ADAPTIVE_ATTEMPTS_EXISTING_NN_ISOLATED_NUMERICS_ORIGINAL_DEADLINE'
PRIMARY_DRIVER_SHA='b28ffe701b3fb452cf6d3c04fb049d77afc8c6ed4eeb450da513e338717bba73'
# Exact root-frozen and independently reviewed source; every replay rehashes it.
REVIEWED_ADAPTIVE_DRIVER_SHA='f252d2061bf443b0703f904a892d321f56356c2b963ebc2d252c8b702d050c68'
REVIEWED_ASSESSORS={
    'assess_height_family_stage3_v1.py':'77dae64d7a81f78ec9e338e2d3e1c576756a0aa7c91977ecbba89cc87db2d08f',
    'assess_height_family_stage3_v2.py':'96dd3849f2d3c5997d6c05524ac5503452694cd78819619cee766a878d1f7b4c'}


def adaptive_dimensions(e,job):
    """Frozen pure setup helper; its module explicitly imports no Meep/GPU."""
    worker=ROOT/'scripts/height_family_maxwell_worker_v2.py'
    if e.protect(worker)!=e.sources['scripts/height_family_maxwell_worker_v2.py']:raise ValueError('Frozen setup source changed')
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('stage3_adaptive_pure_dimensions',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.configured_dimensions(e.config,job['H_um'],job['resolution'],pml=job['pml_um'],padding=job['padding_um'],duration=job['duration'],m=1)


def normalized_argv(args):
    if not isinstance(args,list) or len(args)<3:raise ValueError('Full actual worker argv required')
    result=list(args)
    for index in (0,2):result[index]=str(local_path(args[index]).resolve())
    paths={'--config','--output','--design','--reference'}
    numbers={'--H','--resolution','--pml','--padding','--duration'}
    for index,token in enumerate(args):
        if token in paths|numbers:
            if args.count(token)!=1 or index+1>=len(args):raise ValueError('One complete worker argument required')
            result[index+1]=str(local_path(args[index+1]).resolve()) if token in paths else str(float(args[index+1]))
    return result


def reviewed_assessment_proof(e,request,document,ledger_sha,snapshot):
    """Reuse frozen f252 read-only proof policy; never construct/mutate Stage."""
    path=ROOT/'scripts/run_height_family_adaptive_v1.py'
    if e.protect(path)!=REVIEWED_ADAPTIVE_DRIVER_SHA:raise ValueError('Reviewed adaptive proof source changed')
    spec=importlib.util.spec_from_file_location('stage3_reviewed_adaptive_proof_helpers',path)
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    def cache(job):
        entry=e.entries.get(('stage3',job['proposed_id']))
        if entry is None or entry.get('status')!='completed' or identity(entry)!=identity(job):return None
        rawpath,raw=e.raw('stage3',entry)
        if entry['method']=='NN':
            analysis=contained(e.folder,entry['analysis_file'])
            if entry.get('analysis_status')!='completed' or e.protect(analysis)!=entry['analysis_sha256']:
                raise ValueError('Primary analysis cache proof changed')
        return {'entry':entry,'stage_folder':str(e.folder),'record':raw,'path':rawpath,'stage_id':'stage3'}
    catalog=getattr(e,'catalog',[{'stage_id':sid,'stage_name':e.stages[sid]['stage_name'],'entry':entry}
                               for (sid,_),entry in e.entries.items() if sid!='stage3'])
    stage=types.SimpleNamespace(plan=e.plan,catalog=catalog,cache=cache)
    scope=dict(driver.assessment_proof.__globals__)
    # Only the read-only artifact-folder dependency is supplied. The imported
    # base module and all primary-controller globals remain unchanged.
    scope['base']=types.SimpleNamespace(FOLDER=e.folder)
    proof=types.FunctionType(driver.assessment_proof.__code__,scope,driver.assessment_proof.__name__,driver.assessment_proof.__defaults__)
    proof(request,stage,document,ledger_sha,snapshot)
    for key in ('input_sha256','analysis_source_sha256'):
        if key not in document:continue
        for source,digest in driver.normalized_inventory(document[key]).items():
            if e.protect(source)!=digest:raise ValueError('Reviewed assessment input changed after proof')
    return document


def adaptive_command(e,entry,job):
    label=f"{entry['id']:02d}_{job['method']}_H{job['H_um']:g}_r{job['resolution']:g}_ADAPTIVE_{job['job_type']}"
    output=e.folder/'records'/(label+'.json')
    expected=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
        '--config',str(e.folder/'config.json'),'--output',str(output),'--H',str(job['H_um']),
        '--resolution',str(job['resolution']),'--pml',str(job['pml_um']),'--padding',str(job['padding_um']),'--duration',str(job['duration'])]
    cases=[c for c in e.cases if c['case_id']==entry.get('paired_NN_case_id')]
    if len(cases)!=1:raise ValueError('One already charged paired NN profile required')
    case=cases[0]
    if any(entry.get('paired_NN_'+k)!=case[k] for k in ('case_id','design_sha256','physical_geometry_id')):raise ValueError('Paired frozen geometry differs')
    if job['method']=='NN':
        if any(entry.get(k)!=case[k] for k in ('case_id','design_sha256','physical_geometry_id')):raise ValueError('NN frozen geometry differs')
        sid=entry.get('reference_stage_id');ref=e.entries.get((sid,entry.get('reference_id')))
        if ref is None or ref.get('method')!='reference' or ref.get('status')!='completed' or identity(ref)[1:]!=identity(job)[1:]:
            raise ValueError('Actual adaptive reference origin/settings are not completed and matching')
        reference=contained(e.stages[sid]['folder'],ref['record_file'])
        expected+=['--design',str(contained(e.folder,case['design_file'])),'--reference',str(reference)]
    elif job['method']=='reference':
        if any(entry.get(k) is not None for k in ('case_id','design_sha256','physical_geometry_id','reference_id')):
            raise ValueError('Homogeneous adaptive reference must not claim device geometry/reference')
    else:raise ValueError('Only reference/NN adaptive jobs allowed')
    if contained(e.folder,entry['record_file'])!=output or normalized_argv(entry.get('command'))!=normalized_argv(expected):
        raise ValueError('Actual adaptive worker argv/output differs from the exact selected job')


def copied_primary_assessment(e,path,receipt,request,snapshot):
    original=local_path(request['primary_assessment_file']).resolve()
    if not original.is_relative_to(e.folder/'assessment') or original==e.folder/'assessment':
        raise ValueError('Primary assessment must be an actual stage3 assessment/version output')
    if e.protect(original)!=receipt.get('primary_assessment_sha256'):
        raise ValueError('Actual original primary assessment differs from prospective copy')
    utility=local_path(request['primary_assessment_utility_file']).resolve()
    expected=REVIEWED_ASSESSORS.get(utility.name)
    if utility!=ROOT/'scripts'/utility.name or expected is None or e.protect(utility)!=expected:
        raise ValueError('Actual reviewed primary assessor path/SHA required')
    if receipt.get('primary_assessment_utility_sha256')!=expected or request.get('primary_assessment_utility_sha256')!=expected:
        raise ValueError('Assessor proof differs from reviewed utility')
    if e.protect(path.parent/'assessment_source_snapshots/reviewed_assessor.py')!=expected:raise ValueError('Copied reviewed assessor source differs')
    copied=path.parent/'primary_assessment.json'
    if e.protect(copied)!=receipt.get('primary_assessment_sha256') or request.get('primary_assessment_sha256')!=receipt.get('primary_assessment_sha256'):
        raise ValueError('Copied primary assessment SHA differs')
    snapshot_sha=receipt['prior_ledger_snapshot_sha256']
    if request.get('primary_assessment_ledger_sha256')!=snapshot_sha:
        raise ValueError('Primary assessment snapshot differs from prospective selection snapshot')
    return reviewed_assessment_proof(e,request,e.document(copied),snapshot_sha,snapshot)


def validate_adaptive_pair(e,pair,requested):
    if not isinstance(pair.get('reason'),str) or not pair['reason'].strip() or pair.get('reason')!=requested.get('reason'):
        raise ValueError('Exact prospective per-pair scientific reason required')
    descriptor=pair['baseline'];sid=descriptor.get('stage_id');baseline=e.entries.get((sid,descriptor.get('record_id')))
    if descriptor!=requested.get('baseline') or pair.get('pair_id')!=requested.get('pair_id'):
        raise ValueError('Selected baseline/pair differs from exact root request')
    if baseline is None or baseline.get('method')!='NN' or baseline.get('status')!='completed' or baseline.get('analysis_status')!='completed':
        raise ValueError('Completed frozen NN baseline required')
    cases=[c for c in e.cases if close(c['H_um'],baseline['H_um'])]
    if len(cases)!=1 or baseline.get('design_sha256')!=cases[0]['design_sha256']:
        raise ValueError('Adaptive baseline geometry is outside the frozen NN cohort')
    rawpath,raw=e.raw(sid,baseline);ap=contained(e.stages[sid]['folder'],baseline['analysis_file'])
    if e.protect(rawpath)!=pair.get('baseline_record_sha256') or e.protect(ap)!=pair.get('baseline_analysis_sha256'):
        raise ValueError('Baseline raw/analysis proof differs')
    analysis=e.document(ap)
    if not close(analysis.get('angular_second_moment_ratio'),pair.get('baseline_Q')) or not close(analysis.get('normalized_curvature'),pair.get('baseline_C')):
        raise ValueError('Baseline Q/C diagnostics differ from unchanged analysis')
    minimum=max([j['resolution'] for j in e.plan['primary_jobs'] if j['method']=='NN' and close(j['H_um'],baseline['H_um'])] or [32.])
    if baseline['resolution']<minimum:raise ValueError('Adaptive baseline cannot fall below primary fine resolution')
    jobs=pair.get('jobs',[])
    if len(jobs)!=2 or [j.get('method') for j in jobs]!=['reference','NN'] or identity(jobs[0])[1:]!=identity(jobs[1])[1:]:
        raise ValueError('Exact matching reference/NN pair required')
    requested_jobs=requested.get('jobs',[])
    if (len(requested_jobs)!=2 or any(set(j)!={'method',*NUMERICS} for j in requested_jobs)
        or any(set(j)!={'method',*NUMERICS,'job_type','phase','pair_id'} for j in jobs)
        or any(any(job.get(k)!=original.get(k) for k in ('method',*NUMERICS)) for job,original in zip(jobs,requested_jobs))):
        raise ValueError('Selected jobs differ from exact root-requested jobs')
    changes=[k for k in ('resolution','pml_um','padding_um','duration') if not close(jobs[1].get(k),baseline.get(k))]
    if len(changes)!=1 or pair.get('changed_variable')!=changes[0] or requested.get('changed_variable')!=changes[0] or not close(jobs[1]['H_um'],baseline['H_um']):
        raise ValueError('Exactly one declared numerical variable may change on the same frozen H')
    changed=changes[0];old=baseline[changed];new=jobs[1][changed]
    if not finite(new) or new<=old:raise ValueError('Adaptive numerical setting must increase')
    allowed=(new in (40.,48.) if changed=='resolution' else new in (3.,3.5) if changed=='pml_um'
             else new in (15.,20.) if changed=='padding_um' else close(new/old,1.5) or close(new/old,2.))
    if not allowed:raise ValueError('Adaptive numerical value exceeds reviewed bounds')
    kind={'resolution':'refinement','pml_um':'pml','padding_um':'padding','duration':'duration'}[changed]
    if any(j.get('job_type')!=kind or j.get('phase')!='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC' or j.get('pair_id')!=pair.get('pair_id') for j in jobs):
        raise ValueError('Adaptive labels hide the changed numerical variable')
    effective=adaptive_dimensions(e,jobs[1])
    if pair.get('effective_settings')!=effective:raise ValueError('Selected effective dimensions differ from frozen pure setup')
    matches=[]
    for key,ref in e.entries.items():
        if ref.get('method')=='reference' and ref.get('status')=='completed':
            candidate=contained(e.stages[key[0]]['folder'],ref['record_file'])
            if candidate==local_path(raw.get('reference_file','')).resolve():matches.append((key[0],ref))
    if len(matches)!=1:raise ValueError('One authenticated baseline reference origin required')
    refsid,ref=matches[0];_,reference=e.raw(refsid,ref)
    if refsid!=pair.get('baseline_reference_stage') or ref['id']!=pair.get('baseline_reference_id') or reference['settings']!=raw['settings']:
        raise ValueError('Baseline reference proof differs')
    if not finite(reference.get('incident_power')) or reference['incident_power']<=0 or not finite(reference.get('power_balance_error')) or not 0<=reference['power_balance_error']<.05:
        raise ValueError('Baseline reference native normalization is invalid')
    scale=(effective['cell_radial_extent_um']/raw['settings']['cell_radial_extent_um'])*(effective['cell_axial_extent_um']/raw['settings']['cell_axial_extent_um'])*(effective['resolution']/raw['settings']['resolution'])**3*effective['duration']/raw['settings']['duration']
    forecasts=pair.get('forecast',[])
    if len(forecasts)!=2:raise ValueError('Two measured worker forecasts required')
    total=0.
    for source,origin,job,row in zip((ref,baseline),(refsid,sid),jobs,forecasts):
        predicted=source['wrapper_wall_s']*scale*1.15;allowance=120. if job['method']=='NN' else 0.
        if not finite(predicted) or not 0<predicted<=21600:raise ValueError('Measured forecast exceeds worker bound')
        if (row.get('method')!=job['method'] or row.get('baseline_stage')!=origin or row.get('baseline_id')!=source['id']
            or not all(close(row.get(k),v) for k,v in [('measured_wrapper_s',source['wrapper_wall_s']),('scale',scale),('margin',1.15),('worker_forecast_s',predicted),('analysis_allowance_s',allowance)])):
            raise ValueError('Measured forecast provenance/formula differs')
        total+=predicted+allowance
    if not close(pair.get('forecast_s'),total) or not finite(pair.get('forecast_ceiling_s')) or pair['forecast_ceiling_s']<total or requested.get('forecast_ceiling_s')!=pair['forecast_ceiling_s']:
        raise ValueError('Prospective measured forecast/explicit ceiling differs')
    return total


def adaptive_attempt_receipt(e,entry):
    if REVIEWED_ADAPTIVE_DRIVER_SHA is None:
        return checked({},['Adaptive driver has not completed independent source review/freeze'])
    try:
        if not 80<=entry.get('id',0)<=83 or entry.get('phase')!='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC':
            raise ValueError('Adaptive jobs occupy only IDs80..83 with explicit phase labels')
        links=[r for r in e.ledger.get('adaptive_selections',[]) if r.get('receipt_file')==entry.get('selection_receipt_file') and r.get('receipt_sha256')==entry.get('selection_receipt_sha256')]
        if len(links)!=1:raise ValueError('One registry-linked prospective adaptive selection required')
        link=links[0];path=contained(e.folder,link['receipt_file'])
        if e.protect(path)!=link['receipt_sha256']:raise ValueError('Adaptive receipt changed')
        receipt=e.document(path)
        expected={'status':'PROSPECTIVE_STAGE3_ADAPTIVE_SELECTION_BEFORE_DATA','selected_mode':ADAPTIVE_MODE,
            'execution_scope':ADAPTIVE_SCOPE,'decision_origin':'EXPLICIT_ROOT_SELECTION_REASON',
            'driver_sha256':REVIEWED_ADAPTIVE_DRIVER_SHA,'frozen_primary_controller_sha256':PRIMARY_DRIVER_SHA,
            'approval_receipt_sha256':AUTH_SHA,'approved_proposal_sha256':PROPOSAL_SHA,
            'adaptive_launch_authorized_by_this_receipt':True,'adaptive_attempt_cap':4,'new_attempt_cap':18,
            'cumulative_attempt_cap':83,'max_started_pairs':2,'adaptive_autoexecution':False,
            'original_first_started_utc':ORIGINAL_START,'original_absolute_deadline_utc':DEADLINE,
            'geometry_physics_thresholds_unchanged':True,'no_new_Q_gate':True,'no_coarse_to_fine_correction':True,'no_fully_steady_claim':True}
        if any(receipt.get(k)!=v for k,v in expected.items()) or entry.get('selection_scope')!=ADAPTIVE_SCOPE:
            raise ValueError('Adaptive reviewed authority, scope, clock or unchanged-gate flags differ')
        driver=ROOT/'scripts/run_height_family_adaptive_v1.py'
        if e.protect(driver)!=REVIEWED_ADAPTIVE_DRIVER_SHA or e.protect(path.parent/driver.name)!=REVIEWED_ADAPTIVE_DRIVER_SHA or link.get('driver_sha256')!=REVIEWED_ADAPTIVE_DRIVER_SHA:
            raise ValueError('Actual reviewed adaptive driver and copied source must match')
        for key,name in [('preparation_manifest_sha256','preparation_manifest.json'),('source_manifest_sha256','source_manifest.json'),('frozen_cases_sha256','frozen_cases.json')]:
            if receipt.get(key)!=e.protect(e.folder/name):raise ValueError('Frozen stage3 adaptive proof changed')
        request_path=path.parent/'selection_request_exact.json'
        if e.protect(request_path)!=receipt.get('request_sha256'):raise ValueError('Exact root adaptive request changed')
        request=e.document(request_path);reason=receipt.get('decision_text_exact')
        if (not isinstance(reason,str) or not reason.strip() or reason!=request.get('reason')
            or receipt.get('decision_text_sha256')!=hashlib.sha256(reason.encode()).hexdigest()
            or not isinstance(receipt.get('fine_time_priority_review'),str) or not receipt['fine_time_priority_review'].strip()
            or receipt['fine_time_priority_review']!=request.get('fine_time_priority_review')):
            raise ValueError('Exact prospective root reason and fine-time priority review required')
        snapshot_path=path.parent/'prior_ledger.json'
        if e.protect(snapshot_path)!=receipt.get('prior_ledger_snapshot_sha256') or receipt.get('primary_assessment_ledger_sha256')!=receipt.get('prior_ledger_snapshot_sha256'):
            raise ValueError('Prospective primary ledger snapshot proof differs')
        snapshot=e.document(snapshot_path);before=snapshot.get('entries',[])
        if (not 14<=len(before)<=16 or [r.get('id') for r in before]!=list(range(66,66+len(before)))
            or any(r.get('id')==entry['id'] for r in before) or 66+len(before)>entry['id']
            or snapshot.get('first_started_utc')!=ORIGINAL_START or snapshot.get('absolute_deadline_utc')!=DEADLINE):
            raise ValueError('Adaptive snapshot must precede this attempt and preserve the original clock/IDs')
        for job in e.plan['primary_jobs']:
            found=[r for r in before if r['id']==job['proposed_id']]
            current=e.entries.get(('stage3',job['proposed_id']))
            if (len(found)!=1 or found[0]!=current or found[0].get('status')!='completed' or identity(found[0])!=identity(job)
                or job['method']=='NN' and found[0].get('analysis_status')!='completed'):
                raise ValueError('All14 unchanged primary jobs/analyses must precede adaptive selection')
        copied_primary_assessment(e,path,receipt,request,snapshot)
        if utc(receipt['created_utc'])>utc(entry['started_utc']):raise ValueError('Adaptive selection was not prospective')
        pairs=receipt.get('selected_pairs',[]);requested=request.get('pairs',[])
        if not 1<=len(pairs)<=2 or len(requested)!=len(pairs) or len({p.get('pair_id') for p in pairs})!=len(pairs):
            raise ValueError('One or two distinct prospective pairs required')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',str(pair.get('pair_id',''))) for pair in pairs):raise ValueError('Safe explicit pair IDs required')
        if receipt.get('selected_jobs')!=[j for pair in pairs for j in pair.get('jobs',[])]:raise ValueError('Flattened selected jobs differ')
        targets=[identity(pair['jobs'][1]) for pair in pairs]
        if len(set(targets))!=len(targets):raise ValueError('Distinct pair IDs cannot reuse the same target numerical identity')
        if any(identity(old) in targets and old.get('method')=='NN' for old in before):raise ValueError('Adaptive target already existed before selection')
        total=0.;selected=[]
        for pair,original in zip(pairs,requested):
            total+=validate_adaptive_pair(e,pair,original)
            matches=[j for j in pair['jobs'] if j.get('method')==entry.get('method') and identity(j)==identity(entry)
                     and j.get('pair_id')==entry.get('pair_id') and j.get('job_type')==entry.get('job_type')]
            if matches:selected.append((pair,matches[0]))
            baseline=pair['baseline']
            if baseline['stage_id']=='stage3' and not any(r.get('id')==baseline['record_id'] for r in before):
                raise ValueError('Adaptive baseline was not in the prospective preselection snapshot')
        remaining=receipt.get('remaining_minus185s')
        max_remaining=(utc(DEADLINE)-utc(receipt['created_utc'])).total_seconds()-185.
        if (not close(receipt.get('selected_forecast_s'),total) or not finite(remaining) or not 0<remaining<=max_remaining+1e-6 or total>=remaining):
            raise ValueError('Measured selection forecast cannot fit the original UTC deadline/reporting reserve')
        runs=e.ledger.get('adaptive_pair_runs',[])
        if len(runs)>2 or len([r for r in e.ledger['entries'] if r.get('phase')=='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC'])>4:
            raise ValueError('Two-started-pair/four-attempt allocation exhausted')
        matching_runs=[r for r in runs if r.get('selection_sha256')==link['receipt_sha256'] and r.get('pair_id')==entry.get('pair_id')]
        if len(matching_runs)!=1 or not utc(receipt['created_utc'])<=utc(matching_runs[0]['started_utc'])<=utc(entry['started_utc']):
            raise ValueError('Each adaptive worker requires one explicit prospective pair-run record')
        if len(selected)!=1:raise ValueError('Worker does not match exactly one selected pair job')
        pair,job=selected[0]
        if entry.get('effective_settings')!=pair['effective_settings'] or entry.get('requested_settings')!={k:job[k] for k in NUMERICS}:
            raise ValueError('Adaptive requested/effective settings differ from selected pair')
        forecast=[r for r in pair['forecast'] if r['method']==entry['method']]
        if len(forecast)!=1 or entry.get('scheduling_forecast')!=forecast[0]:raise ValueError('Actual worker scheduling forecast differs')
        pending_s=forecast[0]['worker_forecast_s']+forecast[0]['analysis_allowance_s']
        if pending_s>=(utc(DEADLINE)-utc(entry['started_utc'])).total_seconds()-185.:
            raise ValueError('Pending measured forecast did not fit original UTC allowance at this worker start')
        adaptive_command(e,entry,job)
        return {**checked({'reviewed_adaptive_scope_and_provenance':True}),
                'receipt_file':str(path),'receipt_sha256':link['receipt_sha256'],'pair_id':pair['pair_id'],
                'changed_variable':pair['changed_variable'],'forecast_s':pair['forecast_s']}
    except (OSError,KeyError,TypeError,ValueError) as error:
        return checked({'valid_reviewed_adaptive_receipt':False},[str(error)])


def utc(value):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


def checked(checks,missing=()):
    return {'status':'FAIL' if any(value is False for value in checks.values()) else ('UNRESOLVED' if missing else 'PASS'),
        'checks':checks,'failed_checks':[key for key,value in checks.items() if value is False],
        'missing_evidence':list(missing)}


def identity(job):return (job.get('method'),*(float(job[key]) for key in NUMERICS))


def validate_primary_plan(plan,proposal):
    planned=plan.get('primary_jobs',[]);approved=proposal.get('primary_jobs',[])
    if len(planned)!=14 or len(approved)!=14 or [j.get('proposed_id') for j in planned]!=list(range(66,80)):
        raise ValueError('Exactly approved primary IDs66..79 required')
    for job,original in zip(planned,approved):
        if any(job.get(k)!=original.get(k) for k in ('proposed_id','method','job_type',*NUMERICS,'expected_actual_settings',
            'worker_argv','output_record','matching_reference_record','configuration_sha256','m')):
            raise ValueError('Prepared primary plan differs from actual approved proposal')
    if plan.get('adaptive_jobs') or plan.get('adaptive_autoexecution') is not False:
        raise ValueError('Adaptive selection cannot be inferred from the primary plan')


def validate_prior_links(approval,priors):
    matches=[p for p in priors.get('stages',[]) if p.get('stage_id')=='stage2']
    if len(matches)!=1:raise ValueError('One immutable approved stage2 origin required')
    prior=matches[0]
    if (prior.get('ledger_sha256')!=approval.get('prior_ledger_sha256')
        or prior.get('delivery_manifest_sha256')!=approval.get('prior_delivery_manifest_sha256')
        or priors.get('stage2_archive_sha256')!=approval.get('prior_archive_sha256')):
        raise ValueError('Prior ledger/manifest/archive must match actual user-approved seals')
    return prior


def command_binding(entry,job,folder,case=None):
    """Bind actual executable/argv/output to the actual approved job."""
    args=entry.get('command');expected=job.get('worker_argv');folder=Path(folder).resolve()
    if not isinstance(args,list) or not args or args!=expected:raise ValueError('Actual worker argv differs from approved argv')
    if len(args)<3 or local_path(args[0]).resolve()!=(ROOT/'.venv-meep/bin/python').resolve() or args[1]!='-B' or local_path(args[2]).resolve()!=(ROOT/'scripts/height_family_maxwell_worker_v2.py').resolve():
        raise ValueError('Approved native worker executable/source required')
    def value(flag):
        if args.count(flag)!=1:raise ValueError('One exact worker argument required: '+flag)
        index=args.index(flag)
        if index+1>=len(args):raise ValueError('Worker argument value missing: '+flag)
        return args[index+1]
    if local_path(value('--config')).resolve()!=folder/'config.json':raise ValueError('Actual configuration argv differs')
    output=contained(folder,entry.get('record_file'))
    if local_path(value('--output')).resolve()!=output or local_path(job.get('output_record','')).resolve()!=output:
        raise ValueError('Actual/approved output-record path differs')
    for flag,key in (('--H','H_um'),('--resolution','resolution'),('--pml','pml_um'),('--padding','padding_um'),('--duration','duration')):
        if not close(float(value(flag)),entry.get(key)):raise ValueError('Actual numerical argv differs: '+flag)
    if entry.get('method')=='reference':
        if '--design' in args or '--reference' in args:raise ValueError('Plain reference argv cannot contain device/reference inputs')
    elif entry.get('method')=='NN':
        if case is None or local_path(value('--design')).resolve()!=contained(folder,case['design_file']):
            raise ValueError('Actual NN profile argv differs from frozen geometry')
        if local_path(value('--reference')).resolve()!=local_path(job.get('matching_reference_record','')).resolve():
            raise ValueError('Actual reference argv differs from the approved pair')
    else:raise ValueError('Unapproved worker method')


def budget_assessment(approval,ledger,priors,*,now=None):
    now=datetime.now(timezone.utc) if now is None else now
    entries=ledger.get('entries',[]);missing=[]
    checks={'explicit_user_approval':approval.get('status')=='APPROVED_BY_USER' and approval.get('decision_origin')=='EXPLICIT_USER_REPLY',
        'additional18':approval.get('additional_attempt_ceiling')==18 and len(entries)<=18,
        'cumulative83':approval.get('cumulative_attempt_ceiling')==83 and 65+len(entries)<=83,
        'preserved_prior65':approval.get('prior_cumulative_attempts')==65 and priors.get('prior_attempt_count')==65 and ledger.get('prior_attempt_count')==65,
        'continued_IDs66': [entry.get('id') for entry in entries]==list(range(66,66+len(entries))),
        'no_new_physical_cases':approval.get('new_physical_cases_authorized')==0 and priors.get('actual_prior_physical_cases')==25,
        'one_worker_contract':approval.get('one_local_CPU_optical_worker') is True,
        'no_RAM_gates':approval.get('all_RAM_gates') is False,
        'no_time_extension':approval.get('time_extension_authorized') is False}
    elapsed=None
    try:
        anchors=[approval['original_first_started_utc'],priors['original_first_started_utc'],ledger['first_started_utc'],ledger['original_first_started_utc']]
        deadlines=[approval['original_absolute_deadline_utc'],priors['original_absolute_deadline_utc'],ledger['absolute_deadline_utc']]
        checks['immutable_original_start']=all(value==ORIGINAL_START for value in anchors)
        checks['immutable_original_deadline']=all(value==DEADLINE for value in deadlines)
        recorded=ledger.get('elapsed_s')
        checks['finite_nonnegative_recorded_elapsed']=finite(recorded) and recorded>=0
        actual=(now-utc(ORIGINAL_START)).total_seconds()
        elapsed=max(actual,recorded) if finite(recorded) else actual
        checks['all_idle_time_counts']=0<=actual and elapsed<=172800 and now<=utc(DEADLINE)
        for entry in entries:
            started=utc(entry['started_utc'])
            if not utc(ORIGINAL_START)<=started<=now or started>utc(DEADLINE):checks['attempt_timestamps_within_original_clock']=False
    except (KeyError,TypeError,ValueError):missing.append('Immutable original clock and attempt timestamps required')
    result=checked(checks,missing)
    result.update(prior_attempts=65,additional_attempts=len(entries),cumulative_attempts=65+len(entries),
        absolute_elapsed_s_including_idle=elapsed,original_first_started_utc=ORIGINAL_START,
        absolute_deadline_utc=DEADLINE,actual_cumulative_cases=25,reserved_case_ceiling=28)
    return result


class Evidence:
    def __init__(self,folder):
        self.folder=local_path(folder).resolve();self.inputs={};self.source_checks={}
        self.preparation=self.document(self.folder/'preparation_manifest.json')
        for name,digest in self.preparation.items():
            if self.protect(contained(self.folder,name))!=digest:raise ValueError('Stage3 preparation bytes changed: '+name)
        self.approval=self.document(self.folder/'approval_receipt.json')
        if self.protect(self.folder/'approval_receipt.json')!=AUTH_SHA:raise ValueError('Actual stage3 user approval changed')
        if self.protect(self.folder/'authorization/approval_receipt.json')!=AUTH_SHA:raise ValueError('Authorization copy changed')
        for stem in ('question','user_reply'):
            path=self.folder/'authorization'/(stem+'_exact.txt');data=path.read_bytes();self.protect(path)
            if hashlib.sha256(data).hexdigest()!=self.approval[stem+'_sha256'] or data.decode('utf-8')!=self.approval[stem+'_exact']:
                raise ValueError('Exact approval question/reply bytes differ')
        self.proposal=self.document(self.folder/'approved_proposal.json')
        if self.protect(self.folder/'approved_proposal.json')!=PROPOSAL_SHA or self.approval.get('approved_proposal_sha256')!=PROPOSAL_SHA:
            raise ValueError('Approved proposal changed')
        for key in ('unchanged_physics','unchanged_designs','unchanged_acceptance_thresholds','idle_and_approval_wait_count'):
            if self.approval.get(key) is not True:raise ValueError('Required unchanged scope missing: '+key)
        self.priors=self.document(self.folder/'prior_stages.json')
        approved_prior=validate_prior_links(self.approval,self.priors)
        if self.protect(local_path(self.priors['stage2_archive_file']))!=self.approval['prior_archive_sha256']:
            raise ValueError('Approved stage2 archive bytes changed')
        if approved_prior.get('final_assessment_sha256')!=self.proposal['prerequisites']['final65_assessment_sha256']:
            raise ValueError('Prior assessment differs from actual approved proposal')
        self.plan=self.document(self.folder/'plan.json');self.cases=self.document(self.folder/'frozen_cases.json')['cases']
        validate_primary_plan(self.plan,self.proposal)
        self.sources=self.document(self.folder/'source_manifest.json');self.config=self.document(self.folder/'config.json')
        if self.protect(self.folder/'config.json')!=self.proposal['frozen_inputs']['configuration_sha256']:
            raise ValueError('Stage3 physical configuration differs')
        for name,digest in self.sources.items():
            if self.protect(ROOT/name)!=digest or self.protect(self.folder/'source_snapshots'/name)!=digest:
                raise ValueError('Frozen execution source changed: '+name)
        if len(self.cases)!=8 or any(c.get('new_case') is not False or c.get('method')!='NN' for c in self.cases):
            raise ValueError('Only eight already charged NN profiles are allowed')
        for case in self.cases:
            path=contained(self.folder,case['design_file'])
            if self.protect(path)!=case['design_sha256']:raise ValueError('Frozen stage3 profile changed')
            design=self.document(path)
            physical=fingerprint({'H_um':float(design['H_um']),'cylinder_diameter_um':design['cylinder_diameter_um'],
                                  'heights_um':design['heights_um'],'materials':self.config['materials']})
            if physical!=case['physical_geometry_id']:raise ValueError('Physical profile identity changed')
        self.stages={};self.entries={};self.prior_verified_files=0
        for prior in self.priors['stages']:
            sid=prior['stage_id'];path=local_path(prior['folder']).resolve()
            if sid not in ('stage1','stage2') or sid in self.stages:raise ValueError('Exactly two distinct prior origins required')
            ledger_path=local_path(prior['ledger_file']).resolve()
            if ledger_path!=path/'ledger.json' or self.protect(ledger_path)!=prior['ledger_sha256']:raise ValueError('Immutable prior ledger changed')
            ledger=self.document(ledger_path)
            manifest=self.document(path/'delivery_manifest.json')
            if self.protect(path/'delivery_manifest.json')!=prior['delivery_manifest_sha256']:raise ValueError('Prior seal changed')
            copied=self.document(self.folder/'prior_delivery_manifests'/(sid+'.json'))
            if copied!=manifest:raise ValueError('Copied prior delivery inventory differs')
            for name,row in manifest['files'].items():
                if self.protect(contained(path,name))!=row['sha256']:raise ValueError('Sealed prior payload changed: '+sid+':'+name)
                self.prior_verified_files+=1
            expected=45 if sid=='stage1' else 20
            if len(ledger['entries'])!=expected or prior['attempt_count']!=expected:raise ValueError('Prior attempt counts changed')
            self.stages[sid]={'folder':path,'ledger':ledger,'stage_name':path.name}
        if set(self.stages)!={'stage1','stage2'}:raise ValueError('Both immutable prior stages required')
        self.ledger_bytes=(self.folder/'ledger.json').read_bytes()
        self.ledger=json.loads(self.ledger_bytes.decode('utf-8-sig'))
        self.stages['stage3']={'folder':self.folder,'ledger':self.ledger,'stage_name':self.folder.name}
        for sid,stage in self.stages.items():
            expected_start={'stage1':1,'stage2':46,'stage3':66}[sid]
            if [e.get('id') for e in stage['ledger']['entries']]!=list(range(expected_start,expected_start+len(stage['ledger']['entries']))):
                raise ValueError('Duplicate/reset attempt IDs in '+sid)
            for entry in stage['ledger']['entries']:
                self.entries[sid,entry['id']]=entry
        self.catalog=self.document(self.folder/'cache_catalog.json')['records']
        self.raw_cache={};self.authorization_cache={}
    def protect(self,path):
        path=local_path(path).resolve();value=sha(path)
        if str(path) in self.inputs and self.inputs[str(path)]!=value:raise ValueError('Evidence changed between authenticated reads: '+str(path))
        self.inputs[str(path)]=value;return value
    def document(self,path):self.protect(path);return read(path)
    def raw(self,sid,entry):
        key=sid,entry['id']
        if key in self.raw_cache:return self.raw_cache[key]
        stage=self.stages[sid];path=contained(stage['folder'],entry['record_file'])
        if entry['status']!='completed' or self.protect(path)!=entry['record_sha256']:raise ValueError('Completed raw record/hash required')
        record=self.document(path)
        if record.get('config_fingerprint')!=fingerprint(self.config) or not all(close(record['settings'].get(k),entry.get(k)) for k in NUMERICS):
            raise ValueError('Actual configuration/settings differ from ledger')
        if sid=='stage3':
            if record['settings']!=entry.get('effective_settings') or entry.get('requested_settings')!={k:entry[k] for k in NUMERICS}:
                raise ValueError('Requested/effective/actual stage3 settings differ')
            if record.get('worker_sha256')!=self.sources['scripts/height_family_maxwell_worker_v2.py']:
                raise ValueError('Actual worker source differs')
            inventory=entry['required_file_sha256']
        else:
            rows=[r for r in self.catalog if r['stage_id']==sid and r['entry']['id']==entry['id']]
            if len(rows)!=1 or rows[0]['entry']['record_sha256']!=entry['record_sha256']:raise ValueError('Authenticated cache row missing')
            inventory=rows[0]['required_file_sha256']
        keys=['fields_file','power_audit_fields_file']+(['flux_data_file'] if 'incident_power' in record else [])
        required={str(local_path(record[key]).resolve()) for key in keys}
        hashes={str(local_path(name).resolve()):digest for name,digest in inventory.items()}
        if set(hashes)!=required:raise ValueError('Incomplete native field SHA inventory')
        for name,digest in hashes.items():
            if self.protect(name)!=digest:raise ValueError('Native field bytes changed')
        self.raw_cache[key]=(path,record);return path,record


def attempt_receipt(evidence,entry):
    """Independent primary prelaunch authorization, including failed attempts."""
    if entry.get('selection_mode')==ADAPTIVE_MODE:
        return adaptive_attempt_receipt(evidence,entry)
    e=evidence;checks={};missing=[]
    try:
        link=[r for r in e.ledger.get('selection_receipts',[]) if r.get('attempt_id')==entry['id']
              and r.get('receipt_file')==entry.get('selection_receipt_file') and r.get('receipt_sha256')==entry.get('selection_receipt_sha256')]
        if len(link)!=1:raise ValueError('One hash-linked prospective receipt required')
        link=link[0];path=contained(e.folder,link['receipt_file'])
        if e.protect(path)!=link['receipt_sha256']:raise ValueError('Prelaunch receipt hash mismatch')
        receipt=e.document(path)
        if receipt.get('selected_mode')!=PRIMARY_MODE:
            return checked({},['Adaptive receipt schema requires explicit independent review; attempts remain charged'])
        checks['primary_status']=receipt.get('status')=='PROSPECTIVE_STAGE3_PRIMARY_ATTEMPT_BEFORE_POPEN'
        checks['exact_mode_scope']=receipt.get('execution_scope')==PRIMARY_SCOPE and entry.get('selection_mode')==PRIMARY_MODE and entry.get('selection_scope')==PRIMARY_SCOPE
        checks['actual_user_approval']=receipt.get('decision_origin')=='EXPLICIT_USER_REPLY' and receipt.get('approval_receipt_sha256')==AUTH_SHA
        fields={'approved_proposal_sha256':'approved_proposal.json','preparation_manifest_sha256':'preparation_manifest.json',
            'frozen_plan_sha256':'plan.json','source_manifest_sha256':'source_manifest.json','frozen_cases_sha256':'frozen_cases.json','prior_stages_sha256':'prior_stages.json'}
        checks['all_prepared_hashes']=all(receipt.get(k)==e.protect(e.folder/name) for k,name in fields.items())
        jobs=receipt.get('selected_jobs',[])
        planned=[job for job in e.plan['primary_jobs'] if job.get('proposed_id')==entry['id']]
        checks['approved_exact_job']=len(jobs)==len(planned)==1 and jobs==planned and identity(jobs[0])==identity(entry) and entry.get('job_type')=='refinement'
        if len(planned)==1:
            checks['approved_full_effective_settings']=entry.get('effective_settings')==planned[0].get('expected_actual_settings')==receipt.get('effective_settings')
        checks['requested_settings']=receipt.get('requested_settings')==entry.get('requested_settings')=={k:entry[k] for k in NUMERICS}
        checks['attempt_ID_and_caps']=receipt.get('attempt_id')==entry['id'] and receipt.get('prior_attempts')==65 and receipt.get('new_attempt_cap')==18 and receipt.get('cumulative_cap')==83
        checks['unchanged_clock']=receipt.get('original_first_started_utc')==ORIGINAL_START and receipt.get('original_absolute_deadline_utc')==DEADLINE
        checks['unchanged_physics_no_adaptive_autoexecution']=receipt.get('geometry_physics_thresholds_unchanged') is True and receipt.get('adaptive_autoexecution') is False
        checks['receipt_precedes_attempt']=utc(receipt['created_utc'])<=utc(entry['started_utc'])
        snapshot_path=path.parent/'prior_ledger.json'
        checks['prior_snapshot_hash']=e.protect(snapshot_path)==receipt.get('prior_ledger_snapshot_sha256')
        snapshot=e.document(snapshot_path)
        checks['attempt_absent_from_snapshot']=not any(r.get('id')==entry['id'] for r in snapshot.get('entries',[])) and len(snapshot.get('entries',[]))==entry['id']-66
        checks['snapshot_IDs_continue66']=[r.get('id') for r in snapshot.get('entries',[])]==list(range(66,entry['id']))
        checks['original_clock_in_snapshot']=snapshot.get('first_started_utc')==ORIGINAL_START and snapshot.get('absolute_deadline_utc')==DEADLINE
        drivers=[p for p in path.parent.glob('*.py') if e.protect(p)==receipt.get('driver_sha256')]
        checks['driver_snapshot']=len(drivers)==1 and receipt.get('driver_sha256')==link.get('driver_sha256')==e.sources.get('scripts/run_height_family_maxwell_v3.py')
        cases=[c for c in e.cases if c['case_id']==entry.get('paired_NN_case_id')]
        checks['paired_frozen_profile']=len(cases)==1 and all(receipt.get('paired_NN_'+k)==entry.get('paired_NN_'+k)==cases[0].get(k)
            for k in ('case_id','design_sha256','physical_geometry_id'))
        if entry.get('method')=='NN':
            checks['device_profile']=len(cases)==1 and all(receipt.get(k)==entry.get(k)==cases[0].get(k) for k in ('case_id','design_sha256','physical_geometry_id'))
        else:
            checks['homogeneous_reference_not_a_device']=entry.get('method')=='reference' and all(receipt.get(k) is None and entry.get(k) is None for k in ('case_id','design_sha256','physical_geometry_id'))
            checks['reference_has_no_reference']=receipt.get('reference') is None and entry.get('reference_id') is None
        if len(planned)==1:
            try:
                command_binding(entry,planned[0],e.folder,cases[0] if len(cases)==1 else None)
                checks['approved_actual_argv_and_output']=True
            except (OSError,KeyError,TypeError,ValueError) as error:
                checks['approved_actual_argv_and_output']=False;missing.append(str(error))
        result=checked(checks,missing)
        return {**result,'receipt_file':str(path),'receipt_sha256':link['receipt_sha256']}
    except (OSError,KeyError,TypeError,ValueError) as error:
        return checked({'valid_receipt':False},[str(error)])


def new_attempt(entry,folder):
    return {'identity':folder.name+':'+str(entry['id']),'stage_folder':str(folder),**{key:entry.get(key) for key in
        ('id','method','H_um','resolution','job_type','status','record_file','analysis_status','pml_um','padding_um',
         'duration','phase','case_id','design_sha256','reference_id','reference_stage','selection_receipt_file',
         'selection_receipt_sha256','started_utc')}}


def validate_runtime_inputs(raw,folder,method,design_path=None,reference_path=None):
    if local_path(raw.get('requested_config','')).resolve()!=Path(folder)/'config.json' or raw.get('solver')!='CPU Meep 1.35.0':
        raise ValueError('Exact immutable configuration path and native solver required')
    if method=='reference':
        incident=raw.get('incident_power');closure=raw.get('power_balance_error')
        if ('design_file' in raw or 'reference_file' in raw or not finite(incident) or incident<=0
            or not finite(closure) or not 0<=closure<.05):
            raise ValueError('Homogeneous native reference must have positive incident power, closure and no device inputs')
    elif method=='NN':
        if ('incident_power' in raw or local_path(raw.get('design_file','')).resolve()!=Path(design_path).resolve()
            or local_path(raw.get('reference_file','')).resolve()!=Path(reference_path).resolve()):
            raise ValueError('Actual NN design/reference input paths differ from the frozen files')
    else:raise ValueError('Only reference and frozen NN methods are authorized')


def new_record(e,entry):
    path,raw=e.raw('stage3',entry)
    if entry.get('analysis_status')!='completed':raise ValueError('Completed independent saved-field analysis required')
    cases=[c for c in e.cases if c['case_id']==entry.get('case_id')]
    if len(cases)!=1 or entry['design_sha256']!=cases[0]['design_sha256']:raise ValueError('Unchanged frozen NN geometry required')
    settings=raw['settings'];sid=entry.get('reference_stage_id');refentry=e.entries.get((sid,entry.get('reference_id')))
    if refentry is None or refentry.get('method')!='reference':raise ValueError('Authenticated reference origin/ID required')
    refpath,reference=e.raw(sid,refentry)
    validate_runtime_inputs(raw,e.folder,'NN',e.folder/cases[0]['design_file'],refpath)
    if (local_path(raw.get('reference_file','')).resolve()!=refpath or reference['settings']!=settings
        or local_path(entry.get('reference_stage_folder','')).resolve()!=e.stages[sid]['folder']):
        raise ValueError('Reference path, origin or full settings differ')
    ap=contained(e.folder,entry['analysis_file'])
    if e.protect(ap)!=entry['analysis_sha256']:raise ValueError('Independent analysis SHA mismatch')
    saved=e.document(ap)
    if saved.get('settings')!=settings or local_path(saved.get('record','')).resolve()!=path:raise ValueError('Analysis names different settings/record')
    design=e.document(e.folder/cases[0]['design_file'])
    if not close(saved.get('declared_target_diameter_um'),design['target_diameter_um']):raise ValueError('Frozen declared target differs')
    receipt=e.document(e.folder/entry['selection_receipt_file'])
    if entry.get('selection_mode')!=ADAPTIVE_MODE:
        proof=receipt.get('reference') or {}
        if (proof.get('stage_id')!=sid or proof.get('id')!=refentry['id'] or proof.get('record_sha256')!=refentry['record_sha256']
            or local_path(proof.get('record_file','')).resolve()!=refpath or not close(proof.get('native_closure'),reference.get('power_balance_error'))):
            raise ValueError('Prospective reference proof differs from actual native reference')
    native=saved.get('numerical_checks',{})
    checks={key:native.get(key) is True for key in NATIVE_KEYS}
    missing=['Native analysis check '+key for key in NATIVE_KEYS if key not in native]
    for label,record in [('reference',reference),('device',raw)]:
        value=record.get('power_balance_error');checks[label+'_power_closure']=finite(value) and 0<=value<.05
        if not finite(value):missing.append('Finite '+label+' closure')
    incident=reference.get('incident_power');checks['positive_incident_reference']=finite(incident) and incident>0
    if not finite(incident):missing.append('Finite native incident power')
    planes=[]
    for z in e.config['evaluation']['processing_planes_um']:
        matches=[p for p in saved.get('metrics',{}).get('planes',[]) if close(p.get('z_um'),z)]
        p=matches[0] if len(matches)==1 else {};valid=p.get('valid_crossings')==[True,True]
        planes.append({'z_um':z,'diameter_x_um':p.get('diameter_x_um') if valid and finite(p.get('diameter_x_um')) else None,
                       'diameter_y_um':p.get('diameter_y_um') if valid and finite(p.get('diameter_y_um')) else None})
    return {'identity':e.folder.name+':'+str(entry['id']),'method':'NN','H_um':entry['H_um'],'resolution':entry['resolution'],
        'job_type':entry['job_type'],'phase':entry.get('phase'),'stage_folder':str(e.folder),
        'selection_receipt_file':entry['selection_receipt_file'],'selection_receipt_sha256':entry['selection_receipt_sha256'],
        'record_file':str(path),'record_sha256':entry['record_sha256'],'analysis_file':str(ap),'analysis_sha256':entry['analysis_sha256'],
        'design_sha256':entry['design_sha256'],'reference_record_file':str(refpath),'reference_record_sha256':refentry['record_sha256'],
        'settings':settings,'native_checks':checks,'native_all_pass':all(checks.values()),'native_missing_evidence':missing,
        'optical_gates':{key:saved.get('optical_gates',{}).get(key) for key in OPTICAL_KEYS},
        'eta':saved.get('conservative_useful_efficiency'),'C':saved.get('normalized_curvature'),'Q':saved.get('angular_second_moment_ratio'),
        'planes':planes,'core_half_angles_deg':core_angles(saved.get('metrics',{}).get('fitted_core_divergence',{}))}


def adaptive_progress(e,verified_devices):
    runs=e.ledger.get('adaptive_pair_runs',[]);rows=[]
    for run in runs:
        entries=[item for item in e.ledger['entries'] if item.get('pair_id')==run.get('pair_id') and item.get('selection_receipt_sha256')==run.get('selection_sha256')]
        devices=[item for item in entries if item.get('method')=='NN']
        failed=run.get('status')=='stopped_requires_diagnosis' or any(item.get('status') in ('failed','timeout','interrupted') for item in entries)
        complete=run.get('status')=='completed' and len(devices)==1 and devices[0]['id'] in verified_devices
        rows.append({'pair_id':run.get('pair_id'),'selection_sha256':run.get('selection_sha256'),
                     'status':'FAIL' if failed else 'PASS' if complete else 'UNRESOLVED'})
    pending=[];link=e.ledger.get('current_adaptive_selection')
    if link:
        path=contained(e.folder,link['receipt_file'])
        if e.protect(path)!=link['receipt_sha256']:raise ValueError('Current adaptive selection changed')
        receipt=e.document(path)
        completed={row['pair_id'] for row in rows if row['selection_sha256']==link['receipt_sha256'] and row['status']=='PASS'}
        pending=[pair['pair_id'] for pair in receipt.get('selected_pairs',[]) if pair['pair_id'] not in completed]
    state=aggregate({str(i):row for i,row in enumerate(rows)}) if rows else {'status':'PASS','components':{}}
    if pending and state['status']!='FAIL':state['status']='UNRESOLVED'
    state.update(started_pair_runs=rows,pending_current_selected_pairs=pending,
        reserved_but_unselected_attempts_are_not_required=True)
    return state


def selected_pair_diagnostics(e,data,limits,science):
    """Keep exact receipt-baseline comparisons separate from original controls."""
    records={r['identity']:r for r in data['Meep_records']};output=[]
    for run in e.ledger.get('adaptive_pair_runs',[]):
        links=[r for r in e.ledger.get('adaptive_selections',[]) if r['receipt_sha256']==run.get('selection_sha256')]
        if len(links)!=1:raise ValueError('Adaptive diagnostic selection registry missing')
        path=contained(e.folder,links[0]['receipt_file'])
        if e.protect(path)!=links[0]['receipt_sha256']:raise ValueError('Adaptive diagnostic receipt changed')
        receipt=e.document(path);pairs=[p for p in receipt['selected_pairs'] if p['pair_id']==run['pair_id']]
        if len(pairs)!=1:raise ValueError('One explicit adaptive diagnostic pair required')
        pair=pairs[0];baseline_id=e.stages[pair['baseline']['stage_id']]['stage_name']+':'+str(pair['baseline']['record_id'])
        targets=[entry for entry in e.ledger['entries'] if entry.get('method')=='NN' and entry.get('pair_id')==run['pair_id'] and entry.get('selection_receipt_sha256')==links[0]['receipt_sha256']]
        target_id=e.folder.name+':'+str(targets[0]['id']) if len(targets)==1 else None
        base=records.get(baseline_id);target=records.get(target_id)
        row={'pair_id':pair['pair_id'],'changed_variable':pair['changed_variable'],'baseline_identity':baseline_id,
             'target_identity':target_id,'receipt_sha256':links[0]['receipt_sha256'],
             'scope':'Exact prospectively selected baseline/target, in addition to unchanged original coarse controls'}
        if base is None or target is None:
            output.append({**row,'native_status':'UNRESOLVED','numerical_status':'UNRESOLVED','target_nominal_status':'UNRESOLVED',
                           'missing_evidence':['Completed verified baseline and adaptive target analyses required']});continue
        analyses=[]
        for record in (base,target):
            if e.protect(record['analysis_file'])!=record['analysis_sha256']:raise ValueError('Adaptive diagnostic analysis changed')
            analyses.append(e.document(record['analysis_file']))
        comparison=science.compare_analyses(*analyses,data['config'],limits,require_collimation=True)
        # Replay the unchanged v3 point-wise optical gates on this specific
        # target. This does not alter the preferred nominal-field selector.
        scope=dict(science.build_comparison.__globals__);scope.update(HEIGHTS=(target['H_um'],),EXCLUDED_VARIANTS=set())
        classify=types.FunctionType(science.build_comparison.__code__,scope,science.build_comparison.__name__,science.build_comparison.__defaults__)
        point_data=dict(data,Meep_records=[target],all_attempts=[])
        point=classify(point_data,limits)['rows'][0]
        base_flags=analyses[0].get('optical_gates',{});target_flags=analyses[1].get('optical_gates',{})
        native=base['native_all_pass'] and target['native_all_pass']
        output.append({**row,'native_status':'PASS' if native else 'FAIL','numerical_status':comparison['status'],
            'target_nominal_status':point['nominal_optical']['status'],'target_nominal_optical':point['nominal_optical'],
            'comparison':comparison,'baseline_settings':base['settings'],'target_settings':target['settings'],
            'baseline_saved_optical_flags':base_flags,'target_saved_optical_flags':target_flags,
            'baseline_Q':base['Q'],'target_Q':target['Q'],
            'Q_gate_crossing':base_flags.get('angular_second_moment')!=target_flags.get('angular_second_moment'),
            'Q_change_is_diagnostic_not_a_new_convergence_gate':True,
            'no_overall_optical_acceptance_inferred_from_execution_completion':True})
    return output


def collect_stage3(folder,*,now=None):
    """Return normalized evidence plus independent authorization and accounting."""
    e=Evidence(folder);science=load_science()
    for source in (Path(__file__),ROOT/'scripts/assess_height_family_stage3_v1.py',Path(science.__file__)):
        e.protect(source)
    old2=e.stages['stage2'];legacy=science.collect(old2['folder'],heights=HEIGHTS,methods=('NN',),include_variants=True)
    if fingerprint(legacy['config'])!=fingerprint(e.config):raise ValueError('Prior/stage3 physical configurations differ')
    if local_path(legacy['prior_stage_linkage']['folder']).resolve()!=e.stages['stage1']['folder']:
        raise ValueError('Stage1 origin differs from sealed stage2 linkage')
    original_sources=e.document(old2['folder']/'source_manifest.json')
    if any(e.sources.get(name)!=digest for name,digest in original_sources.items()):
        raise ValueError('Stage3 changes or omits previously frozen physics/execution sources')
    original_profiles={r['H_um']:r['design_sha256'] for r in legacy['proxy_records'] if r['method']=='NN'}
    if any(original_profiles.get(case['H_um'])!=case['design_sha256'] for case in e.cases):
        raise ValueError('Stage3 profiles differ from the original selected NN geometry')
    for name,digest in legacy['input_sha256'].items():
        if name in e.inputs and e.inputs[name]!=digest:raise ValueError('Sealed prior input changed during legacy collection')
        e.inputs[name]=digest
    stage2_link=next(p for p in e.priors['stages'] if p['stage_id']=='stage2')
    assessment_path=local_path(stage2_link['final_assessment_file']).resolve()
    if not assessment_path.is_relative_to(old2['folder']) or e.protect(assessment_path)!=stage2_link['final_assessment_sha256']:
        raise ValueError('Authenticated final stage2 assessment required')
    old_assessment=e.document(assessment_path)
    if old_assessment.get('protocol',{}).get('status')!='PASS':raise ValueError('Prior prospective-authorization protocol is not accepted')
    registry=[];old_records={r['identity']:r for r in legacy['Meep_records']}
    for row in old_assessment['boundary_duration_checks']:
        record=old_records.get(row.get('variation_identity'))
        if record is None:raise ValueError('Original coarse control field missing from sealed evidence')
        registry.append({'job_type':row['job_type'],'H_um':row['H_um'],'variation_identity':record['identity'],
                         'nominal_identity':row['nominal_identity'],'settings':record['settings'],
                         'sealed_assessment_file':str(assessment_path),'sealed_assessment_sha256':stage2_link['final_assessment_sha256']})
    if len(registry)!=4:raise ValueError('Four authenticated original controls required')
    records=list(legacy['Meep_records']);attempts=list(legacy['all_attempts']);missing=list(legacy['missing_evidence'])
    record_authorizations={r['identity']:{'status':'PASS','basis':'Immutable prior delivery and accepted prospective protocol'} for r in records}
    new_authorizations=[];verified_refs=set();verified_devices=set()
    for entry in e.ledger['entries']:
        attempt=new_attempt(entry,e.folder);attempts.append(attempt)
        authorization=attempt_receipt(e,entry);new_authorizations.append({'identity':attempt['identity'],**authorization})
        if entry.get('status')!='completed':continue
        try:
            if entry.get('method')=='reference':
                _,raw=e.raw('stage3',entry)
                validate_runtime_inputs(raw,e.folder,'reference')
                verified_refs.add(entry['id']);continue
            if entry.get('method')!='NN':raise ValueError('No new physical candidate/method is authorized')
            record=new_record(e,entry);records.append(record);verified_devices.add(entry['id'])
            ref_auth=({'status':'PASS'} if entry['reference_stage_id']!='stage3' else
                      attempt_receipt(e,e.entries['stage3',entry['reference_id']]))
            record_authorizations[record['identity']]=aggregate({'device':authorization,'reference':ref_auth})
        except (OSError,KeyError,TypeError,ValueError) as error:
            missing.append({'scope':'stage3','identity':attempt['identity'],'reason':str(error)})
            record_authorizations[attempt['identity']]={'status':'FAIL','reason':str(error)}
    budget=budget_assessment(e.approval,e.ledger,e.priors,now=now)
    known={case['case_id']:case for case in e.cases}
    only_frozen=all(entry.get('case_id') in known and entry.get('design_sha256')==known[entry['case_id']]['design_sha256']
                    for entry in e.ledger['entries'] if entry.get('method')=='NN')
    budget['checks']['only_previously_charged_frozen_devices']=only_frozen
    if not only_frozen:
        budget['status']='FAIL';budget['actual_cumulative_cases']=None
        budget['failed_checks'].append('only_previously_charged_frozen_devices')
    authorization=aggregate({r['identity']:r for r in new_authorizations}) if new_authorizations else {'status':'PASS','components':{}}
    # Source/record violations cannot disappear merely because a record was
    # excluded from plotting. Pending analyses are execution progress only.
    invalid=[r for r in missing if r.get('scope')=='stage3' and 'Completed independent saved-field analysis' not in r.get('reason','')]
    if invalid:authorization=aggregate({'prospective_receipts':authorization,'saved_record_integrity':'FAIL'})
    authorization.update(record_authorizations=record_authorizations,new_attempt_authorizations=new_authorizations,
        prior_sealed_payload_files_verified=e.prior_verified_files,prior_protocol_status=old_assessment['protocol']['status'])
    primary=[]
    for job in e.plan['primary_jobs']:
        matches=[entry for entry in e.ledger['entries'] if entry['id']==job['proposed_id'] and identity(entry)==identity(job)]
        verified=verified_refs if job['method']=='reference' else verified_devices
        primary.append({'proposed_id':job['proposed_id'],'completed_verified':len(matches)==1 and job['proposed_id'] in verified,
                        'status':matches[0]['status'] if len(matches)==1 else 'not_started'})
    failed=any(row['status'] in ('failed','timeout','interrupted') for row in primary)
    progress={'status':'FAIL' if failed else ('PASS' if len(primary)==14 and all(r['completed_verified'] for r in primary) else 'UNRESOLVED'),
              'primary_jobs':primary,'completed_verified':sum(r['completed_verified'] for r in primary),'required':14}
    adaptive=adaptive_progress(e,verified_devices)
    normalized=dict(legacy)
    normalized.update(folder=str(e.folder),Meep_records=records,all_attempts=attempts,missing_evidence=missing,
        budget=budget,original_boundary_definitions=registry,input_sha256=e.inputs,
        ledger_snapshot_sha256={**legacy['ledger_snapshot_sha256'],str(e.folder/'ledger.json'):hashlib.sha256(e.ledger_bytes).hexdigest()})
    limits=e.document(local_path(e.stages['stage1']['folder'])/'maxwell_validation_proposal.json')['numerical_acceptance']
    diagnostics=selected_pair_diagnostics(e,normalized,limits,science)
    if not all(sha(path)==digest for path,digest in e.inputs.items()):raise RuntimeError('Immutable input changed during stage3 collection')
    return {'data':safe_json(normalized),'numerical_limits':limits,
            'authorization':safe_json(authorization),'budget':budget,'primary_execution':progress,
            'adaptive_execution':adaptive,'selected_adaptive_diagnostics':safe_json(diagnostics)}
