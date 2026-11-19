"""Draft prospective isolated adaptive-pair controller; no automatic choices.

Default inspection is read only. After all14 primary jobs complete, ROOT may
provide a reason, authenticated assessment and exact pair jobs/forecast ceiling
to --prepare-selection. --run-next-pair explicitly runs at most ONE selected
pair. The original deadline,18/83 attempt caps, frozen NN-only cases and base
worker/analyzer sources remain unchanged. No primary globals are monkeypatched.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'scripts/run_height_family_maxwell_v3.py'
BASE_SHA='b28ffe701b3fb452cf6d3c04fb049d77afc8c6ed4eeb450da513e338717bba73'
MODE='stage3_adaptive_isolated_pairs'
SCOPE='APPROVED_FOUR_ADAPTIVE_ATTEMPTS_EXISTING_NN_ISOLATED_NUMERICS_ORIGINAL_DEADLINE'
NUMERICS=('resolution','pml_um','padding_um','duration')
COARSE_Q_TIME_EFFECT=.003522181559152626
ASSESSOR_CORE_SHA='77dae64d7a81f78ec9e338e2d3e1c576756a0aa7c91977ecbba89cc87db2d08f'
SCIENCE_SHA='aafa34734fbdd6c959278d935e43bfe0ae7e81cf877041b6380ac658d3d3d043'
REVIEWED_ASSESSORS={
    'assess_height_family_stage3_v1.py':ASSESSOR_CORE_SHA,
    'assess_height_family_stage3_v2.py':'96dd3849f2d3c5997d6c05524ac5503452694cd78819619cee766a878d1f7b4c'}
V1_COLLECTOR_SHA='957cfc2213fb9e4f708f42acb15537e04758471c13a20359b26737cf227c71b8'
HEIGHTS=(40.,50.,60.,70.,75.,80.,90.,100.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def local_path(value):
    """Resolve native Windows or WSL evidence paths on the current host."""
    value=str(value)
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()
def normalized_inventory(inventory):
    if not isinstance(inventory,dict) or not inventory:raise ValueError('Nonempty assessment integrity inventory required')
    result={}
    for name,digest in inventory.items():
        path=local_path(name)
        if path in result or not re.fullmatch('[a-f0-9]{64}',str(digest)):
            raise ValueError('Duplicate normalized path or invalid assessment inventory SHA')
        result[path]=digest
    return result
def load_base():
    if sha(BASE_PATH)!=BASE_SHA:raise ValueError('Frozen stage3 primary source changed')
    spec=importlib.util.spec_from_file_location('frozen_stage3_for_adaptive',BASE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
base=load_base()
def primary_complete(stage):
    for job in stage.plan['primary_jobs']:
        found=stage.cache(job)
        if found is None or found['stage_id']!='stage3' or found['entry']['status']!='completed':
            raise ValueError('All14 primary raw jobs must complete before adaptive selection')
        entry=found['entry']
        if job['method']=='NN' and (entry.get('analysis_status')!='completed'
                or sha(base.FOLDER/entry['analysis_file'])!=entry['analysis_sha256']):
            raise ValueError('Every primary independent analysis must be complete and SHA-verified')
def adaptive_entries(stage):return [e for e in stage.ledger['entries'] if e.get('phase')=='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC']
def pair_runs(stage):return stage.ledger.get('adaptive_pair_runs',[])
def frozen_reference(stage,item):
    found=stage.cache(dict(item,method='reference'))
    if found is None:raise ValueError('A matching verified baseline reference is required')
    error=found['record']['power_balance_error'];incident=found['record'].get('incident_power')
    if not math.isfinite(error) or not 0<=error<.05 or not isinstance(incident,(int,float)) or not math.isfinite(incident) or incident<=0:
        raise ValueError('Baseline/matching native reference closure or incident power invalid')
    return found
def baseline_record(stage,descriptor):
    if descriptor.get('stage_id')=='stage3':
        entries=stage.ledger['entries']
    else:entries=[row['entry'] for row in stage.catalog if row['stage_id']==descriptor.get('stage_id')]
    matches=[entry for entry in entries if entry['id']==descriptor.get('record_id') and entry['method']=='NN' and entry['status']=='completed']
    if len(matches)!=1:raise ValueError('Explicit completed NN baseline stage/id required')
    entry=matches[0];found=stage.cache(entry)
    if found is None or found['stage_id']!=descriptor['stage_id'] or found['entry']['id']!=entry['id']:
        raise ValueError('Baseline must be the latest SHA-verified evidence at its full settings')
    if entry.get('analysis_status')!='completed':raise ValueError('Completed baseline analysis required')
    origin=Path(found['stage_folder']);analysis_path=origin/entry['analysis_file']
    if sha(analysis_path)!=entry['analysis_sha256']:raise ValueError('Baseline analysis SHA changed')
    case=stage.case(entry['H_um'])
    if entry['design_sha256']!=case['design_sha256']:raise ValueError('Baseline frozen case differs')
    primary=[job['resolution'] for job in stage.plan['primary_jobs'] if job['method']=='NN' and job['H_um']==entry['H_um']]
    minimum=max(primary) if primary else 32.
    if entry['resolution']<minimum:raise ValueError('Adaptive baseline must be at least the completed primary fine grid')
    return found,frozen_reference(stage,entry),read(analysis_path)
def isolated_jobs(pair,stage):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',str(pair.get('pair_id',''))):raise ValueError('Safe explicit pair_id required')
    if not isinstance(pair.get('reason'),str) or not pair['reason'].strip():raise ValueError('Explicit per-pair scientific reason required')
    nn,ref,analysis=baseline_record(stage,pair['baseline']);baseline=nn['entry']
    jobs=pair.get('jobs')
    if not isinstance(jobs,list) or len(jobs)!=2 or [j.get('method') for j in jobs]!=['reference','NN']:
        raise ValueError('Exactly one reference/NN pair is required')
    jobs=[dict(job) for job in jobs]
    if any(set(job)!={'method','H_um','resolution','pml_um','padding_um','duration'} for job in jobs):
        raise ValueError('Root must supply exact requested settings only; no hidden job fields')
    if any(not isinstance(job[key],(int,float)) or isinstance(job[key],bool) or not math.isfinite(job[key]) or job[key]<=0
           for job in jobs for key in ('H_um',)+NUMERICS):raise ValueError('Positive finite requested settings required')
    if any(job['H_um']!=baseline['H_um'] for job in jobs) or base.identity(jobs[0])[1:]!=base.identity(jobs[1])[1:]:
        raise ValueError('Adaptive reference and NN must match the same frozen H/settings')
    changed=[key for key in NUMERICS if jobs[1][key]!=baseline[key]]
    if len(changed)!=1 or pair.get('changed_variable')!=changed[0]:raise ValueError('Exactly one declared numerical variable may change')
    changed=changed[0];value=float(jobs[1][changed]);old=float(baseline[changed])
    if not math.isfinite(value) or value<=old:raise ValueError('Bounded numerical diagnostic must increase its selected variable')
    if changed=='resolution' and value not in (40.,48.):raise ValueError('Higher-mesh choices are40 or48 only')
    if changed=='pml_um' and value not in (3.,3.5):raise ValueError('Bounded PML choices are3 or3.5um')
    if changed=='padding_um' and value not in (15.,20.):raise ValueError('Bounded padding choices are15 or20um')
    if changed=='duration' and not math.isclose(value/old,1.5,rel_tol=1e-10) and not math.isclose(value/old,2.,rel_tol=1e-10):
        raise ValueError('Bounded same-grid duration factors are1.5 or2 only')
    kind={'resolution':'refinement','pml_um':'pml','padding_um':'padding','duration':'duration'}[changed]
    effective=base.u.settings_for(stage.config,jobs[1])
    for job in jobs:job.update(job_type=kind,phase='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC',pair_id=pair['pair_id'])
    settings=nn['record']['settings'];scale=(effective['cell_radial_extent_um']/settings['cell_radial_extent_um'])*(
        effective['cell_axial_extent_um']/settings['cell_axial_extent_um'])*(effective['resolution']/settings['resolution'])**3
    scale*=effective['duration']/settings['duration']
    forecast=[]
    for job,source in zip(jobs,[ref,nn]):
        cost=float(source['entry']['wrapper_wall_s'])*scale*1.15
        if not math.isfinite(cost) or cost<=0 or cost>base.CHILD_MAX:raise ValueError('Measured scaled worker forecast must fit the6h cap')
        forecast.append(dict(method=job['method'],baseline_stage=source['stage_id'],baseline_id=source['entry']['id'],
            measured_wrapper_s=source['entry']['wrapper_wall_s'],scale=scale,margin=1.15,
            worker_forecast_s=cost,analysis_allowance_s=120. if job['method']=='NN' else 0.))
    total=sum(row['worker_forecast_s']+row['analysis_allowance_s'] for row in forecast)
    ceiling=pair.get('forecast_ceiling_s')
    if not isinstance(ceiling,(int,float)) or not math.isfinite(ceiling) or ceiling<total:
        raise ValueError('Explicit root forecast ceiling must cover the dynamically measured forecast')
    return dict(pair_id=pair['pair_id'],reason=pair['reason'],changed_variable=changed,jobs=jobs,effective_settings=effective,
        baseline=pair['baseline'],baseline_record_sha256=nn['entry']['record_sha256'],baseline_analysis_sha256=nn['entry']['analysis_sha256'],
        baseline_reference_stage=ref['stage_id'],baseline_reference_id=ref['entry']['id'],
        baseline_Q=analysis['angular_second_moment_ratio'],baseline_C=analysis['normalized_curvature'],
        forecast=forecast,forecast_s=total,forecast_ceiling_s=ceiling,
        padding_scope='Combined radial-domain/collection-radius diagnostic; not fixed-aperture boundary-only' if changed=='padding_um' else None)
def assessment_proof(request,stage,document,ledger_sha,snapshot):
    """Authenticate reviewed sources and evidence; optical PASS is not required.

    The v2 collector is intentionally not digest-pinned here: that collector
    pins this driver. Its exact current bytes must instead match the report's
    fixed-path source inventory, and both can be sealed by an external review
    manifest after this driver freezes, avoiding a circular hash dependency.
    """
    path=local_path(request['primary_assessment_file']);assessment_root=(base.FOLDER/'assessment').resolve()
    if (not path.is_relative_to(assessment_root) or len(path.relative_to(assessment_root).parts)<2
            or path.name!='assessment.json'):
        raise ValueError('Primary assessment must be a contained stage3/assessment/version/assessment.json')
    utility=local_path(request['primary_assessment_utility_file']);digest=REVIEWED_ASSESSORS.get(utility.name)
    if (digest is None or utility!=(ROOT/'scripts'/utility.name).resolve() or sha(utility)!=digest
            or request.get('primary_assessment_utility_sha256')!=digest
            or document.get('assessment_utility_sha256')!=digest):
        raise ValueError('Primary assessment requires the actual reviewed saved-analysis utility path/SHA')
    if (document.get('status')!='SAVED_THREE_STAGE_ASSESSMENT'
            or document.get('scientific_source_sha256')!=SCIENCE_SHA
            or document.get('scientific_function_bytecode_changed') is not False
            or document.get('reporting_height_enumeration')!=list(HEIGHTS)):
        raise ValueError('Reviewed assessment schema/scientific source/eight-height coverage differs')
    is_v2=utility.name.endswith('_v2.py')
    if is_v2 and (document.get('assessment_schema')!='height_family_stage3_v2'
            or document.get('assessment_core_sha256')!=ASSESSOR_CORE_SHA):
        raise ValueError('Reviewed adaptive assessment schema/core differs')
    if not is_v2 and ('assessment_schema' in document or 'assessment_core_sha256' in document):
        raise ValueError('Original reviewed v1 report schema differs')
    snapshots=normalized_inventory(document.get('ledger_snapshot_sha256'))
    if request.get('primary_assessment_ledger_sha256')!=ledger_sha or snapshots.get((base.FOLDER/'ledger.json').resolve())!=ledger_sha:
        raise ValueError('Primary assessment must authenticate the exact current stage3 ledger snapshot')
    if any(document.get(key,{}).get('status')!='PASS' for key in ['primary_execution','authorization','budget']):
        raise ValueError('Primary execution, authorization and budget assessments must PASS')
    progress=document['primary_execution'];jobs=progress.get('primary_jobs',[])
    expected_ids={job['proposed_id'] for job in stage.plan['primary_jobs']}
    if (progress.get('completed_verified')!=14 or progress.get('required')!=14 or len(jobs)!=14
            or {job.get('proposed_id') for job in jobs}!=expected_ids
            or any(job.get('completed_verified') is not True or job.get('status')!='completed' for job in jobs)):
        raise ValueError('Assessment must prove all14 exact primary jobs completed and verified')
    if (document['budget'].get('additional_attempts')!=len(snapshot['entries'])
            or document['budget'].get('cumulative_attempts')!=65+len(snapshot['entries'])):
        raise ValueError('Assessment budget accounting differs from its ledger snapshot')
    inputs=normalized_inventory(document.get('input_sha256'))
    if (base.FOLDER/'ledger.json').resolve() in inputs:
        raise ValueError('Mutable current ledger belongs only in ledger_snapshot_sha256')
    for filename,digest_value in inputs.items():
        if sha(filename)!=digest_value:raise ValueError('Assessment immutable input inventory changed')
    required=[base.FOLDER/name for name in ('config.json','preparation_manifest.json','source_manifest.json','frozen_cases.json','plan.json')]
    core=(ROOT/'scripts/assess_height_family_stage3_v1.py').resolve()
    science=(ROOT/'scripts/plot_height_family_nn_forward_v3.py').resolve()
    if sha(core)!=ASSESSOR_CORE_SHA or sha(science)!=SCIENCE_SHA:
        raise ValueError('Reviewed scientific/core source changed')
    if is_v2:
        sources=normalized_inventory(document.get('analysis_source_sha256'))
        collector=(ROOT/'scripts/collect_height_family_stage3_v2.py').resolve()
        if set(sources)!={utility,core,collector}:
            raise ValueError('Assessment fixed-path analysis source inventory differs')
        if any(sha(filename)!=value for filename,value in sources.items()):
            raise ValueError('Assessment analysis source inventory changed')
        if sources[utility]!=REVIEWED_ASSESSORS[utility.name] or sources[core]!=ASSESSOR_CORE_SHA:
            raise ValueError('Reviewed assessment wrapper/core inventory differs')
        required.append(science)
    else:
        collector=ROOT/'scripts/collect_height_family_stage3_v1.py'
        if sha(collector)!=V1_COLLECTOR_SHA:raise ValueError('Reviewed original assessment collector changed')
        required.extend([utility,science,collector])
    for job in stage.plan['primary_jobs']:
        found=stage.cache(job)
        if found is None:raise ValueError('Assessment primary evidence is no longer available')
        entry=found['entry'];origin=local_path(found['stage_folder'])
        required.append(origin/entry['record_file'])
        required.extend(local_path(filename) for filename in entry.get('required_file_sha256',{}))
        if job['method']=='NN':required.append(origin/entry['analysis_file'])
    for filename in required:
        filename=filename.resolve()
        if inputs.get(filename)!=sha(filename):raise ValueError('Assessment required immutable source/evidence inventory missing or changed')
    rows=document.get('NN_heights',[])
    if len(rows)!=8 or sorted(row.get('H_um') for row in rows)!=list(HEIGHTS):
        raise ValueError('Assessment must retain all eight measured-height rows')
    entries=[('stage3',base.FOLDER.name,entry) for entry in snapshot['entries']]
    entries.extend((row['stage_id'],row['stage_name'],row['entry']) for row in stage.catalog)
    for row in rows:
        measured=row.get('Meep') or {}
        matches=[entry for sid,name,entry in entries if name+':'+str(entry['id'])==measured.get('identity')]
        if (len(matches)!=1 or matches[0].get('method')!='NN' or matches[0].get('status')!='completed'
                or matches[0].get('H_um')!=row['H_um']
                or matches[0].get('record_sha256')!=measured.get('record_sha256')
                or matches[0].get('analysis_sha256')!=measured.get('analysis_sha256')):
            raise ValueError('Assessment selected measured-record source inventory differs')
    return document
def verify_assessment(request,stage):
    path=local_path(request['primary_assessment_file']);document=read(path)
    if sha(path)!=request['primary_assessment_sha256']:raise ValueError('Primary assessment SHA differs')
    return assessment_proof(request,stage,document,sha(base.FOLDER/'ledger.json'),read(base.FOLDER/'ledger.json'))
def reject_duplicate_targets(selected,prior_pairs=()):
    targets=set()
    for pair in [*prior_pairs,*selected]:
        target=base.identity(pair['jobs'][1])
        if target in targets:raise ValueError('Duplicate adaptive target numerical identity across pair allocations')
        targets.add(target)
def allocated_pairs(stage,exclude_sha=None):
    result=[]
    for link in stage.ledger.get('adaptive_selections',[]):
        if link['receipt_sha256']==exclude_sha:continue
        started={run['pair_id'] for run in pair_runs(stage) if run['selection_sha256']==link['receipt_sha256']}
        if not started:continue
        path=base.FOLDER/link['receipt_file']
        if sha(path)!=link['receipt_sha256']:raise ValueError('Prior allocated pair receipt changed')
        receipt=read(path)
        if receipt.get('selected_mode')!=MODE:raise ValueError('Prior allocated adaptive scope differs')
        pairs=[pair for pair in receipt['selected_pairs'] if pair['pair_id'] in started]
        if len(pairs)!=len(started):raise ValueError('Prior pair allocation is absent from its prospective receipt')
        result.extend(pairs)
    return result
def require_forecast_fit(stage,forecast):
    worker=forecast.get('worker_forecast_s');analysis=forecast.get('analysis_allowance_s')
    if any(not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0 for value in (worker,analysis)) or worker<=0:
        raise ValueError('Finite measured worker and analysis scheduling forecast required')
    if worker+analysis>=stage.remaining():
        raise RuntimeError('Adaptive worker deferred: measured worker plus analysis forecast no longer fits original remaining deadline minus185s')
def request_selection(request,stage):
    stage.verify();primary_complete(stage)
    base.u.require_no_live_children(stage.all_prior_entries+stage.ledger['entries'])
    for key in ['reason','fine_time_priority_review']:
        if not isinstance(request.get(key),str) or not request[key].strip():raise ValueError('Explicit root reason and fine-time-priority review required')
    verify_assessment(request,stage)
    pairs=request.get('pairs');used=len(pair_runs(stage))
    if not isinstance(pairs,list) or not 1<=len(pairs)<=2-used:raise ValueError('At most two started isolated adaptive pairs in total')
    if len({pair['pair_id'] for pair in pairs})!=len(pairs):raise ValueError('Pair IDs must be unique')
    prior=stage.ledger.get('current_adaptive_selection')
    if prior:
        prior_receipt=verify_selection_receipt(stage,base.FOLDER/prior['receipt_file'],prior)
        done={run['pair_id'] for run in pair_runs(stage) if run['selection_sha256']==prior['receipt_sha256'] and run.get('status')=='completed'}
        if any(pair['pair_id'] not in done for pair in prior_receipt['selected_pairs']):
            if request.get('supersedes_unstarted_selection_sha256')!=prior['receipt_sha256']:
                raise ValueError('Explicit supersession required for unstarted prior selection')
            if any(run['selection_sha256']==prior['receipt_sha256'] and run.get('status')!='completed' for run in pair_runs(stage)):
                raise ValueError('Recover/diagnose a started incomplete pair before replacing its selection')
    selected=[isolated_jobs(pair,stage) for pair in pairs]
    reject_duplicate_targets(selected,allocated_pairs(stage))
    for pair in selected:
        if stage.cache(pair['jobs'][1]) is not None:raise ValueError('Target NN raw result already exists; use saved analysis/recovery under its original receipt')
        for item in pair['jobs']:
            if any(base.identity(e)==base.identity(item) and e['status']!='completed' for e in stage.ledger['entries']):
                raise ValueError('Failed target settings require diagnosis, not a fresh unchanged selection')
    if len(stage.ledger['entries'])+2*len(selected)>18 or len(adaptive_entries(stage))+2*len(selected)>4:
        raise RuntimeError('Adaptive4/global18/cumulative83 attempt cap cannot fit selection')
    total=sum(pair['forecast_s'] for pair in selected)
    if total>=stage.remaining():raise RuntimeError('Selected measured forecasts cannot fit the original remaining deadline minus185s')
    return selected
def verify_selection_receipt(stage,path,link):
    path=Path(path).resolve()
    if not path.is_relative_to(base.FOLDER.resolve()) or sha(path)!=link['receipt_sha256']:
        raise ValueError('Adaptive receipt is outside this stage or changed')
    receipt=read(path)
    expected=dict(status='PROSPECTIVE_STAGE3_ADAPTIVE_SELECTION_BEFORE_DATA',selected_mode=MODE,execution_scope=SCOPE,
        decision_origin='EXPLICIT_ROOT_SELECTION_REASON',driver_sha256=sha(__file__),frozen_primary_controller_sha256=BASE_SHA,
        approval_receipt_sha256=base.AUTH_SHA,approved_proposal_sha256=base.PROPOSAL_SHA,adaptive_launch_authorized_by_this_receipt=True,
        adaptive_attempt_cap=4,new_attempt_cap=18,cumulative_attempt_cap=83,max_started_pairs=2,adaptive_autoexecution=False,
        original_first_started_utc=base.ORIGINAL_START,original_absolute_deadline_utc=base.DEADLINE)
    if any(receipt.get(key)!=value for key,value in expected.items()):raise ValueError('Adaptive authority/scope/clock limits differ')
    reason=receipt.get('decision_text_exact')
    if (not isinstance(reason,str) or not reason.strip()
            or receipt.get('decision_text_sha256')!=hashlib.sha256(reason.encode()).hexdigest()
            or not isinstance(receipt.get('fine_time_priority_review'),str) or not receipt['fine_time_priority_review'].strip()
            or receipt.get('no_new_Q_gate') is not True or receipt.get('no_coarse_to_fine_correction') is not True
            or receipt.get('no_fully_steady_claim') is not True):
        raise ValueError('Prospective root reason/fine-time review or unchanged-diagnostic scope differs')
    if sha(path.parent/Path(__file__).name)!=receipt['driver_sha256']:raise ValueError('Prospective adaptive driver snapshot changed')
    for key,name in [('preparation_manifest_sha256','preparation_manifest.json'),('source_manifest_sha256','source_manifest.json'),('frozen_cases_sha256','frozen_cases.json')]:
        if receipt[key]!=sha(base.FOLDER/name):raise ValueError('Adaptive frozen stage/source/cases changed')
    if sha(path.parent/'selection_request_exact.json')!=receipt['request_sha256']:
        raise ValueError('Root exact adaptive request changed')
    if (sha(path.parent/'primary_assessment.json')!=receipt['primary_assessment_sha256']
            or sha(path.parent/'assessment_source_snapshots/reviewed_assessor.py')!=receipt['primary_assessment_utility_sha256']
            or sha(path.parent/'prior_ledger.json')!=receipt['prior_ledger_snapshot_sha256']
            or receipt['prior_ledger_snapshot_sha256']!=receipt['primary_assessment_ledger_sha256']):
        raise ValueError('Prospective primary assessment/utility/ledger proof changed')
    request=read(path.parent/'selection_request_exact.json')
    if (request.get('primary_assessment_sha256')!=receipt['primary_assessment_sha256']
            or request.get('primary_assessment_utility_sha256')!=receipt['primary_assessment_utility_sha256']):
        raise ValueError('Copied assessment proof differs from exact root request')
    assessment_proof(request,stage,read(path.parent/'primary_assessment.json'),
        receipt['prior_ledger_snapshot_sha256'],read(path.parent/'prior_ledger.json'))
    if receipt['selected_jobs']!=[job for pair in receipt['selected_pairs'] for job in pair['jobs']]:
        raise ValueError('Adaptive jobs differ from frozen selected pairs')
    reject_duplicate_targets(receipt['selected_pairs'],allocated_pairs(stage,link['receipt_sha256']))
    return receipt


class AdaptiveStage(base.Stage):
    """Adds an honestly labelled adaptive executor; frozen base methods intact."""
    def execute_adaptive_job(self,item,path,digest,forecast):
        self.verify();cached=self.cache(item)
        if cached is not None:
            if cached['stage_id']=='stage3':self.analyze(cached['entry'])
            return cached
        primary_complete(self)
        links=[link for link in self.ledger.get('adaptive_selections',[])
               if (base.FOLDER/link['receipt_file']).resolve()==Path(path).resolve() and link['receipt_sha256']==digest]
        if len(links)!=1:raise ValueError('New adaptive worker requires an explicit registry-linked prospective selection')
        selection=verify_selection_receipt(self,path,links[0])
        if item not in selection['selected_jobs']:raise ValueError('Worker job is outside its exact prospective adaptive selection')
        selected_forecasts=[row for pair in selection['selected_pairs'] for job,row in zip(pair['jobs'],pair['forecast']) if job==item]
        if selected_forecasts!=[forecast]:raise ValueError('Worker scheduling forecast differs from prospective measured forecast')
        require_forecast_fit(self,forecast)
        if any(base.identity(e)==base.identity(item) for e in self.ledger['entries']):raise RuntimeError('Failed settings require diagnosis; no unchanged retry loop')
        if len(adaptive_entries(self))>=4:raise RuntimeError('Four charged adaptive attempts exhausted')
        timeout=base.capacity(len(self.ledger['entries']),self.elapsed())
        base.u.require_no_live_children(self.all_prior_entries+self.ledger['entries'])
        if sha(path)!=digest or read(path)['driver_sha256']!=sha(__file__):raise ValueError('Adaptive selection source/receipt changed')
        reference=None if item['method']=='reference' else frozen_reference(self,item)
        case=self.case(item['H_um']);effective=base.u.settings_for(self.config,item)
        number=66+len(self.ledger['entries'])
        if not 80<=number<=83:raise ValueError('Adaptive attempts may occupy only IDs80..83')
        label=f"{number:02d}_{item['method']}_H{item['H_um']:g}_r{item['resolution']:g}_ADAPTIVE_{item['job_type']}"
        output=base.FOLDER/'records'/(label+'.json');log=base.FOLDER/'logs'/(label+'.log')
        if output.exists() or log.exists():raise FileExistsError('Preserve all previous adaptive evidence')
        args=[str(base.ROOT/'.venv-meep/bin/python'),'-B',str(base.WORKER),'--config',str(base.FOLDER/'config.json'),
            '--output',str(output),'--H',str(item['H_um']),'--resolution',str(item['resolution']),
            '--pml',str(item['pml_um']),'--padding',str(item['padding_um']),'--duration',str(item['duration'])]
        if reference:args+=['--design',str(base.FOLDER/case['design_file']),'--reference',str(reference['path'])]
        entry=dict(item,id=number,status='started',started_utc=utc(),record_file=str(output.relative_to(base.FOLDER)),
            log_file=str(log.relative_to(base.FOLDER)),command=args,requested_settings={key:item[key] for key in base.u.KEYS},
            effective_settings=effective,**base.geometry_metadata(item,case),selection_receipt_file=str(path.relative_to(base.FOLDER)),
            selection_receipt_sha256=digest,selection_mode=MODE,selection_scope=SCOPE,solver_launched=False,
            reference_id=None if reference is None else reference['entry']['id'],reference_stage=None if reference is None else reference['stage_name'],
            reference_stage_id=None if reference is None else reference['stage_id'],reference_stage_folder=None if reference is None else reference['stage_folder'],
            scheduling_forecast=forecast)
        self.ledger['entries'].append(entry);self.save();start=time.monotonic()
        try:
            self.verify()
            if sha(path)!=digest or read(path)['driver_sha256']!=sha(__file__):raise ValueError('Adaptive prelaunch receipt/source changed')
            require_forecast_fit(self,forecast)
            timeout=min(timeout,self.remaining())
            if timeout<=1:raise RuntimeError('Original deadline/reporting reserve reached')
            self.active=None
            with log.open('x') as stream:
                self.active=subprocess.Popen(args,cwd=base.ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                entry.update(owned_pid=self.active.pid,solver_launched=True,effective_child_timeout_s=timeout);self.save()
                code=self.wait_owned(timeout,args);entry['solver_exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Adaptive worker failed; charged attempt preserved')
            record=read(output)
            base.validate_worker_output(record,item,effective,self.config_fingerprint,
                base.read(base.FOLDER/'source_manifest.json')['scripts/height_family_maxwell_worker_v2.py'],
                base.FOLDER/'config.json',base.sha(base.FOLDER/'config.json'),
                design_path=base.FOLDER/case['design_file'] if reference else None,design_sha=case['design_sha256'] if reference else None,
                reference_path=reference['path'] if reference else None,reference_sha=reference['entry']['record_sha256'] if reference else None)
            entry.update(status='completed',record_sha256=sha(output),required_file_sha256=base.u.required_files(record),power_balance_error=record['power_balance_error'])
        except BaseException as error:
            self.terminate();entry['status']='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed';entry['error']=repr(error)
            if self.active is not None:entry['solver_exit_code']=self.active.returncode
            raise
        finally:entry.update(wrapper_wall_s=time.monotonic()-start,finished_utc=utc());self.save()
        self.analyze(entry)
        return dict(entry=entry,path=output,record=record,stage_id='stage3',stage_name=base.FOLDER.name,stage_folder=str(base.FOLDER))


def prepare_selection(request_file):
    stage=AdaptiveStage();data=Path(request_file).read_bytes();request=json.loads(data.decode('utf-8'));pairs=request_selection(request,stage)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=base.FOLDER/'adaptive_selections'/stamp;folder.mkdir(parents=True,exist_ok=False)
    snapshot=(base.FOLDER/'ledger.json').read_bytes()
    for name,blob in [('selection_request_exact.json',data),('prior_ledger.json',snapshot)]:
        with (folder/name).open('xb') as stream:stream.write(blob)
    shutil.copyfile(__file__,folder/Path(__file__).name)
    shutil.copyfile(local_path(request['primary_assessment_file']),folder/'primary_assessment.json')
    target=folder/'assessment_source_snapshots/reviewed_assessor.py';target.parent.mkdir();shutil.copyfile(local_path(request['primary_assessment_utility_file']),target)
    receipt=dict(status='PROSPECTIVE_STAGE3_ADAPTIVE_SELECTION_BEFORE_DATA',created_utc=utc(),selected_mode=MODE,execution_scope=SCOPE,
        decision_origin='EXPLICIT_ROOT_SELECTION_REASON',decision_text_exact=request['reason'],decision_text_sha256=hashlib.sha256(request['reason'].encode()).hexdigest(),
        driver_sha256=sha(__file__),frozen_primary_controller_sha256=BASE_SHA,approval_receipt_sha256=base.AUTH_SHA,
        approved_proposal_sha256=base.PROPOSAL_SHA,preparation_manifest_sha256=sha(base.FOLDER/'preparation_manifest.json'),
        source_manifest_sha256=sha(base.FOLDER/'source_manifest.json'),frozen_cases_sha256=sha(base.FOLDER/'frozen_cases.json'),
        prior_ledger_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),primary_assessment_sha256=request['primary_assessment_sha256'],
        primary_assessment_ledger_sha256=request['primary_assessment_ledger_sha256'],
        primary_assessment_utility_sha256=request['primary_assessment_utility_sha256'],request_sha256=hashlib.sha256(data).hexdigest(),
        selected_pairs=pairs,selected_jobs=[job for pair in pairs for job in pair['jobs']],adaptive_launch_authorized_by_this_receipt=True,
        max_started_pairs=2,adaptive_attempt_cap=4,new_attempt_cap=18,cumulative_attempt_cap=83,
        original_first_started_utc=base.ORIGINAL_START,original_absolute_deadline_utc=base.DEADLINE,
        elapsed_s=stage.elapsed(),remaining_minus185s=stage.remaining(),selected_forecast_s=sum(pair['forecast_s'] for pair in pairs),
        fine_time_priority_review=request['fine_time_priority_review'],coarse_Q_time_effect_diagnostic=COARSE_Q_TIME_EFFECT,
        no_new_Q_gate=True,no_coarse_to_fine_correction=True,no_fully_steady_claim=True,
        adaptive_autoexecution=False,geometry_physics_thresholds_unchanged=True)
    path=folder/'selection_receipt.json';base.write_new(path,receipt);digest=sha(path)
    link=dict(mode=MODE,receipt_file=str(path.relative_to(base.FOLDER)),receipt_sha256=digest,driver_sha256=receipt['driver_sha256'])
    stage.ledger.setdefault('adaptive_selections',[]).append(link);stage.ledger['current_adaptive_selection']=link;stage.save()
    print(json.dumps(dict(status='ADAPTIVE_SELECTION_PREPARED_NO_WORKERS',receipt=str(path),pairs=len(pairs),forecast_s=receipt['selected_forecast_s'],worker_launches=0)))
def run_next_pair(selection):
    stage=AdaptiveStage();stage.verify();primary_complete(stage)
    path=Path(selection).resolve();link=stage.ledger.get('current_adaptive_selection')
    if link is None or path!=(base.FOLDER/link['receipt_file']).resolve() or sha(path)!=link['receipt_sha256']:
        raise ValueError('Run only the current explicit hash-linked adaptive selection')
    receipt=verify_selection_receipt(stage,path,link)
    runs=pair_runs(stage);completed={run['pair_id'] for run in runs if run['selection_sha256']==link['receipt_sha256'] and run['status']=='completed'}
    remaining=[pair for pair in receipt['selected_pairs'] if pair['pair_id'] not in completed]
    if not remaining:print(json.dumps(dict(status='ALL_SELECTED_PAIRS_ALREADY_COMPLETE',worker_launches=0)));return
    pair=remaining[0];run=next((run for run in runs if run['selection_sha256']==link['receipt_sha256'] and run['pair_id']==pair['pair_id']),None)
    if run is None and len(runs)>=2:raise RuntimeError('Two isolated adaptive pair allocations already started')
    # Revalidate exact baseline, isolation, forecast ceiling and current time.
    request=read(path.parent/'selection_request_exact.json');original=next(p for p in request['pairs'] if p['pair_id']==pair['pair_id'])
    checked=isolated_jobs(original,stage)
    if checked!=pair:raise ValueError('Adaptive baseline/jobs/forecast differ from prospective frozen selection')
    pending=0.;new_calls=0
    for item,row in zip(checked['jobs'],checked['forecast']):
        cached=stage.cache(item)
        if cached is None:
            if any(base.identity(e)==base.identity(item) for e in stage.ledger['entries']):raise RuntimeError('Failed adaptive settings require diagnosis, not unchanged retry')
            pending+=row['worker_forecast_s']+row['analysis_allowance_s'];new_calls+=1
        elif item['method']=='NN' and cached['entry'].get('analysis_status')!='completed':pending+=120.
    if len(stage.ledger['entries'])+new_calls>18 or len(adaptive_entries(stage))+new_calls>4:raise RuntimeError('18/83 or adaptive4 cap exhausted')
    if pending>=stage.remaining():print(json.dumps(dict(status='PAIR_DEFERRED_CANNOT_FIT_ORIGINAL_DEADLINE',worker_launches=0,forecast_s=pending)));return
    base.u.require_no_live_children(stage.all_prior_entries+stage.ledger['entries'])
    if run is None:
        run=dict(pair_id=pair['pair_id'],selection_sha256=link['receipt_sha256'],status='started',started_utc=utc())
        stage.ledger.setdefault('adaptive_pair_runs',[]).append(run)
    stage.ledger['status']='RUNNING_ONE_EXPLICIT_ADAPTIVE_PAIR';stage.save()
    try:
        for item,row in zip(checked['jobs'],checked['forecast']):stage.execute_adaptive_job(item,path,link['receipt_sha256'],row)
        run.update(status='completed',finished_utc=utc());stage.ledger['status']='ADAPTIVE_PAIR_COMPLETE_ASSESS_BEFORE_NEXT_SELECTION_OR_RUN'
    except BaseException as error:
        stage.terminate();run.update(status='stopped_requires_diagnosis',error=repr(error));stage.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS';raise
    finally:stage.save()
    print(json.dumps(dict(status=stage.ledger['status'],pair_id=pair['pair_id'],new_attempts=len(stage.ledger['entries']),cumulative_attempts=65+len(stage.ledger['entries']),pairs_this_invocation=1)))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group();action.add_argument('--inspect-selection',type=Path)
    action.add_argument('--prepare-selection',type=Path);action.add_argument('--run-next-pair',type=Path)
    args=parser.parse_args()
    if args.prepare_selection or args.run_next_pair:
        if os.name!='posix':raise RuntimeError('Execution/preparation requires original Ubuntu-20.04 WSL')
        import fcntl
        with (base.STAGE1/'runner.lock').open('r+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.prepare_selection:prepare_selection(args.prepare_selection)
            else:run_next_pair(args.run_next_pair)
        return
    try:
        stage=AdaptiveStage();stage.verify();primary_complete(stage)
        if args.inspect_selection:
            pairs=request_selection(read(args.inspect_selection),stage)
            print(json.dumps(dict(status='READ_ONLY_SELECTION_VALID',pairs=pairs,worker_launches=0,selection_created=False),indent=2))
        else:print(json.dumps(dict(status='READ_ONLY_AWAIT_EXPLICIT_ROOT_ADAPTIVE_SELECTION',worker_launches=0,selection_created=False,remaining_h=stage.remaining()/3600)))
    except (OSError,KeyError,ValueError,RuntimeError) as error:print(json.dumps(dict(status='READ_ONLY_NOT_READY',worker_launches=0,reason=str(error))))


if __name__=='__main__':main()
