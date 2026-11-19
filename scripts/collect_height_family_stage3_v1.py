"""Authenticate saved stage1/2/3 NN evidence; no execution or ledger writes."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

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
        path=local_path(path).resolve();value=sha(path);self.inputs[str(path)]=value;return value
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
    receipt=e.document(e.folder/entry['selection_receipt_file']);proof=receipt.get('reference') or {}
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
    e.inputs.update(legacy['input_sha256'])
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
    normalized=dict(legacy)
    normalized.update(folder=str(e.folder),Meep_records=records,all_attempts=attempts,missing_evidence=missing,
        budget=budget,original_boundary_definitions=registry,input_sha256=e.inputs,
        ledger_snapshot_sha256={**legacy['ledger_snapshot_sha256'],str(e.folder/'ledger.json'):hashlib.sha256(e.ledger_bytes).hexdigest()})
    if not all(sha(path)==digest for path,digest in e.inputs.items()):raise RuntimeError('Immutable input changed during stage3 collection')
    return {'data':safe_json(normalized),'numerical_limits':e.document(local_path(e.stages['stage1']['folder'])/'maxwell_validation_proposal.json')['numerical_acceptance'],
            'authorization':safe_json(authorization),'budget':budget,'primary_execution':progress}
