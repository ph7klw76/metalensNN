"""Bounded approved stage3 primary refinement; no adaptive autoexecution.

Preparation preserves the actual approval and immutable post65 proposal.
The original Nov13 clock/Nov15 deadline never resets. Only --run-primary
launches workers; --preflight/--smoke are read only and launch zero workers.
Future adaptive selection requires a separate prospective controller/receipt.
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
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
STAGE1=CAMPAIGN/'maxwell_validation_v1'
STAGE2=CAMPAIGN/'maxwell_validation_v2'
FOLDER=CAMPAIGN/'maxwell_validation_v3'
PROPOSAL=CAMPAIGN/'prepared_maxwell_continuation_v4/revisions/post65_v1/PROPOSAL.json'
WORKER=ROOT/'scripts/height_family_maxwell_worker_v2.py'
ANALYZER=ROOT/'scripts/height_family_maxwell_analysis_v2.py'
AUTH_SHA='51249da9f94d4eb066ddf028e55c145f80f01cedd5a2ae714f5eb3dc10e3413c'
PROPOSAL_SHA='49f5301f2d7167d039bab39a086d4866f413e3e3274e49e1aff6e2d3accb90b0'
PRIOR_LEDGER_SHA='00ce8db6903e941f05139453f203beb7e0f0f8a580309066f81dec2d9d075623'
ORIGINAL_START='2026-11-13T08:59:13.107420+00:00'
DEADLINE='2026-11-15T08:59:13.107420+00:00'
MAX_NEW=18;PRIOR_ATTEMPTS=65;MAX_CUMULATIVE=83;MAX_PRIMARY=14
RESERVE=185.;CHILD_MAX=21600.
MODE='stage3_primary_mesh_refinement'
SCOPE='APPROVED_EIGHTEEN_AMENDMENT_PRIMARY_FROZEN_NN_ONLY_ORIGINAL_DEADLINE'


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def parse(value):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware original clock required')
    return result.astimezone(timezone.utc)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_utils():
    path=ROOT/'scripts/run_height_family_continuation_v3.py'
    if sha(path)!='a287f5ef647d105c413761aa0b9dd7449263d4e7d8e06960c2c2e079d07d2586':
        raise ValueError('Frozen stage2 utility source changed')
    spec=importlib.util.spec_from_file_location('frozen_stage2_utilities_for_stage3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
u=load_utils()
def identity(item):return u.identity(item)
def case_id(H):return 'NN_H'+format(float(H),'g')
def absolute_elapsed(ledger,now=None):
    now=datetime.now(timezone.utc) if now is None else parse(now)
    if ledger.get('first_started_utc')!=ORIGINAL_START or ledger.get('absolute_deadline_utc')!=DEADLINE:
        raise ValueError('Stage3 cannot change/reset the original clock')
    elapsed=(now-parse(ORIGINAL_START)).total_seconds();recorded=float(ledger.get('elapsed_s',0))
    if elapsed<0 or recorded<0 or not math.isfinite(recorded):raise ValueError('Invalid original elapsed clock')
    return max(elapsed,recorded)
def capacity(attempts,elapsed):
    if attempts>=18 or 65+attempts>=83:raise RuntimeError('Approved18/83 attempt ceiling reached')
    remaining=(parse(DEADLINE)-parse(ORIGINAL_START)).total_seconds()-float(elapsed)-RESERVE
    if remaining<=1:raise RuntimeError('Original deadline/reporting reserve reached')
    return min(CHILD_MAX,remaining)
def validate_authorization(folder=FOLDER):
    auth=Path(folder)/'authorization';path=auth/'approval_receipt.json'
    if sha(path)!=AUTH_SHA:raise ValueError('Actual approved amendment receipt changed')
    receipt=read(path)
    for key,name in [('user_reply','user_reply_exact.txt'),('question','question_exact.txt')]:
        data=(auth/name).read_bytes()
        if data.decode('utf-8')!=receipt[key+'_exact'] or hashlib.sha256(data).hexdigest()!=receipt[key+'_sha256']:
            raise ValueError('Exact approval question/reply bytes changed')
    required=dict(status='APPROVED_BY_USER',decision_origin='EXPLICIT_USER_REPLY',prior_cumulative_attempts=65,
        additional_attempt_ceiling=18,cumulative_attempt_ceiling=83,primary_attempts=14,adaptive_attempt_ceiling=4,
        original_first_started_utc=ORIGINAL_START,original_absolute_deadline_utc=DEADLINE,
        time_extension_authorized=False,idle_and_approval_wait_count=True,one_local_CPU_optical_worker=True,
        all_RAM_gates=False,unchanged_physics=True,unchanged_designs=True,unchanged_acceptance_thresholds=True,
        new_physical_cases_authorized=0,actual_prior_physical_cases=25,reserved_case_ceiling=28,
        adaptive_specific_jobs_require_prospective_selection=True,elapsed_used_as_approval=False)
    if any(receipt.get(key)!=value for key,value in required.items()):raise ValueError('Amendment limits/scope changed')
    if receipt['approved_proposal_sha256']!=PROPOSAL_SHA or sha(PROPOSAL)!=PROPOSAL_SHA:
        raise ValueError('Approved immutable proposal changed')
    if receipt['approved_proposal_file'].replace('\\','/')!=PROPOSAL.relative_to(ROOT).as_posix():
        raise ValueError('Approval names a different proposal')
    return receipt
def proposal_plan(proposal):
    jobs=[]
    for original in proposal['primary_jobs']:
        item=dict(original);item.pop('authority_status',None)
        item.update(phase='PRIMARY_MESH_REFINEMENT',job_authorization='APPROVED_STAGE3_PRIMARY')
        jobs.append(item)
    expected=[(100.,32.,335.),(75.,32.,259.),(70.,40.,243.),(80.,32.,274.),(50.,32.,182.),(60.,32.,213.),(90.,32.,304.)]
    if len(jobs)!=14 or [item['proposed_id'] for item in jobs]!=list(range(66,80)):
        raise ValueError('Approved primary job IDs must be66..79')
    for pair,(H,res,duration) in zip([jobs[i:i+2] for i in range(0,14,2)],expected):
        if [item['method'] for item in pair]!=['reference','NN']:raise ValueError('Matching primary reference/NN pair required')
        if any(identity(item)!=(item['method'],H,res,2.,10.,duration) or item['job_type']!='refinement' for item in pair):
            raise ValueError('Primary physics/numerical settings changed')
    return dict(status='APPROVED_PRIMARY_ONLY_ADAPTIVE_UNSELECTED',primary_jobs=jobs,
        adaptive_attempt_slots=[80,81,82,83],adaptive_jobs=[],adaptive_autoexecution=False,
        proposal_sha256=PROPOSAL_SHA,primary_forecast_s=proposal['forecasts']['primary_with_margin_and_analysis_s'],
        old_full_envelope_s=proposal['forecasts']['total_with_adaptive_and_reporting_cleanup_s'])
def metadata_inventory(folder=FOLDER):
    """Read-only saved metadata helper; never instantiates a runner."""
    folder=Path(folder)
    return {key:read(folder/name) for key,name in [('approval','approval_receipt.json'),('prior_stages','prior_stages.json'),
        ('cases','frozen_cases.json'),('catalog','cache_catalog.json'),('plan','plan.json'),('ledger','ledger.json'),
        ('sources','source_manifest.json')]}
def verify_delivered(path,folder,manifest,digest=None):
    path=Path(path).resolve();folder=Path(folder).resolve()
    if not path.is_relative_to(folder):raise ValueError('Delivered evidence path escapes prior stage')
    name=path.relative_to(folder).as_posix();entry=manifest['files'].get(name)
    if entry is None or (sha(path) if digest is None else digest)!=entry['sha256']:
        raise ValueError('Prior evidence differs from sealed delivery: '+name)
def geometry_metadata(item,case):
    device=item['method']=='NN'
    return dict(case_id=case['case_id'] if device else None,
        design_sha256=case['design_sha256'] if device else None,
        physical_geometry_id=case['physical_geometry_id'] if device else None,
        paired_NN_case_id=case['case_id'],paired_NN_design_sha256=case['design_sha256'],
        paired_NN_physical_geometry_id=case['physical_geometry_id'])
def validate_worker_output(record,item,effective,config_fingerprint,worker_sha,config_path,config_sha,
                           design_path=None,design_sha=None,reference_path=None,reference_sha=None):
    if record['settings']!=effective or record['config_fingerprint']!=config_fingerprint:
        raise ValueError('Actual output differs from requested/effective frozen settings')
    if record['worker_sha256']!=worker_sha or record.get('solver')!='CPU Meep 1.35.0':
        raise ValueError('Worker source/runtime differs from frozen Meep1.35.0 contract')
    if (not isinstance(record.get('requested_config'),str)
            or Path(record['requested_config']).resolve()!=Path(config_path).resolve() or sha(config_path)!=config_sha):
        raise ValueError('Worker requested_config path/bytes differ from exact stage3 configuration')
    closure=record.get('power_balance_error')
    if not isinstance(closure,(int,float)) or not math.isfinite(closure) or closure<0:
        raise ValueError('Finite nonnegative native closure required')
    if item['method']=='reference':
        incident=record.get('incident_power')
        if ('design_file' in record or 'reference_file' in record or not isinstance(incident,(int,float))
                or not math.isfinite(incident) or incident<=0 or closure>=.05):
            raise ValueError('Homogeneous reference requires positive incident/native closure and no device/ref inputs')
    else:
        if ('incident_power' in record or not isinstance(record.get('design_file'),str)
                or not isinstance(record.get('reference_file'),str)
                or Path(record['design_file']).resolve()!=Path(design_path).resolve()
                or Path(record['reference_file']).resolve()!=Path(reference_path).resolve()
                or sha(design_path)!=design_sha or sha(reference_path)!=reference_sha):
            raise ValueError('NN design/reference input paths or frozen source-file SHA differ')
def require_empty_preparation_directory(folder):
    folder=Path(folder)
    if not (folder/'authorization').is_dir() or set(path.name for path in folder.iterdir())-{'authorization'}:
        raise FileExistsError('Preparation allows only the pre-existing authorization directory')


def prepare():
    auth=validate_authorization();require_empty_preparation_directory(FOLDER)
    if sha(STAGE2/'ledger.json')!=PRIOR_LEDGER_SHA or auth['prior_ledger_sha256']!=PRIOR_LEDGER_SHA:
        raise ValueError('Sealed closed65 ledger changed')
    old2=read(STAGE2/'ledger.json');old1=read(STAGE1/'ledger.json')
    if len(old1['entries'])!=45 or len(old2['entries'])!=20 or any(e['status']!='completed' for e in old2['entries']):
        raise ValueError('Closed65 cumulative attempts required')
    u.require_no_live_children(old1['entries']+old2['entries'])
    archive=CAMPAIGN/'height_specific_Maxwell_continuation_v2_evidence.zip'
    if sha(archive)!=auth['prior_archive_sha256'] or sha(STAGE2/'delivery_manifest.json')!=auth['prior_delivery_manifest_sha256']:
        raise ValueError('Sealed stage2 archive/manifest changed')
    with zipfile.ZipFile(archive) as zipped:
        matches=[name for name in zipped.namelist() if name.endswith('delivery_manifest.json')]
        if not any(zipped.read(name)==(STAGE2/'delivery_manifest.json').read_bytes() for name in matches):
            raise ValueError('Current stage2 manifest is absent from sealed archive')
    sources=read(STAGE2/'source_manifest.json')
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen prior physics source changed')
    proposal=read(PROPOSAL);plan=proposal_plan(proposal);config=read(STAGE2/'config.json')
    if sha(STAGE2/'config.json')!=proposal['frozen_inputs']['configuration_sha256']:raise ValueError('Frozen configuration changed')
    manifests={'stage1':read(STAGE1/'delivery_manifest.json'),'stage2':read(STAGE2/'delivery_manifest.json')}
    link=read(STAGE2/'prior_stage.json')
    old_archive=CAMPAIGN/'height_specific_Maxwell_validation_v1.zip'
    if (sha(STAGE1/'ledger.json')!=link['ledger_sha256']
            or sha(STAGE1/'delivery_manifest.json')!=link['delivery_manifest_sha256']
            or sha(old_archive)!=link['archive_sha256']):
        raise ValueError('Stage1 history differs from the sealed stage2 linkage')
    catalog=[];prior_stages=[]
    for sid,folder,ledger in [('stage1',STAGE1,old1),('stage2',STAGE2,old2)]:
        for name,entry in manifests[sid]['files'].items():
            if sha(folder/name)!=entry['sha256']:raise ValueError('Delivered prior bytes changed: '+sid+':'+name)
        prior_stages.append(dict(stage_id=sid,stage_name=folder.name,folder=str(folder.resolve()),
            ledger_file=str((folder/'ledger.json').resolve()),ledger_sha256=sha(folder/'ledger.json'),
            delivery_manifest_sha256=sha(folder/'delivery_manifest.json'),attempt_count=len(ledger['entries']),
            archive_file=str((old_archive if sid=='stage1' else archive).resolve()),
            archive_sha256=link['archive_sha256'] if sid=='stage1' else auth['prior_archive_sha256']))
        if sid=='stage2':
            assessment=folder/'assessment/final_v1/assessment.json'
            if sha(assessment)!=proposal['prerequisites']['final65_assessment_sha256']:
                raise ValueError('Final65 assessment seal changed')
            prior_stages[-1].update(final_assessment_file=str(assessment.resolve()),final_assessment_sha256=sha(assessment))
        for entry in ledger['entries']:
            if entry['status']!='completed' or entry['method'] not in ('reference','NN'):continue
            path,record=u.checked_record(entry,folder);verify_delivered(path,folder,manifests[sid],entry['record_sha256'])
            fields=u.required_files(record)
            for filename,digest in fields.items():verify_delivered(filename,folder,manifests[sid],digest)
            if entry['method']=='NN':
                if entry.get('analysis_status')!='completed':raise ValueError('Prior NN cache requires completed independent analysis')
                ap=folder/entry['analysis_file'];verify_delivered(ap,folder,manifests[sid],entry['analysis_sha256'])
            catalog.append(dict(stage_id=sid,stage_name=folder.name,stage_folder=str(folder.resolve()),entry=entry,
                record_path=str(path.resolve()),required_file_sha256=fields))
    shutil.copyfile(FOLDER/'authorization/approval_receipt.json',FOLDER/'approval_receipt.json')
    shutil.copyfile(STAGE2/'config.json',FOLDER/'config.json')
    shutil.copyfile(PROPOSAL,FOLDER/'approved_proposal.json')
    write(FOLDER/'prior_stages.json',dict(prior_attempt_count=65,actual_prior_physical_cases=25,reserved_case_ceiling=28,
        original_first_started_utc=ORIGINAL_START,original_absolute_deadline_utc=DEADLINE,stages=prior_stages,
        stage2_archive_file=str(archive.resolve()),stage2_archive_sha256=auth['prior_archive_sha256']))
    for sid,manifest in manifests.items():write(FOLDER/'prior_delivery_manifests'/(sid+'.json'),manifest)
    target=FOLDER/'prior_assessments/stage2_final.json';target.parent.mkdir(exist_ok=True)
    shutil.copyfile(STAGE2/'assessment/final_v1/assessment.json',target)
    cases=[]
    for oldcase in read(STAGE2/'frozen_cases.json')['cases']:
        if oldcase['method']!='NN':continue
        source=STAGE2/oldcase['design_file']
        if sha(source)!=oldcase['design_sha256']:raise ValueError('Frozen NN case changed')
        target=FOLDER/'frozen_designs'/(oldcase['case_id']+'.json');target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
        case=dict(oldcase,new_case=False,historical_stage2_new_case=oldcase['new_case'],design_file=str(target.relative_to(FOLDER)),
                  source_file=str(source.resolve()))
        if u.geometry_identity(read(target),config)!=case['physical_geometry_id']:raise ValueError('Physical geometry identity changed')
        cases.append(case)
    if len(cases)!=8:raise ValueError('Exactly8 already validated frozen NN cases required')
    write(FOLDER/'frozen_cases.json',dict(cases=cases,new_physical_cases=0,actual_cumulative_cases=25,reserved_case_ceiling=28))
    write(FOLDER/'cache_catalog.json',dict(status='SEALED_COMPLETED_PRIOR_NN_REFERENCE_ONLY',records=catalog))
    write(FOLDER/'plan.json',plan)
    sources.update({str(Path(__file__).relative_to(ROOT)):sha(__file__),
                    'scripts/record_height_family_stage3_approval_v1.py':sha(ROOT/'scripts/record_height_family_stage3_approval_v1.py')})
    for name in sources:
        target=FOLDER/'source_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(FOLDER/'source_manifest.json',sources)
    environment=CAMPAIGN/'stage3_prelaunch_environment_v1.json'
    if environment.is_file():shutil.copyfile(environment,FOLDER/'runtime_environment.json')
    protected=read(CAMPAIGN/'provenance.json')['protected_before'];write(FOLDER/'protected_historical.json',protected)
    static=['approval_receipt.json','authorization/approval_receipt.json','authorization/user_reply_exact.txt','authorization/question_exact.txt',
            'config.json','approved_proposal.json','prior_stages.json','prior_delivery_manifests/stage1.json','prior_delivery_manifests/stage2.json',
            'prior_assessments/stage2_final.json','frozen_cases.json','cache_catalog.json','plan.json','source_manifest.json','protected_historical.json']
    if (FOLDER/'runtime_environment.json').exists():static.append('runtime_environment.json')
    write(FOLDER/'preparation_manifest.json',{name:sha(FOLDER/name) for name in static})
    ledger=dict(status='PREPARED_APPROVED_PRIMARY_ONLY',entries=[],prior_attempt_count=65,new_attempt_count=0,
        cumulative_attempted_solves=65,first_started_utc=ORIGINAL_START,original_first_started_utc=ORIGINAL_START,
        absolute_deadline_utc=DEADLINE,stage3_prepared_utc=utc(),elapsed_s=0.,approval_sha256=AUTH_SHA,
        adaptive_attempt_slots_unselected=[80,81,82,83])
    ledger['elapsed_s']=absolute_elapsed(ledger)
    write(FOLDER/'ledger.json',ledger)
    print(json.dumps(dict(status=ledger['status'],worker_launches=0,elapsed_s=ledger['elapsed_s'],
        remaining_h=(172800-ledger['elapsed_s'])/3600,primary_forecast_h=plan['primary_forecast_s']/3600,
        full_old_envelope_fits=plan['old_full_envelope_s']<172800-ledger['elapsed_s'])),flush=True)


class Stage:
    """Reusable clock/cache/analysis/executor; current policy authorizes primary only."""
    def __init__(self):
        self.ledger=read(FOLDER/'ledger.json');self.approval=read(FOLDER/'approval_receipt.json');self.plan=read(FOLDER/'plan.json')
        self.priors=read(FOLDER/'prior_stages.json');self.cases=read(FOLDER/'frozen_cases.json')['cases']
        self.config=read(FOLDER/'config.json');self.config_fingerprint=fingerprint(self.config)
        self.catalog=read(FOLDER/'cache_catalog.json')['records'];self.old_entries=[row['entry'] for row in self.catalog]
        # Include the interrupted prior attempt in ownership checks too.
        self.all_prior_entries=[e for stage in self.priors['stages'] for e in read(stage['ledger_file'])['entries']]
        self.active=None;self.attached=time.monotonic();self.elapsed_at_attach=absolute_elapsed(self.ledger)
        self.env=os.environ.copy();self.env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.verify()
    def elapsed(self):return max(absolute_elapsed(self.ledger),self.elapsed_at_attach+time.monotonic()-self.attached)
    def remaining(self):return 172800.-self.elapsed()-RESERVE
    def save(self):
        self.ledger.update(elapsed_s=self.elapsed(),new_attempt_count=len(self.ledger['entries']),
            cumulative_attempted_solves=65+len(self.ledger['entries']),updated_utc=utc())
        write(FOLDER/'ledger.json',self.ledger)
    def verify(self):
        validate_authorization()
        for name,digest in read(FOLDER/'preparation_manifest.json').items():
            if sha(FOLDER/name)!=digest:raise ValueError('Prepared stage3 evidence changed: '+name)
        if sha(FOLDER/'approval_receipt.json')!=AUTH_SHA or sha(FOLDER/'approved_proposal.json')!=PROPOSAL_SHA:
            raise ValueError('Approved proposal/receipt changed')
        if read(FOLDER/'plan.json')!=proposal_plan(read(PROPOSAL)):
            raise ValueError('Prepared primary plan differs from exact immutable approved proposal')
        for name,digest in read(FOLDER/'source_manifest.json').items():
            if sha(ROOT/name)!=digest or sha(FOLDER/'source_snapshots'/name)!=digest:raise ValueError('Frozen execution source changed: '+name)
        for prior in self.priors['stages']:
            if sha(prior['ledger_file'])!=prior['ledger_sha256']:raise ValueError('Prior closed ledger changed')
            if sha(Path(prior['folder'])/'delivery_manifest.json')!=prior['delivery_manifest_sha256']:
                raise ValueError('Prior delivery manifest changed')
            if sha(prior['archive_file'])!=prior['archive_sha256']:raise ValueError('Prior sealed archive changed')
            if prior.get('final_assessment_file') and sha(prior['final_assessment_file'])!=prior['final_assessment_sha256']:
                raise ValueError('Sealed final65 assessment changed')
        for name,digest in read(FOLDER/'protected_historical.json').items():
            if sha(ROOT/name)!=digest:raise ValueError('Protected historical artifact changed')
        for case in self.cases:
            if sha(FOLDER/case['design_file'])!=case['design_sha256'] or u.geometry_identity(read(FOLDER/case['design_file']),self.config)!=case['physical_geometry_id']:
                raise ValueError('Frozen design/physical geometry changed')
        if sha(FOLDER/'config.json')!=read(FOLDER/'approved_proposal.json')['frozen_inputs']['configuration_sha256']:
            raise ValueError('Unchanged physical configuration required')
        entries=self.ledger['entries']
        if self.ledger['prior_attempt_count']!=65 or [e['id'] for e in entries]!=list(range(66,66+len(entries))) or len(entries)>18:
            raise ValueError('Stage3 IDs/count must remain66..83 and preserve prior65')
        absolute_elapsed(self.ledger)
    def case(self,H):return next(case for case in self.cases if case['case_id']==case_id(H))
    def validate_item(self,item):
        matches=[job for job in self.plan['primary_jobs'] if identity(job)==identity(item)]
        if len(matches)!=1 or item['job_type']!='refinement':raise ValueError('This controller authorizes only the14 approved primary jobs')
        proposed=matches[0];effective=u.settings_for(self.config,item)
        if effective!=proposed['expected_actual_settings']:raise ValueError('Requested/effective settings differ from immutable proposal')
        return proposed,effective
    def cache(self,item,verify=True):
        if verify:self.verify()
        desired=u.settings_for(self.config,item);candidates=[]
        for row in self.catalog:
            if row['entry']['status']=='completed' and identity(row['entry'])==identity(item):candidates.append(row)
        for entry in self.ledger['entries']:
            if entry['status']=='completed' and identity(entry)==identity(item):
                candidates.append(dict(stage_id='stage3',stage_name=FOLDER.name,stage_folder=str(FOLDER),entry=entry,
                    record_path=str(FOLDER/entry['record_file']),required_file_sha256=entry['required_file_sha256']))
        if not candidates:return None
        row=candidates[-1];entry=row['entry'];origin=Path(row['stage_folder']);path,record=u.checked_record(entry,origin)
        if record['config_fingerprint']!=self.config_fingerprint or record['settings']!=desired:
            raise ValueError('Cached full physics/numerical settings differ')
        actual_paths=set(u.required_files(record))
        if actual_paths!=set(row['required_file_sha256']):raise ValueError('Incomplete required field SHA inventory')
        for filename,digest in row['required_file_sha256'].items():
            if sha(filename)!=digest:raise ValueError('Cached required field bytes changed')
        if entry['method']=='NN' and entry['design_sha256']!=self.case(item['H_um'])['design_sha256']:
            raise ValueError('Cached frozen geometry differs')
        if row['stage_id']!='stage3':
            delivered=read(FOLDER/'prior_delivery_manifests'/(row['stage_id']+'.json'))
            verify_delivered(path,origin,delivered,entry['record_sha256'])
            for filename,digest in row['required_file_sha256'].items():verify_delivered(filename,origin,delivered,digest)
            if entry['method']=='NN':verify_delivered(origin/entry['analysis_file'],origin,delivered,entry['analysis_sha256'])
        return dict(entry=entry,path=path,record=record,stage_id=row['stage_id'],stage_name=row['stage_name'],stage_folder=str(origin))
    def forecast(self):
        self.verify();remaining_jobs=[];seconds=0.
        for item in self.plan['primary_jobs']:
            cached=self.cache(item,verify=False)
            if cached is not None:
                if item['method']=='NN' and cached['entry'].get('analysis_status')!='completed':seconds+=120.
                continue
            if any(identity(e)==identity(item) for e in self.ledger['entries']):raise RuntimeError('Failed/incomplete primary settings require diagnosis; no unchanged retry loop')
            remaining_jobs.append(item['proposed_id']);seconds+=item['forecast']['predicted_worker_s_with15percent_margin']+item['forecast']['analysis_allowance_s']
        return dict(remaining_primary_job_ids=remaining_jobs,remaining_primary_forecast_s=seconds,
            original_deadline_remaining_minus_reserve_s=self.remaining(),primary_forecast_fits=seconds<self.remaining(),
            full_original_envelope_fit_required=False,adaptive_jobs_selected=0,worker_launches=0)
    def terminate(self):
        if self.active is not None and self.active.poll() is None:
            os.killpg(self.active.pid,signal.SIGTERM)
            try:self.active.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(self.active.pid,signal.SIGKILL);self.active.wait()
    def wait_owned(self,timeout,args):
        """Poll one owned process against fixed child and absolute UTC limits."""
        started=time.monotonic()
        while True:
            child_left=float(timeout)-(time.monotonic()-started)
            absolute_left=self.remaining()
            if child_left<=0 or absolute_left<=0:
                self.terminate()
                raise subprocess.TimeoutExpired(args,timeout)
            interval=min(30.,child_left,absolute_left)
            try:code=self.active.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                # Do not restart the child timer. UTC is reevaluated after
                # every bounded wait, including after host suspend/clock jump.
                continue
            if self.remaining()<=0 or time.monotonic()-started>=float(timeout):
                self.terminate()
                raise subprocess.TimeoutExpired(args,timeout)
            return code
    def analyze(self,entry):
        if entry['method']=='reference':return
        if entry.get('analysis_status')=='completed':
            if sha(FOLDER/entry['analysis_file'])!=entry['analysis_sha256']:raise ValueError('Completed analysis SHA changed')
            return
        self.verify()
        if self.remaining()<=1:raise RuntimeError('Original deadline reached before saved-field analysis')
        label=Path(entry['record_file']).stem;number=len(entry.get('analysis_attempts',[]))+1
        while True:
            stem=f'{label}.analysis-{number:02d}';output=FOLDER/'analyses'/(stem+'.json');log=FOLDER/'logs'/(stem+'.log')
            if not output.exists() and not log.exists():break
            number+=1
        output.parent.mkdir(exist_ok=True);log.parent.mkdir(exist_ok=True)
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ANALYZER),'--record',str(FOLDER/entry['record_file']),
            '--config',str(FOLDER/'config.json'),'--target-diameter',str(read(FOLDER/self.case(entry['H_um'])['design_file'])['target_diameter_um']),
            '--output',str(output)]
        attempt=dict(attempt=number,status='started',started_utc=utc(),command=args,log_file=str(log.relative_to(FOLDER)))
        entry.setdefault('analysis_attempts',[]).append(attempt);entry['analysis_status']='started';self.save()
        try:
            timeout=min(600.,self.remaining())
            if timeout<=1:raise RuntimeError('Original deadline/reporting reserve reached')
            self.active=None
            with log.open('x') as stream:
                self.active=subprocess.Popen(args,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                attempt['owned_analysis_pid']=self.active.pid;self.save()
                code=self.wait_owned(timeout,args)
            attempt['exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Saved-field analysis failed; recover from raw without new solve')
            value=read(output);entry.update(analysis_status='completed',analysis_file=str(output.relative_to(FOLDER)),analysis_sha256=sha(output),
                reconstruction_pass=value['reconstruction_pass'],nominal_optical_pass=value['synthetic_optical_pass'],
                C=value['normalized_curvature'],eta=value['conservative_useful_efficiency'])
            attempt.update(status='completed',analysis_file=entry['analysis_file'],analysis_sha256=entry['analysis_sha256'])
        except BaseException as error:
            self.terminate()
            entry['analysis_status']=attempt['status']='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed'
            attempt['error']=repr(error);raise
        finally:attempt['finished_utc']=utc();self.save()
    def launch_receipt(self,item,effective,number,reference,timeout):
        label=f"{number:02d}_{item['method']}_H{item['H_um']:g}_r{item['resolution']:g}"
        folder=FOLDER/'prelaunch'/label;folder.mkdir(parents=True,exist_ok=False)
        data=(FOLDER/'ledger.json').read_bytes()
        with (folder/'prior_ledger.json').open('xb') as stream:stream.write(data)
        shutil.copyfile(__file__,folder/Path(__file__).name)
        case=self.case(item['H_um']);requested={key:item[key] for key in u.KEYS}
        receipt=dict(status='PROSPECTIVE_STAGE3_PRIMARY_ATTEMPT_BEFORE_POPEN',created_utc=utc(),selected_mode=MODE,
            execution_scope=SCOPE,decision_origin='EXPLICIT_USER_REPLY',approval_receipt_sha256=AUTH_SHA,
            approved_proposal_sha256=PROPOSAL_SHA,preparation_manifest_sha256=sha(FOLDER/'preparation_manifest.json'),
            frozen_plan_sha256=sha(FOLDER/'plan.json'),source_manifest_sha256=sha(FOLDER/'source_manifest.json'),
            frozen_cases_sha256=sha(FOLDER/'frozen_cases.json'),prior_stages_sha256=sha(FOLDER/'prior_stages.json'),
            prior_ledger_snapshot_sha256=hashlib.sha256(data).hexdigest(),driver_sha256=sha(__file__),
            attempt_id=number,selected_jobs=[item],requested_settings=requested,effective_settings=effective,
            **geometry_metadata(item,case),
            reference=None if reference is None else dict(stage_id=reference['stage_id'],stage_name=reference['stage_name'],
                stage_folder=reference['stage_folder'],id=reference['entry']['id'],record_file=str(reference['path']),
                record_sha256=reference['entry']['record_sha256'],native_closure=reference['record']['power_balance_error']),
            original_first_started_utc=ORIGINAL_START,original_absolute_deadline_utc=DEADLINE,
            elapsed_since_original_start_s=self.elapsed(),remaining_minus_report_cleanup_s=self.remaining(),
            child_timeout_ceiling_s=timeout,prior_attempts=65,new_attempts_before_launch=len(self.ledger['entries']),
            cumulative_attempts_before_launch=65+len(self.ledger['entries']),new_attempt_cap=18,cumulative_cap=83,
            adaptive_autoexecution=False,geometry_physics_thresholds_unchanged=True)
        path=folder/'selection_receipt.json';write_new(path,receipt);digest=sha(path)
        link=dict(attempt_id=number,mode=MODE,receipt_file=str(path.relative_to(FOLDER)),receipt_sha256=digest,driver_sha256=receipt['driver_sha256'])
        self.ledger.setdefault('selection_receipts',[]).append(link)
        return path,digest
    def execute(self,item):
        proposed,effective=self.validate_item(item);cached=self.cache(item)
        if cached is not None:
            if cached['stage_id']=='stage3':self.analyze(cached['entry'])
            return cached
        if any(identity(entry)==identity(item) for entry in self.ledger['entries']):raise RuntimeError('Prior failed/incomplete settings cannot silently retry')
        forecast=self.forecast()
        if not forecast['primary_forecast_fits']:raise RuntimeError('Remaining primary forecast no longer fits unchanged original deadline')
        timeout=capacity(len(self.ledger['entries']),self.elapsed());self.verify()
        u.require_no_live_children(self.all_prior_entries+self.ledger['entries'])
        reference=None
        if item['method']=='NN':
            reference=self.cache(dict(item,method='reference'))
            if reference is None:raise ValueError('Exactly matching completed reference required before NN')
            closure=reference['record']['power_balance_error']
            if not math.isfinite(closure) or not 0<=closure<.05:raise ValueError('Reference native closure failed')
        number=66+len(self.ledger['entries'])
        if number!=proposed['proposed_id']:raise ValueError('Primary attempt order/IDs must match approved66..79 plan')
        case=self.case(item['H_um']);label=f"{number:02d}_{item['method']}_H{item['H_um']:g}_r{item['resolution']:g}_refinement"
        output=FOLDER/'records'/(label+'.json');log=FOLDER/'logs'/(label+'.log');output.parent.mkdir(exist_ok=True);log.parent.mkdir(exist_ok=True)
        if output.exists() or log.exists():raise FileExistsError('Preserve prior record/log attempts')
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(WORKER),'--config',str(FOLDER/'config.json'),'--output',str(output),
            '--H',str(item['H_um']),'--resolution',str(item['resolution']),'--pml',str(item['pml_um']),'--padding',str(item['padding_um']),
            '--duration',str(item['duration'])]
        if reference is not None:args+=['--design',str(FOLDER/case['design_file']),'--reference',str(reference['path'])]
        receipt_path,receipt_sha=self.launch_receipt(item,effective,number,reference,timeout)
        entry={key:value for key,value in item.items() if key not in ('worker_argv','expected_actual_settings')}
        entry.update(id=number,status='started',started_utc=utc(),record_file=str(output.relative_to(FOLDER)),log_file=str(log.relative_to(FOLDER)),
            command=args,requested_settings={key:item[key] for key in u.KEYS},effective_settings=effective,
            **geometry_metadata(item,case),
            reference_id=None if reference is None else reference['entry']['id'],reference_stage=None if reference is None else reference['stage_name'],
            reference_stage_id=None if reference is None else reference['stage_id'],reference_stage_folder=None if reference is None else reference['stage_folder'],
            selection_receipt_file=str(receipt_path.relative_to(FOLDER)),selection_receipt_sha256=receipt_sha,selection_mode=MODE,selection_scope=SCOPE,
            solver_launched=False)
        self.ledger['entries'].append(entry);self.save();started=time.monotonic()
        print(json.dumps(dict(event='SOLVE_STARTED',id=number,method=item['method'],H_um=item['H_um'],resolution=item['resolution'],remaining_h=self.remaining()/3600)),flush=True)
        try:
            self.verify()
            if sha(receipt_path)!=receipt_sha:raise ValueError('Prelaunch receipt changed')
            timeout=min(timeout,self.remaining())
            if timeout<=1:raise RuntimeError('Original deadline reached before Popen')
            entry['effective_child_timeout_s']=timeout
            self.active=None
            with log.open('x') as stream:
                self.active=subprocess.Popen(args,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                entry.update(owned_pid=self.active.pid,solver_launched=True);self.save()
                code=self.wait_owned(timeout,args);entry['solver_exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Meep worker failed; preserve charged attempt')
            record=read(output)
            validate_worker_output(record,item,effective,self.config_fingerprint,
                read(FOLDER/'source_manifest.json')['scripts/height_family_maxwell_worker_v2.py'],
                FOLDER/'config.json',sha(FOLDER/'config.json'),
                design_path=FOLDER/case['design_file'] if item['method']=='NN' else None,
                design_sha=case['design_sha256'] if item['method']=='NN' else None,
                reference_path=reference['path'] if reference is not None else None,
                reference_sha=reference['entry']['record_sha256'] if reference is not None else None)
            entry.update(status='completed',record_sha256=sha(output),required_file_sha256=u.required_files(record),power_balance_error=record['power_balance_error'])
        except BaseException as error:
            self.terminate();entry['status']='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed'
            entry['error']=repr(error)
            if self.active is not None:entry['solver_exit_code']=self.active.returncode
            raise
        finally:entry.update(wrapper_wall_s=time.monotonic()-started,finished_utc=utc());self.save()
        self.analyze(entry)
        print(json.dumps(dict(event='SOLVE_COMPLETED',id=number,closure=entry['power_balance_error'],C=entry.get('C'),eta=entry.get('eta'))),flush=True)
        return dict(entry=entry,path=output,record=record,stage_id='stage3',stage_name=FOLDER.name,stage_folder=str(FOLDER))
    def run_primary(self):
        self.verify();u.require_no_live_children(self.all_prior_entries+self.ledger['entries'])
        self.ledger['status']='RUNNING_APPROVED_PRIMARY_ONLY';self.ledger.setdefault('stage3_first_launch_utc',utc());self.save()
        try:
            for item in self.plan['primary_jobs']:self.execute(item)
            self.verify();self.ledger['status']='PRIMARY_EXECUTED_REQUIRES_ASSESSMENT_ADAPTIVE_UNSELECTED'
        except BaseException as error:
            self.terminate();self.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS'
            self.ledger.setdefault('failures',[]).append(dict(utc=utc(),type=type(error).__name__,reason=str(error)));raise
        finally:self.save()
        print(json.dumps(dict(status=self.ledger['status'],new_attempts=len(self.ledger['entries']),cumulative_attempts=65+len(self.ledger['entries']),
            original_deadline=DEADLINE,adaptive_calls_launched=0,full_acceptance_claimed=False)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--preflight',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--run-primary',action='store_true');args=parser.parse_args()
    if not any(vars(args).values()):parser.error('Choose explicit --prepare, --preflight/--smoke or --run-primary; no implicit launch')
    if args.preflight or args.smoke:
        if args.prepare or args.run_primary:parser.error('Read-only preflight/smoke cannot combine with mutations or execution')
        try:
            stage=Stage();forecast=stage.forecast();u.require_no_live_children(stage.all_prior_entries+stage.ledger['entries'])
            print(json.dumps(dict(status='READ_ONLY_PREFLIGHT',worker_launches=0,elapsed_s=stage.elapsed(),remaining_h=stage.remaining()/3600,
                forecast=forecast,adaptive_jobs_selected=0),indent=2))
        except (OSError,KeyError,ValueError,RuntimeError) as error:print(json.dumps(dict(status='READ_ONLY_NOT_READY',worker_launches=0,reason=str(error))))
        return
    if os.name!='posix':raise RuntimeError('Preparation/execution requires original Ubuntu-20.04 WSL environment')
    import fcntl
    with (STAGE1/'runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare()
        if args.run_primary:Stage().run_primary()


if __name__=='__main__':main()
