"""Prospective six-device Meep validation of the existing 500-nm SLC profiles.

Default inspection and --prepare launch zero workers. Preparation is UNAPPROVED.
Only a separately recorded new explicit budget approval permits --run. Frozen
worker/analyzer sources, original optical config and homogeneous references are
reused without changing old campaigns. No reference fallback or automatic retry.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
FOLDER=CAMPAIGN/'slc500_meep_validation_v1'
EXPORT=ROOT/'exports/height_family_cylinder_attached_slc_500nm_v5'
BASE=ROOT/'scripts/run_height_family_maxwell_v3.py'
WORKER=ROOT/'scripts/height_family_maxwell_worker_v2.py'
ANALYZER=ROOT/'scripts/height_family_maxwell_analysis_v2.py'
PRIOR=83;MAX_NEW=6;MAX_CUMULATIVE=89;MAX_WALL=28800.;RESERVE=185.;CHILD_MAX=21600.
QUESTION='if using direct MEEP calculation on Existing 500 nm SCL what is the useful modal efficiency?'
CONFIG_SHA='ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734'
MANIFEST_SHA='94cdb24ba0f71bb072e30f17b8557c3f0955d04bb903ee1e2ef97e8b4e4fcffd'
SOURCE_PINS={
 'scripts/run_height_family_maxwell_v3.py':'b28ffe701b3fb452cf6d3c04fb049d77afc8c6ed4eeb450da513e338717bba73',
 'scripts/run_height_family_continuation_v3.py':'a287f5ef647d105c413761aa0b9dd7449263d4e7d8e06960c2c2e079d07d2586',
 'scripts/height_family_maxwell_worker_v2.py':'59fdb4dee1e7074ca17463e1495b87886563bc0ea2f393f8c539b8727b019008',
 'scripts/height_family_maxwell_analysis_v2.py':'4d1c2cd04c539660928f29d66bce3e701ef41a115512931070dd0f11a6dfc55b',
 'scripts/report_height_family_maxwell_v2.py':'d8773663c8595fc35c6c942c7fda6d95f20ab69e1db9940b8c66ce0bcf447d12',
}
COARSE_SHA={40:'06b642955e9c19a3c7bc6865501640dfe2b57db4cc1754fc8354216e317ac6a1',
 75:'cf782ba2091d29a67022678e7ce1bd7a8060e3084d3038e617e46d2b4ff0145d',
 100:'9a69d7bffce92dc858186ba190536d83b8319a7d27be405f236e3f769ea2aeaa'}
LEDGER_PINS={'maxwell_validation_v1':'7b38ca7d2c5d543fd26284cec9c0370749f335d17cf3c749376ac770ce91b693',
 'maxwell_validation_v2':'00ce8db6903e941f05139453f203beb7e0f0f8a580309066f81dec2d9d075623',
 'maxwell_validation_v3':'27628c8f777c8e76b35feadf82a1851b0ab6fbe1cd6d270dd646771147b8dc37'}
REFERENCE_MAP={(40,24):('maxwell_validation_v1',37),(40,32):('maxwell_validation_v1',42),
 (75,24):('maxwell_validation_v2',56),(75,32):('maxwell_validation_v3',68),
 (100,24):('maxwell_validation_v2',46),(100,32):('maxwell_validation_v3',66)}
DEVICE_TIMES={(40,24):586.503156453,(40,32):1345.0837590340016,
 (75,24):1260.8405970020103,(75,32):2959.1940989869763,
 (100,24):1944.667958842998,(100,32):4543.475285036984}
LIMITS={'diameter_relative_mesh_change_max':.02,'eta_relative_mesh_change_max':.05,
        'curvature_absolute_max':.1,'curvature_last_pair_delta_max':.005}


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def parse(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('UTC-aware clock required')
    return result.astimezone(timezone.utc)
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x',encoding='utf-8') as stream:stream.write(data)
    else:
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(data,encoding='utf-8');temporary.replace(path)
def local(path):
    text=str(path).replace('\\','/')
    if text.startswith('/mnt/') and os.name=='nt':text=text[5].upper()+':'+text[6:]
    elif len(text)>2 and text[1]==':' and os.name!='nt':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text).resolve()
def relative(path):return local(path).relative_to(ROOT.resolve()).as_posix()
def protect(path,expected):
    if sha(path)!=expected:raise ValueError('Immutable input differs: '+str(path))
    return expected
def module(path,expected):
    protect(path,expected);spec=importlib.util.spec_from_file_location('slc500_'+Path(path).stem,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
def helpers():return module(BASE,SOURCE_PINS[relative(BASE)])


def plan():
    jobs=[]
    for H in (40,75,100):
        for resolution in (24,32):
            jobs.append(dict(id=84+len(jobs),method='NN',case_id=f'SLC500_NN_H{H}',H_um=float(H),
                resolution=float(resolution),pml_um=2.,padding_um=10.,duration={40:160.,75:259.,100:335.}[H],
                job_type='slc500_nominal_mesh',measured_fine_device_worker_s=DEVICE_TIMES[H,resolution],
                forecast_worker_s=1.25*DEVICE_TIMES[H,resolution],analysis_allowance_s=120.))
    return jobs
def exact_job(job):
    candidates=[row for row in plan() if row['id']==job.get('id')]
    if len(candidates)!=1 or candidates[0]!=job:raise ValueError('Only exact six proposed devices are permitted')
def remaining(approval,now=None):
    now=parse(utc()) if now is None else parse(now)
    start=parse(approval['approved_utc']);deadline=parse(approval['absolute_deadline_utc'])
    if (deadline-start).total_seconds()!=MAX_WALL or now<start:raise ValueError('New allocation clock cannot reset')
    return (deadline-now).total_seconds()-RESERVE
def capacity(attempt_count,approval,now=None):
    if attempt_count>=MAX_NEW or PRIOR+attempt_count>=MAX_CUMULATIVE:raise RuntimeError('Six new /89 cumulative attempt cap reached')
    left=remaining(approval,now)
    if left<=1:raise RuntimeError('Eight-hour deadline/reporting reserve reached')
    return min(CHILD_MAX,left)


def required_files(record):
    keys=['fields_file','power_audit_fields_file']+(['flux_data_file'] if 'incident_power' in record else [])
    return {relative(record[key]):sha(local(record[key])) for key in keys}
def reference_catalog(config,base):
    result=[]
    for job in plan():
        H,res=int(job['H_um']),int(job['resolution']);stage,number=REFERENCE_MAP[H,res]
        origin=CAMPAIGN/stage;ledger_path=origin/'ledger.json'
        protect(ledger_path,LEDGER_PINS[stage]);ledger=read(ledger_path)
        entries=[e for e in ledger['entries'] if e['id']==number]
        if len(entries)!=1 or entries[0]['method']!='reference' or entries[0]['status']!='completed':
            raise ValueError('One actually completed homogeneous reference required')
        entry=entries[0];record_path=origin/entry['record_file'];protect(record_path,entry['record_sha256'])
        record=read(record_path);expected=base.u.settings_for(config,job)
        if (record['settings']!=expected or record['config_fingerprint']!=digest(config)
                or record['worker_sha256']!=SOURCE_PINS[relative(WORKER)]
                or record['solver']!='CPU Meep 1.35.0' or 'design_file' in record or 'reference_file' in record
                or not math.isfinite(record['incident_power']) or record['incident_power']<=0
                or not 0<=record['power_balance_error']<.05):
            raise ValueError('Reference physics/settings/source/power identity differs')
        fields=required_files(record)
        if stage=='maxwell_validation_v1':
            delivery=origin/'delivery_manifest.json'
            protect(delivery,'e72401385363bdc6f3c3d4a509ab41675daa4c96345b8fb0921b4c1cf209d2b3')
            manifest=read(delivery)
            for filename,value in {relative(record_path):entry['record_sha256'],**fields}.items():
                key=local(ROOT/filename).relative_to(origin.resolve()).as_posix()
                if manifest['files'][key]['sha256']!=value:raise ValueError('Reference differs from original sealed stage1 delivery')
        elif {relative(name):value for name,value in entry['required_file_sha256'].items()}!=fields:
            raise ValueError('Reference field/flux SHA differs from completed original ledger')
        result.append(dict(H_um=H,resolution=res,stage=stage,id=number,record_file=relative(record_path),
            record_sha256=entry['record_sha256'],required_file_sha256=fields,settings=expected,
            native_closure=record['power_balance_error'],incident_power=record['incident_power'],
            original_ledger_sha256=LEDGER_PINS[stage],original_wrapper_s=entry['wrapper_wall_s']))
    return result


def prepare():
    if FOLDER.exists():raise FileExistsError('Preserve existing proposal/results')
    base=helpers();cfg_path=CAMPAIGN/'config.json';protect(cfg_path,CONFIG_SHA);config=read(cfg_path)
    source_manifest=CAMPAIGN/'maxwell_validation_v3/source_manifest.json';protect(source_manifest,MANIFEST_SHA)
    sources={**read(source_manifest),**SOURCE_PINS,relative(Path(__file__)):sha(__file__)}
    for name,value in sources.items():protect(ROOT/name,value)
    refs=reference_catalog(config,base);protected={relative(cfg_path):CONFIG_SHA,relative(source_manifest):MANIFEST_SHA}
    targets_path=CAMPAIGN/'train_frozen_train_validation_targets.json'
    protect(targets_path,'2b5f1c36abb49ad9f43c20df99d3710141148060961cddcfb56be9a1669e601c')
    targets=read(targets_path);protected[relative(targets_path)]=sha(targets_path)
    proof=CAMPAIGN/'slc500_meep_independent_preflight_v1'
    for name,value in {'review.json':'ec5ef91720ff1bc8599dc8f00fe26b6c74514d7123c436fa6ad1bba98040383b',
        'review_manifest.json':'3477dd72cd2844aa41c556520169969cb679356578b51c74baa09947af502ee6',
        'plan_count_clarification.json':'6bd20042c6999be50e05d6a8ffa9a979e7846a9e76ab2d6b0e6a444c3ec1ac32'}.items():
        protect(proof/name,value);protected[relative(proof/name)]=value
    cases=[]
    for H in (40,75,100):
        path=EXPORT/f'fabrication_sources/NN_H{H}_500nm_fabrication.json';protect(path,COARSE_SHA[H]);design=read(path)
        if (design['H_um']!=H or design['cylinder_diameter_um']!=50 or len(design['heights_um'])!=50
                or design['materials']!=config['materials'] or design['mfd_um']!=config['mfd_um']
                or design['units']!='um' or any(h<0 or h>3 or h/.5!=round(h/.5) for h in design['heights_um'])):
            raise ValueError('Exact500-nm source/500-nm radial band/material geometry required')
        original=read(CAMPAIGN/f'designs/H{H}/NN.json')
        target=targets[str(float(H))]
        if (design['target_diameter_um']!=original['target_diameter_um']
                or design['target_diameter_um']!=target['declared_target_diameter_um']
                or design['output_reference_z_um']!=original['output_reference_z_um']
                or design['output_reference_z_um']!=target['target_output_z_um']):
            raise ValueError('Original frozen target/reference plane must be unchanged')
        protected[relative(path)]=COARSE_SHA[H]
        for extra in (EXPORT/f'H{H:03d}/C0{(40,75,100).index(H)+1}.slc',
                      EXPORT/f'H{H:03d}/assembly_manifest.json',CAMPAIGN/f'designs/H{H}/NN.json'):
            protected[relative(extra)]=sha(extra)
        cases.append(dict(case_id=f'SLC500_NN_H{H}',H_um=H,new_physical_case=True,
            design_file=f'frozen_designs/NN_H{H}_500nm_fabrication.json',design_sha256=COARSE_SHA[H],
            physical_geometry_id=base.u.geometry_identity(design,config),target_diameter_um=design['target_diameter_um'],
            radial_band_width_um=.5,height_step_um=.5,original_optical_status=design['optical_status']))
    for stage,value in LEDGER_PINS.items():protected[relative(CAMPAIGN/stage/'ledger.json')]=value
    for row in refs:
        protected[row['record_file']]=row['record_sha256'];protected.update(row['required_file_sha256'])
    FOLDER.mkdir(parents=True)
    shutil.copyfile(cfg_path,FOLDER/'config.json')
    for case in cases:
        destination=FOLDER/case['design_file'];destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(EXPORT/'fabrication_sources'/destination.name,destination)
    for name,value in sources.items():
        destination=FOLDER/'source_snapshots'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    proposal=dict(status='PROPOSED_UNAPPROVED',approved=False,created_utc=utc(),user_question_exact=QUESTION,
        additional_device_attempt_ceiling=6,cumulative_attempt_ceiling=89,prior_attempt_count=83,
        wall_time_ceiling_s=MAX_WALL,child_timeout_max_s=CHILD_MAX,cleanup_reserve_s=RESERVE,
        actual_prior_cases=25,new_distinct_SLC500_cases=3,actual_case_total_if_executed=28,
        old_unrun_reserved_case_allocation_is_not_reused=True,one_local_CPU_worker=True,RAM_gates=False,
        reference_solves_authorized=0,no_reference_fallback=True,retraining=False,printing=False,
        jobs=plan(),forecast_measured_fine_device_s=sum(DEVICE_TIMES.values()),
        forecast_with25percent_margin_and_analysis_s=sum(j['forecast_worker_s']+j['analysis_allowance_s'] for j in plan()),
        target='Original per-H flat-phase Gaussian and monitor planes unchanged; imported500nm relief arrays only differ',
        geometry_representation='Axisymmetric circular annuli equivalent to v5 SLC128-segment polygons; not literal128-facet Maxwell',
        polygon_radial_sagitta_nm=25*(1-math.cos(math.pi/128))*1000,
        no_global_acceptance_claim=True,numerical_limits=LIMITS)
    for name,value in [('proposal.json',proposal),('plan.json',plan()),('frozen_cases.json',cases),('cache_catalog.json',refs),
        ('source_manifest.json',sources),('protected_inputs.json',protected)]:write(FOLDER/name,value,True)
    write(FOLDER/'ledger.json',dict(status='PREPARED_UNAPPROVED',approved=False,prior_attempt_count=PRIOR,entries=[]),True)
    prepared={relative(p).split('/slc500_meep_validation_v1/',1)[1]:sha(p) for p in FOLDER.rglob('*')
              if p.is_file() and p.name!='ledger.json'}
    write(FOLDER/'preparation_manifest.json',prepared,True)
    print(json.dumps(dict(status='PREPARED_UNAPPROVED',folder=str(FOLDER),worker_launches=0,proposal_sha256=sha(FOLDER/'proposal.json'))))


def verify():
    for name,value in read(FOLDER/'preparation_manifest.json').items():protect(FOLDER/name,value)
    for name,value in read(FOLDER/'source_manifest.json').items():protect(ROOT/name,value)
    for name,value in read(FOLDER/'protected_inputs.json').items():protect(ROOT/name,value)
    if read(FOLDER/'plan.json')!=plan():raise ValueError('Prepared six-job scope changed')
    if sha(FOLDER/'config.json')!=CONFIG_SHA:raise ValueError('Frozen config changed')
    ledger=read(FOLDER/'ledger.json')
    if ledger['prior_attempt_count']!=PRIOR or len(ledger['entries'])>6 or [e['id'] for e in ledger['entries']]!=list(range(84,84+len(ledger['entries']))):
        raise ValueError('Preserve actual charged attempts84..89')
    for entry in ledger['entries']:exact_job(entry['job'])
    return ledger


def approve(question_file,reply_file):
    ledger=verify()
    if ledger['entries'] or (FOLDER/'approval_receipt.json').exists():raise RuntimeError('Approval cannot reset/reopen allocation')
    question=Path(question_file).read_bytes();reply=Path(reply_file).read_bytes()
    if not question.strip() or not re.match(r'^(approve\b|approved\b|i approve\b|yes\b)',reply.decode('utf-8').strip(),re.I):
        raise ValueError('Actual new budget question and explicitly affirmative user reply required')
    started=datetime.now(timezone.utc)
    (FOLDER/'authorization').mkdir()
    for name,data in [('question_exact.txt',question),('user_reply_exact.txt',reply)]:
        with (FOLDER/'authorization'/name).open('xb') as stream:stream.write(data)
    value=dict(status='APPROVED_BY_USER',decision_origin='NEW_EXPLICIT_USER_BUDGET_APPROVAL',approved_utc=started.isoformat(),
        absolute_deadline_utc=(started+timedelta(seconds=MAX_WALL)).isoformat(),proposal_sha256=sha(FOLDER/'proposal.json'),
        preparation_manifest_sha256=sha(FOLDER/'preparation_manifest.json'),additional_attempt_ceiling=6,cumulative_attempt_ceiling=89,
        question_exact=question.decode('utf-8'),question_sha256=hashlib.sha256(question).hexdigest(),
        user_reply_exact=reply.decode('utf-8'),user_reply_sha256=hashlib.sha256(reply).hexdigest(),
        one_local_CPU_worker=True,RAM_gates=False,no_reference_solves=True,prior_attempts=83,new_case_count=3)
    write(FOLDER/'approval_receipt.json',value,True)
    ledger.update(status='APPROVED_NOT_RUNNING',approved=True,approval_receipt_sha256=sha(FOLDER/'approval_receipt.json'))
    write(FOLDER/'ledger.json',ledger)


def authorization(ledger):
    if ledger.get('approved') is not True:raise RuntimeError('Actual new explicit6-solve/8-hour approval is still required')
    receipt=FOLDER/'approval_receipt.json';protect(receipt,ledger['approval_receipt_sha256']);value=read(receipt)
    required=dict(status='APPROVED_BY_USER',decision_origin='NEW_EXPLICIT_USER_BUDGET_APPROVAL',
        additional_attempt_ceiling=6,cumulative_attempt_ceiling=89,prior_attempts=83,new_case_count=3,
        one_local_CPU_worker=True,RAM_gates=False,no_reference_solves=True,
        proposal_sha256=sha(FOLDER/'proposal.json'),preparation_manifest_sha256=sha(FOLDER/'preparation_manifest.json'))
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('New allocation approval scope differs')
    for key,name in [('question','question_exact.txt'),('user_reply','user_reply_exact.txt')]:
        path=FOLDER/'authorization'/name
        if sha(path)!=value[key+'_sha256'] or path.read_text(encoding='utf-8')!=value[key+'_exact']:
            raise ValueError('Actual budget approval bytes changed')
    remaining(value);return value


def no_owned_worker():
    if sys.platform!='linux':raise RuntimeError('Actual execution requires existing WSL/Linux Meep environment')
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            args=[s.decode() for s in (folder/'cmdline').read_bytes().split(b'\0') if s]
            if any(Path(arg).name==WORKER.name for arg in args):
                cwd=(folder/'cwd').resolve()
                if cwd==ROOT.resolve() or any(str(ROOT.resolve()) in arg for arg in args):
                    raise RuntimeError('A project-owned optical worker is already active')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue


class Runner:
    def __init__(self):
        self.ledger=verify();self.approval=authorization(self.ledger);self.base=helpers();self.active=None
        self.config=read(FOLDER/'config.json');self.refs=read(FOLDER/'cache_catalog.json');self.cases=read(FOLDER/'frozen_cases.json')
        self.env=os.environ.copy();self.env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def save(self):
        self.ledger.update(updated_utc=utc(),additional_attempt_count=len(self.ledger['entries']),
            cumulative_attempt_count=PRIOR+len(self.ledger['entries']),elapsed_since_new_approval_s=(parse(utc())-parse(self.approval['approved_utc'])).total_seconds())
        write(FOLDER/'ledger.json',self.ledger)
    def terminate(self):
        if self.active is not None and self.active.poll() is None:
            os.killpg(self.active.pid,signal.SIGTERM)
            try:self.active.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(self.active.pid,signal.SIGKILL);self.active.wait()
    def wait(self,timeout,args):
        started=time.monotonic()
        while True:
            left=min(timeout-(time.monotonic()-started),remaining(self.approval))
            if left<=0:self.terminate();raise subprocess.TimeoutExpired(args,timeout)
            try:code=self.active.wait(timeout=min(30.,left))
            except subprocess.TimeoutExpired:continue
            if remaining(self.approval)<=0 or time.monotonic()-started>=timeout:
                self.terminate();raise subprocess.TimeoutExpired(args,timeout)
            return code
    def call(self,args,log,timeout,entry,analysis=False):
        self.active=None
        with log.open('x') as stream:
            self.active=subprocess.Popen(args,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            entry['analysis_owned_pid' if analysis else 'owned_pid']=self.active.pid
            if not analysis:entry['solver_launched']=True
            self.save();return self.wait(timeout,args)
    def analyze(self,entry):
        if entry.get('analysis_status')=='completed':
            protect(FOLDER/entry['analysis_file'],entry['analysis_sha256']);return
        verify();authorization(self.ledger)
        n=len(entry.get('analysis_attempts',[]))+1
        output=FOLDER/f'analyses/{entry["id"]}.analysis-{n:02d}.json';log=FOLDER/f'logs/{entry["id"]}.analysis-{n:02d}.log'
        if output.exists() or log.exists():raise FileExistsError('Preserve prior saved-field analysis attempt')
        output.parent.mkdir(exist_ok=True);case=next(c for c in self.cases if c['case_id']==entry['job']['case_id'])
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ANALYZER),'--record',str(FOLDER/entry['record_file']),
            '--config',str(FOLDER/'config.json'),'--target-diameter',str(case['target_diameter_um']),'--output',str(output),'--count','512','--window','128']
        attempt=dict(started_utc=utc(),status='started',command=args,output_file=relative(output),log_file=relative(log))
        entry.setdefault('analysis_attempts',[]).append(attempt);entry['analysis_status']='started';self.save()
        try:
            timeout=min(600.,remaining(self.approval))
            if timeout<=1:raise RuntimeError('Deadline before analysis')
            code=self.call(args,log,timeout,entry,True);attempt['exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Saved-field analysis failed; no new solve is authorized by recovery')
            result=read(output)
            if (local(result['record'])!=local(FOLDER/entry['record_file']) or result['settings']!=entry['effective_settings']
                    or result['declared_target_diameter_um']!=case['target_diameter_um']):raise ValueError('Analysis record/settings/frozen target differs')
            entry.update(analysis_status='completed',analysis_file=str(output.relative_to(FOLDER)),analysis_sha256=sha(output),
                useful_modal_efficiency=result['conservative_useful_efficiency'],native_pass=result['reconstruction_pass'],
                nominal_optical_pass=result['synthetic_optical_pass'])
            attempt['status']='completed'
        except BaseException as error:
            self.terminate();entry['analysis_status']=attempt['status']='failed';attempt['error']=repr(error);raise
        finally:attempt['finished_utc']=utc();self.save()
    def cached(self,job):
        entries=[e for e in self.ledger['entries'] if e['id']==job['id']]
        if not entries:return None
        if len(entries)!=1 or entries[0]['status']!='completed':raise RuntimeError('Failed/incomplete device cannot retry unchanged settings')
        entry=entries[0];self.checked_device(entry)
        return entry
    def checked_device(self,entry):
        job=entry['job'];exact_job(job);path=local(FOLDER/entry['record_file'])
        if not path.is_relative_to(local(FOLDER/'records')):raise ValueError('New device record must remain in its own records folder')
        protect(path,entry['record_sha256']);record=read(path)
        case=next(c for c in self.cases if c['case_id']==job['case_id'])
        ref=next(r for r in self.refs if r['H_um']==job['H_um'] and r['resolution']==job['resolution'])
        effective=self.base.u.settings_for(self.config,job)
        self.base.validate_worker_output(record,job,effective,digest(self.config),SOURCE_PINS[relative(WORKER)],
            FOLDER/'config.json',CONFIG_SHA,FOLDER/case['design_file'],case['design_sha256'],ROOT/ref['record_file'],ref['record_sha256'])
        for key,suffix in [('fields_file','.fields.npz'),('power_audit_fields_file','.power-audit.npz')]:
            if local(record[key])!=path.with_suffix(suffix):raise ValueError('Device NPZ paths must belong to exact new record')
        actual=required_files(record)
        if actual!=entry['required_file_sha256']:raise ValueError('New device required field hashes differ')
        return record
    def execute(self,job):
        exact_job(job);verify();authorization(self.ledger)
        cached=self.cached(job)
        if cached is not None:
            self.analyze(cached)
            if not cached.get('native_pass'):raise RuntimeError('Native reconstruction audit failed; diagnose saved evidence before any further solve')
            return
        timeout=capacity(len(self.ledger['entries']),self.approval)
        pending=[j for j in plan() if not any(e['id']==j['id'] for e in self.ledger['entries'])]
        if sum(j['forecast_worker_s']+j['analysis_allowance_s'] for j in pending)>remaining(self.approval):
            raise RuntimeError('Remaining measured forecast does not fit allocation')
        no_owned_worker();ref=next(r for r in self.refs if r['H_um']==job['H_um'] and r['resolution']==job['resolution'])
        case=next(c for c in self.cases if c['case_id']==job['case_id']);effective=self.base.u.settings_for(self.config,job)
        if ref['settings']!=effective:raise ValueError('No exact matching reference; fallback solves forbidden')
        label=f'{job["id"]}_SLC500_H{job["H_um"]:g}_r{job["resolution"]:g}'
        output=FOLDER/'records'/(label+'.json');log=FOLDER/'logs'/(label+'.log')
        for folder in (output.parent,log.parent):folder.mkdir(exist_ok=True)
        if output.exists() or log.exists():raise FileExistsError('Preserve stale raw/log evidence')
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(WORKER),'--config',str(FOLDER/'config.json'),'--design',str(FOLDER/case['design_file']),
            '--reference',str(ROOT/ref['record_file']),'--output',str(output),'--H',str(job['H_um']),'--resolution',str(job['resolution']),
            '--pml','2.0','--padding','10.0','--duration',str(job['duration'])]
        pre=FOLDER/f'prelaunch/{job["id"]}';pre.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(FOLDER/'ledger.json',pre/'prior_ledger.json')
        receipt=dict(status='PROSPECTIVE_NEW_SLC500_DEVICE_BEFORE_POPEN',created_utc=utc(),job=job,command=args,
            effective_settings=effective,approval_receipt_sha256=sha(FOLDER/'approval_receipt.json'),
            preparation_manifest_sha256=sha(FOLDER/'preparation_manifest.json'),driver_sha256=sha(__file__),
            source_manifest_sha256=sha(FOLDER/'source_manifest.json'),config_sha256=CONFIG_SHA,case=case,reference=ref,
            prior_ledger_sha256=sha(pre/'prior_ledger.json'),charged_before_launch=len(self.ledger['entries']),
            cumulative_before_launch=PRIOR+len(self.ledger['entries']),remaining_s=remaining(self.approval),timeout_s=timeout)
        write(pre/'receipt.json',receipt,True)
        entry=dict(id=job['id'],job=job,status='started',started_utc=utc(),command=args,solver_launched=False,
            record_file=str(output.relative_to(FOLDER)),log_file=str(log.relative_to(FOLDER)),effective_settings=effective,
            prelaunch_receipt_file=str((pre/'receipt.json').relative_to(FOLDER)),prelaunch_receipt_sha256=sha(pre/'receipt.json'),
            case_id=case['case_id'],physical_geometry_id=case['physical_geometry_id'],design_sha256=case['design_sha256'],
            reference_record_sha256=ref['record_sha256'])
        self.ledger['entries'].append(entry);self.save();started=time.monotonic()
        try:
            verify();protect(pre/'receipt.json',entry['prelaunch_receipt_sha256'])
            code=self.call(args,log,min(timeout,remaining(self.approval)),entry);entry['solver_exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Meep device failed; charged attempt retained')
            record=read(output)
            self.base.validate_worker_output(record,job,effective,digest(self.config),SOURCE_PINS[relative(WORKER)],
                FOLDER/'config.json',CONFIG_SHA,FOLDER/case['design_file'],case['design_sha256'],ROOT/ref['record_file'],ref['record_sha256'])
            entry.update(status='completed',record_sha256=sha(output),required_file_sha256=required_files(record))
            self.checked_device(entry)
        except BaseException as error:
            self.terminate();entry.update(status='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed',error=repr(error))
            if self.active is not None:entry['solver_exit_code']=self.active.returncode
            raise
        finally:entry.update(finished_utc=utc(),wrapper_wall_s=time.monotonic()-started);self.save()
        self.analyze(entry)
        if not entry.get('native_pass'):raise RuntimeError('Native reconstruction audit failed; diagnose saved evidence before any further solve')
        print(json.dumps(dict(event='DEVICE_COMPLETED',id=job['id'],H_um=job['H_um'],resolution=job['resolution'],eta=entry['useful_modal_efficiency'])),flush=True)
    def summary(self):
        reporter=module(ROOT/'scripts/report_height_family_maxwell_v2.py',SOURCE_PINS['scripts/report_height_family_maxwell_v2.py'])
        result=dict(status='COMPLETED_SIX_DEVICE_SOLVES_REQUIRES_SCIENTIFIC_REVIEW',additional_attempts=len(self.ledger['entries']),
            cumulative_attempts=PRIOR+len(self.ledger['entries']),reference_solves=0,heights=[],global_acceptance_claimed=False,
            interpretation='Direct CPU Meep of circular-annulus equivalents of existing500nm SLC profiles; synthetic fiber Gaussian, not experiment')
        for H in (40,75,100):
            pair=[e for e in self.ledger['entries'] if e['job']['H_um']==H]
            analyses=[read(FOLDER/e['analysis_file']) for e in pair]
            result['heights'].append(dict(H_um=H,results=[dict(resolution=e['job']['resolution'],analysis_file=e['analysis_file'],
                analysis_sha256=e['analysis_sha256'],useful_modal_efficiency=a['conservative_useful_efficiency'],
                useful_modal_efficiency_percent=100*a['conservative_useful_efficiency'],native_pass=a['reconstruction_pass'],
                nominal_optical_pass=a['synthetic_optical_pass']) for e,a in zip(pair,analyses)],
                mesh_last_pair=reporter.compare_analyses(analyses[0],analyses[1],self.config,LIMITS)))
        version=1
        while (FOLDER/f'results_v{version}.json').exists():version+=1
        write(FOLDER/f'results_v{version}.json',result,True)
    def run(self):
        no_owned_worker();self.ledger['status']='RUNNING_APPROVED_NEW_SLC500_ALLOCATION';self.save()
        try:
            for job in plan():self.execute(job)
            self.summary();self.ledger['status']='COMPLETED_REQUIRES_REVIEW'
        except BaseException as error:
            self.terminate();self.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS';self.ledger.setdefault('failures',[]).append(dict(utc=utc(),reason=repr(error)));raise
        finally:self.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--prepare',action='store_true');mode.add_argument('--approve',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--question-file',type=Path);parser.add_argument('--user-reply-file',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare();return
    if args.approve:
        if not args.question_file or not args.user_reply_file:parser.error('Approval requires actual question and user-reply files')
        approve(args.question_file,args.user_reply_file);return
    if not args.run:
        value=dict(status='INSPECTION_ONLY',worker_launches=0,proposed_jobs=plan(),additional_attempt_cap=6,wall_ceiling_s=MAX_WALL,
            preparation_exists=FOLDER.exists(),approval_received=False)
        if FOLDER.exists():
            ledger=verify();value.update(status=ledger['status'],approval_received=ledger.get('approved',False),charged_attempts=len(ledger['entries']))
        print(json.dumps(value,indent=2));return
    import fcntl
    lock=CAMPAIGN/'maxwell_validation_v1/runner.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Runner()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:(_ for _ in ()).throw(KeyboardInterrupt('Stop requested')))
        runner.run()


if __name__=='__main__':main()
