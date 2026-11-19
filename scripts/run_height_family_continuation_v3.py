"""Approved stage2 CPU validation; only explicit phaseA may launch workers.

The first45 attempts and all stage1 bytes stay intact. References are reused
only after immutable-source, actual-setting, physical-case and raw/field SHA
checks. Every new worker attempt consumes one of20 additional slots, including
failures; saved-field analysis recovery consumes time but never another solve.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
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
PRIOR=CAMPAIGN/'maxwell_validation_v1'
FOLDER=CAMPAIGN/'maxwell_validation_v2'
WORKER=ROOT/'scripts/height_family_maxwell_worker_v2.py'
ANALYZER=ROOT/'scripts/height_family_maxwell_analysis_v2.py'
AUTHORIZATION=CAMPAIGN/'approval_message_2026-11-13.txt'
PRIOR_ATTEMPTS=45
MAX_NEW=20
MAX_CUMULATIVE=65
MAX_WALL=172800.
RESERVE=185.
CHILD_TIMEOUT=21600.
EXPECTED_PRIOR_LEDGER_SHA='7b38ca7d2c5d543fd26284cec9c0370749f335d17cf3c749376ac770ce91b693'
EXPECTED_PRIOR_ARCHIVE_SHA='a18dc723b2513946f6cadc6afabb794a2fcb529f697be997d5beb87c3b3305fd'
KEYS=('H_um','resolution','pml_um','padding_um','duration')


def utc():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)
def parse_time(value):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware stage clock required')
    return result.astimezone(timezone.utc)
def elapsed_absolute(ledger,now=None):
    recorded=float(ledger.get('elapsed_s',0))
    if not math.isfinite(recorded) or recorded<0:raise ValueError('Invalid recorded elapsed time')
    if not ledger.get('first_started_utc'):return recorded
    now=datetime.now(timezone.utc) if now is None else parse_time(now)
    delta=(now-parse_time(ledger['first_started_utc'])).total_seconds()
    if delta<0:raise ValueError('Stage starts in future')
    return max(recorded,delta)
def capacity(new_attempts,elapsed,prior=PRIOR_ATTEMPTS):
    if prior!=PRIOR_ATTEMPTS or new_attempts>=MAX_NEW or prior+new_attempts>=MAX_CUMULATIVE:
        raise RuntimeError('Approved additional/cumulative solve limit reached')
    remaining=MAX_WALL-RESERVE-float(elapsed)
    if remaining<=1:raise RuntimeError('Approved stage deadline/reporting reserve reached')
    return min(CHILD_TIMEOUT,remaining)
def case_id(method,H):return f'{method}_H{float(H):g}'
def identity(job):return (job['method'],)+tuple(float(job[key]) for key in KEYS)
def require_permitted_attempt(item,old_entries,new_entries):
    if any(identity(entry)==identity(item) for entry in new_entries):
        raise RuntimeError('Previous new-stage settings failed/incomplete; unchanged retries are forbidden')
    failed=[entry for entry in old_entries if identity(entry)==identity(item) and entry['status']!='completed']
    if failed:
        allowed=(identity(item)==('reference',100.,24.,2.,10.,335.) and item.get('retry_of_prior_id')==41
                 and [entry['id'] for entry in failed]==[41])
        if not allowed:raise RuntimeError('Only explicitly planned charged retry of prior41 is permitted')
def duration(H):return float(max(160,math.ceil(2*1.5263*(H+3)+20)))
def job(method,H,resolution,phase,kind='nominal',pml=2.,padding=10.,duration_override=None):
    return dict(method=method,H_um=float(H),resolution=float(resolution),phase=phase,job_type=kind,
                pml_um=float(pml),padding_um=float(padding),duration=duration(H) if duration_override is None else float(duration_override))
def prospective_plan():
    phaseA=[]
    for H in (100,50,60,80,90):
        for method in ('reference','NN'):
            item=job(method,H,24,'A')
            if H==100 and method=='reference':item['retry_of_prior_id']=41
            phaseA.append(item)
    phaseB=[job('direct',40,24,'B'),job('reference',75,24,'B'),job('NN',75,24,'B'),
            job('direct',75,24,'B'),job('direct',100,24,'B')]
    return dict(status='ONLY_PHASE_A_EXECUTABLE',phase_A=phaseA,prospective_phase_B=phaseB,
                prospective_phase_C=[job('reference',70,32,'C',kind='refinement'),job('NN',70,32,'C',kind='refinement')],
                prospective_phase_D=[job('direct',40,32,'D')],prospective_adaptive_slots=2,
                planned_additional_attempts=20,
                adaptive_note='Two reserved attempts require a fresh prospective same-geometry numerical choice; not launched by this script',
                incomplete_original_controls='H100 PML3 and H70 padding15 remain unresolved unless later reallocated; no full acceptance promised')


def pure_worker():
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('frozen_worker_setup_for_stage2',WORKER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def settings_for(config,item):
    return pure_worker().configured_dimensions(config,item['H_um'],item['resolution'],
        pml=item['pml_um'],padding=item['padding_um'],duration=item['duration'],m=1)
def geometry_identity(design,config):
    return fingerprint(dict(H_um=float(design['H_um']),cylinder_diameter_um=design['cylinder_diameter_um'],
                            heights_um=design['heights_um'],materials=config['materials']))
def required_files(record):
    keys=['fields_file','power_audit_fields_file']
    if 'incident_power' in record:keys.append('flux_data_file')
    result={}
    for key in keys:
        path=Path(record[key])
        if not path.is_file():raise ValueError('Required saved field evidence missing: '+str(path))
        result[str(path)]=sha(path)
    return result
def checked_record(entry,origin):
    path=Path(origin)/entry['record_file']
    if not path.resolve().is_relative_to(Path(origin).resolve()):raise ValueError('Record path escapes its stage')
    if sha(path)!=entry['record_sha256']:raise ValueError('Saved raw record hash changed')
    record=read(path)
    if any(record['settings'].get(key)!=entry[key] for key in KEYS):raise ValueError('Ledger/raw numerical settings differ')
    return path,record
def check_delivered_file(path,delivery,origin=PRIOR,digest=None):
    path=Path(path).resolve();origin=Path(origin).resolve()
    if not path.is_relative_to(origin):raise ValueError('Prior cached evidence leaves original delivered stage')
    name=path.relative_to(origin).as_posix();expected=delivery['files'].get(name)
    if expected is None or (sha(path) if digest is None else digest)!=expected['sha256']:
        raise ValueError('Prior evidence differs from original sealed delivery manifest: '+name)


def require_no_live_children(entries,probe=None):
    def owned_active(entry):
        pid=entry.get('owned_pid')
        if not pid:return False
        try:os.kill(int(pid),0)
        except ProcessLookupError:return False
        except PermissionError:return True
        try:
            args=[p.decode() for p in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if p]
        except FileNotFoundError:return False
        except PermissionError:return True
        # A reused number belongs to an unrelated process unless its entire
        # worker argv, including unique workspace/config/output, still matches.
        return not args or args==entry.get('command')
    probe=owned_active if probe is None else probe
    for entry in entries:
        if probe(entry):raise RuntimeError('A proven prior/current owned worker is still active')


def prepare(user_message_file=AUTHORIZATION):
    if FOLDER.exists():raise FileExistsError('Preserve existing stage2; use explicit --run-phase A to resume')
    if sha(PRIOR/'ledger.json')!=EXPECTED_PRIOR_LEDGER_SHA:raise ValueError('Prior45 ledger differs from reviewed immutable stage1')
    previous=read(PRIOR/'ledger.json')
    if len(previous['entries'])!=45:raise ValueError('Exactly45 prior attempts required')
    require_no_live_children(previous['entries'])
    if previous['status']=='RUNNING_APPROVED':raise RuntimeError('Original runner is still running')
    message=Path(user_message_file).read_text(encoding='utf-8-sig')
    if not message:raise ValueError('Exact user authorization text required')
    config=read(PRIOR/'config.json');old_manifest=read(PRIOR/'source_manifest.json')
    archive=CAMPAIGN/'height_specific_Maxwell_validation_v1.zip'
    if sha(archive)!=EXPECTED_PRIOR_ARCHIVE_SHA:raise ValueError('Original stage1 archive seal changed')
    with zipfile.ZipFile(archive) as archived:
        if archived.read('delivery_manifest.json')!=(PRIOR/'delivery_manifest.json').read_bytes():
            raise ValueError('Prior delivery manifest differs from original sealed archive')
    delivery=read(PRIOR/'delivery_manifest.json')
    for name,digest in old_manifest.items():
        if sha(ROOT/name)!=digest or sha(PRIOR/'source_snapshots'/name)!=digest:
            raise ValueError('Original frozen physics source changed: '+name)
    FOLDER.mkdir()
    shutil.copyfile(PRIOR/'config.json',FOLDER/'config.json')
    shutil.copyfile(PRIOR/'ledger.json',FOLDER/'prior_ledger.json')
    shutil.copyfile(PRIOR/'delivery_manifest.json',FOLDER/'prior_delivery_manifest.json')
    shutil.copyfile(user_message_file,FOLDER/'user_authorization_exact.txt')
    write(FOLDER/'approval_receipt.json',dict(status='APPROVED_BY_USER',recorded_utc=utc(),
        user_message_exact=message,user_message_sha256=sha(user_message_file),
        limits=dict(additional_wall_s=172800,additional_CPU_solves=20,cumulative_CPU_solves=65,
                    prior_attempts=45,cumulative_unique=28,additional_unique=4,workers=1,RAM_gates=False,
                    child_timeout_s=21600,reporting_cleanup_reserve_s=185),
        interpretation='Local frozen-profile validation and requested NN/direct comparison only; no retraining, printing, external compute or system changes'))
    write(FOLDER/'prior_stage.json',dict(folder=str(PRIOR.resolve()),ledger_file=str((PRIOR/'ledger.json').resolve()),
        ledger_sha256=sha(PRIOR/'ledger.json'),config_sha256=sha(PRIOR/'config.json'),
        source_manifest_sha256=sha(PRIOR/'source_manifest.json'),prior_attempt_count=45,
        delivery_manifest_sha256=sha(PRIOR/'delivery_manifest.json'),archive_sha256=EXPECTED_PRIOR_ARCHIVE_SHA,
        prior_unique_case_count=24,prior_status=previous['status']))
    cases=[]
    old_cases=read(PRIOR/'cumulative_amendment.json')['cases']
    inputs=[(case['method'],case['H_um'],PRIOR/case['design_file'],False,case['design_sha256']) for case in old_cases]
    for method,H in [('direct',40),('NN',75),('direct',75),('direct',100)]:
        path=CAMPAIGN/f'designs/H{H}/{method}.json';inputs.append((method,H,path,True,sha(path)))
    for method,H,source,is_new,digest in inputs:
        if sha(source)!=digest:raise ValueError('Frozen physical case changed')
        output=FOLDER/'frozen_designs'/(case_id(method,H)+'.json');output.parent.mkdir(exist_ok=True)
        shutil.copyfile(source,output);design=read(output)
        cases.append(dict(case_id=case_id(method,H),method=method,H_um=float(H),new_case=is_new,
                          design_file=str(output.relative_to(FOLDER)),design_sha256=digest,
                          physical_geometry_id=geometry_identity(design,config),source_file=str(source.resolve()),
                          height_domain=design.get('height_domain','training'),
                          H75_note='Validation/seed-selection height; not heldout' if H==75 else None))
    if len(cases)!=15 or len({case['physical_geometry_id'] for case in cases})!=15:
        raise ValueError('Expected11 prior and4 distinct new physical cases')
    write(FOLDER/'frozen_cases.json',dict(status='FROZEN_AUTHORIZED_CASES',cases=cases,
          original_stage_cases=11,new_stage_cases=4,prior_cumulative_unique=24,cumulative_unique_ceiling=28))
    catalog=[]
    for entry in previous['entries']:
        if entry['status']!='completed':continue
        path,record=checked_record(entry,PRIOR)
        check_delivered_file(path,delivery,PRIOR,entry['record_sha256'])
        hashes=required_files(record)
        for filename,digest in hashes.items():check_delivered_file(filename,delivery,PRIOR,digest)
        if entry['method']!='reference':
            analysis=PRIOR/entry['analysis_file']
            if entry.get('analysis_status')!='completed' or sha(analysis)!=entry['analysis_sha256']:
                raise ValueError('Prior completed independent analysis SHA changed')
            check_delivered_file(analysis,delivery,PRIOR,entry['analysis_sha256'])
        catalog.append(dict(entry=entry,origin='prior_stage1',record_path=str(path.resolve()),
                            required_file_sha256=hashes))
    write(FOLDER/'prior_cache_catalog.json',dict(status='COMPLETED_OLD_EVIDENCE_ONLY',records=catalog))
    protected=read(CAMPAIGN/'provenance.json')['protected_before']
    sources={**old_manifest,str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for name in sources:
        out=FOLDER/'source_snapshots'/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,out)
    write(FOLDER/'source_manifest.json',sources)
    write(FOLDER/'protected_historical.json',protected)
    write(FOLDER/'plan.json',prospective_plan())
    immutable=['config.json','approval_receipt.json','prior_stage.json','prior_ledger.json','prior_delivery_manifest.json','user_authorization_exact.txt',
               'frozen_cases.json','prior_cache_catalog.json','source_manifest.json','protected_historical.json','plan.json']
    write(FOLDER/'preparation_manifest.json',{name:sha(FOLDER/name) for name in immutable})
    write(FOLDER/'ledger.json',dict(status='PREPARED_APPROVED_PHASE_A_ONLY',entries=[],prior_attempt_count=45,
          total_attempted_solves=0,cumulative_attempted_solves=45,elapsed_s=0,
          approval_sha256=sha(FOLDER/'approval_receipt.json')))
    print(json.dumps(dict(status='PREPARED_APPROVED_PHASE_A_ONLY',folder=str(FOLDER),cases=15,new_cases=4,
                          phase_A_solves=10,no_solver_launched=True)),flush=True)


class Stage:
    def __init__(self):
        self.ledger=read(FOLDER/'ledger.json');self.receipt=read(FOLDER/'approval_receipt.json')
        self.prior=read(FOLDER/'prior_stage.json');self.cases=read(FOLDER/'frozen_cases.json')['cases']
        self.config=read(FOLDER/'config.json');self.config_fingerprint=fingerprint(self.config)
        self.catalog=read(FOLDER/'prior_cache_catalog.json')['records'];self.old_entries=read(FOLDER/'prior_ledger.json')['entries']
        self.active=None;self.clock_start=time.monotonic();self.elapsed_at_start=elapsed_absolute(self.ledger)
        self.env=os.environ.copy();self.env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.verify()
    def elapsed(self):
        return max(elapsed_absolute(self.ledger),self.elapsed_at_start+time.monotonic()-self.clock_start)
    def save(self):
        self.ledger.update(elapsed_s=self.elapsed(),updated_utc=utc(),total_attempted_solves=len(self.ledger['entries']),
                           cumulative_attempted_solves=45+len(self.ledger['entries']))
        write(FOLDER/'ledger.json',self.ledger)
    def verify(self):
        if self.receipt['status']!='APPROVED_BY_USER' or self.ledger['prior_attempt_count']!=45:
            raise ValueError('Original explicit stage2 approval and prior count required')
        limits=self.receipt['limits']
        if (limits['additional_wall_s']!=172800 or limits['additional_CPU_solves']!=20
                or limits['cumulative_CPU_solves']!=65 or limits['cumulative_unique']!=28 or limits['workers']!=1):
            raise ValueError('Approved stage2 limits changed')
        for name,digest in read(FOLDER/'preparation_manifest.json').items():
            if sha(FOLDER/name)!=digest:raise ValueError('Prepared stage2 evidence changed: '+name)
        if sha(PRIOR/'ledger.json')!=self.prior['ledger_sha256'] or self.prior['ledger_sha256']!=EXPECTED_PRIOR_LEDGER_SHA:
            raise ValueError('Protected prior45 ledger changed')
        if sha(PRIOR/'config.json')!=self.prior['config_sha256'] or sha(FOLDER/'config.json')!=self.prior['config_sha256']:
            raise ValueError('Identical physical configuration required for old reference reuse')
        if sha(PRIOR/'source_manifest.json')!=self.prior['source_manifest_sha256']:
            raise ValueError('Prior source manifest changed')
        if (sha(PRIOR/'delivery_manifest.json')!=self.prior['delivery_manifest_sha256']
                or sha(FOLDER/'prior_delivery_manifest.json')!=self.prior['delivery_manifest_sha256']):
            raise ValueError('Original sealed prior delivery manifest changed')
        for name,digest in read(FOLDER/'source_manifest.json').items():
            if sha(ROOT/name)!=digest or sha(FOLDER/'source_snapshots'/name)!=digest:
                raise ValueError('Frozen execution source changed: '+name)
        for name,digest in read(FOLDER/'protected_historical.json').items():
            if sha(ROOT/name)!=digest:raise ValueError('Protected historical evidence changed')
        if len(self.cases)!=15 or sum(case['new_case'] for case in self.cases)!=4:
            raise ValueError('Frozen15-case/cumulative28 allocation changed')
        for case in self.cases:
            if sha(FOLDER/case['design_file'])!=case['design_sha256']:
                raise ValueError('Frozen case bytes changed')
            if geometry_identity(read(FOLDER/case['design_file']),self.config)!=case['physical_geometry_id']:
                raise ValueError('Frozen physical geometry identity changed')
        entries=self.ledger['entries']
        if [entry['id'] for entry in entries]!=list(range(46,46+len(entries))) or len(entries)>20:
            raise ValueError('New attempt IDs/count must remain46..65')
    def case(self,item):
        return next(case for case in self.cases if case['method']==item['method'] and case['H_um']==item['H_um'])
    def evidence(self,entry,origin,hashes):
        path,record=checked_record(entry,origin)
        if record['config_fingerprint']!=self.config_fingerprint:raise ValueError('Cached physics configuration differs')
        for name,digest in hashes.items():
            if not Path(name).is_file() or sha(name)!=digest:raise ValueError('Cached required field SHA changed')
        if set(hashes)!=set(required_files(record)):raise ValueError('Incomplete required-field SHA inventory')
        if entry['method']!='reference':
            case=self.case(entry)
            if entry.get('design_sha256')!=case['design_sha256']:raise ValueError('Cached geometry differs')
            analysis_path=Path(origin)/entry['analysis_file'] if entry.get('analysis_file') else None
            if origin==PRIOR and (entry.get('analysis_status')!='completed' or analysis_path is None
                                  or sha(analysis_path)!=entry['analysis_sha256']):
                raise ValueError('Prior cached device analysis SHA/status differs')
        if origin==PRIOR:
            delivery=read(FOLDER/'prior_delivery_manifest.json')
            check_delivered_file(path,delivery,PRIOR,entry['record_sha256'])
            for name,digest in hashes.items():check_delivered_file(name,delivery,PRIOR,digest)
            if entry['method']!='reference':
                check_delivered_file(analysis_path,delivery,PRIOR,entry['analysis_sha256'])
        return path,record
    def cache(self,item):
        self.verify()
        candidates=[]
        for wrapped in self.catalog:
            entry=wrapped['entry']
            if entry['status']=='completed' and identity(entry)==identity(item):
                candidates.append((entry,PRIOR,wrapped['required_file_sha256'],'prior_stage1'))
        for entry in self.ledger['entries']:
            if entry['status']=='completed' and identity(entry)==identity(item):
                candidates.append((entry,FOLDER,entry['required_file_sha256'],'stage2'))
        if not candidates:return None
        entry,origin,hashes,stage=candidates[-1]
        path,record=self.evidence(entry,origin,hashes)
        if record['settings']!=settings_for(self.config,item):raise ValueError('Cached full actual settings differ')
        return dict(entry=entry,path=path,record=record,stage=stage)
    def terminate(self):
        if self.active is not None and self.active.poll() is None:
            os.killpg(self.active.pid,signal.SIGTERM)
            try:self.active.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(self.active.pid,signal.SIGKILL);self.active.wait()
    def analyze(self,entry):
        if entry['method']=='reference':return
        if entry.get('analysis_status')=='completed':
            path=FOLDER/entry['analysis_file']
            if sha(path)!=entry['analysis_sha256']:raise ValueError('Completed analysis SHA changed')
            return
        self.verify();remaining=MAX_WALL-RESERVE-self.elapsed()
        if remaining<=1:raise RuntimeError('Stage deadline reached before analysis')
        case=self.case(entry);label=Path(entry['record_file']).stem
        attempt=len(entry.get('analysis_attempts',[]))+1
        while True:
            stem=f'{label}.analysis-{attempt:02d}';output=FOLDER/'analyses'/(stem+'.json');log=FOLDER/'logs'/(stem+'.log')
            if not output.exists() and not log.exists():break
            attempt+=1
        output.parent.mkdir(exist_ok=True);log.parent.mkdir(exist_ok=True)
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ANALYZER),'--record',str(FOLDER/entry['record_file']),
              '--config',str(FOLDER/'config.json'),'--target-diameter',str(read(FOLDER/case['design_file'])['target_diameter_um']),
              '--output',str(output)]
        audit=dict(attempt=attempt,status='started',started_utc=utc(),command=args,log_file=str(log.relative_to(FOLDER)))
        entry.setdefault('analysis_attempts',[]).append(audit);entry['analysis_status']='started';self.save()
        try:
            with log.open('x') as stream:
                result=subprocess.run(args,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,
                                      timeout=min(600.,remaining),text=True)
            audit['exit_code']=result.returncode
            if result.returncode!=0 or not output.is_file():raise RuntimeError('Saved-field analysis failed; recover without another solver')
            value=read(output)
            entry.update(analysis_status='completed',analysis_file=str(output.relative_to(FOLDER)),analysis_sha256=sha(output),
                reconstruction_pass=value['reconstruction_pass'],nominal_optical_pass=value['synthetic_optical_pass'],
                C=value['normalized_curvature'],eta=value['conservative_useful_efficiency'])
            audit['status']='completed';audit['analysis_file']=entry['analysis_file'];audit['analysis_sha256']=entry['analysis_sha256']
        except BaseException as error:
            entry['analysis_status']=audit['status']='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed'
            audit['error']=repr(error);raise
        finally:audit['finished_utc']=utc();self.save()
    def execute(self,item):
        cached=self.cache(item)
        if cached:
            if cached['stage']=='stage2':self.analyze(cached['entry'])
            return cached
        require_permitted_attempt(item,self.old_entries,self.ledger['entries'])
        timeout=capacity(len(self.ledger['entries']),self.elapsed())
        require_no_live_children(self.old_entries+self.ledger['entries']);self.verify()
        reference=None
        if item['method']!='reference':
            reference=self.cache(dict(item,method='reference'))
            if reference is None:raise ValueError('Exactly matching completed homogeneous reference required')
            closure=reference['record']['power_balance_error']
            if not math.isfinite(closure) or not 0<=closure<.05:raise RuntimeError('Reference native closure failed')
        number=46+len(self.ledger['entries']);label=f"{number:02d}_{item['method']}_H{item['H_um']:g}_r{item['resolution']:g}_{item['job_type']}"
        output=FOLDER/'records'/(label+'.json');log=FOLDER/'logs'/(label+'.log')
        output.parent.mkdir(exist_ok=True);log.parent.mkdir(exist_ok=True)
        if output.exists() or log.exists():raise FileExistsError('Preserve all prior output/log attempts')
        args=[str(ROOT/'.venv-meep/bin/python'),'-B',str(WORKER),'--config',str(FOLDER/'config.json'),
              '--output',str(output),'--H',str(item['H_um']),'--resolution',str(item['resolution']),
              '--pml',str(item['pml_um']),'--padding',str(item['padding_um']),'--duration',str(item['duration'])]
        entry=dict(item,id=number,status='started',started_utc=utc(),record_file=str(output.relative_to(FOLDER)),
                   log_file=str(log.relative_to(FOLDER)),command=args)
        if reference:
            case=self.case(item);entry.update(case_id=case['case_id'],design_sha256=case['design_sha256'],
                physical_geometry_id=case['physical_geometry_id'],reference_id=reference['entry']['id'],reference_stage=reference['stage'])
            args+=['--design',str(FOLDER/case['design_file']),'--reference',str(reference['path'])]
        self.ledger['entries'].append(entry);self.save();started=time.monotonic()
        print(json.dumps(dict(event='SOLVE_STARTED',id=number,method=item['method'],H_um=item['H_um'],resolution=item['resolution'],timeout_s=timeout)),flush=True)
        try:
            remaining=MAX_WALL-RESERVE-self.elapsed()
            if remaining<=1:raise RuntimeError('Stage deadline/reporting reserve reached before worker launch')
            timeout=min(timeout,remaining)
            with log.open('x') as stream:
                self.active=subprocess.Popen(args,cwd=ROOT,env=self.env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                entry['owned_pid']=self.active.pid;self.save()
                code=self.active.wait(timeout=timeout);entry['solver_exit_code']=code
            if code!=0 or not output.is_file():raise RuntimeError('Meep worker failed; preserve and diagnose')
            record=read(output)
            if record['settings']!=settings_for(self.config,item) or record['config_fingerprint']!=self.config_fingerprint:
                raise ValueError('Worker actual settings/configuration differ')
            entry.update(status='completed',record_sha256=sha(output),required_file_sha256=required_files(record),
                         power_balance_error=record['power_balance_error'])
        except BaseException as error:
            self.terminate();entry['status']='timeout' if isinstance(error,subprocess.TimeoutExpired) else 'failed'
            entry['error']=repr(error)
            if self.active is not None:entry['solver_exit_code']=self.active.returncode
            raise
        finally:entry.update(wrapper_wall_s=time.monotonic()-started,finished_utc=utc());self.save()
        self.analyze(entry)
        print(json.dumps(dict(event='SOLVE_COMPLETED',id=number,closure=entry['power_balance_error'],C=entry.get('C'),eta=entry.get('eta'))),flush=True)
        return dict(entry=entry,path=output,record=record,stage='stage2')
    def run_phase_A(self):
        require_no_live_children(self.old_entries+self.ledger['entries'])
        if not self.ledger.get('first_started_utc'):
            self.ledger['first_started_utc']=utc();self.elapsed_at_start=0.;self.clock_start=time.monotonic()
            self.ledger['absolute_deadline_utc']=(parse_time(self.ledger['first_started_utc'])+timedelta(seconds=MAX_WALL)).isoformat()
        self.ledger['status']='RUNNING_APPROVED_PHASE_A';self.save()
        try:
            plan=read(FOLDER/'plan.json')
            if plan!=prospective_plan():raise ValueError('Prospective phaseA plan changed')
            for item in plan['phase_A']:self.execute(item)
            self.verify();self.ledger['status']='PHASE_A_EXECUTED_REQUIRES_ASSESSMENT_AND_PHASE_B_SELECTION'
        except BaseException as error:
            self.terminate();self.ledger['status']='STOPPED_REQUIRES_DIAGNOSIS'
            self.ledger.setdefault('failures',[]).append(dict(utc=utc(),type=type(error).__name__,reason=str(error)));raise
        finally:self.save()
        print(json.dumps(dict(status=self.ledger['status'],new_attempts=len(self.ledger['entries']),
                              cumulative_attempts=45+len(self.ledger['entries']),elapsed_s=self.elapsed(),acceptance_claimed=False)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--user-message-file',type=Path,default=AUTHORIZATION)
    parser.add_argument('--run-phase',choices=['A'])
    args=parser.parse_args()
    if not args.prepare and not args.run_phase:parser.error('Choose explicit --prepare and/or --run-phase A; no implicit launch')
    if os.name!='posix':raise RuntimeError('Preparation/execution requires original WSL/POSIX environment')
    import fcntl
    # Same inode as the original runner: no simultaneous stage1/stage2 worker.
    with (PRIOR/'runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:prepare(args.user_message_file)
        if args.run_phase:Stage().run_phase_A()


if __name__=='__main__':main()
