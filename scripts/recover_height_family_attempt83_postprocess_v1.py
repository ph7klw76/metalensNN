"""Postprocess already-written attempt83 fields; never launch an optical worker.

Default is inert. ROOT supplies an exact recovery request after independent raw
CRC/provenance review. --recover writes a prospective receipt under a fresh
campaign sibling, executes ONLY the unchanged saved-field analyzer, then writes
an additive overlay. Original ledgers, raw data and all unknown clocks/exit
statuses remain untouched. The expired optical deadline is never reset.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
STAGE=CAMPAIGN/'maxwell_validation_v3'
OUTPUT=CAMPAIGN/'attempt83_postprocess_recovery_v1'
SEAL_PATH=ROOT/'scripts/seal_height_family_stage3_v1.py'
SEAL_SHA='44581b0e0d91adf61136a105be9a0f5649e7af6f138c5608d67d388958b6263e'
ANALYZER=ROOT/'scripts/height_family_maxwell_analysis_v2.py'
ANALYZER_SHA='4d1c2cd04c539660928f29d66bce3e701ef41a115512931070dd0f11a6dfc55b'
WORKER_SHA='59fdb4dee1e7074ca17463e1495b87886563bc0ea2f393f8c539b8727b019008'
DRIVER_SHA='f252d2061bf443b0703f904a892d321f56356c2b963ebc2d252c8b702d050c68'
CONFIG_SHA='ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734'
SOURCE_MANIFEST_SHA='94cdb24ba0f71bb072e30f17b8557c3f0955d04bb903ee1e2ef97e8b4e4fcffd'
CASES_SHA='10ffc13cfcfa8da8108953fa8c02b09e73dea95e498e313041a73b53e8256633'
SELECTION_SHA='6c78928cabe3e28df1b615b772d631bb1c8941a989187759451a4a9320143872'
RAW83_SHA='cb1fdd7ab6da90159f9357cb75d712aa31eae847417fb2fd10894206f6ece116'
RAW_AUDIT_SHA='72cbeb744b3024b483c303e9b6194a437f73d23bf0b7e0dccd88bf2038018822'
START='2026-11-13T08:59:13.107420+00:00'
DEADLINE='2026-11-15T08:59:13.107420+00:00'
SCHEMA='height_family_attempt83_postprocess_overlay_v1'

def local(value):
    value=str(value)
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()
def sha(path):
    digest=hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def read(path):return json.loads(local(path).read_text(encoding='utf-8-sig'))
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def now():return datetime.now(timezone.utc)
def utc(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0:raise ValueError('Aware UTC required')
    return parsed
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def flag(command,name):
    if command.count(name)!=1:raise ValueError('One exact argument required: '+name)
    return command[command.index(name)+1]
def normalize_inventory(values):
    result={}
    for path,digest in values.items():
        path=local(path)
        if path in result or not re.fullmatch('[a-f0-9]{64}',str(digest)):raise ValueError('Invalid/duplicate normalized SHA input')
        result[path]=digest
    return result
def verify_npz(path):
    """Full CRC/member validation, without loading/reconstructing any arrays."""
    with zipfile.ZipFile(path) as bundle:
        names=bundle.namelist()
        if not names or len(names)!=len(set(names)) or any('/' in n or '\\' in n or not n.endswith('.npy') for n in names):
            raise ValueError('Incomplete/noncanonical saved NPZ members')
        if bundle.testzip() is not None:raise ValueError('Saved NPZ CRC failure')
    return dict(status='PASS_ALL_MEMBERS_CRC',members=sorted(names),arrays_loaded=False)
def validate_bundle(stage,ledger,config,case,raw,reference,selection):
    entries=ledger.get('entries',[])
    if len(entries)!=18 or [e.get('id') for e in entries]!=list(range(66,84)) or ledger.get('prior_attempt_count')!=65:
        raise ValueError('Exactly18 charged attempts66..83 /83 cumulative required')
    entry=entries[-1];ref=entries[-2]
    if entry.get('status')!='started' or entry.get('solver_launched') is not True or any(k in entry for k in ('solver_exit_code','finished_utc','analysis_file','analysis_status','analysis_attempts')):
        raise ValueError('Recover only the specific unfinalized83; never overwrite known outcomes')
    if any(e.get('status')!='completed' or e.get('method')=='NN' and e.get('analysis_status')!='completed' for e in entries[:-1]):
        raise ValueError('All earlier17 attempts and their analyses must already be complete')
    if (entry.get('method')!='NN' or entry.get('reference_id')!=82 or ref.get('method')!='reference'
            or entry.get('reference_stage_id')!='stage3' or entry.get('case_id')!='NN_H100'):
        raise ValueError('Exact existing NN83/reference82 pairing required')
    desired=dict(H_um=100.,resolution=40.,pml_um=2.,padding_um=10.,duration=335.)
    if any(entry.get(k)!=v or ref.get(k)!=v for k,v in desired.items()):raise ValueError('Frozen H100/r40 settings changed')
    if raw['settings']!=entry['effective_settings'] or raw['settings']!=reference['settings'] or raw['settings']!=ref['effective_settings']:
        raise ValueError('Saved raw/reference/effective settings differ')
    if raw.get('worker_sha256')!=WORKER_SHA or reference.get('worker_sha256')!=WORKER_SHA:
        raise ValueError('Frozen actual worker source differs')
    if raw.get('solver')!='CPU Meep 1.35.0' or reference.get('solver')!='CPU Meep 1.35.0':raise ValueError('Actual solver identity differs')
    if raw.get('config_fingerprint')!=canonical_sha(config) or reference.get('config_fingerprint')!=canonical_sha(config):
        raise ValueError('Full physical configuration differs')
    if local(raw.get('requested_config',''))!=stage/'config.json' or local(reference.get('requested_config',''))!=stage/'config.json':
        raise ValueError('Actual requested configuration path differs')
    if (local(raw.get('design_file',''))!=stage/case['design_file'] or local(raw.get('reference_file',''))!=stage/ref['record_file']
            or entry.get('design_sha256')!=case['design_sha256'] or entry.get('physical_geometry_id')!=case['physical_geometry_id']):
        raise ValueError('Actual frozen design/reference path or identity differs')
    if any(raw.get(k)!=v or reference.get(k)!=v for k,v in [('input_index',1.45),('polymer_index',1.5263),('ambient_index',1.),('wavelength_um',1.55)]):
        raise ValueError('Explicit physical indices/wavelength differ')
    if ('design_file' in reference or 'reference_file' in reference or not math.isfinite(reference.get('incident_power',float('nan')))
            or reference['incident_power']<=0 or not 0<=reference['power_balance_error']<.05):
        raise ValueError('Reference is not a valid homogeneous incident-power normalization')
    command=entry['command']
    if (local(command[0])!=local(ROOT/'.venv-meep/bin/python') or command[1]!='-B' or local(command[2])!=local(ROOT/'scripts/height_family_maxwell_worker_v2.py')
            or local(flag(command,'--config'))!=stage/'config.json' or local(flag(command,'--output'))!=stage/entry['record_file']
            or local(flag(command,'--design'))!=stage/case['design_file'] or local(flag(command,'--reference'))!=stage/ref['record_file']):
        raise ValueError('Recorded original worker argv/input/output differs')
    if any(float(flag(command,f))!=desired[k] for f,k in [('--H','H_um'),('--resolution','resolution'),('--pml','pml_um'),('--padding','padding_um'),('--duration','duration')]):
        raise ValueError('Recorded original numerical argv differs')
    jobs=selection['selected_pairs'][0]['jobs']
    if selection.get('driver_sha256')!=DRIVER_SHA or selection.get('adaptive_autoexecution') is not False or len(jobs)!=2:
        raise ValueError('Original prospective selection provenance differs')
    if any(job.get(k)!=v for job in jobs for k,v in desired.items()) or [j['method'] for j in jobs]!=['reference','NN']:
        raise ValueError('Original exact isolated pair differs')
    if ledger.get('first_started_utc')!=START or ledger.get('absolute_deadline_utc')!=DEADLINE:
        raise ValueError('Original optical clock changed')
    return entry,ref
def collect_proof(request):
    if (request.get('decision_origin')!='EXPLICIT_ROOT_POSTPROCESS_RECOVERY_OF_EXISTING_ATTEMPT'
            or not isinstance(request.get('reason'),str) or not request['reason'].strip()
            or request.get('postprocess_only') is not True or request.get('new_optical_solves_authorized')!=0
            or request.get('modify_original_ledger') is not False or request.get('old_clock_reset') is not False):
        raise ValueError('Explicit solver-free additive ROOT recovery request required')
    if request.get('helper_source_sha256')!=sha(__file__):raise ValueError('ROOT request must bind reviewed helper source')
    if request.get('analysis_timeout_s')!=600:raise ValueError('Use bounded600s saved-field analysis only')
    if any((STAGE/name).exists() for name in ('delivery_manifest.json','final_closure')):raise ValueError('Do not append to an already closed/sealed stage')
    protected={}
    def protect(path,expected=None):
        path=local(path);digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Immutable source/input SHA differs: '+str(path))
        if str(path) in protected and protected[str(path)]!=digest:raise ValueError('Input changed during verification')
        protected[str(path)]=digest;return digest
    ledger_file=STAGE/'ledger.json';ledger_sha=protect(ledger_file,request['original_ledger_sha256']);ledger=read(ledger_file)
    protect(Path(__file__),request['helper_source_sha256'])
    config=read(STAGE/'config.json');protect(STAGE/'config.json',CONFIG_SHA)
    protect(ANALYZER,ANALYZER_SHA);protect(SEAL_PATH,SEAL_SHA)
    sources=read(STAGE/'source_manifest.json');protect(STAGE/'source_manifest.json',SOURCE_MANIFEST_SHA)
    for name,digest in sources.items():protect(ROOT/name,digest)
    protect(ROOT/'scripts/run_height_family_adaptive_v1.py',DRIVER_SHA)
    cases=read(STAGE/'frozen_cases.json');protect(STAGE/'frozen_cases.json',CASES_SHA);case=next(c for c in cases['cases'] if c['case_id']=='NN_H100')
    protect(STAGE/case['design_file'],case['design_sha256']);design=read(STAGE/case['design_file'])
    physical=canonical_sha(dict(H_um=float(design['H_um']),cylinder_diameter_um=design['cylinder_diameter_um'],heights_um=design['heights_um'],materials=config['materials']))
    if physical!=case['physical_geometry_id']:raise ValueError('Frozen physical NN100 identity differs')
    entry=ledger['entries'][-1];reference_entry=ledger['entries'][-2]
    raw_path=STAGE/entry['record_file'];reference_path=STAGE/reference_entry['record_file'];raw=read(raw_path);reference=read(reference_path)
    if request.get('raw83_record_sha256')!=RAW83_SHA:raise ValueError('Recover only the independently observed existing raw83 bytes')
    raw_sha=protect(raw_path,RAW83_SHA);protect(reference_path,reference_entry['record_sha256'])
    selection_path=STAGE/entry['selection_receipt_file'];selection=read(selection_path);protect(selection_path,SELECTION_SHA)
    if entry['selection_receipt_sha256']!=SELECTION_SHA:raise ValueError('Original entry83 selection receipt differs')
    validate_bundle(STAGE,ledger,config,case,raw,reference,selection)
    required={str(local(raw[k])):protect(local(raw[k])) for k in ('fields_file','power_audit_fields_file')}
    if any(not local(path).is_relative_to(STAGE/'records') for path in required):raise ValueError('Raw83 fields outside its frozen record directory')
    reference_required={str(path):protect(path,digest) for path,digest in normalize_inventory(reference_entry['required_file_sha256']).items()}
    if set(reference_required)!={str(local(reference[k])) for k in ('fields_file','power_audit_fields_file','flux_data_file')}:
        raise ValueError('Reference82 required field/flux inventory incomplete')
    crc={path:verify_npz(local(path)) for path in {**required,**reference_required}}
    log_path=STAGE/entry['log_file'];protect(log_path);log=log_path.read_text(errors='replace')
    if 'run 0 finished at t =' not in log or 'Elapsed run time =' not in log or str(raw['wall_s']) not in log:
        raise ValueError('Frozen worker completion log evidence incomplete')
    if not isinstance(raw.get('wall_s'),(int,float)) or not 0<raw['wall_s']<=21600:raise ValueError('Saved actual worker duration outside6h bound')
    file_stats={}
    for path in [raw_path,log_path,*map(local,required)]:
        stat=path.stat();mtime=datetime.fromtimestamp(stat.st_mtime,timezone.utc)
        if not utc(entry['started_utc'])<=mtime<=utc(DEADLINE):raise ValueError('Raw output filesystem evidence is outside original optical period')
        file_stats[str(path)]=dict(mtime_utc=mtime.isoformat(),bytes=stat.st_size,sha256=protect(path))
    audit_path=local(request['raw_completion_audit_file']);audit=read(audit_path)
    if request.get('raw_completion_audit_sha256')!=RAW_AUDIT_SHA:raise ValueError('Known independent raw83 completion proof required')
    protect(audit_path,RAW_AUDIT_SHA)
    if not str(audit.get('status','')).startswith('PASS'):raise ValueError('Independent raw completion audit is not PASS')
    audit_inputs={}
    for key in ('input_sha256','file_sha256'):
        if isinstance(audit.get(key),dict):audit_inputs.update(normalize_inventory(audit[key]))
    if audit_inputs.get(raw_path)!=raw_sha or audit_inputs.get(ledger_file)!=ledger_sha or any(audit_inputs.get(local(p))!=digest for p,digest in required.items()):
        raise ValueError('Independent audit must bind exact raw83 JSON and both saved NPZ SHA values')
    evidence=dict(status='RAW_OUTPUTS_COMPLETE_OBSERVED_FROM_ARTIFACTS',file_stat_evidence=file_stats,
        raw_worker_wall_s=raw['wall_s'],all_artifact_mtimes_before_original_deadline=True,
        estimated_start_plus_worker_wall_utc=(utc(entry['started_utc'])+timedelta(seconds=raw['wall_s'])).isoformat(),
        estimated_time_is_not_authoritative_completion_UTC=True,exact_worker_exit_and_completion_clock_unknown=True,
        original_deadline_utc=DEADLINE,CRC=crc)
    context=dict(config_file=str(STAGE/'config.json'),config_sha256=CONFIG_SHA,design_file=str(STAGE/case['design_file']),design_sha256=case['design_sha256'],
        physical_geometry_id=physical,reference_id=82,reference_record_file=str(reference_path),reference_record_sha256=reference_entry['record_sha256'],
        reference_required_file_sha256=reference_required,selection_receipt_file=str(selection_path),selection_receipt_sha256=SELECTION_SHA,
        source_sha256={str(ROOT/name):digest for name,digest in sources.items()}|{str(ANALYZER):ANALYZER_SHA,str(ROOT/'scripts/run_height_family_adaptive_v1.py'):DRIVER_SHA})
    return dict(ledger=ledger,ledger_sha256=ledger_sha,entry=entry,raw_path=raw_path,raw_sha256=raw_sha,required=required,context=context,
        target_diameter_um=design['target_diameter_um'],evidence=evidence,protected=protected,request=request,raw_audit_file=str(audit_path))
def analysis_command(proof,output):
    return [str(ROOT/'.venv-meep/bin/python'),'-B',str(ANALYZER),'--record',str(proof['raw_path']),
        '--config',str(STAGE/'config.json'),'--target-diameter',str(proof['target_diameter_um']),
        '--output',str(output),'--count','512','--window','128.0']
def no_owned_processes():
    if sha(SEAL_PATH)!=SEAL_SHA:raise ValueError('Reviewed process-proof source changed')
    spec=importlib.util.spec_from_file_location('frozen_postprocess_process_proof',SEAL_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.require_no_owned_processes()
def recheck(proof):
    if any(sha(path)!=digest for path,digest in proof['protected'].items()):raise ValueError('Protected input/source changed after prospective verification')
def run_analysis(command,expected,log,env,timeout):
    """The only launch site: exact frozen analyzer argv, never an optical worker."""
    if command!=expected or local(command[2])!=local(ANALYZER) or '--count' not in command or flag(command,'--count')!='512':
        raise ValueError('Only the exact frozen saved-field analyzer command is allowed')
    start=time.monotonic();active=None
    try:
        with log.open('x',encoding='utf-8') as stream:
            active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                left=timeout-(time.monotonic()-start)
                if left<=0:raise subprocess.TimeoutExpired(command,timeout)
                try:code=active.wait(timeout=min(30.,left));break
                except subprocess.TimeoutExpired:continue
        if time.monotonic()-start>timeout:raise subprocess.TimeoutExpired(command,timeout)
        if code!=0:raise RuntimeError('Saved-field analysis failed; no optical retry is authorized')
        return dict(exit_code=code,elapsed_s=time.monotonic()-start)
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
def recover(request_file):
    if os.name!='posix':raise RuntimeError('Explicit recovery requires original WSL and shared optical lock')
    import fcntl
    if OUTPUT.exists():raise FileExistsError('Preserve all recovery evidence; do not overwrite/retry unchanged')
    with (CAMPAIGN/'maxwell_validation_v1/runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        request_bytes=local(request_file).read_bytes();request=json.loads(request_bytes);proof=collect_proof(request)
        process_proof=no_owned_processes();recheck(proof);OUTPUT.mkdir(exist_ok=False)
        for path,data in [(OUTPUT/'root_request_exact.json',request_bytes),(OUTPUT/'original_ledger_exact.json',(STAGE/'ledger.json').read_bytes()),
                          (OUTPUT/Path(__file__).name,Path(__file__).read_bytes()),(OUTPUT/'frozen_analyzer_snapshot.py',ANALYZER.read_bytes()),
                          (OUTPUT/'independent_raw_audit_exact.json',local(proof['raw_audit_file']).read_bytes())]:
            with path.open('xb') as stream:stream.write(data)
        output=OUTPUT/'analysis83.json';command=analysis_command(proof,output)
        receipt=dict(status='PROSPECTIVE_POSTPROCESS_ONLY_BEFORE_ANALYZER',schema=SCHEMA,created_utc=now().isoformat(),
            helper_source_sha256=sha(__file__),frozen_analyzer_sha256=ANALYZER_SHA,root_request_sha256=hashlib.sha256(request_bytes).hexdigest(),
            original_ledger_source_file=str(STAGE/'ledger.json'),original_ledger_sha256=proof['ledger_sha256'],original_entry83=proof['entry'],
            original_entry83_sha256=canonical_sha(proof['entry']),raw_record_file=str(proof['raw_path']),raw_record_sha256=proof['raw_sha256'],
            required_file_sha256=proof['required'],context=proof['context'],raw_completion_evidence=proof['evidence'],
            raw_completion_audit_file=proof['raw_audit_file'],raw_completion_audit_sha256=request['raw_completion_audit_sha256'],
            protected_input_sha256=proof['protected'],analysis_command=command,analysis_timeout_s=600,no_owned_process_proof=process_proof,
            new_optical_solves=0,new_training_calls=0,modify_original_ledger=False,old_clock_reset=False,
            old_clock_status='EXPIRED_OPTICAL_PERIOD_NOT_RESET; POSTPROCESSING_RECORDED_AT_ACTUAL_CURRENT_TIME',
            solver_exit_code=None,worker_finished_utc=None,controller_finished_utc=None)
        before=OUTPUT/'receipt_before_analysis.json';write_new(before,receipt);recheck(proof)
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started=now();completion=None;error=None
        try:
            completion=run_analysis(command,analysis_command(proof,output),OUTPUT/'analysis83.log',env,600)
            recheck(proof);analysis=read(output)
            if (local(analysis['record'])!=proof['raw_path'] or analysis['settings']!=proof['entry']['effective_settings']
                    or analysis['declared_target_diameter_um']!=proof['target_diameter_um']):raise ValueError('Frozen analyzer output binding differs')
            finished=now()
            overlay=dict(status='COMPLETE_FROM_SAVED_ARTIFACTS_POSTPROCESS_ONLY',schema=SCHEMA,created_utc=finished.isoformat(),
                original_ledger_source_file=str(STAGE/'ledger.json'),original_ledger_file=str(OUTPUT/'original_ledger_exact.json'),original_ledger_sha256=proof['ledger_sha256'],
                original_entry83=proof['entry'],original_entry83_sha256=canonical_sha(proof['entry']),
                attempt83_result=dict(id=83,verification_status='complete_from_artifacts',raw_record_file=str(proof['raw_path']),raw_record_sha256=proof['raw_sha256'],
                    required_file_sha256=proof['required'],analysis_file=str(output),analysis_sha256=sha(output),analysis_status='completed',
                    solver_exit_code=None,worker_finished_utc=None,controller_finished_utc=None,
                    reconstruction_pass=analysis['reconstruction_pass'],nominal_optical_pass=analysis['synthetic_optical_pass']),
                context=proof['context'],raw_completion_evidence=proof['evidence'],
                postprocessing=dict(started_utc=started.isoformat(),finished_utc=finished.isoformat(),elapsed_s=completion['elapsed_s'],
                    after_original_deadline=started>utc(DEADLINE),old_clock_reset=False,new_optical_solves=0,analysis_exit_code=completion['exit_code']),
                receipt_before_analysis_file=str(before),receipt_before_analysis_sha256=sha(before),helper_source_sha256=sha(__file__),
                raw_completion_audit_file=proof['raw_audit_file'],raw_completion_audit_sha256=request['raw_completion_audit_sha256'],
                original_attempt_count_unchanged=18,cumulative_attempt_count_unchanged=83,workflow_48h_reporting_status_not_promoted_to_PASS=True)
            write_new(OUTPUT/'recovery_overlay.json',overlay)
        except BaseException as caught:
            error=repr(caught);raise
        finally:
            write_new(OUTPUT/'postprocess_attempt_result.json',dict(status='POSTPROCESS_FAILED_PRESERVE_FOR_DIAGNOSIS' if error else 'POSTPROCESS_COMPLETE',
                started_utc=started.isoformat(),finished_utc=now().isoformat(),error=error,completion=completion,new_optical_solves=0,original_ledger_modified=False))
            recheck(proof)
            write_new(OUTPUT/'recovery_manifest.json',dict(status='ADDITIVE_POSTPROCESS_RECOVERY_FILES_ONLY',schema=SCHEMA,
                files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUTPUT.iterdir()) if p.is_file()},
                original_ledger_source_sha256=sha(STAGE/'ledger.json'),original_raw_inputs_unchanged=True,new_optical_solves=0))
        print(json.dumps(dict(status='POSTPROCESS_RECOVERY_COMPLETE',overlay=str(OUTPUT/'recovery_overlay.json'),overlay_sha256=sha(OUTPUT/'recovery_overlay.json'),new_optical_solves=0)))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recover',type=Path);args=parser.parse_args()
    if args.recover:recover(args.recover)
    else:print(json.dumps(dict(status='INERT_POSTPROCESS_ONLY_SOFTWARE_NO_ACTION',analyzer_launched=False,optical_workers_launched=0,original_ledger_modified=False)))
if __name__=='__main__':main()
