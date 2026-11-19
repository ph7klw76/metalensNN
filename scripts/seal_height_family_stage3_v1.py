"""Prepare a final, immutable saved-evidence package; never run optical code.

Default is inert/read-only. ROOT may explicitly --seal only after all selected
work and final reports/audits finish. A separate exact closing decision binds
the final ledger, reviewed assessment, completion artifacts and unused slots.
Closure metadata is additive; the execution ledger and old stages are untouched.
"""
from __future__ import annotations
import argparse
import ast
import base64
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
FOLDER=CAMPAIGN/'maxwell_validation_v3'
START='2026-11-13T08:59:13.107420+00:00'
DEADLINE='2026-11-15T08:59:13.107420+00:00'
ASSESSOR_SHA='96dd3849f2d3c5997d6c05524ac5503452694cd78819619cee766a878d1f7b4c'
CORE_SHA='77dae64d7a81f78ec9e338e2d3e1c576756a0aa7c91977ecbba89cc87db2d08f'
SCIENCE_SHA='aafa34734fbdd6c959278d935e43bfe0ae7e81cf877041b6380ac658d3d3d043'
COLLECTOR_SHA='081f45029b9db2543d1c5f0f3a347f4f274981954df2ffe5089a3b908a89278a'
PLOT_SHA='79c421c77ea49dfbd1d468e635e01c17b317664359e32bd3c9785be411bf6a08'
LAYOUT_SHA='8698d5cd6c70e60baa2361f67385cd0132df79816ad2edde257f9884135542ad'
REVIEWED_SOURCES={
 'scripts/run_height_family_maxwell_v3.py':'b28ffe701b3fb452cf6d3c04fb049d77afc8c6ed4eeb450da513e338717bba73',
 'scripts/run_height_family_adaptive_v1.py':'f252d2061bf443b0703f904a892d321f56356c2b963ebc2d252c8b702d050c68',
 'scripts/assess_height_family_stage3_v2.py':ASSESSOR_SHA,
 'scripts/assess_height_family_stage3_v1.py':CORE_SHA,
 'scripts/plot_height_family_nn_forward_v3.py':SCIENCE_SHA,
 'scripts/collect_height_family_stage3_v2.py':COLLECTOR_SHA,
 'scripts/plot_height_family_stage3_v2.py':'71f5287d2ffc4d5051584d1964f2f34acc9bffa5aeadf56060b901d311d82d97',
 'scripts/plot_height_family_stage3_v3.py':PLOT_SHA,
 'tests/test_height_family_stage3_publication_v3.py':'ef872bb07cf4c78b4ccaf485ef11a2410ec8bc2a97bd7cf3faaea0341f274e28',
 'scripts/plot_height_family_nn_forward_publication_v5.py':LAYOUT_SHA}
PRIOR_PINS={
 'stage1':dict(manifest='e72401385363bdc6f3c3d4a509ab41675daa4c96345b8fb0921b4c1cf209d2b3',
  ledger='7b38ca7d2c5d543fd26284cec9c0370749f335d17cf3c749376ac770ce91b693',
  archive='a18dc723b2513946f6cadc6afabb794a2fcb529f697be997d5beb87c3b3305fd',attempts=45),
 'stage2':dict(manifest='b9b1df9b10c56f4246e2b23456a2d6ef1b0de2f9b621db908eed4010ee807973',
  ledger='00ce8db6903e941f05139453f203beb7e0f0f8a580309066f81dec2d9d075623',
  archive='f00a3acc9db2d11b0b7edf6256980c82b5be1ba55fd704efcefe427a5588e85b',attempts=20)}
HEIGHTS=(40.,50.,60.,70.,75.,80.,90.,100.)
FORBIDDEN_PARTS={'.git','__pycache__','node_modules','.pytest_cache'}
PROCESS_SCRIPTS={'run_height_family_maxwell_v3.py','run_height_family_adaptive_v1.py',
 'height_family_maxwell_worker_v2.py','height_family_maxwell_analysis_v2.py',
 'assess_height_family_stage3_v1.py','assess_height_family_stage3_v2.py',
 'collect_height_family_stage3_v1.py','collect_height_family_stage3_v2.py',
 'plot_height_family_stage3_v1.py','plot_height_family_stage3_v2.py','plot_height_family_stage3_v3.py',
 'plot_height_family_nn_forward_publication_v5.py'}
SCIENTIFIC_ROW_KEYS={'H_um','height_use','proxy','Meep','ideal_declared_Gaussian','selection_rule',
 'same_geometry','unfinished_finer_attempts','native','nominal_optical','selected_mesh'}

def sha(path):
    with Path(path).open('rb') as stream:return stream_sha(stream)
def stream_sha(stream):
    digest=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def local_path(value):
    value=str(value)
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()
def read(path):return json.loads(local_path(path).read_text(encoding='utf-8-sig'))
def utc(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds()!=0:
        raise ValueError('Timezone-aware UTC timestamp required')
    return parsed.astimezone(timezone.utc)
def now():return datetime.now(timezone.utc)
def write_new(path,data):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
def contained(parent,name):
    parent=local_path(parent);path=(parent/str(name).replace('\\','/')).resolve()
    if not path.is_relative_to(parent) or path==parent:raise ValueError('Outside intended evidence directory: '+str(name))
    return path
def normalized_inventory(values):
    if not isinstance(values,dict) or not values:raise ValueError('Nonempty SHA inventory required')
    result={}
    for name,digest in values.items():
        path=local_path(name)
        if path in result or not re.fullmatch('[a-f0-9]{64}',str(digest)):raise ValueError('Duplicate normalized path or invalid SHA')
        result[path]=digest
    return result
def verify_inventory(values):
    values=normalized_inventory(values)
    for path,digest in values.items():
        if sha(path)!=digest:raise ValueError('Immutable input changed: '+str(path))
    return values
def membersafe(name):
    path=PurePosixPath(name)
    if '\\' in name or '\0' in name or re.match(r'^[A-Za-z]:',name) or path.is_absolute() or '..' in path.parts or not path.parts or str(path)!=name:
        raise ValueError('Unsafe/noncanonical archive member')
    return name
def file_inventory(folder):
    folder=local_path(folder);result={}
    for path in sorted(folder.rglob('*')):
        if path.is_symlink():raise ValueError('Symlinks are not deliverable evidence')
        if not path.is_file():continue
        relative=path.relative_to(folder)
        if any(part in FORBIDDEN_PARTS or part.startswith('.venv') for part in relative.parts) or path.suffix=='.pyc':
            raise ValueError('Environment/cache/unrelated material must not enter evidence: '+str(relative))
        name=membersafe(relative.as_posix());result[name]=dict(sha256=sha(path),bytes=path.stat().st_size)
    return result
def processes():
    """Read-only /proc inventory; PID reuse alone never establishes ownership."""
    if os.name!='posix' or not Path('/proc').is_dir():raise RuntimeError('Owned-process proof requires original WSL /proc')
    result=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            argv=(directory/'cmdline').read_bytes().split(b'\0');argv=[v.decode(errors='replace') for v in argv if v]
            cwd=(directory/'cwd').resolve(strict=True)
        except (OSError,PermissionError):continue
        result.append(dict(pid=int(directory.name),argv=argv,cwd=str(cwd)))
    return result
def windows_processes(names):
    """Filtered native Windows proof; never return unrelated command lines.

    WSL /proc cannot see Native Windows Python. Query Python-family executables
    via CIM, filter to this project's distinctive script/module names, and emit
    only PID, executable basename and matched script names. A matching relative
    script with unavailable cwd is conservatively blocking. Unreadable Python
    command lines or failed Windows interop leave proof unavailable and fail shut.
    """
    raw_names=json.dumps(sorted(names)).replace("'","''")
    root_windows=str(ROOT)
    if re.match(r'^/mnt/[a-z]/',root_windows):root_windows=root_windows[5].upper()+':/'+root_windows[7:]
    root_windows=root_windows.replace('/','\\').replace("'","''")
    script="""$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
try {
 $wanted=ConvertFrom-Json 'NAMES_JSON'
 $candidates=@(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%.exe' OR Name = 'py.exe' OR Name LIKE 'pypy%.exe'")
 if (@($candidates | Where-Object { [string]::IsNullOrWhiteSpace($_.CommandLine) }).Count -gt 0) { throw 'UNREADABLE_PYTHON_COMMAND' }
 $found=@()
 foreach($p in $candidates) {
  $command=[string]$p.CommandLine
  $hits=@($wanted | Where-Object { $command -match [regex]::Escape($_) -or $command -match [regex]::Escape([IO.Path]::GetFileNameWithoutExtension($_)) })
  if($hits.Count -gt 0) { $found += [pscustomobject]@{pid=$p.ProcessId;executable_name=$p.Name;script_names=$hits;workspace_match=$command.Contains('WINDOWS_ROOT')} }
 }
 [pscustomobject]@{status='VERIFIED_FILTERED_WINDOWS_PYTHON_PROCESS_SCAN';native_windows_host_queried=$true;all_candidate_command_lines_readable=$true;raw_command_lines_returned=$false;matches=@($found)} | ConvertTo-Json -Depth 5 -Compress
} catch { [pscustomobject]@{status='UNAVAILABLE_WINDOWS_PROCESS_PROOF';raw_command_lines_returned=$false} | ConvertTo-Json -Compress; exit 1 }
""".replace('NAMES_JSON',raw_names).replace('WINDOWS_ROOT',root_windows)
    executable=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe' if os.name=='nt' else Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
    args=[str(executable),'-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode('ascii')]
    try:
        result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',timeout=30,check=False)
        proof=json.loads(result.stdout.lstrip('\ufeff').strip())
        if result.returncode!=0:raise ValueError('Unverified native process query')
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        raise RuntimeError('Native Windows Python process proof unavailable; sealing is blocked (no unrelated command lines disclosed)') from error
    return proof
def require_no_owned_processes(rows=None,windows_proof=None,additional_names=()):
    names=PROCESS_SCRIPTS|set(additional_names)
    rows=processes() if rows is None else rows
    for row in rows:
        args=row.get('argv',[]);cwd=local_path(row.get('cwd','/'))
        for index,arg in enumerate(args[:-1]):
            if arg=='-m' and args[index+1] in {Path(name).stem for name in names}|{'scripts.'+Path(name).stem for name in names}:
                if cwd==ROOT or cwd.is_relative_to(ROOT):raise RuntimeError('Owned WSL project Python module remains active: '+str(row['pid']))
        scripts=[local_path(arg if Path(arg).is_absolute() or re.match(r'^[A-Za-z]:',arg) else cwd/arg)
                 for arg in args if Path(arg).name in names]
        if any(path==ROOT/'scripts'/path.name for path in scripts):
            raise RuntimeError('Owned worker/analyzer/controller is still active: '+str(row['pid']))
        if any(str(FOLDER) in str(arg) for arg in args) and any(Path(arg).name in names for arg in args):
            raise RuntimeError('Stage3 owned command remains active: '+str(row['pid']))
    proof=windows_processes(names) if windows_proof is None else windows_proof
    if (not isinstance(proof,dict) or proof.get('status')!='VERIFIED_FILTERED_WINDOWS_PYTHON_PROCESS_SCAN'
            or proof.get('native_windows_host_queried') is not True or proof.get('all_candidate_command_lines_readable') is not True
            or proof.get('raw_command_lines_returned') is not False or not isinstance(proof.get('matches'),list)):
        raise RuntimeError('Native Windows process proof unavailable or malformed')
    for match in proof['matches']:
        if not isinstance(match,dict) or not isinstance(match.get('script_names'),list) or not match['script_names']:
            raise RuntimeError('Native Windows process proof is malformed')
        if any(name not in names for name in match['script_names']):raise RuntimeError('Native Windows process filter scope differs')
        raise RuntimeError('Native Windows project Python process remains active: PID '+str(match.get('pid'))+'; '+', '.join(match['script_names']))
    return dict(status='PASS_NO_OWNED_WSL_OR_NATIVE_WINDOWS_PROCESSES',checked_utc=now().isoformat(),
        WSL_proc_checked=True,native_Windows_CIM_checked=True,raw_unrelated_command_lines_disclosed=False,matching_processes=0,
        scope='Known frozen execution/analysis/report/plot scripts and declared extra sources; ambiguous Native relative-script matches block closure')
def closed_ledger(ledger,decision,*,at):
    entries=ledger.get('entries',[]);count=len(entries)
    if not 14<=count<=18 or ledger.get('prior_attempt_count')!=65 or 65+count>83:
        raise ValueError('Actual18/83 bounds and preserved prior65 are required')
    if [e.get('id') for e in entries]!=list(range(66,66+count)):raise ValueError('Attempt IDs must remain contiguous66..83')
    if (ledger.get('first_started_utc')!=START or ledger.get('original_first_started_utc')!=START
            or ledger.get('absolute_deadline_utc')!=DEADLINE):raise ValueError('Original absolute clock changed')
    if not isinstance(ledger.get('elapsed_s'),(int,float)) or not math.isfinite(ledger['elapsed_s']) or ledger['elapsed_s']<0:
        raise ValueError('Finite truthful recorded elapsed required')
    if any(e.get('status')!='completed' for e in entries):raise ValueError('Incomplete/failed selected optical work cannot be sealed')
    for entry in entries:
        if not utc(START)<=utc(entry['started_utc'])<=utc(entry['finished_utc'])<=min(utc(DEADLINE),at):
            raise ValueError('Optical attempt exceeded the original deadline')
        if entry.get('method')=='NN':
            if entry.get('analysis_status')!='completed':raise ValueError('Independent saved-field analysis is incomplete')
            if any(a.get('status')=='started' for a in entry.get('analysis_attempts',[])):
                raise ValueError('Analyzer attempt is active/incomplete')
            for attempt in entry.get('analysis_attempts',[]):
                if not utc(START)<=utc(attempt['started_utc'])<=utc(attempt['finished_utc'])<=min(utc(DEADLINE),at):
                    raise ValueError('Saved-field analysis exceeded the original experiment deadline')
        elif entry.get('method')!='reference':raise ValueError('Only frozen NN/reference work belongs to stage3')
    if str(ledger.get('status','')).startswith(('RUNNING','STOPPED')):raise ValueError('Controller ledger is not closed')
    if (decision.get('unused_additional_solve_slots')!=18-count or decision.get('closing_with_unused_slots')!=(count<18)
            or decision.get('no_more_optical_calls_under_this_stage') is not True):
        raise ValueError('Explicit unused-slot accounting and ROOT closure required')
    if count<18 and (not isinstance(decision.get('unused_slot_reason'),str) or not decision['unused_slot_reason'].strip()):
        raise ValueError('Explicit ROOT reason for unused approved slots required')
    elapsed=(at-utc(START)).total_seconds()
    if elapsed<ledger['elapsed_s']-2:raise ValueError('Absolute clock cannot conceal recorded elapsed or reset idle')
    return dict(actual_new_attempts=count,cumulative_attempts=65+count,cumulative_completed=64+count,
        unused_additional_solve_slots=18-count,absolute_elapsed_s_including_idle=elapsed,
        original_first_started_utc=START,original_absolute_deadline_utc=DEADLINE,
        packaging_after_optical_deadline=at>utc(DEADLINE),historical_interrupted_attempt_id=41,
        historical_interrupted_attempt_preserved_and_charged=True,actual_cumulative_unique_cases=25,reserved_case_ceiling=28)
def verify_priors(folder):
    linkage=read(folder/'prior_stages.json');result=[]
    if linkage.get('prior_attempt_count')!=65 or linkage.get('actual_prior_physical_cases')!=25 or linkage.get('reserved_case_ceiling')!=28:
        raise ValueError('Immutable prior attempt/case accounting differs')
    if len(linkage['stages'])!=2 or {p['stage_id'] for p in linkage['stages']}!={'stage1','stage2'}:raise ValueError('Both distinct prior stages required')
    for prior in linkage['stages']:
        sid=prior['stage_id'];pin=PRIOR_PINS[sid];origin=local_path(prior['folder'])
        manifest_path=origin/'delivery_manifest.json';ledger_path=origin/'ledger.json';archive=local_path(prior['archive_file'])
        if (sha(manifest_path)!=pin['manifest'] or sha(ledger_path)!=pin['ledger'] or sha(archive)!=pin['archive']
                or prior['ledger_sha256']!=pin['ledger'] or prior['delivery_manifest_sha256']!=pin['manifest']
                or prior['archive_sha256']!=pin['archive']):raise ValueError('Prior stage seal/archive changed')
        ledger=read(ledger_path);manifest=read(manifest_path)
        if len(ledger['entries'])!=pin['attempts']:raise ValueError('Prior attempt count changed')
        if sid=='stage1' and next(e for e in ledger['entries'] if e['id']==41)['status']=='completed':
            raise ValueError('Historical interrupted41 must not be silently completed')
        for name,row in manifest['files'].items():
            path=contained(origin,name)
            if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Prior delivered payload changed: '+name)
        result.append(dict(stage_id=sid,folder=str(origin),manifest_sha256=pin['manifest'],ledger_sha256=pin['ledger'],
            archive_file=str(archive),archive_sha256=pin['archive'],manifest=manifest,payload_files=len(manifest['files'])))
    return result
def latest_reviewed_assessment(folder):
    reports=[]
    for path in (folder/'assessment').rglob('assessment.json'):
        document=read(path)
        if document.get('assessment_utility_sha256') in (ASSESSOR_SHA,CORE_SHA):
            reports.append((utc(document['created_utc']),path.resolve()))
    if not reports:raise ValueError('Final reviewed assessment is absent')
    reports.sort()
    if len(reports)>1 and reports[-1][0]==reports[-2][0]:raise ValueError('Ambiguous latest assessment timestamp')
    return reports[-1][1]
def verify_assessment(folder,path,ledger,decision,*,at):
    path=local_path(path)
    if path!=latest_reviewed_assessment(folder):raise ValueError('Closing decision must bind the latest reviewed assessment')
    if not path.is_relative_to(folder/'assessment') or path.name!='assessment.json':raise ValueError('Assessment must be contained')
    report=read(path)
    if (report.get('status')!='SAVED_THREE_STAGE_ASSESSMENT' or report.get('assessment_schema')!='height_family_stage3_v2'
            or report.get('assessment_utility_sha256')!=ASSESSOR_SHA or report.get('assessment_core_sha256')!=CORE_SHA
            or report.get('scientific_source_sha256')!=SCIENCE_SHA or report.get('scientific_function_bytecode_changed') is not False
            or report.get('reporting_height_enumeration')!=list(HEIGHTS)
            or report.get('execution_completion_is_not_optical_acceptance') is not True):
        raise ValueError('Latest reviewed adaptive-aware scientific assessment required')
    if not utc(START)<=utc(report['created_utc'])<=at:raise ValueError('Assessment timestamp is invalid')
    for key in ['authorization','budget','primary_execution','adaptive_execution','continuation_execution']:
        if report.get(key,{}).get('status')!='PASS':raise ValueError('Selected computation/protocol/budget is incomplete: '+key)
    progress=report['primary_execution'];jobs=progress.get('primary_jobs',[])
    if (progress.get('required')!=14 or progress.get('completed_verified')!=14 or len(jobs)!=14
            or {j.get('proposed_id') for j in jobs}!=set(range(66,80))
            or any(j.get('completed_verified') is not True or j.get('status')!='completed' for j in jobs)):
        raise ValueError('All14 primary jobs and analyses must be independently verified')
    budget=report['budget'];count=len(ledger['entries'])
    if (budget.get('additional_attempts')!=count or budget.get('cumulative_attempts')!=65+count
            or budget.get('actual_cumulative_cases')!=25 or budget.get('reserved_case_ceiling')!=28):
        raise ValueError('Final assessment budget/case accounting differs')
    snapshots=normalized_inventory(report['ledger_snapshot_sha256'])
    if snapshots.get((folder/'ledger.json').resolve())!=sha(folder/'ledger.json'):raise ValueError('Assessment current ledger snapshot is stale')
    inputs=verify_inventory(report['input_sha256']);sources=verify_inventory(report['analysis_source_sha256'])
    expected={ROOT/name:digest for name,digest in REVIEWED_SOURCES.items() if name.endswith(('assess_height_family_stage3_v2.py','assess_height_family_stage3_v1.py','collect_height_family_stage3_v2.py'))}
    if sources!=expected:raise ValueError('Reviewed assessment wrapper/core/collector source inventory differs')
    for name in ['config.json','plan.json','source_manifest.json','preparation_manifest.json','frozen_cases.json']:
        if inputs.get((folder/name).resolve())!=sha(folder/name):raise ValueError('Final assessment omits immutable stage input')
    for entry in ledger['entries']:
        for name,digest in [(entry['record_file'],entry['record_sha256'])]+(
                [(entry['analysis_file'],entry['analysis_sha256'])] if entry['method']=='NN' else []):
            p=contained(folder,name)
            if inputs.get(p)!=digest or sha(p)!=digest:raise ValueError('Final assessment raw/analysis evidence is missing or changed')
        for p,digest in normalized_inventory(entry['required_file_sha256']).items():
            if inputs.get(p)!=digest or sha(p)!=digest:raise ValueError('Final native field inventory missing or changed')
    runs=ledger.get('adaptive_pair_runs',[]);execution=report['adaptive_execution']
    if (execution.get('pending_current_selected_pairs') or any(r.get('status')!='completed' for r in runs)
            or len(execution.get('started_pair_runs',[]))!=len(runs)
            or any(r.get('status')!='PASS' for r in execution.get('started_pair_runs',[]))):
        raise ValueError('A selected adaptive pair remains incomplete')
    recorded_runs={(r.get('pair_id'),r.get('selection_sha256')) for r in runs}
    reported_runs={(r.get('pair_id'),r.get('selection_sha256')) for r in execution.get('started_pair_runs',[])}
    if len(recorded_runs)!=len(runs) or recorded_runs!=reported_runs:
        raise ValueError('Selected pair execution lineage differs from actual ledger')
    link=ledger.get('current_adaptive_selection')
    if link:
        receipt_path=contained(folder,link['receipt_file'])
        if sha(receipt_path)!=link['receipt_sha256']:raise ValueError('Current adaptive selection changed')
        selected=read(receipt_path)['selected_pairs'];completed={(r['pair_id'],r['selection_sha256']) for r in runs if r['status']=='completed'}
        if any((p['pair_id'],link['receipt_sha256']) not in completed for p in selected):raise ValueError('Unstarted selected pair cannot be discarded by sealing')
    diagnostics=report.get('selected_adaptive_diagnostics')
    if not isinstance(diagnostics,list) or len(diagnostics)!=len(runs):raise ValueError('Explicit selected diagnostic outcomes required')
    if {(r.get('pair_id'),r.get('receipt_sha256')) for r in diagnostics}!=recorded_runs:
        raise ValueError('Selected diagnostic receipt lineage differs from actual executed pairs')
    for row in diagnostics:
        devices=[e for e in ledger['entries'] if e.get('method')=='NN' and e.get('pair_id')==row['pair_id']
                 and e.get('selection_receipt_sha256')==row['receipt_sha256']]
        if len(devices)!=1 or row.get('target_identity')!=folder.name+':'+str(devices[0]['id']):
            raise ValueError('Selected diagnostic is missing its actual completed target')
        if any(row.get(k) not in ('PASS','FAIL','UNRESOLVED') for k in ['native_status','numerical_status','target_nominal_status']):
            raise ValueError('Selected diagnostic outcome must be explicitly classified')
    rows=report.get('NN_heights',[])
    if len(rows)!=8 or sorted(r.get('H_um') for r in rows)!=list(HEIGHTS):raise ValueError('All eight nominal-height outcomes required')
    if any(row.get(key,{}).get('status') not in ('PASS','FAIL','UNRESOLVED') for row in rows for key in ['native','nominal_optical','selected_mesh']):
        raise ValueError('Native/nominal/mesh outcomes must remain explicit, including failures and unresolved optics')
    return report
def optical_summary(report):
    return dict(original_seven_height_family=report['original_seven_height_family_acceptance']['status'],
        all_eight_nominal_heights=report['all_eight_nominal_heights_acceptance']['status'],
        execution_completion_is_optical_acceptance=False,
        nominal_native_mesh=[dict(H_um=r['H_um'],native=r['native']['status'],nominal=r['nominal_optical']['status'],mesh=r['selected_mesh']['status']) for r in report['NN_heights']],
        selected_adaptive_diagnostics=report['selected_adaptive_diagnostics'],
        original_and_fine_boundary_duration_checks=report['boundary_duration_checks'])
def normalized_value(value):
    if isinstance(value,dict):return {k:normalized_value(v) for k,v in value.items()}
    if isinstance(value,list):return [normalized_value(v) for v in value]
    if isinstance(value,str) and (re.match(r'^[A-Za-z]:[\\/]',value) or re.match(r'^/mnt/[a-z]/',value)):
        return str(local_path(value))
    return value
def frozen_csv_mapping():
    """Extract only the reviewed pure CSV mapper, without layout imports."""
    path=ROOT/'scripts/plot_height_family_nn_forward_publication_v5.py'
    if sha(path)!=LAYOUT_SHA:raise ValueError('Reviewed frozen CSV layout source changed')
    tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='CSV_COLUMNS' for target in node.targets):nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name=='csv_rows':nodes.append(node)
    if len(nodes)!=2:raise ValueError('Frozen CSV mapper schema differs')
    scope={'Path':Path}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure_csv_mapping','exec'),scope)
    return scope['CSV_COLUMNS'],scope['csv_rows']
def verify_graph_csv(path,comparison,inputs,canonical,digest):
    columns,mapper=frozen_csv_mapping()
    projected={str(p):value for p,value in inputs.items() if p.name!='training_result.json'};projected[str(canonical)]=digest
    finite=lambda v:isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    science=SimpleNamespace(finite=finite,close=lambda x,y:finite(x) and finite(y) and math.isclose(x,y,rel_tol=1e-8,abs_tol=1e-8))
    expected=mapper(dict(comparison,input_sha256=projected),science)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);actual=list(reader)
        if tuple(reader.fieldnames or ())!=tuple(columns):raise ValueError('Final graph CSV columns/units differ from frozen mapping')
    if len(expected)!=6 or len(actual)!=6:raise ValueError('Final graph CSV must retain six exact proxy/Meep rows')
    path_columns={'design_file','source_training_record_file','Meep_record_file','analysis_file'}
    for observed,reference in zip(actual,expected):
        if set(observed)!=set(columns):raise ValueError('Final graph CSV row shape differs')
        for key in columns:
            value=reference[key];cell=observed[key]
            if value is None or value=='':valid=cell==''
            elif key in path_columns:valid=local_path(cell)==local_path(value)
            elif isinstance(value,(int,float)) and not isinstance(value,bool):
                try:valid=math.isfinite(float(cell)) and float(cell)==float(value)
                except (ValueError,TypeError):valid=False
            else:valid=cell==str(value)
            if not valid:raise ValueError('Final graph CSV source/numeric/status/unit mapping differs: '+key)
def verify_graph_artifact(folder,path,ledger,report):
    """Bind completed figure files to current full, reviewed scientific metadata.

    A final assessment cannot legitimize an old interim graph. ROOT supplies
    comparison.json or its delivery_manifest.json, never a bare figure SHA.
    Both metadata files plus all nine canonical export files must authenticate.
    """
    path=local_path(path);graph_root=(folder/'forward_comparison').resolve()
    if not path.is_relative_to(graph_root) or path.name not in ('comparison.json','delivery_manifest.json'):
        raise ValueError('Final graph artifact must be contained comparison.json or its delivery_manifest.json')
    output=path.parent;manifest_path=output/'delivery_manifest.json';comparison_path=output/'comparison.json'
    manifest=read(manifest_path);comparison=read(comparison_path)
    if (manifest.get('status')!='STAGE3_DESCRIPTIVE_PUBLICATION_NO_OPTICAL_RUN'
            or manifest.get('scientific_source_sha256')!=SCIENCE_SHA or manifest.get('publication_layout_sha256')!=LAYOUT_SHA
            or manifest.get('stage3_adapter_sha256')!=PLOT_SHA or manifest.get('scientific_logic_changed') is not False
            or manifest.get('optical_solves_launched')!=0 or manifest.get('training_runs_launched')!=0):
        raise ValueError('Final graph requires the frozen reviewed plotv3/science/layout export')
    canonical={'comparison.json','REPORT.txt','proxy_vs_selected_Meep.csv',
        'beam_propagation.png','beam_propagation.svg','beam_propagation.pdf',
        'forward_model_metrics.png','forward_model_metrics.svg','forward_model_metrics.pdf'}
    files=manifest.get('files',{})
    if not isinstance(files,dict) or not canonical<=set(files):raise ValueError('Complete canonical graph export is missing')
    for name,item in files.items():
        member=contained(output,name)
        if sha(member)!=item.get('sha256') or member.stat().st_size!=item.get('bytes'):
            raise ValueError('Graph delivery payload SHA/size changed')
    if (comparison.get('status')!='DESCRIPTIVE_SAME_NN_GEOMETRY_PROXY_VS_DIRECT_MAXWELL'
            or comparison.get('utility_sha256')!=SCIENCE_SHA or comparison.get('publication_layout_sha256')!=LAYOUT_SHA
            or comparison.get('stage3_publication_adapter_sha256')!=PLOT_SHA
            or comparison.get('stage3_collector_sha256')!=COLLECTOR_SHA
            or comparison.get('partial_report_reason') not in (None,'')):
        raise ValueError('Final graph metadata source/scope is not the reviewed full comparison')
    for key in ('authorization','budget','primary_execution','adaptive_execution'):
        if comparison.get(key,{}).get('status')!='PASS':raise ValueError('Final graph has incomplete current '+key)
    progress=comparison['primary_execution'];jobs=progress.get('primary_jobs',[])
    if (progress.get('completed_verified')!=14 or progress.get('required')!=14 or len(jobs)!=14
            or {j.get('proposed_id') for j in jobs}!=set(range(66,80))
            or any(j.get('completed_verified') is not True or j.get('status')!='completed' for j in jobs)):
        raise ValueError('Final graph must authenticate all14 completed primary jobs')
    snapshots=normalized_inventory(comparison.get('ledger_snapshot_sha256'))
    if snapshots.get((folder/'ledger.json').resolve())!=sha(folder/'ledger.json') or snapshots!=normalized_inventory(report['ledger_snapshot_sha256']):
        raise ValueError('Final graph current ledger snapshot differs from final assessment')
    inputs=verify_inventory(comparison.get('input_sha256'));report_inputs=normalized_inventory(report['input_sha256'])
    if inputs!=report_inputs or normalized_inventory(manifest.get('input_sha256'))!=inputs:
        raise ValueError('Final graph full immutable input inventory differs from final assessment/delivery')
    priors=read(folder/'prior_stages.json')['stages'];original=[p for p in priors if p['stage_id']=='stage1']
    if len(original)!=1:raise ValueError('Graph requires immutable original numerical proposal origin')
    proposal=local_path(original[0]['folder'])/'maxwell_validation_proposal.json'
    if (comparison.get('config')!=read(folder/'config.json') or inputs.get(proposal.resolve())!=sha(proposal)
            or comparison.get('numerical_limits')!=read(proposal)['numerical_acceptance']):
        raise ValueError('Final graph frozen optical config or original numerical thresholds differ')
    if any(comparison.get(key) is not False for key in ('full_optical_acceptance_claimed','NN_superiority_claimed',
            'global_diffraction_optimum_claimed','experimental_validation_claimed')):
        raise ValueError('Final graph may not introduce unsupported optical/superiority/global/experimental claims')
    for key in ('additional_attempts','cumulative_attempts','actual_cumulative_cases','reserved_case_ceiling',
                'original_first_started_utc','absolute_deadline_utc','checks'):
        if comparison['budget'].get(key)!=report['budget'].get(key):raise ValueError('Final graph budget/clock/checked lineage differs')
    if comparison['budget'].get('additional_attempts')!=len(ledger['entries']):raise ValueError('Final graph attempt count is stale')
    for key in ('primary_execution','adaptive_execution','selected_adaptive_diagnostics','authorization'):
        if normalized_value(comparison.get(key))!=normalized_value(report.get(key)):
            raise ValueError('Final graph current selected execution/status lineage differs')
    rows=comparison.get('rows',[]);selected={row['H_um']:row for row in report['NN_heights']}
    if len(rows)!=3 or sorted(row.get('H_um') for row in rows)!=[40.,75.,100.]:raise ValueError('Final requested H40/H75/H100 graph rows required')
    for row in rows:
        reference=selected[row['H_um']]
        if set(row)!=SCIENTIFIC_ROW_KEYS or not row.get('Meep') or any(normalized_value(value)!=normalized_value(reference.get(key)) for key,value in row.items()):
            raise ValueError('Final graph selected record/settings/native/nominal/mesh lineage differs from assessment')
    proof=comparison.get('CSV_training_result_lineage',{})
    if (proof.get('status')!='VERIFIED_IDENTICAL_TRAINING_RESULT_ALIASES_FOR_CSV_ONLY'
            or proof.get('full_comparison_and_delivery_input_inventory_preserved') is not True
            or proof.get('scientific_records_and_numeric_CSV_mapping_unchanged') is not True
            or normalized_value(manifest.get('CSV_training_result_lineage'))!=normalized_value(proof)):
        raise ValueError('Final graph canonical CSV training alias proof differs')
    canonical_training=(folder.parent/'training_result.json').resolve();digest=inputs.get(canonical_training)
    if digest is None or local_path(proof.get('canonical_file',''))!=canonical_training or proof.get('canonical_sha256')!=digest:
        raise ValueError('Final graph CSV canonical campaign training source differs')
    expected_aliases={p:d for p,d in inputs.items() if p.name=='training_result.json'};aliases={}
    for alias in proof.get('authenticated_aliases',[]):
        key=local_path(alias.get('input_inventory_key',''));resolved=local_path(alias.get('resolved_file',''))
        if key in aliases or key!=resolved or alias.get('sha256')!=digest or expected_aliases.get(key)!=digest:
            raise ValueError('Final graph CSV training aliases differ from full inventory')
        aliases[key]=digest
    if aliases!=expected_aliases:raise ValueError('Final graph CSV proof omits an authenticated training alias')
    verify_graph_csv(output/'proxy_vs_selected_Meep.csv',comparison,inputs,canonical_training,digest)
    return dict(status='PASS_CURRENT_REVIEWED_GRAPH_LINEAGE',comparison_file=str(comparison_path),comparison_sha256=sha(comparison_path),
        delivery_manifest_file=str(manifest_path),delivery_manifest_sha256=sha(manifest_path),verified_export_files=len(files))
def inspect_decision(decision_file,*,at=None,process_rows=None,windows_proof=None):
    at=now() if at is None else at;folder=FOLDER.resolve();decision_file=local_path(decision_file);decision=read(decision_file)
    if decision_file.is_relative_to(folder):raise ValueError('Exact root closing decision must initially remain outside the stage3 seal tree')
    if (decision.get('status')!='ROOT_FINAL_STAGE3_CLOSING_DECISION' or decision.get('decision_origin')!='EXPLICIT_ROOT_CLOSING_DECISION'
            or not isinstance(decision.get('reason'),str) or not decision['reason'].strip()
            or decision.get('all_requested_reports_graphs_and_audits_complete') is not True):
        raise ValueError('Explicit ROOT final closing decision with completion statement required')
    if decision.get('seal_utility_sha256')!=sha(__file__):raise ValueError('ROOT closing decision must bind the exact reviewed sealing utility')
    additional_names=[Path(name).name for name in decision.get('additional_frozen_project_sources',{}) if str(name).endswith('.py')]
    process_proof=require_no_owned_processes(process_rows,windows_proof,additional_names)
    ledger=read(folder/'ledger.json')
    if decision.get('ledger_sha256')!=sha(folder/'ledger.json'):raise ValueError('ROOT closing ledger SHA differs')
    accounting=closed_ledger(ledger,decision,at=at)
    preparation=read(folder/'preparation_manifest.json')
    for name,digest in preparation.items():
        if sha(contained(folder,name))!=digest:raise ValueError('Prepared stage3 input changed')
    for name,digest in read(folder/'protected_historical.json').items():
        if sha(contained(ROOT,name))!=digest:raise ValueError('Protected historical input changed')
    source_pins={**read(folder/'source_manifest.json'),**REVIEWED_SOURCES}
    extras=decision.get('additional_frozen_project_sources')
    if not isinstance(extras,dict) or decision.get('required_source_inventory_complete') is not True:
        raise ValueError('ROOT must identify any additional audit/report sources and attest source inventory completeness')
    for name,digest in extras.items():
        path=contained(ROOT,name)
        if path.suffix not in ('.py','.md','.json','.yaml','.yml','.txt','.xml') or not re.fullmatch('[a-f0-9]{64}',str(digest)):
            raise ValueError('Only explicit project source/document SHA inputs may be snapshotted')
        if name in source_pins and source_pins[name]!=digest:raise ValueError('Additional source cannot override frozen physics/analysis')
        source_pins[name]=digest
    for name,digest in source_pins.items():
        if sha(contained(ROOT,name))!=digest:raise ValueError('Frozen execution/analysis source changed')
    priors=verify_priors(folder)
    cases=read(folder/'frozen_cases.json')['cases'];by_id={case['case_id']:case for case in cases}
    if len(cases)!=8 or len(by_id)!=8 or any(case.get('method')!='NN' or case.get('new_case') is not False for case in cases):
        raise ValueError('Only eight previously charged NN profiles may be packaged')
    for entry in ledger['entries']:
        if entry['method']=='NN':
            case=by_id.get(entry.get('case_id'))
            if case is None or any(entry.get(k)!=case[k] for k in ('design_sha256','physical_geometry_id')):
                raise ValueError('Stage3 device is outside the frozen previously charged NN cohort')
        elif any(entry.get(k) is not None for k in ('case_id','design_sha256','physical_geometry_id')):
            raise ValueError('Homogeneous references must not claim device geometry')
    primary=read(folder/'plan.json')['primary_jobs'];primary_by_id={j['proposed_id']:j for j in primary}
    if set(primary_by_id)!=set(range(66,80)):raise ValueError('Immutable primary14 plan differs')
    keys=('method','H_um','resolution','pml_um','padding_um','duration','phase','job_type')
    for entry in ledger['entries']:
        if entry['id']<80:
            if any(entry.get(key)!=primary_by_id[entry['id']].get(key) for key in keys):
                raise ValueError('Actual primary requested settings differ from frozen plan')
        elif entry.get('phase')!='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC':
            raise ValueError('IDs80..83 must retain their explicit adaptive selection labels')
    assessment_path=local_path(decision['assessment_file'])
    if sha(assessment_path)!=decision.get('assessment_sha256'):raise ValueError('ROOT assessment SHA differs')
    report=verify_assessment(folder,assessment_path,ledger,decision,at=at)
    for name,digest in report['input_sha256'].items():
        path=local_path(name)
        if path.is_relative_to(ROOT) and path.suffix=='.py' and path.relative_to(ROOT).parts[0] in ('scripts','src'):
            relative=path.relative_to(ROOT).as_posix()
            if relative in source_pins and source_pins[relative]!=digest:raise ValueError('Assessment source inventory conflicts with frozen sources')
            source_pins[relative]=digest
    artifacts=decision.get('completion_artifacts',[])
    if not isinstance(artifacts,list) or not {'graph','independent_audit'}<={a.get('role') for a in artifacts}:
        raise ValueError('Completed final graph and independent audit artifacts are required')
    graph_proofs=[]
    for artifact in artifacts:
        path=local_path(artifact['file'])
        if not path.is_relative_to(folder) or sha(path)!=artifact['sha256'] or artifact.get('completion_status')!='COMPLETE':
            raise ValueError('Completion artifact path/hash/status differs')
        if artifact['role']=='graph':graph_proofs.append(verify_graph_artifact(folder,path,ledger,report))
        if artifact['role']=='independent_audit':
            audit=read(path)
            if not str(audit.get(artifact.get('status_field','status'),'' )).startswith('PASS'):
                raise ValueError('Independent completion audit has not passed')
            ledger_proofs=[audit.get('ledger_sha256'),audit.get('stage3_ledger_sha256')]
            for key in ('ledger_snapshot_sha256','input_sha256'):
                if isinstance(audit.get(key),dict) and audit[key]:
                    ledger_proofs.append(normalized_inventory(audit[key]).get((folder/'ledger.json').resolve()))
            if sha(folder/'ledger.json') not in ledger_proofs:
                raise ValueError('Independent final audit must authenticate this exact final stage3 ledger')
    inventory=file_inventory(folder)
    return dict(decision=decision,decision_file=str(decision_file),decision_sha256=sha(decision_file),ledger=ledger,
        ledger_sha256=sha(folder/'ledger.json'),accounting=accounting,process_proof=process_proof,graph_proofs=graph_proofs,assessment_file=str(assessment_path),
        assessment_sha256=sha(assessment_path),report=report,priors=priors,source_pins=source_pins,initial_files=inventory)
def add_member(members,name,path,digest):
    membersafe(name)
    if name in members:raise ValueError('Duplicate archive member')
    members[name]=(Path(path),digest)
def package_archive(archive,members):
    """Exclusive create plus exact member/CRC/every-payload SHA verification."""
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for name,(path,digest) in sorted(members.items()):
            membersafe(name)
            if sha(path)!=digest:raise ValueError('Payload changed before archive: '+name)
            bundle.write(path,name)
    return verify_archive(archive,members)
def verify_archive(archive,members):
    """Read every member; duplicate, missing, extra and corrupt data all fail."""
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(members) or bundle.testzip() is not None:
            raise ValueError('Archive exact membership/CRC verification failed')
        for name,(path,digest) in members.items():
            with bundle.open(name) as stream:
                if stream_sha(stream)!=digest:raise ValueError('Archived payload SHA differs: '+name)
            if sha(path)!=digest:raise ValueError('Original payload changed during archive: '+name)
    return dict(archive_sha256=sha(archive),archive_bytes=Path(archive).stat().st_size,members_verified=len(members))
def seal(decision_file):
    if os.name!='posix':raise RuntimeError('Final sealing requires original Ubuntu-20.04 WSL shared-lock/process proof')
    import fcntl
    archive=CAMPAIGN/'height_specific_Maxwell_continuation_v3_evidence.zip'
    verification=CAMPAIGN/'height_specific_Maxwell_continuation_v3_evidence.verification.json'
    sidecar=Path(str(verification)+'.sha256');manifest_path=FOLDER/'delivery_manifest.json';closure=FOLDER/'final_closure'
    snapshots=FOLDER/'final_source_snapshots'
    if any(p.exists() for p in [archive,verification,sidecar,manifest_path,closure,snapshots]):
        raise FileExistsError('Existing final artifacts must remain intact; no overwrite/resume shortcut')
    with (CAMPAIGN/'maxwell_validation_v1/runner.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        proof=inspect_decision(decision_file)
        additional_names=[Path(name).name for name in proof['source_pins'] if name.endswith('.py')]
        final_process_proof=require_no_owned_processes(additional_names=additional_names)
        if file_inventory(FOLDER)!=proof['initial_files']:raise ValueError('Stage3 changed during final preflight')
        closure.mkdir();source_names=set(proof['source_pins'])
        source_names.update(['scripts/'+Path(__file__).name,'tests/test_height_family_stage3_seal_v1.py'])
        for skill in ['metalens-phase-workflow','fiber-metalens-design-and-slc']:
            source_names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'skills'/skill).rglob('*') if p.is_file())
        copied={}
        for name in sorted(source_names):
            source=contained(ROOT,name);target=snapshots/name;target.parent.mkdir(parents=True,exist_ok=True)
            digest=sha(source)
            expected=proof['source_pins'].get(name)
            if expected is not None and digest!=expected:raise ValueError('Pinned final source changed after preflight')
            if source==Path(__file__).resolve() and digest!=proof['decision']['seal_utility_sha256']:
                raise ValueError('ROOT-reviewed sealing source changed after preflight')
            with target.open('xb') as stream:stream.write(source.read_bytes())
            if sha(target)!=digest or sha(source)!=digest:raise ValueError('Final source/skill snapshot changed')
            copied[name]=digest
        with (closure/'root_closing_decision_exact.json').open('xb') as stream:stream.write(local_path(decision_file).read_bytes())
        if sha(closure/'root_closing_decision_exact.json')!=proof['decision_sha256']:raise ValueError('Closing decision changed')
        metadata=dict(status='ROOT_CLOSED_ALL_SELECTED_STAGE3_WORK_WITH_SEPARATE_OPTICAL_OUTCOMES',created_utc=now().isoformat(),
            ledger_sha256=proof['ledger_sha256'],ledger_status_preserved=proof['ledger']['status'],ledger_mutated=False,
            root_decision_sha256=proof['decision_sha256'],assessment_file=proof['assessment_file'],assessment_sha256=proof['assessment_sha256'],
            accounting=proof['accounting'],optical_outcomes=optical_summary(proof['report']),source_snapshot_sha256=copied,
            initial_cross_host_process_proof=proof.get('process_proof'),final_cross_host_process_proof=final_process_proof,
            final_graph_lineage_proofs=proof.get('graph_proofs'),
            closure_is_new_optical_authorization=False,optical_solves_launched=0)
        write_new(closure/'closure_metadata.json',metadata)
        files=file_inventory(FOLDER)
        if any(files.get(name)!=item for name,item in proof['initial_files'].items()):
            raise ValueError('Existing stage3 payload changed while closure snapshots were added')
        manifest=dict(status='SEALED_SAVED_THREE_STAGE_EVIDENCE_WITH_EXPLICIT_OPTICAL_OUTCOMES',created_utc=now().isoformat(),
            **proof['accounting'],ledger_sha256=proof['ledger_sha256'],assessment_sha256=proof['assessment_sha256'],
            root_decision_sha256=proof['decision_sha256'],optical_outcomes=optical_summary(proof['report']),
            prior_inputs_unchanged=[{k:v for k,v in p.items() if k!='manifest'} for p in proof['priors']],
            archive_scope='Exact stage1/stage2 delivered payloads/manifests plus all actual stage3 evidence and needed frozen source/skill snapshots; environments/caches excluded',
            manifest_excludes_its_own_recursive_SHA=True,files=files)
        # LAST WRITE TO STAGE3: all later reports and self-SHA are outside it.
        write_new(manifest_path,manifest)
        manifest_sha=sha(manifest_path);members={}
        for prior in proof['priors']:
            origin=local_path(prior['folder']);prefix=origin.name+'/'
            for name,row in prior['manifest']['files'].items():add_member(members,prefix+name,contained(origin,name),row['sha256'])
            add_member(members,prefix+'delivery_manifest.json',origin/'delivery_manifest.json',prior['manifest_sha256'])
        for name,row in files.items():add_member(members,FOLDER.name+'/'+name,contained(FOLDER,name),row['sha256'])
        add_member(members,FOLDER.name+'/delivery_manifest.json',manifest_path,manifest_sha)
        result=package_archive(archive,members)
        expected=dict(files);expected['delivery_manifest.json']=dict(sha256=manifest_sha,bytes=manifest_path.stat().st_size)
        if file_inventory(FOLDER)!=expected or sha(FOLDER/'ledger.json')!=proof['ledger_sha256']:
            raise ValueError('Sealed stage tree/ledger changed after final manifest')
        for prior in proof['priors']:
            if sha(local_path(prior['archive_file']))!=prior['archive_sha256']:raise ValueError('Prior archive changed during packaging')
        result.update(status='PASS_EXACT_MEMBERS_ALL_CRC_ALL_PAYLOAD_SHA',archive_file=str(archive),
            delivery_manifest_sha256=manifest_sha,archive_report_file=str(verification),archive_report_self_SHA_sidecar=str(sidecar),
            seal_utility_sha256=sha(__file__),stage3_payload_files=len(files),optical_outcomes=optical_summary(proof['report']),
            actual_accounting_at_preflight=proof['accounting'],actual_elapsed_s_at_package_finish=(now()-utc(START)).total_seconds(),
            stage_tree_writes_after_manifest=0,prior_payloads_and_archives_unchanged=True,optical_solves_launched=0)
        write_new(verification,result)
        digest=sha(verification)
        with sidecar.open('x',encoding='ascii') as stream:stream.write(digest+'  '+verification.name+'\n')
        print(json.dumps(dict(result,archive_report_sha256=digest),indent=2))
def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--inspect-decision',type=Path);action.add_argument('--seal',type=Path)
    args=parser.parse_args()
    if args.seal:seal(args.seal)
    elif args.inspect_decision:
        proof=inspect_decision(args.inspect_decision)
        print(json.dumps(dict(status='READ_ONLY_FINAL_CLOSURE_READY',accounting=proof['accounting'],optical_outcomes=optical_summary(proof['report']),
            seal_created=False,archive_created=False,optical_solves_launched=0),indent=2))
    else:print(json.dumps(dict(status='SOFTWARE_SCAFFOLD_NO_SEAL_REQUESTED',seal_created=False,archive_created=False,optical_solves_launched=0)))
if __name__=='__main__':main()
