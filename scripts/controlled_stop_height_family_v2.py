"""Inspect, or explicitly interrupt, one proven owned height-family runner.

Default inspection sends no signal and writes nothing. --run is narrowly
limited to the completed H70/r24 job40 -> started H100/r24 reference41
boundary, when the remaining H100 pair cannot fit and the approved H40/r32
reserve does fit. Only SIGINT to the proven original parent is permitted;
that parent owns its worker-group cleanup and all ledger writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import time


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v1'
HELPER=ROOT/'scripts/continue_height_family_numerical_checks_v2.py'
EXPECTED_HELPER_SHA='5ea30b0a533bbfa0c5d9134cc32806173d277cdacb5f34d1a60ede1a2e609674'
SETTING_KEYS=('H_um','resolution','pml_um','padding_um','duration')
RESERVE_S=185.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_frozen_helper():
    if sha(HELPER)!=EXPECTED_HELPER_SHA:
        raise ValueError('Prepared continuation helper source hash changed')
    spec=importlib.util.spec_from_file_location('frozen_continuation_for_controlled_stop',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def boundary_entries(ledger):
    if ledger['status']!='RUNNING_APPROVED':
        raise ValueError('Original runner must still be RUNNING_APPROVED')
    entries=ledger['entries']
    if len(entries)!=41 or [entry['id'] for entry in entries]!=list(range(1,42)):
        raise ValueError('Only the original job40/job41 boundary is permitted')
    completed,current=entries[-2:]
    if (completed['method']!='NN' or completed['id']!=40 or completed['job_type']!='nominal'
            or completed['status']!='completed' or completed.get('analysis_status')!='completed'
            or tuple(completed[key] for key in SETTING_KEYS)!=(70.,24.,2.,10.,243.)):
        raise ValueError('NN H70/r24 job40 must have completed independent analysis')
    if (current['method']!='reference' or current['id']!=41 or current['job_type']!='nominal'
            or current['status']!='started'
            or tuple(current[key] for key in SETTING_KEYS)!=(100.,24.,2.,10.,335.)
            or not isinstance(current.get('owned_pid'),int) or current['owned_pid']<=1):
        raise ValueError('Current job must be started nominal H100/r24 reference41 with owned PID')
    return completed,current


def verified_completed(entry,folder=FOLDER):
    if entry['status']!='completed':raise ValueError('Completed observed timing required')
    path=folder/entry['record_file']
    if sha(path)!=entry['record_sha256']:raise ValueError('Observed raw record hash changed')
    record=read(path)
    if any(record['settings'].get(key)!=entry[key] for key in SETTING_KEYS):
        raise ValueError('Observed raw settings differ from ledger')
    if entry['method']!='reference':
        if entry.get('analysis_status')!='completed':raise ValueError('Observed device analysis incomplete')
        path=folder/entry['analysis_file']
        if sha(path)!=entry['analysis_sha256']:raise ValueError('Observed analysis hash changed')
        if read(path).get('settings')!=record['settings']:raise ValueError('Observed analysis settings differ')
    return record


def project_H100_pair(ledger,helper,now=None,folder=FOLDER):
    """Use maximum observed method-specific24/16 slowdown, then15% margin."""
    now=datetime.now(timezone.utc) if now is None else helper.parse_utc(now)
    _,current=boundary_entries(ledger)
    measurements=[]; projected={}
    def measured(method,H,resolution):
        matches=[entry for entry in ledger['entries'] if entry['method']==method and entry['H_um']==H
                 and entry['resolution']==resolution and entry['job_type']=='nominal'
                 and entry['status']=='completed']
        if len(matches)!=1:raise ValueError('Completed unique observed nominal timing required')
        entry=matches[0];verified_completed(entry,folder)
        if entry['pml_um']!=2 or entry['padding_um']!=10:
            raise ValueError('Timing extrapolation requires identical nominal domain controls')
        cost=float(entry['wrapper_wall_s'])
        if not math.isfinite(cost) or cost<=0:raise ValueError('Positive finite observed wrapper time required')
        return entry,cost
    for method in ('reference','NN'):
        ratios=[]
        for H in (40,70):
            old,low=measured(method,H,16.);fine,high=measured(method,H,24.)
            if old['duration']!=fine['duration']:raise ValueError('Timing mesh ratios require identical duration')
            ratios.append(high/low)
            measurements.append(dict(method=method,H_um=H,prior_id=old['id'],fine_id=fine['id'],
                                     wrapper16_s=low,wrapper24_s=high,ratio24_over16=high/low))
        base,cost=measured(method,100.,16.)
        if base['duration']!=current['duration']:raise ValueError('H100 timing duration differs from current job')
        projected[method]=dict(base16_entry_id=base['id'],observed16_wrapper_s=cost,
                               maximum_observed24_over16_ratio=max(ratios),timing_margin=1.15,
                               projected24_wrapper_s=1.15*cost*max(ratios))
    spent=(now-helper.parse_utc(current['started_utc'])).total_seconds()
    if spent<0:raise ValueError('Current reference starts in the future')
    remaining=14400.-helper.absolute_elapsed(ledger,now)
    reference_remaining=max(0.,projected['reference']['projected24_wrapper_s']-spent)
    pair_remaining=reference_remaining+projected['NN']['projected24_wrapper_s']+120.+RESERVE_S
    return dict(measurements=measurements,method_projections=projected,
                current_reference_wall_already_spent_s=spent,
                projected_remaining_reference_s=reference_remaining,analysis_allowance_s=120.,
                reporting_and_cleanup_reserve_s=RESERVE_S,
                projected_remaining_H100_pair_including_reserve_s=pair_remaining,
                original_absolute_wall_remaining_s=remaining,
                projected_H100_pair_exceeds_remaining=pair_remaining>remaining)


def read_process(pid,proc_root=Path('/proc')):
    directory=Path(proc_root)/str(pid)
    cmdline=[part.decode() for part in (directory/'cmdline').read_bytes().split(b'\0') if part]
    status=(directory/'status').read_text()
    parent=next(int(line.split(':',1)[1]) for line in status.splitlines() if line.startswith('PPid:'))
    stat=(directory/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),ppid=parent,cmdline=cmdline,cwd=str((directory/'cwd').resolve(strict=True)),
                start_ticks=stat[19])


def resolved_argument(value,cwd):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (Path(cwd)/path).resolve()


def prove_process_identities(entry,child,parent,root=ROOT,folder=FOLDER):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    command=entry['command']
    if child['pid']!=entry['owned_pid'] or child['ppid']!=parent['pid'] or parent['pid']<=1:
        raise ValueError('Owned child and direct original parent identities differ')
    if child['cmdline']!=command or Path(child['cwd']).resolve()!=root:
        raise ValueError('Child argv/cwd differs from exact ledger command/workspace')
    if (len(command)!=17 or command[3::2]!=['--config','--output','--H','--resolution','--pml','--padding','--duration']
            or resolved_argument(command[0],root)!=(root/'.venv-meep/bin/python').resolve()
            or command[1]!='-B' or resolved_argument(command[2],root)!=(root/'scripts/height_family_maxwell_worker_v2.py').resolve()):
        raise ValueError('Child is not the exact original project Maxwell worker')
    def argument(flag):
        if command.count(flag)!=1:raise ValueError('Unique child command argument required: '+flag)
        return command[command.index(flag)+1]
    if (resolved_argument(argument('--config'),root)!=(folder/'config.json').resolve()
            or resolved_argument(argument('--output'),root)!=(folder/entry['record_file']).resolve()):
        raise ValueError('Child config/output leaves the original workspace evidence')
    expected={'--H':entry['H_um'],'--resolution':entry['resolution'],'--pml':entry['pml_um'],
              '--padding':entry['padding_um'],'--duration':entry['duration']}
    if any(float(argument(flag))!=value for flag,value in expected.items()):
        raise ValueError('Child command physical/numerical settings differ from ledger')
    argv=parent['cmdline'];cwd=Path(parent['cwd']).resolve()
    if cwd!=root or not argv or resolved_argument(argv[0],cwd)!=(root/'.venv-meep/bin/python').resolve():
        raise ValueError('Parent interpreter/cwd is not the original project runner')
    script=root/'scripts/run_height_family_maxwell_v2.py'
    locations=[i for i,value in enumerate(argv[1:],1) if resolved_argument(value,cwd)==script.resolve()]
    if len(locations)!=1:
        raise ValueError('Parent is not the exact original run_height_family_maxwell_v2.py')
    index=locations[0]
    if (any(option not in ('-B','-u') for option in argv[1:index])
            or argv[index+1:] not in (['--run'],['--prepare','--run'])):
        raise ValueError('Parent argv is not the original explicit --run invocation')


def same_process(expected,actual):
    return all(expected[key]==actual[key] for key in ('pid','ppid','cmdline','cwd','start_ticks'))


def preflight(helper,now=None):
    ledger_bytes=(FOLDER/'ledger.json').read_bytes();ledger=json.loads(ledger_bytes.decode('utf-8-sig'))
    completed,current=boundary_entries(ledger)
    validation=helper.SupervisedValidation()
    if validation.ledger!=ledger:raise ValueError('Live ledger changed during inspection')
    helper.verify_allocation(validation)
    verified_completed(completed,FOLDER)
    forecast=project_H100_pair(ledger,helper,now,FOLDER)
    if not forecast['projected_H100_pair_exceeds_remaining']:
        raise ValueError('Projected remaining H100 pair can fit; stopping is forbidden')
    pair=helper.select_pairs(helper.runner.make_plan(),[helper.ADAPTIVE_NAME])[0]
    reserve_evidence=helper.adaptive_selection_evidence(pair,ledger,validation.cases)
    estimate=helper.estimated_pair_cost(pair,ledger)
    elapsed=helper.absolute_elapsed(ledger,now)
    if not helper.pair_fits(pair,ledger,elapsed,estimate['estimated_total_s']):
        raise ValueError('Approved H40/r32 reserve pair cannot fit remaining original budget')
    child=read_process(current['owned_pid']);parent=read_process(child['ppid'])
    prove_process_identities(current,child,parent)
    return dict(ledger_snapshot_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
                original_approval_sha256=sha(FOLDER/'approval_receipt.json'),
                original_source_manifest_sha256=sha(FOLDER/'source_manifest.json'),
                original_source_hashes=read(FOLDER/'source_manifest.json'),
                controlled_stop_utility_sha256=sha(__file__),frozen_helper_sha256=sha(HELPER),
                inspected_utc=datetime.now(timezone.utc).isoformat(),elapsed_including_downtime_s=elapsed,
                forecast=forecast,reserve_selection_evidence=reserve_evidence,reserve_estimate=estimate,
                reserve_fits_original_remaining_budget=True,completed_boundary_entry_id=completed['id'],
                interrupted_attempt_id=current['id'],child_process=child,parent_process=parent,
                reason='Observed H100 remaining-pair estimate exceeds original deadline while approved observed-nonconverged H40/r32 reserve fits',
                action='Parent-only SIGINT; original parent performs its own child cleanup and ledger accounting'),ledger_bytes


def run_controlled_stop(helper):
    if os.name!='posix':raise RuntimeError('Signalling requires original POSIX/WSL environment')
    import fcntl
    with (FOLDER/'controlled_stop.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        _run_proven_stop(helper)


def _run_proven_stop(helper):
    evidence,ledger_bytes=preflight(helper)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=FOLDER/'controlled_stop'/stamp;folder.mkdir(parents=True,exist_ok=False)
    with (folder/'prior_ledger.json').open('xb') as stream:stream.write(ledger_bytes)
    write_new(folder/'pre_signal.json',dict(status='PRE_SIGNAL_IMMUTABLE_RECEIPT',**evidence))
    # Re-evaluate budget/hash evidence and both OS identities immediately before
    # the only signal operation. A completed worker or changed PID aborts.
    current,_=preflight(helper)
    if (current['ledger_snapshot_sha256']!=evidence['ledger_snapshot_sha256']
            or not same_process(evidence['child_process'],current['child_process'])
            or not same_process(evidence['parent_process'],current['parent_process'])):
        raise RuntimeError('Ownership/ledger changed before SIGINT; no signal sent')
    parent_id=evidence['parent_process']['pid']
    started=time.monotonic();stopped=False;child_gone=False;latest=None;error=None;sent=False
    try:
        os.kill(parent_id,signal.SIGINT)
        sent=True
        while time.monotonic()-started<=20.:
            latest=read(FOLDER/'ledger.json')
            stopped=latest.get('status')=='STOPPED_REQUIRES_DIAGNOSIS'
            try:
                child_now=read_process(evidence['child_process']['pid'])
                child_gone=child_now['start_ticks']!=evidence['child_process']['start_ticks']
            except (FileNotFoundError,ProcessLookupError):child_gone=True
            if stopped and child_gone:break
            time.sleep(min(.25,max(0.,20.-(time.monotonic()-started))))
    except (OSError,KeyError,ValueError) as caught:
        error=repr(caught)
    write_new(folder/'post_signal.json',dict(
        status='ORIGINAL_PARENT_STOP_OBSERVED' if stopped and child_gone else 'STOP_UNRESOLVED_WITHIN_20_SECONDS',
        parent_only_signal='SIGINT',signal_sent=sent,error=error,
        parent_pid=parent_id,child_pid=evidence['child_process']['pid'],
        stopped_requires_diagnosis=stopped,owned_child_gone=child_gone,
        ledger_status=(latest or {}).get('status'),ledger_sha256=sha(FOLDER/'ledger.json'),
        attempted_solves=len((latest or {}).get('entries',[])),interrupted_attempt_id=evidence['interrupted_attempt_id'],
        observed_attempt_status=next((entry['status'] for entry in (latest or {}).get('entries',[])
                                    if entry['id']==evidence['interrupted_attempt_id']),None),
        poll_elapsed_s=time.monotonic()-started,ledger_and_raw_records_written_by_utility=False))
    print(json.dumps(dict(receipt_folder=str(folder),stopped=stopped,child_gone=child_gone)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Explicit proven-parent SIGINT; default is read-only inspection')
    args=parser.parse_args();helper=load_frozen_helper()
    if args.run:
        run_controlled_stop(helper)
        return
    try:
        evidence,_=preflight(helper)
        print(json.dumps(dict(status='INSPECTION_ONLY_STOP_ELIGIBLE',signal_sent=False,evidence=evidence),indent=2))
    except (OSError,KeyError,ValueError,RuntimeError) as error:
        print(json.dumps(dict(status='INSPECTION_ONLY_STOP_NOT_PERMITTED',signal_sent=False,reason=str(error))))


if __name__=='__main__':main()
