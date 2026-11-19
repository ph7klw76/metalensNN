"""Pure guard/process-metadata fixtures; no stop utility or signal is executed."""
import copy
from datetime import datetime,timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('controlled_height_stop_test',ROOT/'scripts/controlled_stop_height_family_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def parse(value):return datetime.fromisoformat(str(value).replace('Z','+00:00'))
helper=SimpleNamespace(parse_utc=parse,absolute_elapsed=lambda ledger,now:
    max(ledger.get('elapsed_s',0),(parse(now)-parse(ledger['first_started_utc'])).total_seconds()))


def fixture(folder):
    folder=Path(folder);(folder/'records').mkdir();(folder/'analyses').mkdir()
    entries=[dict(id=i,method='reference',H_um=50.,resolution=8.,pml_um=2.,padding_um=10.,
                  duration=200.,job_type='nominal',status='completed') for i in range(1,42)]
    known=[(19,'reference',40,16,100),(20,'NN',40,16,110),
           (22,'reference',70,16,200),(23,'NN',70,16,210),
           (26,'reference',100,16,400),(27,'NN',100,16,420),
           (37,'reference',40,24,400),(38,'NN',40,24,440),
           (39,'reference',70,24,1000),(40,'NN',70,24,1260)]
    for identity,method,H,resolution,cost in known:
        duration={40:160.,70:243.,100:335.}[H]
        settings=dict(H_um=float(H),resolution=float(resolution),pml_um=2.,padding_um=10.,duration=duration)
        raw=folder/f'records/{identity}.json';raw.write_text(json.dumps(dict(settings=settings)))
        entry=dict(id=identity,method=method,**settings,job_type='nominal',status='completed',
                   wrapper_wall_s=float(cost),record_file=f'records/{identity}.json',record_sha256=module.sha(raw))
        if method=='NN':
            analysis=folder/f'analyses/{identity}.json';analysis.write_text(json.dumps(dict(settings=settings)))
            entry.update(analysis_status='completed',analysis_file=f'analyses/{identity}.json',
                         analysis_sha256=module.sha(analysis))
        entries[identity-1]=entry
    entries[-1]=dict(id=41,method='reference',H_um=100.,resolution=24.,pml_um=2.,padding_um=10.,
                    duration=335.,job_type='nominal',status='started',owned_pid=123,
                    started_utc='2026-11-12T12:14:50Z')
    return dict(status='RUNNING_APPROVED',entries=entries,elapsed_s=100.,first_started_utc='2026-11-12T09:17:00Z')


def process_fixture(root):
    root=Path(root).resolve();folder=root/'artifacts/validation'
    entry=dict(id=41,method='reference',owned_pid=123,record_file='records/41.json',
               H_um=100.,resolution=24.,pml_um=2.,padding_um=10.,duration=335.)
    command=[str(root/'.venv-meep/bin/python'),'-B',str(root/'scripts/height_family_maxwell_worker_v2.py'),
             '--config',str(folder/'config.json'),'--output',str(folder/entry['record_file']),
             '--H','100.0','--resolution','24.0','--pml','2.0','--padding','10.0','--duration','335.0']
    entry['command']=command
    child=dict(pid=123,ppid=999,cmdline=command,cwd=str(root),start_ticks='100')
    parent=dict(pid=999,ppid=1000,cmdline=[str(root/'.venv-meep/bin/python'),'-u',
                  'scripts/run_height_family_maxwell_v2.py','--run'],cwd=str(root),start_ticks='80')
    return entry,child,parent,root,folder


class TestControlledStop(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
    def test_only_exact_completed40_started41_boundary_is_permitted(self):
        ledger=fixture(self.folder)
        self.assertEqual(module.boundary_entries(ledger)[1]['id'],41)
        for index,key,value in [(40,'method','NN'),(40,'status','completed'),(40,'H_um',70),
                                (39,'analysis_status','started'),(39,'duration',160)]:
            bad=copy.deepcopy(ledger);bad['entries'][index][key]=value
            with self.assertRaises(ValueError):module.boundary_entries(bad)
        bad=copy.deepcopy(ledger);bad['status']='STOPPED_REQUIRES_DIAGNOSIS'
        with self.assertRaises(ValueError):module.boundary_entries(bad)
    def test_forecast_uses_larger_observed_method_ratio_and_truthful_remaining_time(self):
        ledger=fixture(self.folder)
        result=module.project_H100_pair(ledger,helper,'2026-11-12T12:15:00Z',self.folder)
        self.assertEqual(result['method_projections']['reference']['maximum_observed24_over16_ratio'],5.)
        self.assertEqual(result['method_projections']['NN']['maximum_observed24_over16_ratio'],6.)
        self.assertEqual(result['current_reference_wall_already_spent_s'],10.)
        self.assertEqual(result['original_absolute_wall_remaining_s'],3720.)
        self.assertAlmostEqual(result['projected_remaining_H100_pair_including_reserve_s'],5493.)
        self.assertTrue(result['projected_H100_pair_exceeds_remaining'])
    def test_pair_that_can_fit_does_not_satisfy_stop_reason(self):
        ledger=fixture(self.folder)
        ledger['entries'][25]['wrapper_wall_s']=100.
        ledger['entries'][26]['wrapper_wall_s']=110.
        result=module.project_H100_pair(ledger,helper,'2026-11-12T12:15:00Z',self.folder)
        self.assertFalse(result['projected_H100_pair_exceeds_remaining'])
    def test_changed_timing_record_hash_is_rejected(self):
        ledger=fixture(self.folder)
        path=self.folder/ledger['entries'][18]['record_file'];path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'hash changed'):
            module.project_H100_pair(ledger,helper,'2026-11-12T12:15:00Z',self.folder)
    def test_exact_owned_child_and_original_parent_are_proven(self):
        entry,child,parent,root,folder=process_fixture(self.folder/'project with spaces')
        module.prove_process_identities(entry,child,parent,root,folder)
        for change in ('child_argv','child_parent','parent_argv','parent_cwd','config_output'):
            e,c,p=copy.deepcopy((entry,child,parent))
            if change=='child_argv':c['cmdline']=c['cmdline']+['--design','elsewhere.json']
            elif change=='child_parent':c['ppid']=888
            elif change=='parent_argv':p['cmdline']=[p['cmdline'][0],'-c','arbitrary code']
            elif change=='parent_cwd':p['cwd']=str(self.folder/'other-project')
            else:e['command'][4]=str(self.folder/'different-config.json')
            with self.assertRaises(ValueError):module.prove_process_identities(e,c,p,root,folder)
    def test_pid_reuse_or_argv_change_fails_immediate_identity_recheck(self):
        _,child,_,_,_=process_fixture(self.folder)
        self.assertTrue(module.same_process(child,copy.deepcopy(child)))
        for key,value in [('start_ticks','101'),('ppid',777),('cmdline',['different'])]:
            other=copy.deepcopy(child);other[key]=value
            self.assertFalse(module.same_process(child,other))
    def test_original_prepare_then_run_parent_suffix_is_accepted_exactly(self):
        entry,child,parent,root,folder=process_fixture(self.folder/'project')
        prefix=['.venv-meep/bin/python','-B','scripts/run_height_family_maxwell_v2.py']
        for suffix in (['--run'],['--prepare','--run']):
            parent['cmdline']=prefix+suffix
            module.prove_process_identities(entry,child,parent,root,folder)
        for suffix in (['--prepare'],['--run','--prepare'],['--prepare','--run','--extra'],
                       ['--prepare','--run','--run']):
            parent['cmdline']=prefix+suffix
            with self.assertRaises(ValueError):
                module.prove_process_identities(entry,child,parent,root,folder)
    def test_proc_start_identity_uses_actual_stat_field22(self):
        directory=self.folder/'321';directory.mkdir();(directory/'cwd').mkdir()
        (directory/'cmdline').write_bytes(b'python\0worker.py\0')
        (directory/'status').write_text('Name:\tworker\nPPid:\t999\n')
        fields=['S','999']+['0']*17+['777']
        (directory/'stat').write_text('321 (worker with spaces) '+' '.join(fields))
        result=module.read_process(321,self.folder)
        self.assertEqual(result['ppid'],999);self.assertEqual(result['start_ticks'],'777')
        self.assertEqual(result['cmdline'],['python','worker.py'])
    def test_default_cli_inspects_without_signal_or_run(self):
        stream=io.StringIO()
        with patch.object(sys,'argv',['controlled-stop']),patch.object(module,'load_frozen_helper',return_value=object()),\
             patch.object(module,'preflight',side_effect=ValueError('Boundary not ready')),\
             patch.object(module,'run_controlled_stop',side_effect=AssertionError('No implicit run')),\
             patch.object(module.os,'kill',side_effect=AssertionError('No signal')),patch('sys.stdout',stream):
            module.main()
        result=json.loads(stream.getvalue())
        self.assertEqual(result['status'],'INSPECTION_ONLY_STOP_NOT_PERMITTED')
        self.assertFalse(result['signal_sent'])
    def test_prepared_continuation_helper_remains_frozen(self):
        self.assertEqual(module.sha(module.HELPER),module.EXPECTED_HELPER_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
