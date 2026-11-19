"""Pure stage3 provenance, bounded clock, receipt and recovery checks."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_controller_test',ROOT/'scripts/run_height_family_maxwell_v3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TestStage3(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.addCleanup(patch.stopall)
        self.real_folder=m.FOLDER;patch.object(m,'FOLDER',self.folder).start()
    def stage(self):
        s=m.Stage.__new__(m.Stage);s.ledger=dict(entries=[],prior_attempt_count=65,first_started_utc=m.ORIGINAL_START,
            absolute_deadline_utc=m.DEADLINE,elapsed_s=67000.)
        s.plan=m.proposal_plan(m.read(m.PROPOSAL));s.config={};s.config_fingerprint='fp';s.catalog=[];s.all_prior_entries=[]
        s.env={};s.active=None;s.verify=lambda:None;s.elapsed=lambda:67000.;s.remaining=lambda:100000.
        case=dict(case_id='NN_H100',design_file='design.json',design_sha256='design-sha',physical_geometry_id='geometry-id')
        (self.folder/'design.json').write_text(json.dumps(dict(target_diameter_um=16.2)))
        case['design_sha256']=m.sha(self.folder/'design.json');s.case=lambda H:case
        return s
    def record(self,method='reference'):
        config=self.folder/'config.json';config.write_text('{}')
        design=self.folder/'lens.json';design.write_text('frozen lens')
        ref=self.folder/'reference.json';ref.write_text('frozen reference')
        effective=dict(resolution=32.,H_um=100.)
        record=dict(settings=effective,config_fingerprint='fp',worker_sha256='worker-sha',solver='CPU Meep 1.35.0',
                    requested_config=str(config),power_balance_error=.001)
        if method=='reference':record['incident_power']=1.
        else:record.update(design_file=str(design),reference_file=str(ref))
        args=dict(item=dict(method=method),effective=effective,config_fingerprint='fp',worker_sha='worker-sha',
                  config_path=config,config_sha=m.sha(config),design_path=design,design_sha=m.sha(design),
                  reference_path=ref,reference_sha=m.sha(ref))
        return record,args
    def test_exact_primary14_and_adaptive_unselected(self):
        proposal=m.read(m.PROPOSAL);before=copy.deepcopy(proposal);plan=m.proposal_plan(proposal)
        self.assertEqual(proposal,before);self.assertEqual(len(plan['primary_jobs']),14)
        self.assertEqual([j['proposed_id'] for j in plan['primary_jobs']],list(range(66,80)))
        self.assertEqual([j['H_um'] for j in plan['primary_jobs'][::2]],[100.,75.,70.,80.,50.,60.,90.])
        self.assertEqual(plan['adaptive_jobs'],[]);self.assertFalse(plan['adaptive_autoexecution'])
        bad=copy.deepcopy(proposal);bad['primary_jobs'][0]['duration']=502.5
        with self.assertRaises(ValueError):m.proposal_plan(bad)
    def test_original_clock_counts_overnight_idle_and_caps18_83(self):
        ledger=dict(first_started_utc=m.ORIGINAL_START,absolute_deadline_utc=m.DEADLINE,elapsed_s=24191.)
        elapsed=m.absolute_elapsed(ledger,'2026-11-14T03:35:14.482786+00:00')
        self.assertGreater(elapsed,18*3600);self.assertLess(m.capacity(17,172800-185-2),3)
        self.assertEqual(m.capacity(0,elapsed),21600.)
        with self.assertRaises(RuntimeError):m.capacity(18,elapsed)
        with self.assertRaises(RuntimeError):m.capacity(0,172800-185)
        ledger['first_started_utc']='2026-11-14T03:35:14.482786+00:00'
        with self.assertRaises(ValueError):m.absolute_elapsed(ledger)
    def test_authorization_question_reply_and_windows_proposal_path_authenticated(self):
        auth=self.folder/'authorization';shutil.copytree(self.real_folder/'authorization',auth)
        receipt=m.validate_authorization(self.folder)
        self.assertEqual(receipt['user_reply_exact'],'Approve up to 18 additional solves')
        (auth/'question_exact.txt').write_text('different question')
        with self.assertRaises(ValueError):m.validate_authorization(self.folder)
    def test_preparation_allows_authorization_only_and_preserves_existing_results(self):
        (self.folder/'authorization').mkdir();m.require_empty_preparation_directory(self.folder)
        (self.folder/'ledger.json').write_text('existing results')
        with self.assertRaises(FileExistsError):m.require_empty_preparation_directory(self.folder)
        self.assertEqual((self.folder/'ledger.json').read_text(),'existing results')
    def test_primary_forecast_fit_not_infeasible_old_full_envelope(self):
        s=self.stage();s.cache=lambda item,verify=True:None;s.remaining=lambda:29*3600
        forecast=s.forecast();self.assertTrue(forecast['primary_forecast_fits'])
        self.assertGreater(s.plan['old_full_envelope_s'],s.remaining())
        self.assertFalse(forecast['full_original_envelope_fit_required'])
        s.remaining=lambda:14*3600;self.assertFalse(s.forecast()['primary_forecast_fits'])
    def test_prior_cache_requires_full_settings_and_original_delivered_field_hashes(self):
        s=self.stage();item=s.plan['primary_jobs'][0];origin=self.folder/'prior';origin.mkdir()
        raw=dict(settings=item['expected_actual_settings'],config_fingerprint='fp',incident_power=1.)
        for key in ['fields_file','power_audit_fields_file','flux_data_file']:
            path=origin/(key+'.npz');path.write_bytes(key.encode());raw[key]=str(path)
        path=origin/'record.json';path.write_text(json.dumps(raw))
        entry=dict(item,id=46,status='completed',record_file=path.name,record_sha256=m.sha(path))
        fields=m.u.required_files(raw)
        s.catalog=[dict(stage_id='stage2',stage_name='old-stage',stage_folder=str(origin),entry=entry,required_file_sha256=fields)]
        delivery=dict(files={path.name:dict(sha256=m.sha(path)),**{Path(name).name:dict(sha256=digest) for name,digest in fields.items()}})
        m.write(self.folder/'prior_delivery_manifests/stage2.json',delivery)
        with patch.object(m.u,'settings_for',return_value=item['expected_actual_settings']):
            self.assertEqual(s.cache(item)['path'],path)
        with patch.object(m.u,'settings_for',return_value={**item['expected_actual_settings'],'Courant':.5}):
            with self.assertRaisesRegex(ValueError,'full physics/numerical settings'):s.cache(item)
        Path(raw['fields_file']).write_bytes(b'tampered')
        with patch.object(m.u,'settings_for',return_value=item['expected_actual_settings']):
            with self.assertRaisesRegex(ValueError,'field bytes changed'):s.cache(item)
        self.assertEqual(m.read(path),raw)
    def test_incomplete_and_wrong_geometry_cache_cannot_be_reused(self):
        s=self.stage();item=s.plan['primary_jobs'][1];entry=dict(item,id=67,status='failed')
        s.catalog=[dict(stage_id='stage2',entry=entry)]
        with patch.object(m.u,'settings_for',return_value=item['expected_actual_settings']):self.assertIsNone(s.cache(item))
        origin=self.folder/'cache';origin.mkdir();raw=dict(settings=item['expected_actual_settings'],config_fingerprint='fp')
        for key in ['fields_file','power_audit_fields_file']:
            path=origin/(key+'.npz');path.write_bytes(b'data');raw[key]=str(path)
        path=origin/'record.json';path.write_text(json.dumps(raw));entry.update(status='completed',record_file=path.name,
            record_sha256=m.sha(path),design_sha256='wrong-geometry')
        s.catalog=[dict(stage_id='stage2',stage_name='old-stage',stage_folder=str(origin),entry=entry,
                        required_file_sha256=m.u.required_files(raw))]
        with patch.object(m.u,'settings_for',return_value=item['expected_actual_settings']):
            with self.assertRaisesRegex(ValueError,'frozen geometry differs'):s.cache(item)
    def test_homogeneous_reference_geometry_is_null_with_paired_context(self):
        case=dict(case_id='NN_H100',design_sha256='hash',physical_geometry_id='geometry')
        actual=m.geometry_metadata(dict(method='reference'),case)
        self.assertIsNone(actual['case_id']);self.assertIsNone(actual['design_sha256']);self.assertIsNone(actual['physical_geometry_id'])
        self.assertEqual(actual['paired_NN_case_id'],'NN_H100')
        self.assertEqual(m.geometry_metadata(dict(method='NN'),case)['design_sha256'],'hash')
    def test_reference_path_incident_and_device_input_counterexamples(self):
        record,args=self.record();m.validate_worker_output(record,**args)
        for key,value in [('requested_config',str(self.folder/'elsewhere.json')),('incident_power',0.),
                          ('incident_power',-1.),('incident_power',float('nan')),('power_balance_error',.05),
                          ('design_file','lens.json'),('reference_file','other-reference.json')]:
            bad=dict(record);bad[key]=value
            with self.assertRaises(ValueError):m.validate_worker_output(bad,**args)
    def test_NN_config_design_reference_paths_and_source_bytes_must_match(self):
        record,args=self.record('NN');m.validate_worker_output(record,**args)
        for key,value in [('requested_config',str(self.folder/'other-config.json')),('design_file',str(self.folder/'other-lens.json')),
                          ('reference_file',str(self.folder/'other-reference.json')),('solver','CPU Meep 2.0')]:
            bad=dict(record);bad[key]=value
            with self.assertRaises(ValueError):m.validate_worker_output(bad,**args)
        Path(args['design_path']).write_text('changed lens')
        with self.assertRaises(ValueError):m.validate_worker_output(record,**args)
    def test_owned_wait_detects_UTC_jump_even_after_successful_wait(self):
        for returns_success in [False,True]:
            s=self.stage();clock=dict(mono=0.,left=1000.,terminated=False)
            def wait(timeout):
                clock['mono']+=1.;clock['left']=-1.
                if returns_success:return 0
                raise subprocess.TimeoutExpired(['owned'],timeout)
            s.active=SimpleNamespace(wait=wait);s.remaining=lambda:clock['left']
            s.terminate=lambda:clock.update(terminated=True)
            with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']):
                with self.assertRaises(subprocess.TimeoutExpired):s.wait_owned(100.,['owned'])
            self.assertTrue(clock['terminated'])
    def test_owned_wait_has_fixed_child_budget_and_ordinary_polling(self):
        s=self.stage();clock=dict(mono=0.,terminated=False);waits=[]
        def wait(timeout):
            waits.append(timeout);clock['mono']+=timeout
            raise subprocess.TimeoutExpired(['owned'],timeout)
        s.active=SimpleNamespace(wait=wait);s.terminate=lambda:clock.update(terminated=True)
        with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']):
            with self.assertRaises(subprocess.TimeoutExpired):s.wait_owned(50.,['owned'])
        self.assertEqual(waits,[30.,20.]);self.assertTrue(clock['terminated'])
        clock.update(mono=0.,terminated=False);waits.clear()
        def normal(timeout):
            waits.append(timeout);clock['mono']+=1.
            if len(waits)<3:raise subprocess.TimeoutExpired(['owned'],timeout)
            return 0
        s.active=SimpleNamespace(wait=normal)
        with patch.object(m.time,'monotonic',side_effect=lambda:clock['mono']):self.assertEqual(s.wait_owned(100.,['owned']),0)
        self.assertFalse(clock['terminated']);self.assertTrue(all(q<=30 for q in waits))
    def test_prelaunch_receipt_exists_before_Popen_and_failure_is_charged(self):
        s=self.stage();item=s.plan['primary_jobs'][0]
        for name in ['preparation_manifest.json','plan.json','source_manifest.json','frozen_cases.json','prior_stages.json']:
            (self.folder/name).write_text('{}')
        m.write(self.folder/'ledger.json',s.ledger)
        s.validate_item=lambda item:(item,item['expected_actual_settings']);s.cache=lambda item,verify=True:None
        s.forecast=lambda:dict(primary_forecast_fits=True)
        def fail(args,**kwargs):
            self.assertEqual(len(s.ledger['entries']),1);entry=s.ledger['entries'][0]
            path=self.folder/entry['selection_receipt_file'];self.assertEqual(m.sha(path),entry['selection_receipt_sha256'])
            receipt=m.read(path);self.assertEqual(receipt['attempt_id'],66);self.assertIsNone(receipt['case_id'])
            self.assertEqual(receipt['paired_NN_case_id'],'NN_H100')
            self.assertEqual(m.read(path.parent/'prior_ledger.json')['entries'],[])
            raise OSError('Synthetic launch failure')
        with patch.object(m.subprocess,'Popen',side_effect=fail):
            with self.assertRaisesRegex(OSError,'Synthetic launch'):s.execute(item)
        self.assertEqual(len(s.ledger['entries']),1);self.assertEqual(s.ledger['entries'][0]['id'],66)
        self.assertEqual(s.ledger['entries'][0]['status'],'failed');self.assertEqual(s.ledger['cumulative_attempted_solves'],66)
        with self.assertRaisesRegex(RuntimeError,'silently retry'):s.execute(item)
    def test_analysis_recovery_launches_only_analyzer_and_preserves_old_logs(self):
        s=self.stage();item=s.plan['primary_jobs'][1]
        entry=dict(item,id=67,status='completed',record_file='records/67_NN.json',analysis_status='failed',analysis_attempts=[dict(status='failed')])
        s.ledger['entries']=[dict(id=66),entry];s.validate_item=lambda item:(item,item['expected_actual_settings'])
        s.cache=lambda item,verify=True:dict(entry=entry,stage_id='stage3')
        (self.folder/'logs').mkdir();old=self.folder/'logs/67_NN.analysis-01.log';old.write_text('old failure')
        calls=[]
        class Child:
            pid=123;returncode=0
            def wait(self,timeout=None):return 0
            def poll(self):return 0
        def analyze(args,**kwargs):
            self.assertIn(str(m.ANALYZER),args);self.assertNotIn(str(m.WORKER),args);calls.append(args)
            output=Path(args[args.index('--output')+1]);output.write_text(json.dumps(dict(
                reconstruction_pass=True,synthetic_optical_pass=False,normalized_curvature=.03,conservative_useful_efficiency=.95)))
            return Child()
        with patch.object(m.subprocess,'Popen',side_effect=analyze):s.execute(item);s.execute(item)
        self.assertEqual(len(calls),1);self.assertEqual(len(s.ledger['entries']),2)
        self.assertEqual(entry['analysis_status'],'completed');self.assertEqual(old.read_text(),'old failure')
    def test_scope_has_no_adaptive_execution_and_smoke_launches_nothing(self):
        s=self.stage();bad=dict(s.plan['primary_jobs'][0],resolution=48.)
        with self.assertRaises(ValueError):s.validate_item(bad)
        out=io.StringIO();s.forecast=lambda:dict(primary_forecast_fits=True)
        with patch.object(sys,'argv',['stage3','--smoke']),patch.object(m,'Stage',return_value=s),\
             patch.object(m.subprocess,'Popen',side_effect=AssertionError('No worker in smoke')),patch('sys.stdout',out):m.main()
        self.assertEqual(json.loads(out.getvalue())['worker_launches'],0)
        for argv in [['stage3'],['stage3','--smoke','--run-primary']]:
            with patch.object(sys,'argv',argv),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):m.main()


if __name__=='__main__':unittest.main(verbosity=2)
