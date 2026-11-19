"""Budget, provenance cache and saved-analysis recovery; no Meep/GPU imports."""
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
spec=importlib.util.spec_from_file_location('stage2_height_continuation_test',ROOT/'scripts/run_height_family_continuation_v3.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TestStage2(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.folder=self.base/'new';self.prior=self.base/'prior'
        self.folder.mkdir();self.prior.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(module,'FOLDER',self.folder).start();patch.object(module,'PRIOR',self.prior).start()
    def stage(self):
        s=module.Stage.__new__(module.Stage)
        s.config={'fixture':'unchanged configuration'};s.config_fingerprint=module.fingerprint(s.config)
        s.ledger={'entries':[]};s.catalog=[];s.old_entries=[];s.cases=[];s.env={};s.active=None
        s.verify=lambda:None;s.elapsed=lambda:100.;s.save=lambda:None
        return s
    def record(self,s,origin,method='reference',number=37):
        item=module.job(method,40.,24.,'A')
        entry=dict(item,id=number,status='completed',record_file=f'records/{number}.json')
        settings={key:entry[key] for key in module.KEYS};settings.update(m=1,Courant=.45,source='unchanged')
        path=origin/entry['record_file'];path.parent.mkdir(exist_ok=True)
        raw=dict(settings=settings,config_fingerprint=s.config_fingerprint,power_balance_error=.001)
        for key in ['fields_file','power_audit_fields_file']+(['flux_data_file'] if method=='reference' else []):
            evidence=origin/f'{number}.{key}.npz';evidence.write_bytes(b'fixed saved evidence '+key.encode());raw[key]=str(evidence)
        if method=='reference':raw['incident_power']=1.
        else:
            design=self.folder/'design.json';design.write_text(json.dumps(dict(target_diameter_um=11.5)))
            case=dict(method=method,H_um=40.,case_id=module.case_id(method,40),design_file=design.name,
                      design_sha256=module.sha(design),physical_geometry_id='physical-case')
            s.cases=[case];entry.update(case_id=case['case_id'],design_sha256=case['design_sha256'])
        path.write_text(json.dumps(raw));entry['record_sha256']=module.sha(path)
        hashes=module.required_files(raw)
        if origin==self.prior:
            delivered={str(path.relative_to(origin)).replace('\\','/'):{'sha256':entry['record_sha256']}}
            delivered.update({str(Path(name).relative_to(origin)).replace('\\','/'):{'sha256':digest}
                              for name,digest in hashes.items()})
            (self.folder/'prior_delivery_manifest.json').write_text(json.dumps(dict(files=delivered)))
        return item,entry,raw,hashes
    def test_budget_caps_all_new_failures_and_prior45(self):
        self.assertEqual(module.capacity(0,0),21600.)
        self.assertEqual(module.capacity(19,172610),5.)
        for count,elapsed,prior in [(20,0,45),(0,172614,45),(19,0,46)]:
            with self.assertRaises(RuntimeError):module.capacity(count,elapsed,prior)
    def test_absolute_clock_includes_resume_downtime(self):
        ledger=dict(first_started_utc='2026-11-13T00:00:00Z',elapsed_s=10.)
        self.assertEqual(module.elapsed_absolute(ledger,'2026-11-14T00:00:00Z'),86400.)
        ledger['elapsed_s']=90000.
        self.assertEqual(module.elapsed_absolute(ledger,'2026-11-14T00:00:00Z'),90000.)
        with self.assertRaises(ValueError):module.elapsed_absolute(ledger,'2026-11-12T23:59:59Z')
    def test_phase_A_exact_ten_and_prospective_total_twenty(self):
        plan=module.prospective_plan();self.assertEqual(len(plan['phase_A']),10)
        self.assertEqual([job['H_um'] for job in plan['phase_A'][::2]],[100.,50.,60.,80.,90.])
        self.assertEqual(plan['phase_A'][0]['retry_of_prior_id'],41)
        self.assertEqual(sum(len(plan[key]) for key in ['phase_A','prospective_phase_B','prospective_phase_C','prospective_phase_D'])
                         +plan['prospective_adaptive_slots'],20)
        self.assertEqual(plan['prospective_phase_B'][1]['duration'],259.)
    def test_old_reference_cache_preserves_original_path_and_all_field_hashes(self):
        s=self.stage();item,entry,raw,hashes=self.record(s,self.prior)
        s.catalog=[dict(entry=entry,required_file_sha256=hashes)]
        before=(self.prior/entry['record_file']).read_bytes()
        with patch.object(module,'settings_for',return_value=raw['settings']):result=s.cache(item)
        self.assertEqual(result['stage'],'prior_stage1');self.assertEqual(result['path'],self.prior/entry['record_file'])
        self.assertEqual(before,(self.prior/entry['record_file']).read_bytes())
        self.assertFalse((self.folder/'records').exists())
    def test_incomplete_cache_cannot_be_reused(self):
        s=self.stage();item,entry,_,hashes=self.record(s,self.prior)
        entry['status']='timeout';s.catalog=[dict(entry=entry,required_file_sha256=hashes)]
        self.assertIsNone(s.cache(item))
    def test_changed_field_or_full_settings_cannot_be_reused(self):
        s=self.stage();item,entry,raw,hashes=self.record(s,self.prior)
        s.catalog=[dict(entry=entry,required_file_sha256=hashes)]
        with patch.object(module,'settings_for',return_value={**raw['settings'],'Courant':.5}):
            with self.assertRaisesRegex(ValueError,'full actual settings'):s.cache(item)
        Path(raw['fields_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'field SHA changed'):s.cache(item)
    def test_today_hash_cannot_replace_original_delivered_field_hash(self):
        path=self.prior/'delivered-fields.npz';path.write_bytes(b'original delivered data')
        delivery=dict(files={path.name:dict(sha256=module.sha(path))})
        module.check_delivered_file(path,delivery,self.prior)
        path.write_bytes(b'altered after delivery')
        with self.assertRaisesRegex(ValueError,'original sealed delivery'):
            module.check_delivered_file(path,delivery,self.prior,module.sha(path))
    def test_geometry_mismatch_cannot_be_cached_as_same_device(self):
        s=self.stage();item,entry,_,hashes=self.record(s,self.folder,method='NN',number=46)
        entry['design_sha256']='0'*64;entry['required_file_sha256']=hashes;s.ledger['entries']=[entry]
        with self.assertRaisesRegex(ValueError,'geometry differs'):s.cache(item)
    def test_only_planned_prior41_retry_is_allowed_and_new_failed_retry_never_loops(self):
        item=module.prospective_plan()['phase_A'][0];old=dict(item,id=41,status='started')
        module.require_permitted_attempt(item,[old],[])
        bad=dict(item);bad.pop('retry_of_prior_id')
        with self.assertRaises(RuntimeError):module.require_permitted_attempt(bad,[old],[])
        with self.assertRaisesRegex(RuntimeError,'unchanged retries'):
            module.require_permitted_attempt(item,[old],[dict(item,id=46,status='timeout')])
    def test_analysis_failure_recovers_from_saved_fields_without_new_worker(self):
        s=self.stage();item,entry,raw,hashes=self.record(s,self.folder,method='NN',number=46)
        entry.update(required_file_sha256=hashes,analysis_status='failed',analysis_attempts=[dict(status='failed')])
        s.ledger['entries']=[entry];(self.folder/'logs').mkdir()
        oldlog=self.folder/'logs/46.analysis-01.log';oldlog.write_text('original failed analysis')
        calls=[]
        def fake_analysis(args,**kwargs):
            self.assertIn(str(module.ANALYZER),args);self.assertLessEqual(kwargs['timeout'],172800-185-100)
            output=Path(args[args.index('--output')+1]);output.write_text(json.dumps(dict(
                reconstruction_pass=True,synthetic_optical_pass=True,normalized_curvature=.01,conservative_useful_efficiency=.95)))
            kwargs['stdout'].write('only saved-field analysis');calls.append(args);return SimpleNamespace(returncode=0)
        with patch.object(module,'settings_for',return_value=raw['settings']),\
             patch.object(module.subprocess,'Popen',side_effect=AssertionError('No Meep launch for analysis recovery')),\
             patch.object(module.subprocess,'run',side_effect=fake_analysis):
            result=s.execute(item);s.execute(item)
        self.assertEqual(len(calls),1);self.assertEqual(len(s.ledger['entries']),1)
        self.assertEqual(result['entry']['analysis_status'],'completed');self.assertEqual(entry['C'],.01)
        self.assertEqual(oldlog.read_text(),'original failed analysis')
        self.assertIn('analysis-02',entry['analysis_file'])
    def test_pid_reuse_does_not_block_unrelated_process_but_exact_owned_argv_does(self):
        entry=dict(owned_pid=123,command=['python','worker','--output','unique-owned-output'])
        with patch.object(module.os,'kill',return_value=None),patch.object(Path,'read_bytes',return_value=b'unrelated\0process\0'):
            module.require_no_live_children([entry])
        with patch.object(module.os,'kill',return_value=None),\
             patch.object(Path,'read_bytes',return_value=b'python\0worker\0--output\0unique-owned-output\0'):
            with self.assertRaisesRegex(RuntimeError,'owned worker'):module.require_no_live_children([entry])
    def test_no_arguments_cannot_prepare_or_launch_and_import_has_no_optical_runtime(self):
        with patch.object(sys,'argv',['stage2']),patch.object(module,'prepare',side_effect=AssertionError('No implicit prepare')),\
             patch.object(module,'Stage',side_effect=AssertionError('No implicit launch')),patch('sys.stderr',io.StringIO()):
            with self.assertRaises(SystemExit):module.main()
        self.assertNotIn('meep',sys.modules);self.assertNotIn('jax',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
