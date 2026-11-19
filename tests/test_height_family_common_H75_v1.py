"""Pure NN75 common-scope/cap/receipt/recovery fixtures; no worker launched."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('common_H75_driver_test',ROOT/'scripts/run_height_family_common_H75_v1.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
framework,module=driver.load_framework()


class FakeStage:
    def __init__(self,folder):
        self.folder=folder;self.ledger=dict(entries=[],first_started_utc='2026-11-13T09:00:00Z',
            absolute_deadline_utc='2026-11-15T09:00:00Z');self.old_entries=[];self.calls=[];self.fail=False;self.terminated=False
        for item in module.prospective_plan()['phase_A']:
            entry=self.complete(item)
            if item['method']!='reference':
                path=folder/f"phaseA-{entry['id']}.json";path.write_text('{}')
                entry.update(analysis_file=path.name,analysis_sha256=module.sha(path))
    def complete(self,item,analysis='completed'):
        entry=dict(item,id=46+len(self.ledger['entries']),status='completed',analysis_status=analysis)
        self.ledger['entries'].append(entry);return entry
    def cache(self,item):
        rows=[entry for entry in self.ledger['entries'] if entry['status']=='completed' and module.identity(entry)==module.identity(item)]
        return None if not rows else dict(entry=rows[-1],stage='stage2')
    def verify(self):pass
    def save(self):pass
    def elapsed(self):return 100.
    def terminate(self):self.terminated=True
    def execute(self,item):
        self.calls.append(item);cached=self.cache(item)
        if cached:cached['entry']['analysis_status']='completed';return cached
        if self.fail:
            self.ledger['entries'].append(dict(item,id=46+len(self.ledger['entries']),status='failed'))
            raise RuntimeError('Synthetic NN75 worker failure')
        return dict(entry=self.complete(item),stage='stage2')


class TestCommonH75(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.addCleanup(patch.stopall)
        patch.object(module,'FOLDER',self.folder).start()
    def receipt(self):
        path=self.folder/'selection_receipt.json';path.write_text(json.dumps(dict(driver_sha256=driver.sha(driver.__file__))))
        return path,module.sha(path)
    def test_exact_scope_rejects_direct_other_heights_resolution_and_duration(self):
        jobs=driver.common_jobs(module);self.assertEqual([j['method'] for j in jobs],['reference','NN'])
        for index,key,value in [(1,'method','direct'),(0,'H_um',70.),(0,'resolution',32.),
                                (0,'duration',243.),(1,'pml_um',3.),(0,'phase','B')]:
            bad=copy.deepcopy(jobs);bad[index][key]=value
            with self.assertRaises(ValueError):driver.require_exact_scope(bad,module)
    def test_NN75_can_precede_H70_and_preserves_both_pending_paths(self):
        stage=FakeStage(self.folder);jobs,estimate=driver.preflight(stage,module,framework)
        self.assertEqual(estimate['common75_new_calls'],2);self.assertEqual(estimate['common70_remaining_calls'],2)
        self.assertEqual(estimate['projected_attempts_after_all_common'],14)
        self.assertEqual(estimate['comparison_remaining_calls'],4);self.assertEqual(estimate['original_controls_remaining_calls'],6)
        self.assertEqual(estimate['original_controls_adaptive_slots'],0)
        self.assertEqual(driver.budget_effect(estimate),dict(all_common_projected_attempts=14,comparison_remaining=4,
            comparison_projected_total=18,comparison_adaptive_remaining=2,original_controls_remaining=6,
            original_projected_total=20,original_adaptive_remaining=0))
        for item in module.prospective_plan()['prospective_phase_C']:stage.complete(item)
        self.assertEqual(driver.budget(stage,module,jobs)['projected_attempts_after_all_common'],14)
    def test_failed_attempt_cost_cannot_spend_original_control_slots(self):
        stage=FakeStage(self.folder)
        stage.ledger['entries'].append(dict(module.job('reference',55.,24.,'fixture'),id=56,status='failed'))
        with self.assertRaisesRegex(RuntimeError,'both pending priority paths'):
            driver.budget(stage,module,driver.common_jobs(module))
    def test_receipt_registry_does_not_select_or_replace_priority(self):
        for previous in [None,dict(mode='repair_only',receipt_file='existing',receipt_sha256='old')]:
            stage=FakeStage(self.folder)
            if previous:stage.ledger['subsequent_phase_selection']=copy.deepcopy(previous)
            path,digest=self.receipt();link=driver.register_receipt(stage,module,path,digest)
            self.assertEqual(stage.ledger.get('subsequent_phase_selection'),previous)
            self.assertEqual(link['mode'],'common_requested_H75')
            self.assertEqual(stage.ledger['subsequent_phase_runs'][-1]['receipt_sha256'],digest)
    def test_receipt_metadata_precedes_execute_and_new_entries(self):
        stage=FakeStage(self.folder);jobs=driver.common_jobs(module);path,digest=self.receipt()
        before=copy.deepcopy(jobs)
        with patch.object(framework,'verify_stage1_delivery',return_value=dict(stage1_unchanged=True)):
            driver.execute_common(stage,module,framework,jobs,path,digest)
        self.assertEqual(jobs,before);self.assertEqual(len(stage.calls),2)
        for entry in stage.ledger['entries'][10:]:
            self.assertEqual(entry['selection_receipt_file'],path.name)
            self.assertEqual(entry['selection_receipt_sha256'],digest)
            self.assertEqual(entry['selection_scope'],driver.SCOPE)
            self.assertEqual(entry['method'] in ('reference','NN'),True)
        self.assertNotIn('subsequent_phase_selection',stage.ledger)
    def test_completed_solver_failed_analysis_recovers_without_another_attempt(self):
        stage=FakeStage(self.folder);jobs=driver.common_jobs(module)
        stage.complete(jobs[0]);stage.complete(jobs[1],analysis='failed')
        path,digest=self.receipt()
        with patch.object(framework,'verify_stage1_delivery',return_value=dict(stage1_unchanged=True)):
            driver.execute_common(stage,module,framework,jobs,path,digest)
        self.assertEqual(len(stage.ledger['entries']),12)
        self.assertEqual(stage.ledger['entries'][-1]['analysis_status'],'completed')
    def test_failure_remains_charged_and_no_retry_loop(self):
        stage=FakeStage(self.folder);stage.fail=True;jobs=driver.common_jobs(module);path,digest=self.receipt()
        with self.assertRaisesRegex(RuntimeError,'Synthetic NN75'):
            driver.execute_common(stage,module,framework,jobs,path,digest)
        self.assertEqual(len(stage.ledger['entries']),11);self.assertEqual(len(stage.calls),1)
        self.assertTrue(stage.terminated)
        self.assertEqual(stage.ledger['entries'][-1]['selection_receipt_sha256'],digest)
        with self.assertRaisesRegex(RuntimeError,'unchanged retries'):driver.budget(stage,module,jobs)
    def test_default_inspection_never_runs_and_execution_needs_reason(self):
        out=io.StringIO()
        with patch.object(sys,'argv',['common75']),patch.object(driver,'load_framework',return_value=(framework,module)),\
             patch.object(module,'Stage',return_value=FakeStage(self.folder)),\
             patch.object(driver,'run_common',side_effect=AssertionError('No implicit execution')),patch('sys.stdout',out):
            driver.main()
        result=json.loads(out.getvalue());self.assertEqual(result['worker_launches'],0)
        with patch.object(sys,'argv',['common75','--run']),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):driver.main()
    def test_frozen_framework_and_active_stage_sources_unchanged(self):
        self.assertEqual(driver.sha(driver.FRAMEWORK_PATH),driver.FRAMEWORK_SHA)
        self.assertEqual(framework.sha(framework.STAGE_PATH),framework.EXPECTED_STAGE_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
