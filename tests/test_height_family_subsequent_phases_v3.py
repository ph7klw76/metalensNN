"""Pure selection/budget/resume fixtures; no Meep or worker is launched."""
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
spec=importlib.util.spec_from_file_location('subsequent_height_driver_test',ROOT/'scripts/run_height_family_subsequent_phases_v3.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
frozen=driver.load_frozen_stage()


def original_plan():
    jobs=[]
    for kind,H,pml,padding,duration in [('pml',100,3,10,335),('padding',70,2,15,243),('duration',100,2,10,502.5)]:
        for method in ('reference','NN'):
            jobs.append(frozen.job(method,H,16,'ORIGINAL',kind=kind,pml=pml,padding=padding,duration_override=duration))
    return dict(jobs=jobs)


class FakeStage:
    def __init__(self,folder):
        self.folder=folder;self.ledger=dict(entries=[],first_started_utc='2026-11-13T09:00:00Z',
            absolute_deadline_utc='2026-11-15T09:00:00Z');self.old_entries=[];self.calls=[];self.terminated=False
        self.fail_next=False
        for item in frozen.prospective_plan()['phase_A']:
            entry=self.complete(item)
            if item['method']!='reference':
                path=folder/f"phaseA-{entry['id']}.json";path.write_text('{}')
                entry.update(analysis_file=path.name,analysis_sha256=frozen.sha(path))
    def complete(self,item,analysis='completed'):
        entry=dict(item,id=46+len(self.ledger['entries']),status='completed',analysis_status=analysis)
        self.ledger['entries'].append(entry);return entry
    def cache(self,item):
        rows=[entry for entry in self.ledger['entries'] if entry['status']=='completed' and frozen.identity(entry)==frozen.identity(item)]
        return None if not rows else dict(entry=rows[-1],stage='stage2')
    def verify(self):pass
    def save(self):pass
    def elapsed(self):return 100.
    def terminate(self):self.terminated=True
    def execute(self,item):
        self.calls.append(item)
        cached=self.cache(item)
        if cached:
            cached['entry']['analysis_status']='completed';return cached
        if self.fail_next:
            self.ledger['entries'].append(dict(item,id=46+len(self.ledger['entries']),status='failed'))
            raise RuntimeError('Synthetic worker failure')
        return dict(entry=self.complete(item),stage='stage2')


class TestSubsequentPhases(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.addCleanup(patch.stopall)
        patch.object(frozen,'FOLDER',self.folder).start()
    def selection(self,mode='comparison_first'):
        return driver.selected_jobs(mode,frozen.prospective_plan(),original_plan())
    def test_comparison_mode_has_exact_eight_and_never_adaptive_jobs(self):
        jobs=self.selection()
        self.assertEqual([(j['method'],j['H_um'],j['resolution']) for j in jobs],
            [('direct',40.,24.),('reference',75.,24.),('NN',75.,24.),('direct',75.,24.),
             ('direct',100.,24.),('reference',70.,32.),('NN',70.,32.),('direct',40.,32.)])
        self.assertTrue(all(not str(job['job_type']).startswith('adaptive') for job in jobs))
    def test_original_mode_keeps_original_controls_and_no_direct_comparison(self):
        jobs=self.selection('original_first')
        self.assertEqual(len(jobs),8)
        self.assertEqual([job['job_type'] for job in jobs],['pml','pml','padding','padding','duration','duration','refinement','refinement'])
        self.assertEqual(jobs[4]['duration'],502.5)
        self.assertTrue(all(job['method'] in ('NN','reference') for job in jobs))
    def test_repair_only_uses_two_common_jobs_and_leaves_eight_slots(self):
        stage=FakeStage(self.folder);jobs=self.selection('repair_only')
        self.assertEqual([(j['method'],j['H_um'],j['resolution']) for j in jobs],
                         [('reference',70.,32.),('NN',70.,32.)])
        budget=driver.budget_plan(stage,frozen,jobs)
        self.assertEqual(budget['projected_new_attempts_after_branch'],12)
        self.assertEqual(budget['unspent_additional_attempt_slots'],8)
        for job in jobs:stage.complete(job)
        for mode in ['comparison_first','original_first']:
            subsequent=driver.budget_plan(stage,frozen,self.selection(mode))
            self.assertEqual(subsequent['additional_solver_attempts_needed'],6)
            self.assertEqual(subsequent['projected_new_attempts_after_branch'],18)
    def test_plan_selection_does_not_mutate_frozen_plans(self):
        prepared=frozen.prospective_plan();original=original_plan()
        before=json.dumps([prepared,original],sort_keys=True)
        driver.selected_jobs('comparison_first',prepared,original)
        driver.selected_jobs('original_first',prepared,original)
        self.assertEqual(before,json.dumps([prepared,original],sort_keys=True))
        original['jobs'].pop()
        with self.assertRaises(ValueError):driver.selected_jobs('original_first',prepared,original)
    def test_phase_A_must_have_all_analyses_before_more_workers(self):
        stage=FakeStage(self.folder);driver.require_phase_A_complete(stage,frozen)
        stage.ledger['entries'][-1]['analysis_status']='failed'
        with self.assertRaisesRegex(ValueError,'PhaseA must complete'):
            driver.require_phase_A_complete(stage,frozen)
    def test_budget_keeps_final_two_and_counts_unrelated_failures(self):
        stage=FakeStage(self.folder);jobs=self.selection()
        result=driver.budget_plan(stage,frozen,jobs)
        self.assertEqual(result['additional_solver_attempts_needed'],8)
        self.assertEqual(result['projected_new_attempts_after_branch'],18)
        self.assertEqual(result['cumulative_attempts_after_branch'],63)
        stage.ledger['entries'].append(dict(frozen.job('reference',55,24,'fixture'),id=56,status='failed'))
        with self.assertRaisesRegex(RuntimeError,'final2'):
            driver.budget_plan(stage,frozen,jobs)
    def test_selected_failed_attempt_cannot_be_silently_retried(self):
        stage=FakeStage(self.folder);jobs=self.selection()
        stage.ledger['entries'].append(dict(jobs[0],id=56,status='timeout'))
        with self.assertRaisesRegex(RuntimeError,'unchanged retries'):
            driver.budget_plan(stage,frozen,jobs)
    def test_mode_and_initial_receipt_are_immutable_after_selection(self):
        stage=FakeStage(self.folder);path=self.folder/'selection.json'
        path.write_text(json.dumps(dict(driver_sha256=driver.sha(driver.__file__))))
        stage.ledger['subsequent_phase_selection']=dict(mode='comparison_first',receipt_file=path.name,
                                                       receipt_sha256=frozen.sha(path))
        driver.verify_existing_selection(stage,frozen,'comparison_first')
        with self.assertRaisesRegex(ValueError,'mode cannot change'):
            driver.verify_existing_selection(stage,frozen,'original_first')
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'receipt changed'):
            driver.verify_existing_selection(stage,frozen,'comparison_first')
    def test_common_repair_selection_can_promote_once_to_confirmed_priority(self):
        stage=FakeStage(self.folder);path=self.folder/'repair-selection.json'
        path.write_text(json.dumps(dict(driver_sha256=driver.sha(driver.__file__))))
        stage.ledger['subsequent_phase_selection']=dict(mode='repair_only',receipt_file=path.name,
                                                       receipt_sha256=frozen.sha(path))
        driver.verify_existing_selection(stage,frozen,'comparison_first')
        driver.verify_existing_selection(stage,frozen,'original_first')
        stage.ledger['subsequent_phase_selection']['mode']='comparison_first'
        with self.assertRaisesRegex(ValueError,'mode cannot change'):
            driver.verify_existing_selection(stage,frozen,'original_first')
    def test_eight_jobs_execute_and_analysis_recovery_does_not_add_attempt(self):
        stage=FakeStage(self.folder);jobs=self.selection();stage.complete(jobs[0],analysis='failed')
        path=self.folder/'receipt.json';path.write_text(json.dumps(dict(driver_sha256=driver.sha(driver.__file__))))
        first,deadline=stage.ledger['first_started_utc'],stage.ledger['absolute_deadline_utc']
        with patch.object(driver,'verify_stage1_delivery',return_value=dict(stage1_unchanged=True)):
            driver.execute_selected(stage,frozen,jobs,'comparison_first',path,frozen.sha(path))
        self.assertEqual(len(stage.calls),8);self.assertEqual(len(stage.ledger['entries']),18)
        self.assertEqual(stage.ledger['entries'][10]['analysis_status'],'completed')
        for entry in stage.ledger['entries'][11:]:
            self.assertEqual(entry['selection_receipt_file'],path.name)
            self.assertEqual(entry['selection_receipt_sha256'],frozen.sha(path))
            self.assertEqual(entry['selection_mode'],'comparison_first')
        self.assertEqual(stage.ledger['first_started_utc'],first);self.assertEqual(stage.ledger['absolute_deadline_utc'],deadline)
        result=json.loads((self.folder/'outcomes.json').read_text())
        self.assertEqual(result['final_adaptive_attempts_automatically_launched'],0)
        self.assertEqual(result['cumulative_attempts'],63);self.assertFalse(result['full_optical_acceptance_claimed'])
    def test_failure_is_charged_and_stops_remaining_selected_jobs(self):
        stage=FakeStage(self.folder);stage.fail_next=True;jobs=self.selection()
        path=self.folder/'receipt.json';path.write_text(json.dumps(dict(driver_sha256=driver.sha(driver.__file__))))
        with self.assertRaisesRegex(RuntimeError,'Synthetic worker failure'):
            driver.execute_selected(stage,frozen,jobs,'comparison_first',path,frozen.sha(path))
        self.assertEqual(len(stage.calls),1);self.assertEqual(len(stage.ledger['entries']),11)
        self.assertEqual(stage.ledger['status'],'STOPPED_REQUIRES_DIAGNOSIS');self.assertTrue(stage.terminated)
        self.assertEqual(stage.ledger['entries'][-1]['status'],'failed')
        self.assertEqual(stage.ledger['entries'][-1]['selection_receipt_file'],path.name)
        self.assertEqual(stage.ledger['entries'][-1]['selection_receipt_sha256'],frozen.sha(path))
    def test_receipt_metadata_is_added_before_execute_without_mutating_frozen_job(self):
        item=self.selection('repair_only')[0];before=copy.deepcopy(item)
        path=self.folder/'repair-receipt.json'
        selected=driver.job_with_receipt(item,frozen,'repair_only',path,'frozen-receipt-sha')
        self.assertEqual(item,before)
        self.assertEqual(selected['selection_receipt_file'],path.name)
        self.assertEqual(selected['selection_receipt_sha256'],'frozen-receipt-sha')
        self.assertEqual(selected['selection_scope'],
            'COMMON_AUTHORIZED_H70_MESH_REPAIR_ONLY_PRIORITY_AND_DIRECT_MEANING_PENDING')
    def test_default_cli_only_inspects_and_explicit_run_requires_mode_and_reason(self):
        out=io.StringIO()
        with patch.object(sys,'argv',['subsequent']),patch.object(driver,'load_frozen_stage',return_value=frozen),\
             patch.object(frozen,'Stage',return_value=FakeStage(self.folder)),\
             patch.object(driver,'preflight',side_effect=ValueError('A still running')),\
             patch.object(driver,'run_mode',side_effect=AssertionError('No implicit worker')),patch('sys.stdout',out):
            driver.main()
        result=json.loads(out.getvalue());self.assertEqual(result['mode'],'comparison_first')
        self.assertEqual(result['worker_launches'],0);self.assertEqual(result['receipt_files_written'],0)
        for argv in [['subsequent','--run'],['subsequent','--run','--mode','comparison_first'],
                     ['subsequent','--run','--mode','original_first','--selection-reason','Waiting time passed']]:
            with patch.object(sys,'argv',argv),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):driver.main()


if __name__=='__main__':unittest.main(verbosity=2)
