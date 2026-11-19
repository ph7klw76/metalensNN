"""Pure exact-scope/14-common/20-cap/receipt/cache fixtures; no workers."""
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
spec=importlib.util.spec_from_file_location('original_controls_request_driver_test',ROOT/'scripts/run_height_family_original_controls_v2.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
framework,module=driver.load_framework()


def original_plan():
    jobs=[]
    for kind,H,pml,padding,duration in [('pml',100,3,10,335),('padding',70,2,15,243),('duration',100,2,10,502.5)]:
        for method in ['reference','NN']:
            jobs.append(module.job(method,H,16,'original',kind=kind,pml=pml,padding=padding,duration_override=duration))
    return dict(jobs=jobs)


class FakeStage:
    def __init__(self,folder):
        self.folder=folder;self.ledger=dict(entries=[],first_started_utc='2026-11-13T09:00:00Z',
            absolute_deadline_utc='2026-11-15T09:00:00Z');self.old_entries=[];self.calls=[];self.fail=False;self.terminated=False
        common=list(module.prospective_plan()['phase_A'])+list(module.prospective_plan()['prospective_phase_C'])
        common.extend(item for item in module.prospective_plan()['prospective_phase_B']
                      if item['H_um']==75 and item['method'] in ('reference','NN'))
        for item in common:
            entry=self.complete(item)
            if item['method']=='NN':
                path=folder/f"analysis-{entry['id']}.json";path.write_text('{}')
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
            raise RuntimeError('Synthetic control worker failure')
        return dict(entry=self.complete(item),stage='stage2')


class TestOriginalControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.addCleanup(patch.stopall)
        patch.object(module,'FOLDER',self.folder).start()
        self.message='Approve the bounded continuation and complete all remaining checks.\n'
        self.root_interpretation='Root assumes direct calculation means independent Meep forward validation of the same frozen NN lens.'
        (self.folder/'user_authorization_exact.txt').write_bytes(self.message.encode())
        (self.folder/'approval_receipt.json').write_text(json.dumps(dict(status='APPROVED_BY_USER',
            user_message_exact=self.message,user_message_sha256=module.sha(self.folder/'user_authorization_exact.txt'))))
    def jobs(self):return driver.selected_controls(original_plan(),module)
    def receipt(self):
        authority=driver.request_authority(module)
        (self.folder/'original_user_request.txt').write_bytes(self.message.encode())
        (self.folder/'root_scope_interpretation.txt').write_bytes(self.root_interpretation.encode())
        path=self.folder/'selection_receipt.json';path.write_text(json.dumps(dict(
            driver_sha256=driver.sha(driver.__file__),selected_mode=driver.MODE,execution_scope=driver.SCOPE,
            approval_sha256=authority['approval_sha256'],**driver.authority_fields(authority,self.root_interpretation))))
        return path,module.sha(path)
    def install_original_plan(self):
        prior=self.folder/'prior';prior.mkdir(exist_ok=True)
        path=prior/'plan.json';path.write_text(json.dumps(original_plan()))
        (self.folder/'prior_delivery_manifest.json').write_text(json.dumps(dict(files={'plan.json':dict(sha256=module.sha(path))})))
        patch.object(module,'PRIOR',prior).start()
    def repair_receipt(self,stage):
        path=self.folder/'repair-selection.json';path.write_text(json.dumps(dict(driver_sha256=framework.sha(framework.__file__))))
        stage.ledger['subsequent_phase_selection']=dict(mode='repair_only',receipt_file=path.name,receipt_sha256=module.sha(path))
        return path
    def test_exact_six_original_jobs_and_no_plan_mutation(self):
        plan=original_plan();before=copy.deepcopy(plan);jobs=driver.selected_controls(plan,module)
        self.assertEqual(plan,before);self.assertEqual(len(jobs),6)
        self.assertEqual([item['job_type'] for item in jobs],['pml','pml','padding','padding','duration','duration'])
        self.assertEqual(jobs[0]['pml_um'],3.);self.assertEqual(jobs[2]['padding_um'],15.)
        self.assertEqual(jobs[4]['duration'],502.5)
        self.assertTrue(all(item['method'] in ('reference','NN') for item in jobs))
    def test_scope_rejects_extra_direct_wrong_mesh_duration_and_reordering(self):
        for index,key,value in [(1,'method','direct'),(0,'H_um',75.),(0,'resolution',24.),
                                (4,'duration',503.),(0,'pml_um',2.)]:
            bad=self.jobs();bad[index][key]=value
            with self.assertRaises(ValueError):driver.require_scope(bad,module)
        bad=self.jobs();bad.reverse()
        with self.assertRaises(ValueError):driver.require_scope(bad,module)
        bad=self.jobs();bad.append(bad[-1])
        with self.assertRaises(ValueError):driver.require_scope(bad,module)
    def test_all_fourteen_common_raw_and_analysis_results_required(self):
        stage=FakeStage(self.folder)
        evidence=driver.require_common_complete(stage,module,framework)
        self.assertEqual(len(evidence),4)
        stage.ledger['entries'][-1]['analysis_status']='failed'
        with self.assertRaisesRegex(ValueError,'common results must be complete'):
            driver.require_common_complete(stage,module,framework)
    def test_changed_common_analysis_hash_is_rejected(self):
        stage=FakeStage(self.folder);entry=stage.ledger['entries'][-1]
        (self.folder/entry['analysis_file']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'analysis SHA changed'):
            driver.require_common_complete(stage,module,framework)
    def test_tampered_current_repair_receipt_blocks_original_control_preflight(self):
        stage=FakeStage(self.folder);self.install_original_plan();path=self.repair_receipt(stage)
        jobs,estimate,_=driver.preflight(stage,module,framework)
        self.assertEqual(len(jobs),6);self.assertEqual(estimate['projected_new_attempts'],20)
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'selection receipt changed'):
            driver.preflight(stage,module,framework)
    def test_partial_original_cache_preserves_registry_and_still_projects_twenty(self):
        stage=FakeStage(self.folder);self.install_original_plan();self.repair_receipt(stage)
        jobs=self.jobs();path,digest=self.receipt()
        linked=dict(mode=driver.MODE,receipt_file=path.name,receipt_sha256=digest,driver_sha256=driver.sha(driver.__file__))
        stage.ledger['subsequent_phase_runs']=[dict(mode='repair_only',receipt_file='historical-repair',receipt_sha256='retained'),linked]
        for item in jobs[:2]:
            stage.complete(driver.with_receipt(item,module,path,digest))
        before=copy.deepcopy(stage.ledger)
        _,estimate,_=driver.preflight(stage,module,framework)
        self.assertEqual(estimate['already_attempted_new'],16);self.assertEqual(estimate['new_calls_needed'],4)
        self.assertEqual(estimate['projected_new_attempts'],20);self.assertEqual(len(estimate['cached_records']),2)
        self.assertEqual(stage.ledger,before)
    def test_fourteen_plus_six_uses_exact20_and65_with_no_adaptive_reserve(self):
        stage=FakeStage(self.folder);budget=driver.budget(stage,module,self.jobs())
        self.assertEqual(budget['projected_new_attempts'],20);self.assertEqual(budget['projected_cumulative_attempts'],65)
        self.assertEqual(budget['adaptive_reserve'],0)
        stage.ledger['entries'].append(dict(module.job('reference',55,24,'fixture'),id=60,status='failed'))
        with self.assertRaisesRegex(RuntimeError,'20/65'):driver.budget(stage,module,self.jobs())
    def test_original_clock_and_reporting_reserve_not_reset(self):
        stage=FakeStage(self.folder);stage.elapsed=lambda:module.MAX_WALL-module.RESERVE
        with self.assertRaisesRegex(RuntimeError,'Original48h'):driver.budget(stage,module,self.jobs())
    def test_receipt_metadata_and_case_scope_precede_all_six_new_attempts(self):
        stage=FakeStage(self.folder);path,digest=self.receipt();first=stage.ledger['first_started_utc'];deadline=stage.ledger['absolute_deadline_utc']
        with patch.object(framework,'verify_stage1_delivery',return_value=dict(stage1_unchanged=True)):
            driver.execute_controls(stage,module,framework,self.jobs(),path,digest)
        self.assertEqual(len(stage.ledger['entries']),20);self.assertEqual(len(stage.calls),6)
        self.assertEqual(stage.ledger['first_started_utc'],first);self.assertEqual(stage.ledger['absolute_deadline_utc'],deadline)
        for entry in stage.ledger['entries'][14:]:
            self.assertEqual(entry['selection_receipt_file'],path.name)
            self.assertEqual(entry['selection_receipt_sha256'],digest)
            self.assertEqual(entry['selection_scope'],driver.SCOPE)
        outcome=json.loads((self.folder/'outcomes.json').read_text())
        self.assertEqual(outcome['adaptive_reserve'],0);self.assertFalse(outcome['full_optical_acceptance_claimed'])
    def test_completed_worker_failed_analysis_recovers_without_extra_attempt(self):
        stage=FakeStage(self.folder);jobs=self.jobs();stage.complete(jobs[0]);stage.complete(jobs[1],analysis='failed')
        path,digest=self.receipt()
        with patch.object(framework,'verify_stage1_delivery',return_value=dict(stage1_unchanged=True)):
            driver.execute_controls(stage,module,framework,jobs,path,digest)
        self.assertEqual(len(stage.ledger['entries']),20)
        self.assertEqual(stage.ledger['entries'][15]['analysis_status'],'completed')
    def test_failure_is_charged_and_stops_without_unchanged_retry(self):
        stage=FakeStage(self.folder);stage.fail=True;path,digest=self.receipt()
        with self.assertRaisesRegex(RuntimeError,'Synthetic control'):
            driver.execute_controls(stage,module,framework,self.jobs(),path,digest)
        self.assertEqual(len(stage.ledger['entries']),15);self.assertEqual(len(stage.calls),1)
        self.assertEqual(stage.ledger['entries'][-1]['selection_receipt_sha256'],digest)
        self.assertTrue(stage.terminated)
        with self.assertRaisesRegex(RuntimeError,'unchanged retries'):driver.budget(stage,module,self.jobs())
    def test_original_request_authority_and_default_cli_never_launches(self):
        authority=driver.request_authority(module)
        self.assertEqual(authority['message_exact'],self.message)
        for interpretation in ['', '   ',None]:
            with self.assertRaises(ValueError):driver.authority_fields(authority,interpretation)
        out=io.StringIO()
        with patch.object(sys,'argv',['original-controls']),patch.object(driver,'load_framework',return_value=(framework,module)),\
             patch.object(module,'Stage',return_value=FakeStage(self.folder)),\
             patch.object(driver,'preflight',side_effect=ValueError('Common work not complete')),\
             patch.object(driver,'run_controls',side_effect=AssertionError('No implicit launch')),patch('sys.stdout',out):
            driver.main()
        self.assertEqual(json.loads(out.getvalue())['worker_launches'],0)
        for argv in [['original-controls','--run'],['original-controls','--run','--selection-reason','Time passed'],
                     ['original-controls','--run','--user-reply-file','new-priority-answer.txt']]:
            with patch.object(sys,'argv',argv),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):driver.main()
    def test_altered_supplied_or_immutable_request_is_not_authorization(self):
        supplied=self.folder/'supplied-request.txt';supplied.write_text('A different request')
        with self.assertRaisesRegex(ValueError,'Original request bytes'):
            driver.request_authority(module,supplied)
        (self.folder/'user_authorization_exact.txt').write_text(self.message+'changed')
        with self.assertRaisesRegex(ValueError,'Original request bytes'):
            driver.request_authority(module)
    def test_receipt_rejects_new_reply_elapsed_approval_or_user_quote_claims(self):
        path,_=self.receipt();original=json.loads(path.read_text())
        driver.validate_request_receipt(original,module,self.folder)
        for key,value in [('decision_origin','EXPLICIT_USER_REPLY'),('new_priority_reply_received',True),
                          ('clarification_pending',False),('elapsed_used_as_authorization',True),
                          ('root_scope_interpretation_is_user_quote',True),
                          ('root_scope_interpretation_origin','EXPLICIT_USER_REPLY'),
                          ('user_direct_calculation_meaning_claimed',True),
                          ('root_scope_interpretation',''),('decision_text_exact','A new priority answer'),
                          ('direct_calculation_assumption','SEPARATELY_OPTIMIZED_LENS')]:
            receipt=copy.deepcopy(original);receipt[key]=value
            with self.assertRaises(ValueError):driver.validate_request_receipt(receipt,module,self.folder)
    def test_root_interpretation_sidecar_is_separate_and_hash_protected(self):
        path,_=self.receipt();receipt=json.loads(path.read_text())
        self.assertNotEqual(receipt['root_scope_interpretation'],receipt['decision_text_exact'])
        (self.folder/'root_scope_interpretation.txt').write_text('Changed root interpretation')
        with self.assertRaisesRegex(ValueError,'sidecar bytes changed'):
            driver.validate_request_receipt(receipt,module,self.folder)
    def test_frozen_framework_and_active_stage_sources_unchanged(self):
        self.assertEqual(driver.sha(driver.FRAMEWORK_PATH),driver.FRAMEWORK_SHA)
        self.assertEqual(framework.sha(framework.STAGE_PATH),framework.EXPECTED_STAGE_SHA)
        self.assertEqual(driver.sha(ROOT/'scripts/run_height_family_original_controls_v1.py'),driver.V1_DRIVER_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
