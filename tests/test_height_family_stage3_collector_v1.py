"""Pure18/83 clock, prospective-receipt and immutable-plan fixtures."""
import copy
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_collector_test',ROOT/'scripts/collect_height_family_stage3_v1.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


def budget_fixture(count=14):
    approval={'status':'APPROVED_BY_USER','decision_origin':'EXPLICIT_USER_REPLY',
        'additional_attempt_ceiling':18,'cumulative_attempt_ceiling':83,'prior_cumulative_attempts':65,
        'new_physical_cases_authorized':0,'one_local_CPU_optical_worker':True,'all_RAM_gates':False,
        'time_extension_authorized':False,'original_first_started_utc':collector.ORIGINAL_START,
        'original_absolute_deadline_utc':collector.DEADLINE}
    priors={'prior_attempt_count':65,'actual_prior_physical_cases':25,
        'original_first_started_utc':collector.ORIGINAL_START,'original_absolute_deadline_utc':collector.DEADLINE}
    ledger={'entries':[{'id':i,'started_utc':'2026-11-14T02:00:00Z','status':'completed'} for i in range(66,66+count)],
        'prior_attempt_count':65,'first_started_utc':collector.ORIGINAL_START,'original_first_started_utc':collector.ORIGINAL_START,
        'absolute_deadline_utc':collector.DEADLINE,'elapsed_s':70000.}
    return approval,ledger,priors


class ReceiptFixture:
    def __init__(self,folder):
        self.folder=Path(folder);self.receipt_dir=self.folder/'prelaunch/66_reference_H100_r32';self.receipt_dir.mkdir(parents=True)
        self.sources={};self.cases=[{'case_id':'NN_H100','design_sha256':'a'*64,'physical_geometry_id':'b'*64,'design_file':'frozen_designs/NN_H100.json'}]
        self.job={'proposed_id':66,'method':'reference','H_um':100.,'resolution':32.,'pml_um':2.,'padding_um':10.,
            'duration':335.,'job_type':'refinement','expected_actual_settings':{'resolution':32.,'H_um':100.,'output_z_um':104.}}
        self.job['output_record']=str(self.folder/'records/66_reference_H100_r32_refinement.json')
        self.job['worker_argv']=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
            '--config',str(self.folder/'config.json'),'--output',self.job['output_record'],'--H','100.0','--resolution','32.0',
            '--pml','2.0','--padding','10.0','--duration','335.0']
        self.plan={'primary_jobs':[self.job]}
        self.entry={**self.job,'id':66,'started_utc':'2026-11-14T02:01:00Z','case_id':None,'design_sha256':None,'physical_geometry_id':None,
            'paired_NN_case_id':'NN_H100','paired_NN_design_sha256':'a'*64,'paired_NN_physical_geometry_id':'b'*64,
            'selection_mode':collector.PRIMARY_MODE,'selection_scope':collector.PRIMARY_SCOPE,'reference_id':None,
            'record_file':'records/66_reference_H100_r32_refinement.json','command':list(self.job['worker_argv']),
            'effective_settings':self.job['expected_actual_settings'],'requested_settings':{k:self.job[k] for k in collector.NUMERICS}}
        for name in ('approved_proposal.json','preparation_manifest.json','plan.json','source_manifest.json','frozen_cases.json','prior_stages.json'):
            (self.folder/name).write_text('{}')
        driver=self.receipt_dir/'run_height_family_maxwell_v3.py';driver.write_text('# fixture only\n')
        digest=collector.sha(driver);self.sources['scripts/run_height_family_maxwell_v3.py']=digest
        self.snapshot={'entries':[],'first_started_utc':collector.ORIGINAL_START,'absolute_deadline_utc':collector.DEADLINE}
        self.write(self.receipt_dir/'prior_ledger.json',self.snapshot)
        field_paths={'approved_proposal_sha256':'approved_proposal.json','preparation_manifest_sha256':'preparation_manifest.json',
            'frozen_plan_sha256':'plan.json','source_manifest_sha256':'source_manifest.json','frozen_cases_sha256':'frozen_cases.json','prior_stages_sha256':'prior_stages.json'}
        self.receipt={'status':'PROSPECTIVE_STAGE3_PRIMARY_ATTEMPT_BEFORE_POPEN','created_utc':'2026-11-14T02:00:59Z',
            'selected_mode':collector.PRIMARY_MODE,'execution_scope':collector.PRIMARY_SCOPE,'decision_origin':'EXPLICIT_USER_REPLY',
            'approval_receipt_sha256':collector.AUTH_SHA,'selected_jobs':[self.job],
            'effective_settings':self.entry['effective_settings'],'requested_settings':self.entry['requested_settings'],
            'attempt_id':66,'prior_attempts':65,'new_attempt_cap':18,'cumulative_cap':83,
            'original_first_started_utc':collector.ORIGINAL_START,'original_absolute_deadline_utc':collector.DEADLINE,
            'geometry_physics_thresholds_unchanged':True,'adaptive_autoexecution':False,'driver_sha256':digest,
            'prior_ledger_snapshot_sha256':collector.sha(self.receipt_dir/'prior_ledger.json'),
            **{k:self.entry[k] for k in ('case_id','design_sha256','physical_geometry_id','paired_NN_case_id','paired_NN_design_sha256','paired_NN_physical_geometry_id')},
            'reference':None,**{key:collector.sha(self.folder/name) for key,name in field_paths.items()}}
        self.ledger={'entries':[self.entry]};self.save_receipt()
    def write(self,path,data):Path(path).write_text(json.dumps(data),encoding='utf-8')
    def protect(self,path):return collector.sha(path)
    def document(self,path):return collector.read(path)
    def save_receipt(self):
        path=self.receipt_dir/'selection_receipt.json';self.write(path,self.receipt)
        relative=str(path.relative_to(self.folder));digest=collector.sha(path)
        self.entry.update(selection_receipt_file=relative,selection_receipt_sha256=digest)
        self.ledger['selection_receipts']=[{'attempt_id':66,'receipt_file':relative,'receipt_sha256':digest,'driver_sha256':self.receipt['driver_sha256']}]
    def result(self):return collector.attempt_receipt(self,self.entry)


class TestBudget(unittest.TestCase):
    now=datetime(2026,11,14,5,tzinfo=timezone.utc)
    def test_all18_attempts_including_failure_stay_charged(self):
        approval,ledger,priors=budget_fixture(18);ledger['entries'][-1]['status']='failed'
        result=collector.budget_assessment(approval,ledger,priors,now=self.now)
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['cumulative_attempts'],83)
        self.assertAlmostEqual(result['absolute_elapsed_s_including_idle'],self.now.timestamp()-collector.utc(collector.ORIGINAL_START).timestamp(),places=6)
    def test_attempt19_or_reset_ID_cannot_pass(self):
        args=budget_fixture(19);self.assertEqual(collector.budget_assessment(*args,now=self.now)['status'],'FAIL')
        args=budget_fixture();args[1]['entries'][0]['id']=1
        self.assertEqual(collector.budget_assessment(*args,now=self.now)['status'],'FAIL')
    def test_shifting_start_and_deadline_cannot_reset_original_allowance(self):
        approval,ledger,priors=budget_fixture()
        ledger['first_started_utc']=ledger['original_first_started_utc']='2026-11-14T05:00:00Z'
        ledger['absolute_deadline_utc']='2026-11-16T05:00:00Z'
        self.assertEqual(collector.budget_assessment(approval,ledger,priors,now=self.now)['status'],'FAIL')
    def test_idle_beyond_original_deadline_fails_even_if_ledger_stopped(self):
        args=budget_fixture();args[1].update(status='STOPPED',elapsed_s=10.)
        result=collector.budget_assessment(*args,now=collector.utc(collector.DEADLINE)+timedelta(seconds=1))
        self.assertEqual(result['status'],'FAIL')
    def test_missing_attempt_clock_is_unresolved(self):
        args=budget_fixture();args[1]['entries'][0].pop('started_utc')
        self.assertEqual(collector.budget_assessment(*args,now=self.now)['status'],'UNRESOLVED')
    def test_RAM_or_worker_or_new_case_scope_cannot_change(self):
        for key,value in [('all_RAM_gates',True),('one_local_CPU_optical_worker',False),('new_physical_cases_authorized',1)]:
            args=budget_fixture();args[0][key]=value
            with self.subTest(key=key):self.assertEqual(collector.budget_assessment(*args,now=self.now)['status'],'FAIL')


class TestReceipt(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=ReceiptFixture(self.temp.name)
    def test_reference_is_homogeneous_with_paired_profile_context_only(self):self.assertEqual(self.f.result()['status'],'PASS')
    def test_both_expected_and_actual_worker_paths_are_fully_resolved(self):
        with patch.object(collector,'ROOT',ROOT/'artifacts'/'..'):
            self.assertEqual(self.f.result()['status'],'PASS')
    def test_reference_cannot_be_charged_or_presented_as_NN_device(self):
        self.f.entry['case_id']=self.f.receipt['case_id']='NN_H100';self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_paired_geometry_or_effective_settings_change_fails(self):
        self.f.receipt['paired_NN_design_sha256']='wrong';self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.receipt['paired_NN_design_sha256']='a'*64
        self.f.receipt['effective_settings']={'resolution':64.};self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_changed_receipt_bytes_are_not_rehashed_into_acceptance(self):
        path=self.f.receipt_dir/'selection_receipt.json';path.write_text(path.read_text()+'\n')
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_receipt_after_start_or_snapshot_with_future_attempt_fails(self):
        self.f.receipt['created_utc']='2026-11-14T02:02:00Z';self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.receipt['created_utc']='2026-11-14T02:00:59Z'
        self.f.snapshot['entries']=[{'id':66}];self.f.write(self.f.receipt_dir/'prior_ledger.json',self.f.snapshot)
        self.f.receipt['prior_ledger_snapshot_sha256']=collector.sha(self.f.receipt_dir/'prior_ledger.json');self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_reserved_adaptive_slots_are_not_primary_authorization(self):
        self.f.receipt['selected_mode']='adaptive_not_reviewed';self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'UNRESOLVED')
    def test_missing_or_changed_actual_command_cannot_pass_receipt_validation(self):
        original=self.f.entry.pop('command')
        self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.entry['command']=original[:-1]+['336.0']
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_output_redirect_or_self_consistent_reference_device_argv_is_rejected(self):
        self.f.entry['record_file']='records/redirected.json'
        self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.entry['record_file']='records/66_reference_H100_r32_refinement.json'
        self.f.job['worker_argv']+=['--design','unexpected-profile.json']
        self.f.entry['command']=list(self.f.job['worker_argv']);self.f.save_receipt()
        self.assertEqual(self.f.result()['status'],'FAIL')


class TestPlan(unittest.TestCase):
    def test_prepared_plan_cannot_self_authorize_different_approved_settings(self):
        jobs=[{'proposed_id':i,'method':'reference' if i%2==0 else 'NN','H_um':100.,'resolution':32.,'pml_um':2.,'padding_um':10.,
               'duration':335.,'job_type':'refinement','expected_actual_settings':{'resolution':32.}} for i in range(66,80)]
        proposal={'primary_jobs':jobs};plan={'primary_jobs':copy.deepcopy(jobs),'adaptive_jobs':[],'adaptive_autoexecution':False}
        collector.validate_primary_plan(plan,proposal)
        plan['primary_jobs'][0]['resolution']=64.
        with self.assertRaisesRegex(ValueError,'approved proposal'):collector.validate_primary_plan(plan,proposal)
    def test_prior_seals_must_match_approval_not_only_self_consistent_new_metadata(self):
        approval={'prior_ledger_sha256':'approved-ledger','prior_delivery_manifest_sha256':'approved-manifest','prior_archive_sha256':'approved-archive'}
        prior={'stage_id':'stage2','ledger_sha256':'approved-ledger','delivery_manifest_sha256':'approved-manifest'}
        links={'stages':[prior],'stage2_archive_sha256':'approved-archive'}
        collector.validate_prior_links(approval,links)
        for key in ('ledger_sha256','delivery_manifest_sha256'):
            altered=copy.deepcopy(links);altered['stages'][0][key]='self-consistent-but-unapproved'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'user-approved seals'):
                collector.validate_prior_links(approval,altered)
        links['stage2_archive_sha256']='changed-archive'
        with self.assertRaisesRegex(ValueError,'user-approved seals'):collector.validate_prior_links(approval,links)


class TestActualRuntimeInputs(unittest.TestCase):
    def setUp(self):
        self.folder=(ROOT/'artifacts/known-fixture-stage3').resolve()
        self.raw={'requested_config':str(self.folder/'config.json'),'solver':'CPU Meep 1.35.0',
                  'incident_power':1.,'power_balance_error':.001}
    def test_homogeneous_reference_cannot_hide_device_inputs(self):
        collector.validate_runtime_inputs(self.raw,self.folder,'reference')
        self.raw['design_file']='another-profile.json'
        with self.assertRaisesRegex(ValueError,'Homogeneous'):collector.validate_runtime_inputs(self.raw,self.folder,'reference')
    def test_same_numeric_config_does_not_allow_wrong_snapshot_path(self):
        self.raw['requested_config']=str(self.folder/'other-identical-config.json')
        with self.assertRaisesRegex(ValueError,'configuration path'):collector.validate_runtime_inputs(self.raw,self.folder,'reference')
    def test_NN_actual_profile_and_reference_paths_must_match_frozen_inputs(self):
        design=self.folder/'frozen_designs/NN_H100.json';ref=self.folder/'records/reference.json'
        self.raw.pop('incident_power');self.raw.update(design_file=str(design),reference_file=str(ref))
        collector.validate_runtime_inputs(self.raw,self.folder,'NN',design,ref)
        self.raw['design_file']=str(design.parent/'different.json')
        with self.assertRaisesRegex(ValueError,'Actual NN'):collector.validate_runtime_inputs(self.raw,self.folder,'NN',design,ref)


if __name__=='__main__':unittest.main(verbosity=2)
