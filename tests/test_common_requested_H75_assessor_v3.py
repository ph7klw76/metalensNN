"""Narrow commonNN75 authorization fixtures; no driver/optical execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full_family_common75_fixtures',ROOT/'tests/test_full_family_assessment_stage2_v3.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
assessment=fixtures.assessment


class TestCommon75Authorization(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=fixtures.Fixture(self.temp.name);self.entries,self.path=self.f.common75()
        self.base_receipt=self.f.f.read(self.path);self.base_snapshot=self.f.f.read(self.path.parent/'prior_ledger.json')
        self.data=self.f.evidence();self.stage=self.f.f.stage
        self.ids={entry['id'] for entry in self.entries}
    def authorized(self,receipt_change=None,attempt_change=None,snapshot_change=None):
        receipt=copy.deepcopy(self.base_receipt);snapshot=copy.deepcopy(self.base_snapshot)
        if snapshot_change:
            snapshot_change(snapshot);self.f.f.write(self.path.parent/'prior_ledger.json',snapshot)
            receipt['prior_ledger_snapshot_sha256']=fixtures.comparison.report.sha(self.path.parent/'prior_ledger.json')
        else:self.f.f.write(self.path.parent/'prior_ledger.json',snapshot)
        if receipt_change:receipt_change(receipt)
        self.f.f.write(self.path,receipt);digest=fixtures.comparison.report.sha(self.path)
        data=copy.deepcopy(self.data)
        for stage in data['stage_metadata']:
            for linked in [stage.get('subsequent_phase_selection'),*(stage.get('subsequent_phase_runs') or [])]:
                if linked and linked.get('receipt_file')==str(self.path.relative_to(self.stage)):linked['receipt_sha256']=digest
        attempts=[row for row in data['all_attempts'] if row['id'] in self.ids and Path(row['stage_folder'])==self.stage]
        for attempt in attempts:
            attempt['selection_receipt_sha256']=digest
            if attempt_change:attempt_change(attempt)
        auth=assessment.Authorization(self.stage,data)
        return [auth.attempt(attempt) for attempt in attempts]
    def test_exact_common75_root_reason_valid_without_common70_first(self):
        self.assertTrue(all(row['status']=='PASS' for row in self.authorized()))
        result=self.f.assess();self.assertEqual(result['protocol']['status'],'PASS')
        self.assertEqual(result['budget']['charged_new_unique'],1)
        self.assertEqual(result['budget']['actual_cumulative_unique'],25)
        self.assertEqual(result['requested_NN_direct_comparisons']['status'],'UNRESOLVED')
        self.assertNotIn(75.,[row['H_um'] for row in result['NN_heights']])
    def test_common75_history_survives_later_current_priority_selection(self):
        # The per-entry receipt and registry history, not the mutable latest
        # selection, anchor this already-authorized common forward pair.
        for stage in self.data['stage_metadata']:
            if Path(stage['folder'])==self.stage:
                stage['subsequent_phase_selection']={'mode':'original_controls_after_common',
                    'receipt_file':'later_selection/not_needed_for_this_pair.json',
                    'receipt_sha256':'b'*64,'driver_sha256':'c'*64}
        self.assertTrue(all(row['status']=='PASS' for row in self.authorized()))
    def test_optimizer_direct_cannot_use_common75_root_authority(self):
        def change_receipt(receipt):receipt['selected_jobs'][1]['method']='direct'
        def change_attempt(attempt):
            if attempt['method']=='NN':attempt['method']='direct'
        results=self.authorized(change_receipt,change_attempt)
        self.assertTrue(all(row['status']!='PASS' for row in results))
    def test_wrong_height_mesh_boundary_duration_and_phase_are_rejected(self):
        for key,value in [('H_um',100.),('resolution',32.),('pml_um',3.),('padding_um',15.),('duration',260.),('phase','B')]:
            with self.subTest(field=key):
                def change_receipt(receipt):
                    for job in receipt['selected_jobs']:job[key]=value
                def change_attempt(attempt):attempt[key]=value
                self.assertTrue(all(row['status']!='PASS' for row in self.authorized(change_receipt,change_attempt)))
    def test_budget_headroom_and_pending_choice_flags_must_be_exact(self):
        def bad_effect(receipt):receipt['common75_budget_effect']['original_adaptive_remaining']=2
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(bad_effect)))
        for flag in ('priority_selected','direct_calculation_meaning_selected','separately_optimized_direct_launched','original_control_deferral_selected'):
            with self.subTest(flag=flag):
                self.assertTrue(all(row['status']!='PASS' for row in self.authorized(lambda receipt:receipt.update({flag:True}))))
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(lambda receipt:receipt.pop('common75_budget_effect'))))
    def test_frozen_geometry_and_prelaunch_entry_link_required(self):
        def wrong_geometry(attempt):
            if attempt['method']=='NN':attempt['design_sha256']='b'*64
        self.assertNotEqual(self.authorized(attempt_change=wrong_geometry)[1]['status'],'PASS')
        for key in ('selection_receipt_file','selection_receipt_sha256','selection_mode','selection_scope'):
            with self.subTest(field=key):
                self.assertTrue(all(row['status']!='PASS' for row in self.authorized(attempt_change=lambda row:row.update({key:None}))))
    def test_approval_plan_source_and_prospective_snapshot_guards_still_apply(self):
        for key in ('approval_sha256','preparation_manifest_sha256','frozen_plan_sha256','frozen_stage_runner_sha256'):
            with self.subTest(hash=key):
                self.assertTrue(all(row['status']!='PASS' for row in self.authorized(lambda receipt:receipt.update({key:'b'*64}))))
        results=self.authorized(snapshot_change=lambda snapshot:snapshot['entries'].append(self.entries[0]))
        self.assertNotEqual(results[0]['status'],'PASS')


if __name__=='__main__':unittest.main(verbosity=2)
