"""Pure original-control authorization fixtures; no worker or driver imports."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('common75_assessor_fixtures',ROOT/'tests/test_common_requested_H75_assessor_v3.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
fixtures=common.fixtures


class TestOriginalControlAuthorization(unittest.TestCase):
    authorized=common.TestCommon75Authorization.authorized
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=fixtures.Fixture(self.temp.name);self.stage=self.f.f.stage
        plan=self.f.f.read(self.stage/'plan.json');plan['phase_A']=[]
        for H in (100,50,60,80,90):
            pair=self.f.pair(H,24,kind='nominal')
            for entry in pair:
                entry['phase']='A'
                plan['phase_A'].append({key:entry.get(key) for key in ('method',*fixtures.assessment.NUMERICS,'job_type','phase')})
        pair70=self.f.pair(70,32);pair75=self.f.pair(75,24,kind='nominal')
        plan['prospective_phase_C']=[{key:e.get(key) for key in ('method',*fixtures.assessment.NUMERICS,'job_type','phase')} for e in pair70]
        plan['prospective_phase_B']=[{key:e.get(key) for key in ('method',*fixtures.assessment.NUMERICS,'job_type','phase')} for e in pair75]
        self.entries=[]
        for kind,H,changed in [('pml',100,{'pml_um':3.}),('padding',70,{'padding_um':15.}),('duration',100,{'duration':502.5})]:
            self.entries.extend(self.f.pair(H,16,kind=kind,changed=changed))
        for entry in self.entries:entry['phase']='ORIGINAL_CONTROLS_AFTER_COMMON'
        self.f.f.write(self.stage/'plan.json',plan)
        oldplan=self.f.f.read(self.f.f.prior/'plan.json')
        oldplan['jobs'].extend({key:e.get(key) for key in ('method',*fixtures.assessment.NUMERICS,'job_type')} for e in self.entries)
        self.f.f.write(self.f.f.prior/'plan.json',oldplan);self.f.seal_preparation()
        self.path=self.f.authorize(self.entries,mode='original_controls_after_common')
        receipt=self.f.f.read(self.path)
        scope='USER_SELECTED_ORIGINAL_CONTROLS_AFTER_COMMON_NO_ADAPTIVE_RESERVE'
        receipt.update(execution_scope=scope,frozen_cases_sha256=fixtures.comparison.report.sha(self.stage/'frozen_cases.json'),
            adaptive_reserve=0,separate20cap_selection_after_common=True,frozen18cap_driver_unchanged=True,
            separately_optimized_direct_launched=False,direct_calculation_interpretation_claimed=False,
            original_controls_budget_effect={'common_completed_attempts':14,'original_control_calls':6,
                'projected_new_attempts':20,'projected_cumulative_attempts':65,'adaptive_remaining':0})
        self.f.f.write(self.path,receipt);digest=fixtures.comparison.report.sha(self.path)
        for linked in self.f.metadata['subsequent_phase_runs']:
            if linked['receipt_file']==str(self.path.relative_to(self.stage)):linked['receipt_sha256']=digest
        for entry in self.entries:entry.update(selection_receipt_sha256=digest,
            selection_mode='original_controls_after_common',selection_scope=scope)
        self.base_receipt=receipt;self.base_snapshot=self.f.f.read(self.path.parent/'prior_ledger.json')
        self.data=self.f.evidence();self.ids={entry['id'] for entry in self.entries}
    def test_exact_user_only_controls_pass_after_all14_completed(self):
        results=self.authorized()
        self.assertEqual(len(results),6)
        self.assertTrue(all(row['status']=='PASS' for row in results))
        self.assertEqual(len(self.base_snapshot['entries']),14)
    def test_root_reason_or_wrong_job_scope_cannot_authorize_original_controls(self):
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda r:r.update(decision_origin='EXPLICIT_ROOT_SELECTION_REASON'))))
        def change_jobs(r):r['selected_jobs'][0]['pml_um']=2.
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(change_jobs)))
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            attempt_change=lambda row:row.update(selection_scope='OTHER_SCOPE'))))
    def test_no_adaptive_reserve_or_optimizer_claim_can_be_added(self):
        changes=[lambda r:r.update(adaptive_reserve=2),
            lambda r:r['original_controls_budget_effect'].update(adaptive_remaining=2),
            lambda r:r.update(separately_optimized_direct_launched=True),
            lambda r:r.update(direct_calculation_interpretation_claimed=True)]
        for change in changes:
            self.assertTrue(all(row['status']!='PASS' for row in self.authorized(change)))
    def test_incomplete_or_unsealed_common_results_do_not_pass(self):
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            snapshot_change=lambda snapshot:snapshot['entries'][0].update(status='running'))))
        def bad_analysis(snapshot):
            next(e for e in snapshot['entries'] if e['method']=='NN')['analysis_sha256']='b'*64
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(snapshot_change=bad_analysis)))
    def test_geometry_receipt_link_and_frozen_case_seal_required(self):
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda r:r.update(frozen_cases_sha256='b'*64)) ))
        results=self.authorized(attempt_change=lambda row:row.update(design_sha256='b'*64) if row['method']=='NN' else None)
        self.assertTrue(all(row['status']!='PASS' for row in results[1::2]))
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            attempt_change=lambda row:row.update(selection_receipt_sha256=None))))


if __name__=='__main__':unittest.main(verbosity=2)
