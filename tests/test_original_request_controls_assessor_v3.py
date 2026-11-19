"""Original-request scope fixtures; no Meep, training or driver execution."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_control_assessor_fixtures',ROOT/'tests/test_original_controls_assessor_v3.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
fixtures=original.fixtures


class TestOriginalRequestControls(original.TestOriginalControlAuthorization):
    def setUp(self):
        super().setUp()
        self.request='Complete the remaining checks for all heights within20 additional local Meep solves and48 more hours.\n'
        self.interpretation='Root interprets direct calculation as independent Meep forward of the same frozen NN; clarification remains pending.'
        user_hash=hashlib.sha256(self.request.encode()).hexdigest()
        approval=self.f.f.read(self.stage/'approval_receipt.json')
        approval.update(user_message_exact=self.request,user_message_sha256=user_hash)
        self.f.f.write(self.stage/'approval_receipt.json',approval)
        (self.stage/'user_authorization_exact.txt').write_bytes(self.request.encode())
        self.f.seal_preparation()
        prep=self.f.f.read(self.stage/'preparation_manifest.json');prep['user_authorization_exact.txt']=user_hash
        self.f.f.write(self.stage/'preparation_manifest.json',prep)
        (self.path.parent/'original_user_request.txt').write_bytes(self.request.encode())
        (self.path.parent/'root_scope_interpretation.txt').write_bytes(self.interpretation.encode())
        mode='original_controls_from_request'
        scope='ORIGINAL_EXPLICIT_USER_REQUEST_REMAINING_SIX_CONTROLS_SAME_NN_FORWARD_NO_ADAPTIVE_RESERVE'
        receipt=self.f.f.read(self.path)
        receipt.pop('direct_calculation_interpretation_claimed')
        receipt.update(selected_mode=mode,execution_scope=scope,decision_origin='EXPLICIT_USER_REQUEST',
            decision_text_exact=self.request,decision_text_sha256=user_hash,
            original_user_request_sha256=user_hash,frozen_v1_actual_reply_driver_unchanged=True,
            approval_sha256=fixtures.comparison.report.sha(self.stage/'approval_receipt.json'),
            preparation_manifest_sha256=fixtures.comparison.report.sha(self.stage/'preparation_manifest.json'),
            original_user_request_file='original_user_request.txt',root_scope_interpretation_file='root_scope_interpretation.txt',
            root_scope_interpretation=self.interpretation,
            root_scope_interpretation_sha256=hashlib.sha256(self.interpretation.encode()).hexdigest(),
            root_scope_interpretation_origin='ROOT_SCOPE_INTERPRETATION_NOT_USER_QUOTE',
            root_scope_interpretation_is_user_quote=False,new_priority_reply_received=False,
            clarification_pending=True,elapsed_used_as_authorization=False,user_direct_calculation_meaning_claimed=False,
            direct_calculation_assumption='INDEPENDENT_MEEP_FORWARD_OF_SAME_FROZEN_NN_GEOMETRY',
            separate_non_neural_comparison_scope='PROXY_ONLY_UNLESS_USER_STEERS')
        self.f.f.write(self.path,receipt);digest=fixtures.comparison.report.sha(self.path)
        for linked in self.f.metadata['subsequent_phase_runs']:
            if linked['receipt_file']==str(self.path.relative_to(self.stage)):linked.update(mode=mode,receipt_sha256=digest)
        for entry in self.entries:entry.update(selection_receipt_sha256=digest,selection_mode=mode,selection_scope=scope)
        self.base_receipt=receipt;self.data=self.f.evidence()
    def test_actual_original_request_is_distinct_from_root_interpretation(self):
        rows=self.authorized();self.assertTrue(all(row['status']=='PASS' for row in rows))
        self.assertTrue(all(row['decision_origin']=='EXPLICIT_USER_REQUEST' for row in rows))
        self.assertTrue(all(row['root_scope_interpretation']==self.interpretation for row in rows))
        self.assertTrue(all(row['new_priority_reply_received'] is False for row in rows))
    def test_fabricated_new_reply_or_elapsed_authorization_is_rejected(self):
        for key,value in [('decision_origin','EXPLICIT_USER_REPLY'),('new_priority_reply_received',True),
                ('elapsed_used_as_authorization',True),('clarification_pending',False),
                ('root_scope_interpretation_is_user_quote',True),('user_direct_calculation_meaning_claimed',True)]:
            with self.subTest(key=key):
                self.assertTrue(all(row['status']!='PASS' for row in self.authorized(lambda receipt:receipt.update({key:value}))))
    def test_root_interpretation_cannot_replace_original_user_words(self):
        def change(receipt):
            receipt['decision_text_exact']=self.interpretation
            receipt['decision_text_sha256']=hashlib.sha256(self.interpretation.encode()).hexdigest()
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(change)))
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda receipt:receipt.update(root_scope_interpretation_sha256='b'*64))))
    def test_same_geometry_forward_assumption_cannot_authorize_non_neural_optimization(self):
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda receipt:receipt.update(direct_calculation_assumption='NON_NEURAL_DIRECT_OPTIMIZER'))))
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda receipt:receipt.update(separate_non_neural_comparison_scope='RUN_NOW'))))
    def test_v1_reply_claim_is_not_a_v2_user_meaning_field(self):
        self.assertNotIn('direct_calculation_interpretation_claimed',self.base_receipt)
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized(
            lambda receipt:receipt.update(original_user_request_sha256='b'*64))))
    def test_no_adaptive_reserve_or_optimizer_claim_can_be_added(self):
        # The v2 explicitly distinguishes a root assumption from a user claim.
        for change in [lambda r:r.update(adaptive_reserve=2),
                lambda r:r['original_controls_budget_effect'].update(adaptive_remaining=2),
                lambda r:r.update(separately_optimized_direct_launched=True),
                lambda r:r.update(user_direct_calculation_meaning_claimed=True)]:
            self.assertTrue(all(row['status']!='PASS' for row in self.authorized(change)))
    def test_original_request_copy_and_interpretation_sidecars_must_be_exact(self):
        path=self.path.parent/'original_user_request.txt';path.write_bytes(b'Changed request\n')
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized()))
        path.write_bytes(self.request.encode())
        (self.path.parent/'root_scope_interpretation.txt').write_bytes(b'Changed root interpretation\n')
        self.assertTrue(all(row['status']!='PASS' for row in self.authorized()))


if __name__=='__main__':unittest.main(verbosity=2)
