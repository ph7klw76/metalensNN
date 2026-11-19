"""Synthetic report semantics only; no actual final report, graph or optics."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuation_report_fixture',ROOT/'scripts/write_height_family_continuation_report_v1.py')
reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)


def fixture():
    def row(H):
        return {'H_um':H,'Meep':{'resolution':48. if H==100 else 32.,'eta':.955,'C':.02,'Q':1.04,
            'planes':[{'z_um':0.,'diameter_x_um':12+H/20,'diameter_y_um':11.9+H/20}]},
            'native':{'status':'PASS'},'nominal_optical':{'status':'PASS'},
            'selected_mesh':{'status':'PASS','resolutions':[32.,48.] if H==100 else [24.,32.]}}
    rows=[row(H) for H in reporter.HEIGHTS]
    budget={'additional_attempts':18,'cumulative_attempts':83,'actual_cumulative_cases':25,'reserved_case_ceiling':28,
            'status':'PASS','absolute_elapsed_s_including_idle':140000.,
            'original_first_started_utc':'2026-11-13T08:59:13.107420+00:00','absolute_deadline_utc':'2026-11-15T08:59:13.107420+00:00'}
    r={'NN_heights':rows,'budget':budget,'original_seven_height_family_acceptance':{'status':'PASS'},
       'all_eight_nominal_heights_acceptance':{'status':'PASS'},'authorization':{'status':'PASS'},
       'primary_execution':{'status':'PASS','completed_verified':14},'adaptive_execution':{'status':'PASS','pending_current_selected_pairs':[]},
       'continuation_execution':{'status':'PASS'},'execution_completion_is_not_optical_acceptance':True,
       'NN_superiority_claimed':False,'global_diffraction_optimum_claimed':False,'full_M2_claimed':False,'experimental_validation_claimed':False,
       'all_attempts':[{'id':i,'status':'started' if i==41 else 'completed'} for i in range(1,84)],
       'selected_adaptive_diagnostics':[{'pair_id':name,'native_status':'PASS','numerical_status':'PASS','target_nominal_status':'PASS'} for name in ('H60_fixture','H100_fixture')],
       'boundary_duration_checks':[{'required_original_sampled_control':True,'job_type':kind,'H_um':H,'status':'PASS','Q_prior':1.12,'Q_variant':1.1235} for kind,H in [('pml',40.),('pml',100.),('padding',70.),('duration',100.)]]}
    ledger={'entries':[{'id':i,'method':'reference' if i%2==0 else 'NN','status':'completed','analysis_status':'completed'} for i in range(66,84)]}
    old=copy.deepcopy(r)
    reporter.row(old,80.)['Meep']['Q']=1.11020712;reporter.row(old,100.)['Meep']['Q']=1.10336477
    reporter.row(old,90.)['selected_mesh']['comparison']={'measurements':{'maximum_diameter_relative_change':.0200556839}}
    primary=copy.deepcopy(r);reporter.row(primary,60.)['selected_mesh']['comparison']={'measurements':{'curvature_absolute_change':.013246714}}
    first=copy.deepcopy(r);reporter.row(first,60.)['selected_mesh']['comparison']={'measurements':{'curvature_absolute_change':.00294367157}}
    ctx={'historical41_status':'started','config':{'wavelength_um':1.55,'mfd_um':10.4,'cylinder_diameter_um':50.,'materials':{'polymer':{'n':1.5263}}},
         'learning':{'model':{'hidden_layers':[64,64,64],'outputs':50},'objective_weights':{'width':.1,'phase':.5,'angular':.25,'sidelobe':.2,'roughness':.0002},
                     'matched_training_validation_recovery_objective_calls':972},
         'prior':old,'primary':primary,'first16':first,
         'q_diagnosis':{'Q_reduction_decomposition':{'groups':[{'label':'20_deg_and_above_retained','signed_share_of_total_Q_reduction':1.13187,
                       'r24_power_fraction':.0009176,'r32_power_fraction':.0003824}]},'r32_max_abs_Q_sampling_shift':.00035714},
         'clock_tests':{'successful_check':{'passed':19}},'collector_tests':{'pure_tests':{'WSL':{'count':42}}},'alias_tests':{'pure_tests':{'native':5}}}
    paths={key:str(ROOT/'artifacts/synthetic_fixture'/key) for key in ('assessment','graphs','audits','learning','replay','q_diagnosis')}
    return r,ledger,ctx,paths


class TestContinuationReport(unittest.TestCase):
    def test_completed_saved_fixture_report_is_readable_and_uses_actual_mesh(self):
        r,l,c,p=fixture();reporter.validate_finished(r,l);text=reporter.render_report(r,l,c,p)
        self.assertTrue(800<=reporter.words(text)<=1200)
        self.assertIn('all-eight-height outcome PASS',text)
        self.assertIn('100       48',text)
        self.assertIn('observed, not held out',text)
        self.assertIn('972 matched',' '.join(text.split()))
        self.assertIn('82 completed calculations',text)
    def test_finer_failed_result_is_retained_and_never_hardcoded_as_final_PASS(self):
        r,l,c,p=fixture();x=reporter.row(r,100.);x['Meep'].update(C=.123456,Q=1.234567,eta=.912345)
        x['nominal_optical']['status']='FAIL';x['selected_mesh']['status']='FAIL'
        r['all_eight_nominal_heights_acceptance']['status']=r['original_seven_height_family_acceptance']['status']='FAIL'
        r['selected_adaptive_diagnostics'][1]['target_nominal_status']='FAIL'
        reporter.validate_finished(r,l);text=reporter.render_report(r,l,c,p)
        self.assertIn('all-eight-height outcome FAIL',text)
        self.assertIn('earlier 16-call snapshot, the all-eight classification was PASS',' '.join(text.split()))
        self.assertIn('+0.123456',text);self.assertIn('1.234567',text);self.assertIn('91.23450',text)
        self.assertIn('target nominal FAIL',text)
    def test_unfinished_final_pair_or_protocol_cannot_be_published_as_final(self):
        for mutate in [lambda r,l:l['entries'].pop(),lambda r,l:l['entries'][-1].update(status='started'),
                       lambda r,l:r['adaptive_execution'].update(status='UNRESOLVED'),
                       lambda r,l:r['budget'].update(cumulative_attempts=82),
                       lambda r,l:r['all_attempts'][40].update(status='completed'),
                       lambda r,l:r.update(global_diffraction_optimum_claimed=True)]:
            r,l,_,_=fixture();mutate(r,l)
            with self.assertRaises(ValueError):reporter.validate_finished(r,l)
    def test_audit_must_authenticate_exact_current_ledger_without_calling_inspection(self):
        p=ROOT/'artifacts/synthetic_fixture/ledger.json';digest='a'*64
        self.assertTrue(reporter.audit_binds_ledger({'status':'PASS_RAW_AUDIT','stage3_ledger_sha256':digest},p,digest))
        self.assertFalse(reporter.audit_binds_ledger({'status':'PASS_RAW_AUDIT','stage3_ledger_sha256':'b'*64},p,digest))
        self.assertFalse(reporter.audit_binds_ledger({'status':'FAIL_RAW_AUDIT','stage3_ledger_sha256':digest},p,digest))
        graph=ROOT/'artifacts/synthetic_fixture/delivery_manifest.json'
        self.assertTrue(reporter.audit_binds_graph({'status':'PASS_GRAPH_AUDIT','graph_delivery_manifest_sha256':digest},graph,digest))
        self.assertTrue(reporter.audit_binds_graph({'status':'PASS_GRAPH_AUDIT','input_sha256':{str(graph):digest}},graph,digest))
        self.assertFalse(reporter.audit_binds_graph({'status':'PASS_GRAPH_AUDIT','notes':{'unrelated_expected_string':digest}},graph,digest))
        self.assertFalse(reporter.audit_binds_graph({'status':'PASS_GRAPH_AUDIT','input_sha256':{str(graph.parent/'wrong.json'):digest}},graph,digest))
    def test_changed_context_file_cannot_silently_rebaseline_authenticated_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            p=Path(temp)/'context.json';p.write_text('{"original":true}')
            inputs=reporter.ProtectedInputs();self.assertEqual(inputs.document(p),{'original':True})
            original=inputs.values[str(p.resolve())];p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed between'):inputs.document(p)
            self.assertEqual(inputs.values[str(p.resolve())],original)
            self.assertFalse(inputs.unchanged())
    def test_legacy_started_marker_is_preserved_and_bound_to_immutable_stage1(self):
        r,l,c,p=fixture();reporter.validate_finished(r,l)
        immutable={'entries':[{'id':41,'status':'started'}]}
        self.assertEqual(reporter.historical_marker(r,immutable),'started')
        r['all_attempts'][40]['status']='interrupted'
        with self.assertRaisesRegex(ValueError,'exact immutable'):reporter.historical_marker(r,immutable)
    def test_changed_historical_context_is_rejected_even_on_first_read(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            p=Path(temp)/'q_diagnosis.json';p.write_text('{"reviewed":true}')
            reviewed=reporter.sha(p);p.write_text('{"tampered_Q_share":999}')
            inputs=reporter.ProtectedInputs()
            with self.assertRaisesRegex(ValueError,'Pinned historical'):inputs.document(p,reviewed)
            self.assertFalse(inputs.values)
    def test_primary_context_must_match_exact_prospective_receipt_copy_and_request(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            stage=Path(temp)/'campaign/maxwell_validation_v3'
            context_path=stage/'assessment/first16/assessment.json';context_path.parent.mkdir(parents=True)
            context={'assessment_utility_sha256':'reviewed_utility','ledger_snapshot_sha256':{str(stage/'ledger.json'):'reviewed_ledger'}}
            context_path.write_text(json.dumps(context));context_sha=reporter.sha(context_path)
            parent=stage/'adaptive_selections/fixture';parent.mkdir(parents=True)
            (parent/'primary_assessment.json').write_bytes(context_path.read_bytes())
            request_path=parent/'selection_request_exact.json'
            request_path.write_text(json.dumps({'primary_assessment_file':str(context_path),'primary_assessment_sha256':context_sha}))
            request_sha=reporter.sha(request_path);receipt_path=parent/'selection_receipt.json'
            receipt_path.write_text(json.dumps({'primary_assessment_sha256':context_sha,'primary_assessment_ledger_sha256':'reviewed_ledger',
                'primary_assessment_utility_sha256':'reviewed_utility','request_sha256':request_sha}))
            receipt_sha=reporter.sha(receipt_path);inputs={receipt_path.resolve():receipt_sha,request_path.resolve():request_sha}
            reporter.bind_context_to_receipt(stage,context_path,context,receipt_path,receipt_sha,inputs,reporter.ProtectedInputs())
            context_path.write_text(json.dumps(dict(context,changed_classification=True)))
            with self.assertRaisesRegex(ValueError,'prospective receipt'):
                reporter.bind_context_to_receipt(stage,context_path,context,receipt_path,receipt_sha,inputs,reporter.ProtectedInputs())
    def test_stale_PASS_headline_cannot_hide_selected_or_original_control_failure(self):
        for mutate in [lambda r:reporter.row(r,100.)['native'].update(status='FAIL'),
                       lambda r:reporter.row(r,60.)['selected_mesh'].update(status='UNRESOLVED'),
                       lambda r:r['boundary_duration_checks'][0].update(status='FAIL')]:
            r,l,_,_=fixture();mutate(r)
            with self.assertRaisesRegex(ValueError,'Family PASS contradicts'):reporter.validate_finished(r,l)
        r,l,_,_=fixture();reporter.row(r,75.)['nominal_optical']['status']='FAIL'
        r['all_eight_nominal_heights_acceptance']['status']='FAIL'
        reporter.validate_finished(r,l)  # Original-seven excludes validation H75.
    def test_unresolved_missing_metrics_are_NA_with_available_width_context(self):
        r,l,c,p=fixture();x=reporter.row(r,100.)
        x['Meep'].update(eta=None,C=None,Q=None)
        x['Meep']['planes'][0].update(diameter_x_um=None,diameter_y_um=None)
        x['nominal_optical']['status']=x['selected_mesh']['status']='UNRESOLVED'
        r['all_eight_nominal_heights_acceptance']['status']=r['original_seven_height_family_acceptance']['status']='UNRESOLVED'
        reporter.validate_finished(r,l);text=reporter.render_report(r,l,c,p)
        self.assertIn('all-eight-height outcome UNRESOLVED',text)
        self.assertIn('      NA        NA         NA',text)
        self.assertIn('Available output x/y diameters',' '.join(text.split()))
    def test_sealed_or_closing_tree_is_rejected_before_report_write(self):
        for sentinel in ('delivery_manifest.json','final_closure','final_source_snapshots'):
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
                stage=Path(temp);reporter.require_unsealed(stage)
                (stage/sentinel).write_text('immutable fixture marker')
                with self.assertRaisesRegex(ValueError,'must remain immutable'):reporter.require_unsealed(stage)
    def test_separate_diagnostic_failure_does_not_redefine_frozen_family_components(self):
        r,l,c,p=fixture()
        r['selected_adaptive_diagnostics'][1].update(numerical_status='FAIL',target_nominal_status='FAIL')
        reporter.validate_finished(r,l);text=reporter.render_report(r,l,c,p)
        self.assertIn('all-eight-height outcome PASS',text)
        self.assertIn('target nominal FAIL',' '.join(text.split()))


if __name__=='__main__':unittest.main(verbosity=2)
