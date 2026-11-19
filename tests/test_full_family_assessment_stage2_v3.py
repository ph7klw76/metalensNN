"""Pure saved-record/prospective-receipt fixtures; no optical worker imports."""
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
assessment=load('full_family_assessment',ROOT/'scripts/assess_height_family_stage2_v3.py')
comparison=load('comparison_schema_fixtures',ROOT/'tests/test_height_family_nn_direct_comparison_v2.py')


class Fixture:
    def __init__(self,path):
        with patch.object(comparison.report,'HEIGHTS',assessment.COLLECT_HEIGHTS):self.f=comparison.Fixture(path)
        self.now=datetime.now(timezone.utc);self.metadata={
            'status':'RUNNING_APPROVED_PHASE_A','first_started_utc':(self.now-timedelta(seconds=100)).isoformat(),
            'absolute_deadline_utc':(self.now-timedelta(seconds=100)+timedelta(seconds=172800)).isoformat()}
        self.f.config['natural_training']['train_H_um']=list(assessment.FAMILY)
        for p in (self.f.base/'config.json',self.f.prior/'config.json',self.f.stage/'config.json'):self.f.write(p,self.f.config)
        # The fixture changes its own declared height domain, never a real file.
        for entry in self.f.entries[self.f.prior]:
            path=self.f.prior/entry['record_file'];raw=self.f.read(path)
            raw['config_fingerprint']=comparison.report.fingerprint(self.f.config);self.f.write(path,raw)
            entry['record_sha256']=comparison.report.sha(path)
            entry['job_type']='reserve_mesh' if entry['resolution']==32 else 'nominal'
        delivery=self.f.read(self.f.base/'delivery_manifest.json')
        delivery['files']['config.json']['sha256']=comparison.report.sha(self.f.base/'config.json')
        self.f.write(self.f.base/'delivery_manifest.json',delivery)
        for H in assessment.FAMILY:
            for res in (16,24):
                if any(e['method']=='NN' and e['H_um']==H and e['resolution']==res for e in self.f.entries[self.f.prior]):continue
                ref=self.f.reference(self.f.prior,H,res);e=self.f.device(self.f.prior,'NN',H,res,ref)
                e['job_type']='nominal'
        while len(self.f.entries[self.f.prior])<45:self.f.reference(self.f.prior,40,8)
        self.f.flush(self.f.prior);self.f.seal_prior()
        oldjobs=[{key:e.get(key) for key in ('method',*assessment.NUMERICS,'job_type')}
                 for e in self.f.entries[self.f.prior] if e['job_type']!='reserve_mesh']
        self.f.write(self.f.prior/'plan.json',{'jobs':oldjobs})
        self.f.write(self.f.prior/'approval_receipt.json',{'status':'APPROVED_BY_USER'})
        cases=self.f.read(self.f.stage/'frozen_cases.json')['cases']
        selected=[]
        for c in cases:
            method,H=c['method'],c['H_um']
            if method=='NN' or method=='direct' and H in (40,75,100) or method=='baseline' and H in (40,70,100):
                c.update(case_id=f'{method}_H{H:g}',new_case=method=='direct' or method=='NN' and H==75)
                selected.append(c)
        flat=self.f.stage/'frozen_designs/flat_cylinder_H70.json'
        self.f.write(flat,{'H_um':70,'heights_um':[0.]*4,'units':'um'})
        selected.append({'case_id':'flat_cylinder_H70','method':'flat_cylinder','H_um':70.,'new_case':False,
            'design_file':str(flat.relative_to(self.f.stage)),'design_sha256':comparison.report.sha(flat)})
        self.f.write(self.f.stage/'frozen_cases.json',{'cases':selected,'cumulative_unique_ceiling':28,
            'original_stage_cases':11,'new_stage_cases':4,'prior_cumulative_unique':24})
        receipt=self.f.read(self.f.stage/'approval_receipt.json')
        receipt['limits'].update(additional_unique=4,cumulative_unique=28)
        self.f.write(self.f.stage/'approval_receipt.json',receipt)
        self.f.write(self.f.stage/'plan.json',{'status':'ONLY_PHASE_A_EXECUTABLE','phase_A':[]})
        self.f.write(self.f.stage/'source_manifest.json',{'scripts/run_height_family_continuation_v3.py':
            comparison.report.sha(ROOT/'scripts/run_height_family_continuation_v3.py')})
        self.seal_preparation();self.flush()
    def seal_preparation(self):
        names=['config.json','approval_receipt.json','prior_stage.json','frozen_cases.json','plan.json','source_manifest.json']
        self.f.write(self.f.stage/'preparation_manifest.json',{name:comparison.report.sha(self.f.stage/name) for name in names})
    def flush(self):self.f.write(self.f.stage/'ledger.json',{'entries':self.f.entries[self.f.stage],'elapsed_s':.2,**self.metadata})
    def evidence(self):
        self.flush()
        return comparison.report.collect(self.f.stage,heights=assessment.COLLECT_HEIGHTS,
            methods=('NN','direct','baseline'),include_variants=True)
    def assess(self,now=None):return assessment.assess_evidence(self.f.stage,self.evidence(),now=now or self.now)
    def pair(self,H,res,kind='refinement',changed=None):
        ref=self.f.reference(self.f.stage,H,res);device=self.f.device(self.f.stage,'NN',H,res,ref)
        entries=[ref[1],device]
        for entry in entries:
            entry.update(job_type=kind,phase='C',started_utc=(self.now-timedelta(seconds=30)).isoformat())
            if entry['method']=='NN':entry['case_id']=f'NN_H{H:g}'
            if changed:
                rp=self.f.stage/entry['record_file'];raw=self.f.read(rp)
                for key,value in changed.items():entry[key]=value;raw['settings'][key]=value
                self.f.write(rp,raw);entry['record_sha256']=comparison.report.sha(rp)
                if entry['method']=='NN':
                    ap=self.f.stage/entry['analysis_file'];a=self.f.read(ap);a['settings']=raw['settings'];self.f.write(ap,a)
                    entry['analysis_sha256']=comparison.report.sha(ap)
        return entries
    def authorize(self,entries,mode='repair_only',adaptive=False):
        count=len(self.metadata.get('subsequent_phase_runs',[]));sub=self.f.stage/f'subsequent_phase_selections/fixture{count}'
        sub.mkdir(parents=True);driver=sub/'run_height_family_subsequent_phases_v3.py';driver.write_text('# fixture, never executed\n')
        self.flush();snapshot=self.f.read(self.f.stage/'ledger.json')
        selected_ids={entry['id'] for entry in entries}
        snapshot['entries']=[row for row in snapshot['entries'] if row['id'] not in selected_ids]
        prior_bytes=(json.dumps(snapshot,indent=2)+'\n').encode();(sub/'prior_ledger.json').write_bytes(prior_bytes)
        text='Pure fixture prospective decision'
        jobs=[{key:e.get(key) for key in ('method',*assessment.NUMERICS,'job_type','phase')} for e in entries]
        receipt={'status':'PROSPECTIVE_ADAPTIVE_SELECTION_BEFORE_NEW_DATA' if adaptive else 'PROSPECTIVE_MODE_SELECTION_BEFORE_SUBSEQUENT_DATA',
            'created_utc':(self.now-timedelta(seconds=40)).isoformat(),'selected_mode':mode,
            'decision_origin':'EXPLICIT_ROOT_SELECTION_REASON' if mode in ('repair_only','common_requested_H75') else 'EXPLICIT_USER_REPLY',
            'decision_text_exact':text,'decision_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'driver_sha256':comparison.report.sha(driver),'approval_sha256':comparison.report.sha(self.f.stage/'approval_receipt.json'),
            'frozen_stage_runner_sha256':comparison.report.sha(ROOT/'scripts/run_height_family_continuation_v3.py'),
            'preparation_manifest_sha256':comparison.report.sha(self.f.stage/'preparation_manifest.json'),
            'frozen_plan_sha256':comparison.report.sha(self.f.stage/'plan.json'),
            'prior_ledger_snapshot_sha256':hashlib.sha256(prior_bytes).hexdigest(),'selected_jobs':jobs,
            'unchanged_deadline':True,'unchanged_thresholds':True,'physics_source_changes':False,
            'adaptive_launch_authorized_by_this_receipt':adaptive}
        if mode=='common_requested_H75':
            scope='COMMON_REQUESTED_FROZEN_NN75_FORWARD_ONLY_PRIORITY_AND_DIRECT_MEANING_PENDING'
            receipt.update(execution_scope=scope,priority_selected=False,direct_calculation_meaning_selected=False,
                separately_optimized_direct_launched=False,original_control_deferral_selected=False,
                no_original_controls_deferred_by_this_selection=True,
                common75_budget_effect={'all_common_projected_attempts':14,'comparison_remaining':4,'comparison_projected_total':18,
                    'comparison_adaptive_remaining':2,'original_controls_remaining':6,'original_projected_total':20,'original_adaptive_remaining':0})
        path=self.f.write(sub/'selection_receipt.json',receipt);digest=comparison.report.sha(path)
        linked={'mode':mode,'receipt_file':str(path.relative_to(self.f.stage)),'receipt_sha256':digest,'driver_sha256':receipt['driver_sha256']}
        self.metadata.setdefault('subsequent_phase_runs',[]).append(linked);self.metadata['subsequent_phase_selection']=linked
        for entry in entries:
            entry.update(selection_receipt_file=linked['receipt_file'],selection_receipt_sha256=digest)
            if mode=='common_requested_H75':entry.update(selection_mode=mode,selection_scope=scope)
        return path
    def common75(self):
        entries=self.pair(75,24,kind='nominal')
        for entry in entries:entry['phase']='COMMON_REQUESTED_H75'
        plan=self.f.read(self.f.stage/'plan.json')
        plan['prospective_phase_B']=[{key:e.get(key) for key in ('method',*assessment.NUMERICS,'job_type','phase')} for e in entries]
        self.f.write(self.f.stage/'plan.json',plan);self.seal_preparation()
        path=self.authorize(entries,mode='common_requested_H75')
        return entries,path
    def boundary(self,kind,H,changed):
        entries=self.pair(H,16,kind=kind,changed=changed);self.authorize(entries,mode='original_first');return entries


class TestFullFamily(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def row(self,result,H):return next(r for r in result['NN_heights'] if r['H_um']==H)
    def test_seven_heights_and_separate_missing_boundaries(self):
        r=self.f.assess();self.assertEqual(len(r['NN_heights']),7)
        self.assertEqual(self.row(r,40)['selected_numerics']['resolutions'],[24.,32.])
        self.assertEqual(self.row(r,40)['selected_numerics']['status'],'PASS')
        self.assertEqual(r['full_family_acceptance']['status'],'UNRESOLVED')
        self.assertTrue(all(x['status']=='UNRESOLVED' for x in r['boundary_duration_checks']))
    def test_new32_without_selection_cannot_contribute_acceptance(self):
        self.f.pair(70,32)
        r=self.f.assess();row=self.row(r,70)
        self.assertEqual(row['latest_resolution'],32)
        self.assertEqual(row['selected_numerics']['status'],'UNRESOLVED')
        self.assertEqual(r['protocol']['status'],'UNRESOLVED')
    def test_exact_prospective_refinement32_and_old_failed_pair_preserved(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        old=next(e for e in self.f.f.entries[self.f.f.prior] if e['method']=='NN' and e['H_um']==70 and e['resolution']==16)
        ap=self.f.f.prior/old['analysis_file'];a=self.f.f.read(ap);a['normalized_curvature']=-.04
        self.f.f.write(ap,a);old['analysis_sha256']=comparison.report.sha(ap)
        self.f.f.flush(self.f.f.prior);self.f.f.seal_prior()
        receipt=self.f.f.read(self.f.f.stage/'approval_receipt.json');receipt['limits'].update(additional_unique=4,cumulative_unique=28)
        self.f.f.write(self.f.f.stage/'approval_receipt.json',receipt)
        # Source fixture resealing precedes a new prospective receipt.
        self.f.seal_preparation();entries=self.f.f.entries[self.f.f.stage];self.f.metadata.pop('subsequent_phase_runs',None);self.f.metadata.pop('subsequent_phase_selection',None)
        for e in entries:e.pop('selection_receipt_file',None);e.pop('selection_receipt_sha256',None)
        # Use a fresh fixture receipt directory index.
        self.f.metadata['subsequent_phase_runs']=[{'receipt_file':'nonexistent','receipt_sha256':'missing'}]
        self.f.authorize(entries)
        r=self.f.assess();row=self.row(r,70)
        self.assertEqual(row['selected_numerics']['resolutions'],[24.,32.]);self.assertEqual(row['selected_numerics']['status'],'PASS')
        self.assertTrue(any(p['resolutions']==[16.,24.] and p['status']=='FAIL' for p in row['all_prior_pairs']))
    def test_different_duration_branches_do_not_form_mesh_pair(self):
        entries=self.f.pair(70,32,changed={'duration':364.5});self.f.authorize(entries)
        row=self.row(self.f.assess(),70)
        self.assertEqual(row['latest_resolution'],32);self.assertEqual(row['selected_numerics']['status'],'UNRESOLVED')
        self.assertNotIn('resolutions',row['selected_numerics'])
    def test_finest_completed_native_failure_does_not_select_older_pass(self):
        entries=self.f.pair(40,40);self.f.authorize(entries,mode='adaptive',adaptive=True)
        device=entries[1];self.f.f.modify_analysis(device,lambda a:a['numerical_checks'].update(power_closure=False))
        row=self.row(self.f.assess(),40)
        self.assertEqual(row['latest_resolution'],40);self.assertEqual(row['selected_numerics']['resolutions'],[32.,40.])
        self.assertNotEqual(row['selected_numerics']['status'],'PASS')
    def test_changed_two_boundary_controls_fails_isolation(self):
        self.f.boundary('padding',70,{'padding_um':15.,'pml_um':3.})
        r=self.f.assess();row=next(x for x in r['boundary_duration_checks'] if x['job_type']=='padding')
        self.assertFalse(row['checks']['only_intended_numerical_setting_changed']);self.assertNotEqual(row['status'],'PASS')
        self.assertIn('output/control-aperture',row['variation_scope'])
        self.assertIs(row['fixed_aperture_boundary_only_established'],False)
        self.assertIn('saved E/H',row['fixed_aperture_followup'])
        self.assertIn('no native35um flux monitor',row['fixed_aperture_followup'])
    def test_idle_gaps_count_against_fixed48h_and_deadline_cannot_reset(self):
        future=self.f.now+timedelta(seconds=172801)
        r=self.f.assess(now=future);self.assertEqual(r['budget']['status'],'FAIL')
        self.assertFalse(r['budget']['checks']['wall_clock_including_idle_gaps'])
        self.f.metadata['absolute_deadline_utc']=(future+timedelta(seconds=172800)).isoformat()
        self.assertFalse(self.f.assess()['budget']['checks']['fixed_absolute_deadline'])
    def test_references_failures_and_retries_all_count20(self):
        for _ in range(21):
            _,entry=self.f.f.reference(self.f.f.stage,100,24);entry.update(status='failed',phase='A')
        r=self.f.assess();self.assertEqual(r['budget']['additional_attempts'],21)
        self.assertEqual(r['budget']['cumulative_attempts'],66);self.assertEqual(r['budget']['status'],'FAIL')
    def test_new_case_reserved_vs_charged_distinct(self):
        r=self.f.assess();self.assertEqual(r['budget']['reserved_additional_unique'],4)
        self.assertEqual(r['budget']['charged_new_unique'],0);self.assertEqual(r['budget']['actual_cumulative_unique'],24)
    def test_h75_comparison_is_outside_seven_height_aggregate(self):
        r=self.f.assess();self.assertNotIn(75.,[row['H_um'] for row in r['NN_heights']])
        self.assertIn('Validation-observed',r['requested_NN_direct_comparisons']['H75_note'])
        self.assertIs(r['requested_NN_direct_comparisons']['required_for_full_family_acceptance'],False)
        self.assertIn('non-neural optimizer',r['requested_NN_direct_comparisons']['scope_clarification'])
        self.assertIs(r['requested_NN_direct_comparisons']['same_NN_forward_evidence_generated_here'],False)
        self.assertFalse(r['NN_advantage_claimed'])
    def test_all_declared_nn_and_boundary_checks_can_pass_without_comparison_claim(self):
        self.f.boundary('pml',40,{'pml_um':3.})
        self.f.boundary('pml',100,{'pml_um':3.})
        self.f.boundary('padding',70,{'padding_um':15.})
        self.f.boundary('duration',100,{'duration':502.5})
        r=self.f.assess();self.assertEqual(r['full_family_acceptance']['status'],'PASS')
        self.assertEqual(r['protocol']['status'],'PASS')
        self.assertEqual(r['requested_NN_direct_comparisons']['status'],'UNRESOLVED')
        self.assertFalse(r['NN_advantage_claimed'])
    def test_receipt_snapshot_after_attempt_is_not_prospective(self):
        entries=self.f.pair(70,32);path=self.f.authorize(entries)
        receipt=self.f.f.read(path);snapshot=self.f.f.read(path.parent/'prior_ledger.json')
        snapshot['entries'].append(entries[0]);self.f.f.write(path.parent/'prior_ledger.json',snapshot)
        receipt['prior_ledger_snapshot_sha256']=comparison.report.sha(path.parent/'prior_ledger.json')
        self.f.f.write(path,receipt);digest=comparison.report.sha(path)
        linked=self.f.metadata['subsequent_phase_selection'];linked['receipt_sha256']=digest
        for entry in entries:entry['selection_receipt_sha256']=digest
        r=self.f.assess();self.assertNotEqual(r['protocol']['status'],'PASS')
        self.assertTrue(any('attempt_absent_before_selection' in reason for reason in r['protocol']['missing_evidence']))
    def test_missing_native_check_remains_unresolved(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        self.f.f.modify_analysis(entries[1],lambda a:a['numerical_checks'].pop('native_quadrature'))
        row=self.row(self.f.assess(),70)
        self.assertEqual(row['native_reference']['status'],'UNRESOLVED')
        self.assertEqual(row['selected_numerics']['status'],'UNRESOLVED')
    def test_id_reset_and_unregistered_device_are_protocol_budget_failures(self):
        self.f.f.entries[self.f.f.stage].append({'id':1,'method':'direct','H_um':42.5,'resolution':24.,'status':'failed',
            'job_type':'comparison','design_sha256':'b'*64,'case_id':'unlisted'})
        budget=self.f.assess()['budget'];self.assertEqual(budget['status'],'FAIL')
        self.assertFalse(budget['checks']['new_IDs_continue46']);self.assertFalse(budget['checks']['only_listed_device_cases'])
        self.assertIsNone(budget['actual_cumulative_unique'])
    def test_finer_unfinished_attempt_is_visible_and_prevents_final_pass(self):
        entries=self.f.pair(70,32);entries[1]['status']='started';self.f.authorize(entries)
        row=self.row(self.f.assess(),70)
        self.assertEqual(row['selected_numerics']['resolutions'],[16.,24.])
        self.assertEqual(row['selected_numerics']['status'],'PASS')
        self.assertEqual(row['status'],'UNRESOLVED');self.assertEqual(len(row['unfinished_finer_attempts']),1)
    def test_matched_comparison_requires_native_validity_and_charges_four_cases(self):
        stage=self.f.f.stage;old=self.f.f.prior
        direct40=self.f.f.device(stage,'direct',40,24,self.f.f.reference40)
        ref75=self.f.f.reference(stage,75,24)
        nn75=self.f.f.device(stage,'NN',75,24,ref75)
        direct75=self.f.f.device(stage,'direct',75,24,ref75)
        ref100=next(e for e in self.f.f.entries[old] if e['method']=='reference' and e['H_um']==100 and e['resolution']==24)
        direct100=self.f.f.device(stage,'direct',100,24,(old,ref100))
        entries=self.f.f.entries[stage]
        for entry in entries:
            entry.update(phase='B',started_utc=(self.f.now-timedelta(seconds=30)).isoformat())
            if entry['method']!='reference':entry['case_id']=f"{entry['method']}_H{entry['H_um']:g}"
        self.f.authorize(entries,mode='comparison_first')
        r=self.f.assess();self.assertEqual(r['requested_NN_direct_comparisons']['status'],'PASS')
        self.assertEqual(r['budget']['charged_new_unique'],4);self.assertEqual(r['budget']['actual_cumulative_unique'],28)
        self.f.f.modify_analysis(direct40,lambda a:a['numerical_checks'].update(power_closure=False))
        r=self.f.assess();self.assertEqual(r['requested_NN_direct_comparisons']['status'],'FAIL')
        self.assertIn('both_native_references_for_each_pair',r['requested_NN_direct_comparisons']['failed_checks'])
    def test_unverified_new_same_finest_record_cannot_hide_behind_old_pass(self):
        entries=self.f.pair(40,32);self.f.authorize(entries,mode='adaptive',adaptive=True)
        entries[1]['analysis_sha256']='b'*64
        row=self.row(self.f.assess(),40)
        self.assertEqual(row['selected_numerics']['resolutions'],[24.,32.])
        self.assertEqual(row['selected_numerics']['status'],'PASS')
        self.assertEqual(row['status'],'UNRESOLVED');self.assertEqual(len(row['unfinished_finer_attempts']),1)
    def test_missing_gated_observables_cannot_hide_behind_true_flags(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        entry=entries[1];path=self.f.f.stage/entry['analysis_file'];original=self.f.f.read(path)
        for key in ('angular_second_moment_ratio','conservative_T','declared_overlap','max_sidelobe_fraction'):
            with self.subTest(missing_scalar=key):
                saved=json.loads(json.dumps(original));saved.pop(key);self.f.f.write(path,saved)
                entry['analysis_sha256']=comparison.report.sha(path)
                row=self.row(self.f.assess(),70)
                self.assertEqual(row['nominal_optical']['status'],'UNRESOLVED')
                self.assertNotEqual(row['status'],'PASS')
        saved=json.loads(json.dumps(original));saved['angular_second_moment_ratio']=1.5
        self.f.f.write(path,saved);entry['analysis_sha256']=comparison.report.sha(path)
        nominal=self.row(self.f.assess(),70)['nominal_optical']
        self.assertEqual(nominal['status'],'FAIL');self.assertFalse(nominal['recomputed_flags']['angular_second_moment'])
    def test_moving_start_and_deadline_together_cannot_erase_idle_time(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        self.f.metadata['first_started_utc']=(assessment.utc(self.f.metadata['first_started_utc'])+timedelta(seconds=60)).isoformat()
        self.f.metadata['absolute_deadline_utc']=(assessment.utc(self.f.metadata['absolute_deadline_utc'])+timedelta(seconds=60)).isoformat()
        budget=self.f.assess()['budget'];self.assertEqual(budget['status'],'FAIL')
        self.assertFalse(budget['checks']['first_start_matches_immutable_snapshots'])
        self.assertFalse(budget['checks']['deadline_matches_immutable_snapshots'])
    def test_recorded_consumed_time_is_a_lower_bound(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        self.f.flush();ledger=self.f.f.read(self.f.f.stage/'ledger.json');ledger['elapsed_s']=172801.
        self.f.f.write(self.f.f.stage/'ledger.json',ledger)
        evidence=comparison.report.collect(self.f.f.stage,heights=assessment.COLLECT_HEIGHTS,
            methods=('NN','direct','baseline'),include_variants=True)
        budget=assessment.assess_evidence(self.f.f.stage,evidence,now=self.f.now)['budget']
        self.assertEqual(budget['status'],'FAIL');self.assertEqual(budget['absolute_elapsed_s_including_idle'],172801.)
    def test_nonfinite_saved_nominal_scalar_is_unresolved_and_json_safe(self):
        entries=self.f.pair(70,32);self.f.authorize(entries)
        self.f.f.modify_analysis(entries[1],lambda a:a.update(conservative_T=float('nan')))
        result=self.f.assess();row=self.row(result,70)
        self.assertEqual(row['nominal_optical']['status'],'UNRESOLVED')
        self.assertIsNone(row['nominal_optical']['saved_scalar_observables']['transmission'])
        json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main(verbosity=2)
