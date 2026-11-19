"""Pure saved-analysis fixtures for stage3 reporting, not optical solves."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

assessment=load('stage3_assessment_test',ROOT/'scripts/assess_height_family_stage3_v1.py')
fixtures=load('stage3_saved_record_fixture',ROOT/'tests/test_height_family_nn_direct_comparison_v2.py')


class Fixture:
    def __init__(self,folder):
        with patch.object(fixtures.report,'HEIGHTS',assessment.HEIGHTS):self.f=fixtures.Fixture(folder)
        self.devices={}
        for H in assessment.HEIGHTS:
            for resolution in (16,24,32)+( (40,) if H==70 else () ):
                existing=[e for e in self.f.entries[self.f.prior] if e['method']=='NN' and e['H_um']==H and e['resolution']==resolution]
                if existing:continue
                self.devices[H,resolution]=self.pair(H,resolution)
        for kind,H,changed in [('pml',40,{'pml_um':3.}),('pml',100,{'pml_um':3.}),
                               ('padding',70,{'padding_um':15.}),('duration',100,{'duration':502.5})]:
            self.pair(H,16,kind,changed)
    def pair(self,H,resolution,kind='refinement',changed=None):
        ref=self.f.reference(self.f.stage,H,resolution);entry=self.f.device(self.f.stage,'NN',H,resolution,ref)
        for item in (ref[1],entry):
            item['job_type']=kind
            rawpath=self.f.stage/item['record_file'];raw=self.f.read(rawpath)
            for key,value in (changed or {}).items():item[key]=value;raw['settings'][key]=value
            self.f.write(rawpath,raw);item['record_sha256']=fixtures.report.sha(rawpath)
            if item['method']=='NN':
                path=self.f.stage/item['analysis_file'];saved=self.f.read(path);saved['settings']=raw['settings']
                self.f.write(path,saved);item['analysis_sha256']=fixtures.report.sha(path)
        return entry
    def data(self):
        self.f.flush(self.f.stage)
        data=fixtures.report.collect(self.f.stage,heights=assessment.HEIGHTS,methods=('NN',),include_variants=True)
        data['original_boundary_definitions']=[{'job_type':r['job_type'],'H_um':r['H_um'],
            'variation_identity':r['identity'],'settings':r['settings']} for r in data['Meep_records']
            if r['job_type'] in assessment.VARIANTS and r['resolution']==16.]
        return data
    def assess(self,data=None,budget='PASS'):
        data=self.data() if data is None else data
        authorization={'status':'PASS','record_authorizations':{r['identity']:{'status':'PASS'} for r in data['Meep_records']}}
        return assessment.assess_data(data,{},authorization,{'status':budget},{'status':'PASS'})


class TestStage3Assessment(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)
    def row(self,result,H):return next(r for r in result['NN_heights'] if r['H_um']==H)
    def test_reporting_scope_expands_without_mutating_pinned_module_or_gate_bytecode(self):
        science=assessment.load_science();original=science.HEIGHTS
        data=self.f.data();result=assessment.scientific_rows(data,{},science)
        self.assertEqual(science.HEIGHTS,original)
        self.assertEqual([r['H_um'] for r in result['rows']],list(assessment.HEIGHTS))
        self.assertEqual(science.sha(assessment.SCIENTIFIC_SOURCE),assessment.SCIENTIFIC_SHA)
    def test_declared_last_pairs_and_old_h40_are_selected(self):
        result=self.f.assess()
        self.assertEqual(self.row(result,70)['selected_mesh']['resolutions'],[32.,40.])
        self.assertEqual(self.row(result,75)['selected_mesh']['resolutions'],[24.,32.])
        self.assertEqual(self.row(result,40)['Meep']['identity'],'maxwell_validation_v1:4')
        self.assertEqual(result['original_seven_height_family_acceptance']['status'],'PASS')
        self.assertEqual(result['all_eight_nominal_heights_acceptance']['status'],'PASS')
    def test_finest_native_failure_never_falls_back_to_older_passing_32(self):
        entry=self.f.devices[70.,40.]
        self.f.f.modify_analysis(entry,lambda a:a['numerical_checks'].update(crosscheck=False))
        row=self.row(self.f.assess(),70)
        self.assertEqual(row['Meep']['resolution'],40.)
        self.assertEqual(row['native']['status'],'FAIL')
        self.assertEqual(row['status'],'FAIL')
    def test_nonrequested_graph_height_still_gets_finite_nominal_checks(self):
        entry=self.f.devices[80.,32.]
        self.f.f.modify_analysis(entry,lambda a:a.update(conservative_T=None))
        row=self.row(self.f.assess(),80)
        self.assertEqual(row['nominal_optical']['status'],'UNRESOLVED')
        self.assertNotEqual(row['status'],'PASS')
    def test_duration_variant_cannot_replace_latest_nominal(self):
        self.f.pair(100,64,'duration',{'duration':502.5})
        row=self.row(self.f.assess(),100)
        self.assertEqual(row['Meep']['resolution'],32.)
    def test_original_coarse_control_and_new_fine_variant_are_distinct(self):
        self.f.pair(100,32,'duration',{'duration':502.5})
        rows=[r for r in self.f.assess()['boundary_duration_checks'] if r['job_type']=='duration' and r['H_um']==100.]
        self.assertEqual(len(rows),2)
        self.assertEqual([(r['resolution'],r['required_original_sampled_control']) for r in rows],[(16.,True),(32.,False)])
    def test_missing_original_registry_cannot_be_replaced_by_a_fine_pass(self):
        self.f.pair(100,32,'duration',{'duration':502.5})
        data=self.f.data();data['original_boundary_definitions']=[]
        result=self.f.assess(data)
        required=[r for r in result['boundary_duration_checks'] if r['required_original_sampled_control']]
        self.assertEqual(len(required),4);self.assertTrue(all(r['status']=='UNRESOLVED' for r in required))
        self.assertNotEqual(result['original_seven_height_family_acceptance']['status'],'PASS')
    def test_original_control_identity_alone_cannot_override_changed_settings(self):
        data=self.f.data();data['original_boundary_definitions'][0]['settings']=dict(data['original_boundary_definitions'][0]['settings'],resolution=32.)
        rows=assessment.boundary_assessments(data,{},assessment.load_science())
        self.assertFalse(rows[0]['checks']['authenticated_original_settings'])
        self.assertEqual(rows[0]['status'],'FAIL')
    def test_finer_unverified_attempt_keeps_current_curve_but_blocks_pass(self):
        data=self.f.data();current=next(r for r in data['Meep_records'] if r['H_um']==90 and r['resolution']==32)
        data['all_attempts'].append({'identity':'maxwell_validation_v3:80','method':'NN','H_um':90.,'resolution':40.,
            'job_type':'refinement',**{k:current['settings'][k] for k in ('pml_um','padding_um','duration')}})
        row=self.row(self.f.assess(data),90)
        self.assertEqual(row['Meep']['resolution'],32.)
        self.assertEqual(row['selected_mesh']['status'],'UNRESOLVED')
    def test_failure_is_visible_despite_other_missing_evidence_and_budget_is_required(self):
        self.assertEqual(assessment.aggregate({'known':'FAIL','other':'UNRESOLVED'})['status'],'FAIL')
        self.assertEqual(self.f.assess(budget='FAIL')['original_seven_height_family_acceptance']['status'],'FAIL')
    def test_all_prior_pairs_and_retained_band_q_diagnostics_remain(self):
        result=self.f.assess();pairs=[r for r in result['all_prior_mesh_pairs'] if r['H_um']==70]
        self.assertTrue(any(r['resolutions']==[16.,24.] for r in pairs))
        self.assertTrue(any(r['resolutions']==[32.,40.] for r in pairs))
        self.assertTrue(all('retained_band_angular_ratio_absolute_change' in r['comparison']['measurements'] for r in pairs))
    def test_CLI_cannot_write_unrequested_assessment_before_primary_completion(self):
        output=self.f.f.stage/'assessment/not_authorized_partial'
        with patch.object(sys,'argv',['assessment','--folder',str(self.f.f.stage),'--output',str(output)]),\
             patch('collect_height_family_stage3_v1.collect_stage3',return_value={'primary_execution':{'status':'UNRESOLVED'}}),\
             self.assertRaisesRegex(RuntimeError,'Primary is incomplete'):
            assessment.main()
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
