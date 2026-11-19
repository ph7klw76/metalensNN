"""Pure known-record fixtures for reporting; no optical imports or solves."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('height_maxwell_report',ROOT/'scripts/report_height_family_maxwell_v2.py')
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


class Fixture:
    def __init__(self,folder):
        self.folder=Path(folder);self.entries=[];self.refs={};self.counter=0
        self.config={'evaluation':{'processing_planes_um':[0,25,50,75,100]},
                     'natural_training':{'train_H_um':[40,50,60,70,80,90,100]}}
        self.proposal={'nominal_heights_um':[40,50,60,70,80,90,100],
            'device_cases':[{'method':'NN','H_um':h} for h in [40,50,60,70,80,90,100]]+
                           [{'method':'baseline','H_um':h} for h in [40,70,100]]+
                           [{'method':'flat_cylinder','H_um':70}],
            'new_unique_device_ceiling':11,'maximum_total_CPU_solves_including_references_and_retries':60,
            'maximum_total_wall_time_s':14400,'numerical_acceptance':{
                'power_closure_error_max':.05,'diameter_relative_mesh_change_max':.02,
                'eta_relative_mesh_change_max':.05,'curvature_absolute_max':.1,
                'curvature_last_pair_delta_max':.005}}
        self.fingerprint=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()
        for h in self.proposal['nominal_heights_um']:
            for resolution in [8,16,24]:self.device('NN',h,resolution)
        for method,H in [('baseline',40),('baseline',70),('baseline',100),('flat_cylinder',70)]:
            for resolution in [8,16]:self.device(method,H,resolution)
        self.device('NN',40,16,job_type='pml',pml=3)
        self.device('NN',100,16,job_type='pml',pml=3)
        self.device('NN',70,16,job_type='padding',padding=15)
        self.device('NN',100,16,job_type='duration',duration=450)
        self.write('config.json',self.config);self.write('maxwell_validation_proposal.json',self.proposal)
    def write(self,name,value):
        p=self.folder/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');return p
    def read(self,name):return json.loads((self.folder/name).read_text())
    def settings(self,H,res,pml,padding,duration):
        return {'H_um':float(H),'resolution':float(res),'pml_um':float(pml),
                'padding_um':float(padding),'duration':float(duration)}
    def reference(self,settings,job_type):
        key=tuple(settings.items())
        if key in self.refs:return self.refs[key]
        self.counter+=1;identity='ref'+str(self.counter);filename='records/'+identity+'.json'
        entry={'id':identity,'method':'reference','status':'completed','job_type':job_type,
               'record_file':filename,**settings}
        self.entries.append(entry);self.refs[key]=entry
        self.write(filename,{'settings':settings,'config_fingerprint':self.fingerprint,
                             'incident_power':1.,'power_balance_error':.001})
        return entry
    def device(self,method,H,res,job_type='nominal',pml=2,padding=10,duration=300):
        settings=self.settings(H,res,pml,padding,duration);ref=self.reference(settings,job_type)
        self.counter+=1;identity='dev'+str(self.counter);filename='records/'+identity+'.json'
        analysis_file='analyses/'+identity+'.json'
        entry={'id':identity,'method':method,'status':'completed','analysis_status':'completed',
               'job_type':job_type,'record_file':filename,'analysis_file':analysis_file,
               'reference_id':ref['id'],'design_sha256':hashlib.sha256(f'{method}:{H}'.encode()).hexdigest(),**settings}
        self.entries.append(entry)
        self.write(filename,{'settings':settings,'config_fingerprint':self.fingerprint,
                             'reference_file':'/mnt/c/example/'+Path(ref['record_file']).name,
                             'power_balance_error':.002})
        optical={key:True for key in report.OPTICAL_KEYS}
        if method=='flat_cylinder':optical['curvature']=False;optical['diameter']=False
        self.write(analysis_file,{'record':'/mnt/c/example/'+Path(filename).name,'settings':settings,
            'metrics':{'planes':[{'z_um':z,'diameter_x_um':15+z*.05,'diameter_y_um':15+z*.05,
                                 'valid_crossings':[True,True]} for z in [0,25,50,75,100]],
                       'fitted_core_divergence':{'valid':True,'axes':[{'half_angle_rad':.07},{'half_angle_rad':.071}]}},
            'conservative_T':.95,'conservative_useful_efficiency':.94,'declared_overlap':.99,
            'normalized_curvature':1. if method=='flat_cylinder' else .02,
            'declared_target_diameter_um':15.,'max_sidelobe_fraction':.01,'angular_second_moment_ratio':1.02,
            'optical_gates':optical,'numerical_checks':{key:True for key in report.NATIVE_KEYS}})
        return entry
    def find(self,method='NN',H=40,res=24,job_type='nominal'):
        return next(e for e in self.entries if e['method']==method and e['H_um']==H and e['resolution']==res and e['job_type']==job_type)
    def modify_analysis(self,entry,callback):
        a=self.read(entry['analysis_file']);callback(a);self.write(entry['analysis_file'],a)
    def summary(self):return report.summarize(self.folder,self.config,self.proposal,{'entries':self.entries,'elapsed_s':1000})


class TestHeightMaxwellReport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)
    def test_complete_known_records_pass_while_flat_optical_failure_is_diagnostic(self):
        s=self.f.summary();self.assertEqual(s['status'],'SYNTHETIC_NN_FAMILY_PASS')
        self.assertEqual(s['budget']['attempted_solves'],58)
        self.assertEqual(len(s['NN_heights']),7)
        flat=next(x for x in s['control_comparisons'] if x['method']=='flat_cylinder')
        self.assertEqual(flat['status'],'PASS');self.assertEqual(flat['nominal_optical']['status'],'FAIL')
        self.assertTrue(all(v['status']=='PASS' for v in s['boundary_duration_checks']))
        self.assertTrue(s['raw_inputs_unchanged_during_report'])
    def test_missing_analysis_never_becomes_pass(self):
        e=self.f.find();(self.f.folder/e['analysis_file']).unlink()
        s=self.f.summary();self.assertEqual(s['NN_heights'][0]['status'],'UNRESOLVED')
        self.assertNotEqual(s['status'],'SYNTHETIC_NN_FAMILY_PASS')
    def test_nonoutput_processing_plane_change_is_detected(self):
        self.f.modify_analysis(self.f.find(),lambda a:a['metrics']['planes'][-1].update(diameter_x_um=22.))
        c=self.f.summary()['NN_heights'][0]['comparison']
        self.assertFalse(c['checks']['all_processing_diameters'])
    def test_small_curvature_delta_still_fails_remaining_margin(self):
        self.f.modify_analysis(self.f.find(res=16),lambda a:a.update(normalized_curvature=.095))
        self.f.modify_analysis(self.f.find(res=24),lambda a:a.update(normalized_curvature=.099))
        c=self.f.summary()['NN_heights'][0]['comparison']
        self.assertTrue(c['checks']['curvature_last_pair_delta'])
        self.assertFalse(c['checks']['curvature_change_below_remaining_margin'])
    def test_reference_native_closure_is_required_explicitly(self):
        e=self.f.find();ref=next(x for x in self.f.entries if x['id']==e['reference_id'])
        raw=self.f.read(ref['record_file']);raw['power_balance_error']=.06;self.f.write(ref['record_file'],raw)
        s=self.f.summary();self.assertFalse(s['NN_heights'][0]['checks']['both_reference_native_closures'])
    def test_wrong_height_reference_is_rejected(self):
        self.f.find()['reference_id']=self.f.find(H=50)['reference_id']
        self.assertNotEqual(self.f.summary()['NN_heights'][0]['status'],'PASS')
    def test_every_failed_attempt_counts_toward_budget(self):
        for i in range(3):self.f.entries.append({'id':'failed'+str(i),'method':'reference','status':'failed'})
        s=self.f.summary();self.assertEqual(s['budget']['attempted_solves'],61)
        self.assertEqual(s['status'],'PROTOCOL_BUDGET_FAILURE')
    def test_mesh_comparison_rejects_different_geometry(self):
        self.f.find()['design_sha256']='b'*64
        self.assertFalse(self.f.summary()['NN_heights'][0]['checks']['same_frozen_device_geometry'])
    def test_nonfinite_metric_is_unresolved_and_json_safe(self):
        self.f.modify_analysis(self.f.find(),lambda a:a.update(normalized_curvature=float('nan')))
        s=self.f.summary();self.assertEqual(s['NN_heights'][0]['status'],'UNRESOLVED')
        json.dumps(s,allow_nan=False)
    def test_missing_native_check_not_replaced_by_success_flag(self):
        self.f.modify_analysis(self.f.find(),lambda a:a['numerical_checks'].pop('native_quadrature'))
        self.assertEqual(self.f.summary()['NN_heights'][0]['status'],'UNRESOLVED')
    def test_missing_boundary_case_is_unresolved(self):
        self.f.entries.remove(self.f.find(H=70,res=16,job_type='padding'))
        self.assertEqual(self.f.summary()['boundary_duration_checks'][2]['status'],'UNRESOLVED')
    def test_raw_record_cannot_be_an_output_path(self):
        self.f.write('ledger.json',{'entries':self.f.entries,'elapsed_s':1000})
        raw=self.f.folder/self.f.find()['record_file'];before=raw.read_bytes()
        argv=['report','--folder',str(self.f.folder),'--output-json',str(raw),'--overwrite-summary']
        with patch.object(sys,'argv',argv),self.assertRaises(ValueError):report.main()
        self.assertEqual(raw.read_bytes(),before)
    def test_missing_ledger_evidence_produces_unresolved_empty_report(self):
        s=report.summarize(self.f.folder,self.f.config,self.f.proposal,{'entries':[],'elapsed_s':0})
        self.assertNotEqual(s['status'],'SYNTHETIC_NN_FAMILY_PASS')
        self.assertEqual(s['budget']['attempted_solves'],0)
    def test_duration_extension_preserves_failed_attempt_but_cannot_isolate_mesh_change(self):
        e=self.f.find(H=100);raw=self.f.read(e['record_file']);raw['power_balance_error']=.07
        self.f.write(e['record_file'],raw)
        self.f.modify_analysis(e,lambda a:a['numerical_checks'].update(power_closure=False))
        extra=self.f.device('NN',100,24,job_type='reserve_duration',duration=450)
        s=self.f.summary();self.assertNotEqual(s['status'],'SYNTHETIC_NN_FAMILY_PASS')
        self.assertEqual(s['NN_heights'][-1]['fine_id'],extra['id'])
        self.assertEqual(s['NN_heights'][-1]['status'],'UNRESOLVED')
        self.assertFalse(s['NN_heights'][-1]['checks']['same_duration'])
        self.assertEqual(s['budget']['attempted_solves'],60)
        failed=next(x for x in s['all_attempts'] if x['id']==e['id'])
        self.assertEqual(failed['native_power_balance_error'],.07)
        self.assertEqual(failed['evidence_status'],'FAIL')
    def test_completed_approved32_uses_last24_to32_pair_and_retains_failed_prior_pair(self):
        self.f.modify_analysis(self.f.find(),lambda a:a.update(normalized_curvature=-.04))
        fine=self.f.device('NN',40,32,job_type='reserve_mesh')
        self.f.modify_analysis(fine,lambda a:a.update(normalized_curvature=-.041))
        s=self.f.summary();row=s['NN_heights'][0]
        self.assertEqual(s['status'],'SYNTHETIC_NN_FAMILY_PASS')
        self.assertEqual(row['last_pair_resolutions'],[24.,32.])
        self.assertEqual(row['fine_id'],fine['id'])
        self.assertEqual(row['finest_completed_resolution'],32.)
        history=next(x for x in row['mesh_history'] if x['last_pair_resolutions']==[16.,24.])
        self.assertEqual(history['status'],'FAIL')
        self.assertIn('curvature_last_pair_delta',history['comparison']['failed_checks'])
        self.assertIn('Retained mesh history [16.0, 24.0]: FAIL',report.human_report(s))
        self.assertEqual(s['budget']['attempted_solves'],60)
    def test_nominal_optical_gates_use_actual_finest_completed32(self):
        fine=self.f.device('NN',40,32,job_type='reserve_mesh')
        self.f.modify_analysis(fine,lambda a:a['optical_gates'].update(declared_overlap=False))
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['nominal_optical_resolution'],32.)
        self.assertEqual(row['nominal_optical_id'],fine['id'])
        self.assertFalse(row['nominal_optical']['checks']['declared_overlap'])
        self.assertNotEqual(row['status'],'PASS')
    def test_reserve32_with_different_duration_is_unresolved(self):
        self.f.device('NN',40,32,job_type='reserve_mesh',duration=450)
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['last_pair_resolutions'],[24.,32.])
        self.assertFalse(row['checks']['same_duration'])
        self.assertEqual(row['status'],'UNRESOLVED')
    def test_incomplete32_does_not_replace_last_completed24(self):
        fine=self.f.device('NN',40,32,job_type='reserve_mesh')
        fine['status']='timeout'
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['last_pair_resolutions'],[16.,24.])
        self.assertEqual(row['nominal_optical_resolution'],24.)
        self.assertEqual(row['finest_completed_resolution'],24.)
    def test_unapproved_nominal32_is_not_treated_as_approved_mesh_reserve(self):
        self.f.device('NN',40,32,job_type='nominal')
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['last_pair_resolutions'],[16.,24.])
        self.assertEqual(row['finest_completed_resolution'],24.)
    def test_nominal_gates_remain_visible_at16_when24_is_missing(self):
        self.f.entries.remove(self.f.find())
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['status'],'UNRESOLVED')
        self.assertEqual(row['nominal_optical_resolution'],16.)
        self.assertEqual(row['nominal_optical']['status'],'PASS')
    def test_changed_declared_target_is_not_a_valid_convergence_pair(self):
        self.f.modify_analysis(self.f.find(),lambda a:a.update(declared_target_diameter_um=16.))
        self.assertFalse(self.f.summary()['NN_heights'][0]['comparison']['checks']['same_frozen_declared_target'])
    def test_recorded_raw_hash_mismatch_cannot_pass(self):
        self.f.find()['record_sha256']='b'*64
        row=self.f.summary()['NN_heights'][0]
        self.assertEqual(row['status'],'FAIL')
        self.assertIn('fine_native_evidence',row['failed_checks'])
    def test_negative_native_closure_is_invalid(self):
        e=self.f.find();raw=self.f.read(e['record_file']);raw['power_balance_error']=-1
        self.f.write(e['record_file'],raw)
        self.assertEqual(self.f.summary()['NN_heights'][0]['status'],'FAIL')
    def test_boundary_change_must_hold_other_controls_fixed(self):
        e=next(e for e in self.f.entries if e.get('job_type')=='padding' and e['method']=='NN')
        e['pml_um']=3
        row=self.f.summary()['boundary_duration_checks'][2]
        self.assertFalse(row['checks']['other_controlled_settings_unchanged'])
    def test_reference_filename_is_required(self):
        e=self.f.find();raw=self.f.read(e['record_file']);raw.pop('reference_file')
        self.f.write(e['record_file'],raw)
        self.assertEqual(self.f.summary()['NN_heights'][0]['status'],'UNRESOLVED')


if __name__=='__main__':unittest.main(verbosity=2)
