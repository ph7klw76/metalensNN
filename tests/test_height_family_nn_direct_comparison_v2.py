"""Known-record schema/figure fixtures; no training, Meep or live ledger use."""
import hashlib
import importlib.util
import json
import math
from datetime import datetime,timedelta,timezone
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('height_family_comparison',ROOT/'scripts/compare_height_family_nn_direct_v2.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class Fixture:
    def __init__(self,folder):
        self.base=Path(folder);self.prior=self.base/'maxwell_validation_v1';self.stage=self.base/'maxwell_validation_v2'
        self.prior.mkdir();self.stage.mkdir();self.entries={self.prior:[],self.stage:[]};self.counter=0
        self.config={'wavelength_um':1.55,'mfd_um':10.4,'annuli':4,'cylinder_diameter_um':50.,'height_max_um':3.,
            'materials':{'substrate':{'n':1.45,'kappa':0.},'polymer':{'n':1.5263,'kappa':0.},'ambient':{'n':1.,'kappa':0.}},
            'evaluation':{'processing_planes_um':[0.,25.,100.]},
            'natural_training':{'train_H_um':[40.,100.],'validation_H_um':[75.],
                'investigation_gates':{'transmission_min':.9,'declared_Gaussian_overlap_min':.98,'useful_efficiency_min':.9,
                    'output_diameter_relative_tolerance':.1,'max_sidelobe_fraction':.02,
                    'max_abs_normalized_curvature':.1,'max_angular_second_moment_ratio':1.1}}}
        self.signature='known-frozen-signature';selection={'NN':{'seed':47},'direct':{'seed':11}}
        records=[];cases=[]
        for H in report.HEIGHTS:
            for i,method in enumerate(('NN','direct','baseline')):
                values=np.asarray([.5+i*.1,.45+i*.1,.4+i*.1,.35+i*.1],dtype='<f4')
                D=12+(H-40)/60*4
                design={'units':'um','H_um':H,'heights_um':values.tolist(),'source_signature':self.signature,
                        'materials':self.config['materials'],'mfd_um':10.4,'target_diameter_um':D,'cylinder_diameter_um':50.}
                p=self.write(self.base/f'designs/H{H:g}/{method}.json',design)
                frozen=self.stage/f'frozen_designs/{method}_H{H:g}.json';frozen.parent.mkdir(exist_ok=True)
                shutil.copyfile(p,frozen)
                cases.append({'method':method,'H_um':H,'design_file':str(frozen.relative_to(self.stage)),
                              'design_sha256':report.sha(frozen)})
                records.append({'method':method,'H_um':H,'height_domain':'validation' if H==75 else 'training',
                    'profile_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'declared_target_diameter_um':D,
                    'fitted_core_divergence':{'valid':True,'axes':[{'half_angle_rad':.07},{'half_angle_rad':.071}]},
                    'metrics':{'declared_useful_efficiency':.95+i*.00001,'normalized_curvature':.02,
                        'angular_second_moment_ratio':1.02,'valid_crossings':[[True,True]]*3,
                        'widths_um':[[D,D],[D+1,D+1.1],[D+5,D+5.1]]}})
        self.write(self.base/'config.json',self.config);self.write(self.prior/'config.json',self.config)
        self.write(self.stage/'config.json',self.config)
        self.write(self.base/'selection_before_heldout.json',{'signature':self.signature,'selection':selection})
        self.write(self.base/'training_result.json',{'selection':selection,'records':records})
        self.write(self.base/'delivery_manifest.json',{'files':{name:{'sha256':report.sha(self.base/name)}
            for name in ('config.json','training_result.json','selection_before_heldout.json')}})
        self.write(self.prior/'source_manifest.json',{'fixture':'saved-source-hash'})
        self.write(self.prior/'maxwell_validation_proposal.json',{'numerical_acceptance':{}})
        self.write(self.stage/'frozen_cases.json',{'cases':cases})
        ref=self.reference(self.prior,40,24);self.device(self.prior,'NN',40,24,ref)
        ref32=self.reference(self.prior,40,32);self.device(self.prior,'NN',40,32,ref32)
        self.reference40=ref;self.reference40_32=ref32
        self.flush(self.prior);self.flush(self.stage)
        self.seal_prior()
    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');return path
    def read(self,path):return json.loads(Path(path).read_text())
    def settings(self,H,res):return {'H_um':float(H),'resolution':float(res),'pml_um':2.,'padding_um':10.,'duration':float(math.ceil(2*1.5263*(H+3)+20))}
    def reference(self,stage,H,res):
        self.counter+=1;identity=self.counter;settings=self.settings(H,res)
        rp=self.write(stage/f'records/{identity}_reference.json',{'settings':settings,'config_fingerprint':report.fingerprint(self.config),
            'incident_power':1.,'power_balance_error':.001})
        entry={'id':identity,'method':'reference','status':'completed','job_type':'nominal',
               'record_file':str(rp.relative_to(stage)),'record_sha256':report.sha(rp),**settings}
        self.entries[stage].append(entry);return stage,entry
    def device(self,stage,method,H,res,reference,status='completed'):
        self.counter+=1;identity=self.counter;refstage,ref=reference;settings=self.settings(H,res)
        design=self.read(self.base/f'designs/H{H:g}/{method}.json');D=design['target_diameter_um']
        rp=self.write(stage/f'records/{identity}_{method}.json',{'settings':settings,'config_fingerprint':report.fingerprint(self.config),
            'reference_file':str((refstage/ref['record_file']).resolve()),'power_balance_error':.002})
        gates={key:True for key in report.OPTICAL_KEYS};native={key:True for key in report.NATIVE_KEYS}
        ap=self.write(stage/f'analyses/{identity}_{method}.json',{'record':str(rp.resolve()),'settings':settings,
            'conservative_T':.96,'declared_overlap':.99,'max_sidelobe_fraction':.01,
            'declared_target_diameter_um':D,'conservative_useful_efficiency':.95+(0.00001 if method=='direct' else 0),
            'normalized_curvature':.02,'angular_second_moment_ratio':1.02,'optical_gates':gates,'numerical_checks':native,
            'metrics':{'target_widths_um':[[D,D],[D+1,D+1],[D+5,D+5]],
                'fitted_core_divergence':{'valid':True,'axes':[{'half_angle_rad':.07},{'half_angle_rad':.071}]},
                'planes':[{'z_um':z,'diameter_x_um':D+z*.05,'diameter_y_um':D+z*.05,'valid_crossings':[True,True]} for z in [0.,25.,100.]]}})
        entry={'id':identity,'method':method,'status':status,'analysis_status':'completed','job_type':'comparison',
            'record_file':str(rp.relative_to(stage)),'record_sha256':report.sha(rp),'analysis_file':str(ap.relative_to(stage)),
            'analysis_sha256':report.sha(ap),'reference_id':ref['id'],
            'design_sha256':report.sha(self.base/f'designs/H{H:g}/{method}.json'),**settings}
        self.entries[stage].append(entry);return entry
    def flush(self,stage):self.write(stage/'ledger.json',{'entries':self.entries[stage],'elapsed_s':0.})
    def seal_prior(self):
        prior_count=len(self.entries[self.prior])
        self.write(self.stage/'prior_stage.json',{'folder':str(self.prior.resolve()),'ledger_file':str((self.prior/'ledger.json').resolve()),
            'ledger_sha256':report.sha(self.prior/'ledger.json'),'config_sha256':report.sha(self.prior/'config.json'),
            'source_manifest_sha256':report.sha(self.prior/'source_manifest.json')})
        self.write(self.stage/'approval_receipt.json',{'status':'APPROVED_BY_USER','limits':{'prior_attempts':prior_count,
            'additional_CPU_solves':20,'additional_wall_s':172800,'cumulative_CPU_solves':prior_count+20}})
    def collect(self):self.flush(self.stage);return report.collect(self.stage)
    def modify_analysis(self,entry,callback):
        p=self.stage/entry['analysis_file'];a=self.read(p);callback(a);self.write(p,a);entry['analysis_sha256']=report.sha(p)


class TestComparison(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=Fixture(self.temp.name)
    def test_proxy_only_keeps_direct_and_baseline_distinct(self):
        data=self.f.collect();self.assertEqual(len(data['proxy_records']),9)
        self.assertEqual(len(data['matched_Meep_pairs']),0)
        self.assertEqual(len(data['missing_comparisons']),4)
        self.assertFalse(data['NN_advantage_claimed'])
        self.assertEqual(next(r['height_domain'] for r in data['proxy_records'] if r['method']=='NN' and r['H_um']==75),'validation')
    def test_exact_matched24_and_separate32(self):
        self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        data=self.f.collect();self.assertEqual([(r['H_um'],r['resolution']) for r in data['matched_Meep_pairs']],[(40.,24.)])
        self.f.device(self.f.stage,'direct',40,32,self.f.reference40_32)
        data=self.f.collect();self.assertEqual(len(data['matched_Meep_pairs']),2)
        self.assertEqual(len(data['Meep_records']),4)
    def test_unmatched_meshes_never_form_a_pair(self):
        self.f.entries[self.f.prior]=[e for e in self.f.entries[self.f.prior] if not(e['method']=='NN' and e['resolution']==24)]
        self.f.flush(self.f.prior);self.f.seal_prior()
        self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        data=self.f.collect();self.assertFalse(data['matched_Meep_pairs'])
        self.assertEqual({r['resolution'] for r in data['Meep_records']},{24.,32.})
    def test_missing_direct_is_blank_not_baseline_substitution(self):
        (self.f.base/'designs/H75/direct.json').unlink()
        data=self.f.collect();self.assertNotIn(('direct',75.),[(r['method'],r['H_um']) for r in data['proxy_records']])
        self.assertIn(('baseline',75.),[(r['method'],r['H_um']) for r in data['proxy_records']])
    def test_changed_profile_and_source_signature_are_rejected(self):
        p=self.f.base/'designs/H75/direct.json';d=self.f.read(p);d['heights_um'][0]+=.05;self.f.write(p,d)
        p2=self.f.base/'designs/H100/NN.json';d2=self.f.read(p2);d2['source_signature']='changed';self.f.write(p2,d2)
        data=self.f.collect();bad={(x['method'],x['H_um']) for x in data['missing_evidence'] if x['scope']=='proxy'}
        self.assertIn(('direct',75.),bad);self.assertIn(('NN',100.),bad)
    def test_invalid_crossings_and_core_fit_remain_gaps(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        def change(a):
            a['metrics']['planes'][0]['valid_crossings']=[False,False]
            a['metrics']['fitted_core_divergence']['valid']=False
        self.f.modify_analysis(e,change);data=self.f.collect()
        row=next(r for r in data['Meep_records'] if r['method']=='direct')
        self.assertIsNone(row['planes'][0]['diameter_x_um']);self.assertEqual(row['core_half_angles_deg'],[None,None])
    def test_prior_ledger_hash_mismatch_stops_collection(self):
        self.f.write(self.f.prior/'ledger.json',{'entries':[]})
        with self.assertRaisesRegex(ValueError,'Prior-stage'):self.f.collect()
    def test_failed_attempt_counted_and_provenance_mismatch_not_plotted(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40,status='failed')
        other=self.f.device(self.f.stage,'direct',40,32,self.f.reference40_32);other['analysis_sha256']='b'*64
        data=self.f.collect();self.assertEqual(data['budget']['additional_attempts'],2)
        self.assertFalse(any(r['method']=='direct' for r in data['Meep_records']))
        self.assertEqual(len([r for r in data['all_attempts'] if r['identity'].startswith('maxwell_validation_v2:')]),2)
    def test_reused_numeric_ids_across_stages_are_namespaced(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40);e['id']=2
        data=self.f.collect();identities=[r['identity'] for r in data['Meep_records']]
        self.assertIn('maxwell_validation_v1:2',identities);self.assertIn('maxwell_validation_v2:2',identities)
        self.assertEqual(len(data['matched_Meep_pairs']),1)
    def test_target_change_and_duration_variant_do_not_create_comparison(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        self.f.modify_analysis(e,lambda a:a.update(declared_target_diameter_um=17.))
        variant=self.f.device(self.f.stage,'direct',40,32,self.f.reference40_32);variant['job_type']='reserve_duration'
        data=self.f.collect();self.assertFalse(data['matched_Meep_pairs'])
    def test_missing_processing_plane_is_an_explicit_gap(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        self.f.modify_analysis(e,lambda a:a['metrics']['planes'].pop(1))
        row=next(r for r in self.f.collect()['Meep_records'] if r['method']=='direct')
        self.assertEqual(len(row['planes']),3);self.assertEqual(row['planes'][1]['z_um'],25.)
        self.assertIsNone(row['planes'][1]['diameter_x_um'])
    def test_reference_stage_declaration_must_match_actual_reference(self):
        e=self.f.device(self.f.stage,'direct',40,24,self.f.reference40);e['reference_stage']='stage2'
        data=self.f.collect();self.assertFalse(data['matched_Meep_pairs'])
        self.assertTrue(any('reference_stage' in row['reason'] for row in data['missing_evidence']))
    def test_missing_authorized_budget_is_unresolved_not_inferred(self):
        receipt=self.f.read(self.f.stage/'approval_receipt.json');receipt['limits'].pop('additional_CPU_solves')
        self.f.write(self.f.stage/'approval_receipt.json',receipt)
        self.assertEqual(self.f.collect()['budget']['status'],'UNRESOLVED')
    def test_running_elapsed_is_not_frozen_at_last_ledger_write(self):
        ledger={'entries':[],'elapsed_s':.2,'status':'RUNNING_APPROVED_PHASE_A',
                'first_started_utc':(datetime.now(timezone.utc)-timedelta(seconds=100)).isoformat()}
        self.f.write(self.f.stage/'ledger.json',ledger)
        budget=report.collect(self.f.stage)['budget']
        self.assertGreaterEqual(budget['additional_elapsed_s'],100.)
        self.assertEqual(budget['recorded_elapsed_s'],.2)
        self.assertEqual(budget['status'],'PASS')
    def test_pending_or_stopped_elapsed_includes_idle_and_cannot_pass_after_deadline(self):
        for status in ('PHASE_A_EXECUTED_REQUIRES_ASSESSMENT_AND_PHASE_B_SELECTION',
                       'COMMON_REPAIR_EXECUTED_PRIORITY_SELECTION_PENDING',
                       'STOPPED_REQUIRES_DIAGNOSIS'):
            with self.subTest(status=status):
                ledger={'entries':[],'elapsed_s':100.,'status':status,
                        'first_started_utc':(datetime.now(timezone.utc)-timedelta(seconds=172801)).isoformat()}
                self.f.write(self.f.stage/'ledger.json',ledger)
                budget=report.collect(self.f.stage)['budget']
                self.assertGreaterEqual(budget['additional_elapsed_s'],172801.)
                self.assertEqual(budget['recorded_elapsed_s'],100.)
                self.assertFalse(budget['checks']['additional_wall_cap'])
                self.assertEqual(budget['status'],'FAIL')
    def test_pending_clock_cannot_regress_below_recorded_elapsed(self):
        ledger={'entries':[],'elapsed_s':500.,'status':'COMMON_REPAIR_EXECUTED_PRIORITY_SELECTION_PENDING',
                'first_started_utc':(datetime.now(timezone.utc)-timedelta(seconds=100)).isoformat()}
        self.f.write(self.f.stage/'ledger.json',ledger)
        self.assertEqual(report.collect(self.f.stage)['budget']['additional_elapsed_s'],500.)
    def test_stopped_stage_without_absolute_clock_is_unresolved(self):
        self.f.write(self.f.stage/'ledger.json',{'entries':[],'elapsed_s':.2,'status':'STOPPED_REQUIRES_DIAGNOSIS'})
        budget=report.collect(self.f.stage)['budget']
        self.assertIsNone(budget['additional_elapsed_s'])
        self.assertEqual(budget['status'],'UNRESOLVED')
    def test_no_stage2_creation_before_prepare(self):
        missing=self.f.base/'not_prepared';output=missing/'comparison/v1'
        with patch.object(sys,'argv',['compare','--folder',str(missing),'--output',str(output)]),self.assertRaises(RuntimeError):report.main()
        self.assertFalse(missing.exists())
    def test_render_and_fresh_output_only(self):
        self.f.device(self.f.stage,'direct',40,24,self.f.reference40)
        self.f.device(self.f.stage,'direct',40,32,self.f.reference40_32)
        self.f.flush(self.f.stage);output=self.f.stage/'comparison/fixture_v1'
        argv=['compare','--folder',str(self.f.stage),'--output',str(output)]
        with patch.object(sys,'argv',argv):self.assertEqual(report.main(),0)
        for name in ['geometry_and_propagation.png','efficiency_angles_and_mesh.png']:
            self.assertEqual((output/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        self.assertTrue((output/'comparison.json').exists());self.assertTrue((output/'REPORT.txt').exists())
        with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):report.main()


if __name__=='__main__':unittest.main(verbosity=2)
