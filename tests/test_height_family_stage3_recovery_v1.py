"""Pure JSON/layout regressions; no field reconstruction or optical worker."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
recovery=load('recovery_test',ROOT/'scripts/collect_height_family_stage3_recovery_v1.py')
plot=load('recovery_plot_test',ROOT/'scripts/plot_height_family_stage3_recovery_v1.py')
fixtures=load('recovery_saved_fixture',ROOT/'tests/test_height_family_stage3_assessment_v1.py')

class RecoveryTests(unittest.TestCase):
    def test_finalized_outputs_reject_self_consistent_rehashed_substitution(self):
        root=Path('recovery');expected=recovery.FINALIZED_OUTPUT_SHA
        recovery.finalized_output_binding(root,lambda p:expected[p.name])
        for name in expected:
            with self.subTest(name=name),self.assertRaises(ValueError):
                recovery.finalized_output_binding(root,lambda p:('0'*64 if p.name==name else expected[p.name]))
    def overlay(self):
        entry={'id':83,'status':'started'}
        return entry,{'schema':recovery.SCHEMA,'status':'COMPLETE_FROM_SAVED_ARTIFACTS_POSTPROCESS_ONLY',
          'original_entry83':entry,'original_entry83_sha256':hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest(),
          'attempt83_result':{'id':83,'verification_status':'complete_from_artifacts','analysis_status':'completed',
            'solver_exit_code':None,'worker_finished_utc':None,'controller_finished_utc':None},
          'raw_completion_evidence':{'exact_worker_exit_and_completion_clock_unknown':True,'all_artifact_mtimes_before_original_deadline':True},
          'postprocessing':{'after_original_deadline':True,'old_clock_reset':False,'new_optical_solves':0,'analysis_exit_code':0,
            'started_utc':'2026-11-16T04:16:57+00:00','finished_utc':'2026-11-16T04:17:00+00:00'}}
    def validate(self,entry,overlay):
        return recovery.validate_recovery_semantics(overlay,entry,deadline='2026-11-15T08:59:13+00:00')
    def test_original_started_unknown_exit_late_postprocessing_accepted(self):
        self.validate(*self.overlay())
    def test_invented_exit_finish_and_backdated_postprocessing_rejected(self):
        for section,key,value in [('attempt83_result','solver_exit_code',0),('attempt83_result','worker_finished_utc','2026-11-14T23:49:28Z'),
                ('attempt83_result','controller_finished_utc','2026-11-16T04:17:00Z'),('postprocessing','old_clock_reset',True),
                ('postprocessing','new_optical_solves',1),('postprocessing','started_utc','2026-11-14T23:49:28Z')]:
            entry,overlay=self.overlay();overlay[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(entry,overlay)
    def test_changed_original_entry_rejected(self):
        entry,overlay=self.overlay();entry=copy.deepcopy(entry);entry['status']='completed'
        with self.assertRaises(ValueError):self.validate(entry,overlay)
    def test_missing_unknown_exit_is_not_null(self):
        entry,overlay=self.overlay();del overlay['attempt83_result']['solver_exit_code']
        with self.assertRaises(ValueError):self.validate(entry,overlay)
    def test_exact_frozen_analysis_argv_and_counterexamples(self):
        class Paths:
            local_path=staticmethod(Path)
        root=ROOT;record=root/'raw.json';config=root/'config.json';output=root/'new.json'
        args=[str(root/'.venv-meep/bin/python'),'-B',str(root/'scripts/height_family_maxwell_analysis_v2.py'),
            '--record',str(record),'--config',str(config),'--target-diameter','16.2','--output',str(output),'--count','512','--window','128.0']
        kwargs=dict(root=root,record=record,config=config,output=output,target=16.2,c=Paths)
        recovery.analysis_argv_binding(args,**kwargs)
        for flag,value in [('--count','1024'),('--window','256'),('--record',str(root/'different.json')),('--output',str(record))]:
            changed=list(args);changed[changed.index(flag)+1]=value
            with self.subTest(flag=flag),self.assertRaises(ValueError):recovery.analysis_argv_binding(changed,**kwargs)
        with self.assertRaises(ValueError):recovery.analysis_argv_binding(args+['--new-flag'],**kwargs)
    def test_record_append_keeps_original_execution_state_and_rejects_duplicate(self):
        base={'data':{'Meep_records':[{'identity':'old'}],'all_attempts':[{'identity':'stage:83','status':'started'}]}}
        result=recovery.add_recovered_record(base,{'identity':'stage:83','resolution':40})
        self.assertEqual(result['all_attempts'][0]['status'],'started');self.assertEqual(len(base['data']['Meep_records']),1)
        with self.assertRaises(ValueError):recovery.add_recovered_record({'data':result},{'identity':'stage:83'})
    def test_newest_failed40_selected_over_older_passing32(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            f=fixtures.Fixture(tmp);entry=f.pair(100,40)
            f.f.modify_analysis(entry,lambda a:a['numerical_checks'].update(crosscheck=False))
            data=f.data();new=next(r for r in data['Meep_records'] if r['H_um']==100 and r['resolution']==40)
            new['identity']='maxwell_validation_v3:83';new['original_ledger_status']='started'
            result=f.assess(data=data,budget='FAIL');row=next(r for r in result['NN_heights'] if r['H_um']==100)
            self.assertEqual(row['Meep']['identity'],'maxwell_validation_v3:83');self.assertEqual(row['native']['status'],'FAIL')
            self.assertEqual(result['all_eight_nominal_heights_acceptance']['status'],'FAIL')
    def test_scoped_optical_pass_does_not_promote_expired_workflow(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            f=fixtures.Fixture(tmp);result=f.assess(budget='FAIL')
            scoped=recovery.scoped_optical_only(result,fixtures.assessment)
            self.assertEqual(scoped['status'],'PASS');self.assertFalse(scoped['full_workflow_acceptance_claimed'])
            self.assertEqual(result['all_eight_nominal_heights_acceptance']['status'],'FAIL')
            missing=copy.deepcopy(result);missing['NN_heights'].pop()
            self.assertEqual(recovery.scoped_optical_only(missing,fixtures.assessment)['status'],'UNRESOLVED')
            missing=copy.deepcopy(result);missing['boundary_duration_checks']=[]
            self.assertEqual(recovery.scoped_optical_only(missing,fixtures.assessment)['status'],'UNRESOLVED')
    def test_footer_discloses_date_and_unknown_controller_exit(self):
        _,overlay=self.overlay();label=plot.recovery_footer(overlay)
        self.assertIn('16 Nov postprocessing',label);self.assertIn('controller exit unrecorded',label)
        self.assertIn('no workflow PASS',label)
    def test_footer_preserves_curve_objects_and_original_text(self):
        class Text:
            def __init__(self,y):self.pos=(.5,y)
            def get_position(self):return self.pos
            def set_position(self,pos):self.pos=pos
        class Fig:
            def __init__(self):self.texts=[Text(.02),Text(.06),Text(.98)];self.added=[]
            def text(self,*args,**kwargs):self.added.append((args,kwargs))
        fig=Fig();original=list(fig.texts);_,overlay=self.overlay();plot.decorate_recovery_footer(fig,overlay)
        self.assertEqual(fig.texts,original);self.assertEqual([t.pos[1] for t in fig.texts],[.038,.078,.98])
        self.assertEqual(len(fig.added),1)

if __name__=='__main__':unittest.main(verbosity=2)
