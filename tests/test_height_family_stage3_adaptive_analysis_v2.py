"""Prospective adaptive receipt and explicit diagnostic-baseline fixtures."""
import copy
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
collector=load('adaptive_analysis_collector_v2',ROOT/'scripts/collect_height_family_stage3_v2.py')
wrapper=load('adaptive_analysis_wrapper_v2',ROOT/'scripts/assess_height_family_stage3_v2.py')
driver_tests=load('adaptive_analysis_driver_fixture',ROOT/'tests/test_height_family_adaptive_v1.py')
core_tests=load('adaptive_analysis_diagnostic_fixture',ROOT/'tests/test_height_family_stage3_assessment_v1.py')


class AdaptiveFixture:
    def __init__(self,folder):
        self.folder=Path(folder)/'maxwell_validation_v3';self.folder.mkdir()
        self.stage=driver_tests.FakeStage(self.folder)
        self.config=self.stage.config;self.plan=self.stage.plan
        self.cases=[dict(case,H_um=float(case['case_id'].split('_H')[1])) for case in self.stage.cases.values()]
        self.sources=collector.read(ROOT/'artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v3/source_manifest.json')
        self.stages={'stage3':{'folder':self.folder,'stage_name':'maxwell_validation_v3'}}
        self.entries={('stage3',entry['id']):entry for entry in self.stage.ledger['entries']}
        self.ledger=self.stage.ledger
        (self.folder/'config.json').write_text(json.dumps(self.config))
        fields=self.folder/'native_fixture';fields.mkdir()
        for item in self.ledger['entries']:
            item.update(started_utc='2026-11-14T03:00:00Z',finished_utc='2026-11-14T03:30:00Z')
            raw=collector.read(self.folder/item['record_file'])
            if item['method']=='NN':raw['reference_file']=str(self.folder/self.entries['stage3',item['id']-1]['record_file'])
            self.write(self.folder/item['record_file'],raw);item['record_sha256']=collector.sha(self.folder/item['record_file'])
            field=fields/(str(item['id'])+'.fields.bin');field.write_bytes(('fixture-only '+str(item['id'])).encode())
            item['required_file_sha256']={str(field):collector.sha(field)}
        # Full report schema includes the unchanged historicalH40 result.
        prior=self.folder/'prior';prior.mkdir();self.stages['stage1']={'folder':prior,'stage_name':'maxwell_validation_v1'}
        old=dict(self.entries['stage3',67],id=43,H_um=40.,record_file='43.json',analysis_file='43.analysis.json')
        self.write(prior/old['record_file'],{'settings':{},'power_balance_error':.001})
        self.write(prior/old['analysis_file'],{'angular_second_moment_ratio':1.02})
        old.update(record_sha256=collector.sha(prior/old['record_file']),analysis_sha256=collector.sha(prior/old['analysis_file']))
        self.entries['stage1',43]=old
        snapshot={'entries':copy.deepcopy(self.ledger['entries']),'first_started_utc':collector.ORIGINAL_START,'absolute_deadline_utc':collector.DEADLINE}
        self.snapshot=snapshot;self.selection=self.folder/'adaptive_selections/fixture';self.selection.mkdir(parents=True)
        self.write(self.selection/'prior_ledger.json',snapshot)
        self.driver=ROOT/'scripts/run_height_family_adaptive_v1.py';self.driver_sha=collector.sha(self.driver)
        shutil.copyfile(self.driver,self.selection/self.driver.name)
        utility=ROOT/'scripts/assess_height_family_stage3_v1.py';utility_sha=collector.sha(utility)
        target=self.selection/'assessment_source_snapshots/reviewed_assessor.py';target.parent.mkdir();shutil.copyfile(utility,target)
        self.assessment_document={'status':'SAVED_THREE_STAGE_ASSESSMENT','assessment_utility_sha256':utility_sha,
            'scientific_source_sha256':'aafa34734fbdd6c959278d935e43bfe0ae7e81cf877041b6380ac658d3d3d043',
            'reporting_height_enumeration':list(collector.HEIGHTS),'scientific_function_bytecode_changed':False,
            'primary_execution':{'status':'PASS','completed_verified':14,'required':14,
                'primary_jobs':[{'proposed_id':j['proposed_id'],'completed_verified':True,'status':'completed'} for j in self.plan['primary_jobs']]},
            'authorization':{'status':'PASS'},'budget':{'status':'PASS','additional_attempts':14,'cumulative_attempts':79},
            'ledger_snapshot_sha256':{str(self.folder/'ledger.json'):collector.sha(self.selection/'prior_ledger.json')},
            'input_sha256':{str(utility):utility_sha,str(self.folder/'config.json'):collector.sha(self.folder/'config.json')},'NN_heights':[]}
        for source in (ROOT/'scripts/collect_height_family_stage3_v1.py',ROOT/'scripts/plot_height_family_nn_forward_v3.py'):
            self.assessment_document['input_sha256'][str(source)]=collector.sha(source)
        for H in collector.HEIGHTS:
            sid,entry=('stage1',old) if H==40 else ('stage3',next(v for v in self.ledger['entries'] if v['method']=='NN' and v['H_um']==H))
            self.assessment_document['NN_heights'].append({'H_um':H,'Meep':{'identity':self.stages[sid]['stage_name']+':'+str(entry['id']),
                'record_sha256':entry['record_sha256'],'analysis_sha256':entry['analysis_sha256']}})
        base=self.entries['stage3',67]
        settings={key:base[key] for key in collector.NUMERICS};settings['duration']*=1.5
        requested_jobs=[dict(method=method,**settings) for method in ('reference','NN')]
        self.request={'reason':'Prospective root fine-time diagnosis','fine_time_priority_review':'Fine Q margin reviewed without a new Q gate or coarse correction',
            'primary_assessment_file':str(self.folder/'assessment/primary_v1/assessment.json'),
            'primary_assessment_utility_file':str(utility),'primary_assessment_utility_sha256':utility_sha,
            'pairs':[{'pair_id':'time100','reason':'Inspect primary fine-grid settling','baseline':{'stage_id':'stage3','record_id':67},
                'changed_variable':'duration','jobs':requested_jobs,'forecast_ceiling_s':20000.}]}
        effective=collector.adaptive_dimensions(self,requested_jobs[1]);scale=1.5
        forecasts=[{'method':method,'baseline_stage':'stage3','baseline_id':record_id,'measured_wrapper_s':measured,
            'scale':scale,'margin':1.15,'worker_forecast_s':measured*scale*1.15,'analysis_allowance_s':allowance}
            for method,record_id,measured,allowance in [('reference',66,100.,0.),('NN',67,120.,120.)]]
        self.pair={**self.request['pairs'][0],'jobs':[dict(job,job_type='duration',phase='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC',pair_id='time100') for job in requested_jobs],
            'effective_settings':effective,'baseline_record_sha256':base['record_sha256'],'baseline_analysis_sha256':base['analysis_sha256'],
            'baseline_reference_stage':'stage3','baseline_reference_id':66,'baseline_Q':1.099,'baseline_C':.02,
            'forecast':forecasts,'forecast_s':sum(r['worker_forecast_s']+r['analysis_allowance_s'] for r in forecasts)}
        self.selected_pairs=[self.pair]
        for name in ('preparation_manifest.json','source_manifest.json','frozen_cases.json'):
            (self.folder/name).write_text('{}')
        self.write(self.folder/'plan.json',self.plan)
        required=[self.folder/name for name in ('config.json','preparation_manifest.json','source_manifest.json','frozen_cases.json','plan.json')]
        for entry in self.ledger['entries']:
            required.append(self.folder/entry['record_file'])
            required.extend(Path(name) for name in entry['required_file_sha256'])
            if entry['method']=='NN':required.append(self.folder/entry['analysis_file'])
        self.assessment_document['input_sha256'].update({str(path):collector.sha(path) for path in required})
        created='2026-11-14T04:00:00Z';elapsed=(collector.utc(created)-collector.utc(collector.ORIGINAL_START)).total_seconds()
        self.receipt={'status':'PROSPECTIVE_STAGE3_ADAPTIVE_SELECTION_BEFORE_DATA','created_utc':created,
            'selected_mode':collector.ADAPTIVE_MODE,'execution_scope':collector.ADAPTIVE_SCOPE,'decision_origin':'EXPLICIT_ROOT_SELECTION_REASON',
            'decision_text_exact':self.request['reason'],'decision_text_sha256':hashlib.sha256(self.request['reason'].encode()).hexdigest(),
            'driver_sha256':self.driver_sha,'frozen_primary_controller_sha256':collector.PRIMARY_DRIVER_SHA,
            'approval_receipt_sha256':collector.AUTH_SHA,'approved_proposal_sha256':collector.PROPOSAL_SHA,
            'adaptive_launch_authorized_by_this_receipt':True,'adaptive_attempt_cap':4,'new_attempt_cap':18,'cumulative_attempt_cap':83,
            'max_started_pairs':2,'adaptive_autoexecution':False,'original_first_started_utc':collector.ORIGINAL_START,'original_absolute_deadline_utc':collector.DEADLINE,
            'geometry_physics_thresholds_unchanged':True,'no_new_Q_gate':True,'no_coarse_to_fine_correction':True,'no_fully_steady_claim':True,
            'fine_time_priority_review':self.request['fine_time_priority_review'],'primary_assessment_utility_sha256':utility_sha,
            'elapsed_s':elapsed,'remaining_minus185s':172800-elapsed-185,'selected_forecast_s':self.pair['forecast_s'],
            **{key:collector.sha(self.folder/name) for key,name in [('preparation_manifest_sha256','preparation_manifest.json'),('source_manifest_sha256','source_manifest.json'),('frozen_cases_sha256','frozen_cases.json')]}}
        case=next(c for c in self.cases if c['case_id']=='NN_H100');job=self.pair['jobs'][0]
        self.entry={**job,'id':80,'status':'started','started_utc':'2026-11-14T04:00:02Z','requested_settings':{k:job[k] for k in collector.NUMERICS},
            'effective_settings':effective,'case_id':None,'design_sha256':None,'physical_geometry_id':None,'reference_id':None,
            'paired_NN_case_id':case['case_id'],'paired_NN_design_sha256':case['design_sha256'],'paired_NN_physical_geometry_id':case['physical_geometry_id'],
            'selection_mode':collector.ADAPTIVE_MODE,'selection_scope':collector.ADAPTIVE_SCOPE,'scheduling_forecast':forecasts[0],
            'record_file':'records/80_reference_H100_r32_ADAPTIVE_duration.json'}
        self.entry['command']=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
            '--config',str(self.folder/'config.json'),'--output',str(self.folder/self.entry['record_file']),
            '--H','100.0','--resolution','32.0','--pml','2.0','--padding','10.0','--duration','502.5']
        self.ledger['entries'].append(self.entry);self.entries['stage3',80]=self.entry
        self.sign()
    def write(self,path,value):Path(path).write_text(json.dumps(value),encoding='utf-8')
    def protect(self,path):return collector.sha(path)
    def document(self,path):return collector.read(path)
    def raw(self,sid,entry):
        path=self.stages[sid]['folder']/entry['record_file'];return path,collector.read(path)
    def sign(self):
        snapsha=collector.sha(self.selection/'prior_ledger.json')
        inventories=self.assessment_document['ledger_snapshot_sha256']
        aliases=[name for name in inventories if collector.local_path(name).resolve()==self.folder/'ledger.json']
        for name in aliases or [str(self.folder/'ledger.json')]:inventories[name]=snapsha
        original=Path(self.request['primary_assessment_file']);original.parent.mkdir(parents=True,exist_ok=True)
        self.write(original,self.assessment_document)
        self.write(self.selection/'primary_assessment.json',self.assessment_document)
        self.request.update(primary_assessment_sha256=collector.sha(self.selection/'primary_assessment.json'),primary_assessment_ledger_sha256=snapsha)
        self.write(self.selection/'selection_request_exact.json',self.request)
        self.receipt.update(primary_assessment_sha256=self.request['primary_assessment_sha256'],primary_assessment_ledger_sha256=snapsha,
            prior_ledger_snapshot_sha256=snapsha,request_sha256=collector.sha(self.selection/'selection_request_exact.json'),selected_pairs=self.selected_pairs,selected_jobs=[j for p in self.selected_pairs for j in p['jobs']])
        self.write(self.selection/'selection_receipt.json',self.receipt)
        digest=collector.sha(self.selection/'selection_receipt.json');relative=str((self.selection/'selection_receipt.json').relative_to(self.folder))
        self.entry.update(selection_receipt_file=relative,selection_receipt_sha256=digest)
        link={'receipt_file':relative,'receipt_sha256':digest,'driver_sha256':self.driver_sha}
        self.ledger.update(adaptive_selections=[link],current_adaptive_selection=link,
            adaptive_pair_runs=[{'pair_id':'time100','selection_sha256':digest,'status':'started','started_utc':'2026-11-14T04:00:01Z'}])
    def result(self):
        with patch.object(collector,'REVIEWED_ADAPTIVE_DRIVER_SHA',self.driver_sha):return collector.attempt_receipt(self,self.entry)
    def use_NN_entry(self):
        reference=dict(self.entry,status='completed')
        self.write(self.folder/reference['record_file'],{'settings':reference['effective_settings'],'incident_power':1.,'power_balance_error':.001})
        reference['record_sha256']=collector.sha(self.folder/reference['record_file'])
        self.ledger['entries'][-1]=reference;self.entries['stage3',80]=reference
        case=next(c for c in self.cases if c['case_id']=='NN_H100')
        self.entry={**reference,**self.pair['jobs'][1],'id':81,'status':'started','started_utc':'2026-11-14T04:05:00Z',
            'record_file':'records/81_NN_H100_r32_ADAPTIVE_duration.json','reference_id':80,'reference_stage_id':'stage3',
            'reference_stage_folder':str(self.folder),'case_id':case['case_id'],'design_sha256':case['design_sha256'],
            'physical_geometry_id':case['physical_geometry_id'],'scheduling_forecast':self.pair['forecast'][1]}
        args=list(reference['command']);args[6]=str(self.folder/self.entry['record_file'])
        self.entry['command']=args+['--design',str(self.folder/case['design_file']),'--reference',str(self.folder/reference['record_file'])]
        self.ledger['entries'].append(self.entry);self.entries['stage3',81]=self.entry;self.sign()
    def use_cached_prior_reference(self):
        prior=self.folder.parent/'maxwell_validation_v2';prior.mkdir()
        ref=dict(self.entry,id=64,status='completed',record_file='64_reference.json')
        raw={'settings':ref['effective_settings'],'incident_power':1.,'power_balance_error':.001}
        self.write(prior/ref['record_file'],raw);ref['record_sha256']=collector.sha(prior/ref['record_file'])
        field=prior/'64.fields.bin';field.write_bytes(b'cached-prior-reference-fixture')
        ref['required_file_sha256']={str(field):collector.sha(field)}
        self.stages['stage2']={'folder':prior,'stage_name':'maxwell_validation_v2'};self.entries['stage2',64]=ref
        case=next(c for c in self.cases if c['case_id']=='NN_H100')
        self.entry={**self.entry,**self.pair['jobs'][1],'case_id':case['case_id'],'design_sha256':case['design_sha256'],
            'physical_geometry_id':case['physical_geometry_id'],'record_file':'records/80_NN_H100_r32_ADAPTIVE_duration.json',
            'reference_id':64,'reference_stage_id':'stage2','reference_stage_folder':str(prior),'scheduling_forecast':self.pair['forecast'][1]}
        args=list(self.entry['command']);args[6]=str(self.folder/self.entry['record_file'])
        self.entry['command']=args+['--design',str(self.folder/case['design_file']),'--reference',str(prior/ref['record_file'])]
        self.ledger['entries'][-1]=self.entry;self.entries['stage3',80]=self.entry
        for source in (prior/ref['record_file'],field):self.assessment_document['input_sha256'][str(source)]=collector.sha(source)
        self.sign()


class TestAdaptiveAuthorization(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=AdaptiveFixture(self.temp.name)
    def test_reviewed_prospective_isolated_pair_is_authenticated_without_execution(self):
        result=self.f.result();self.assertEqual(result['status'],'PASS',result)
    def test_NN_job_binds_matching_new_reference_and_frozen_profile(self):
        self.f.use_NN_entry();result=self.f.result();self.assertEqual(result['status'],'PASS',result)
        self.f.entry['reference_id']=66
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_cached_prior_reference_is_bound_by_origin_path_and_settings(self):
        self.f.use_cached_prior_reference();result=self.f.result();self.assertEqual(result['status'],'PASS',result)
        self.f.entry['reference_stage_id']='stage1';self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.entry['reference_stage_id']='stage2';self.f.entry['command'][-1]=str(self.f.folder/'records/wrong-origin.json')
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_v2_assessor_schema_and_source_pin_are_supported_explicitly(self):
        utility=ROOT/'scripts/assess_height_family_stage3_v2.py';digest=collector.sha(utility)
        self.f.request.update(primary_assessment_utility_file=str(utility),primary_assessment_utility_sha256=digest)
        self.f.receipt['primary_assessment_utility_sha256']=digest
        self.f.assessment_document.update(assessment_utility_sha256=digest,assessment_schema='height_family_stage3_v2',
            assessment_core_sha256=collector.REVIEWED_ASSESSORS['assess_height_family_stage3_v1.py'],
            analysis_source_sha256={str(source):collector.sha(source) for source in (utility,ROOT/'scripts/assess_height_family_stage3_v1.py',ROOT/'scripts/collect_height_family_stage3_v2.py')})
        shutil.copyfile(utility,self.f.selection/'assessment_source_snapshots/reviewed_assessor.py');self.f.sign()
        result=self.f.result();self.assertEqual(result['status'],'PASS',result)
    def test_distinct_pair_names_cannot_alias_the_same_target(self):
        other=copy.deepcopy(self.f.pair);other['pair_id']='same_target_other_name'
        for job in other['jobs']:job['pair_id']=other['pair_id']
        request=copy.deepcopy(self.f.request['pairs'][0]);request['pair_id']=other['pair_id']
        self.f.selected_pairs.append(other);self.f.request['pairs'].append(request);self.f.sign()
        result=self.f.result();self.assertEqual(result['status'],'FAIL')
        self.assertTrue(any('same target' in reason for reason in result['missing_evidence']))
    def test_unreviewed_driver_is_unresolved(self):
        with patch.object(collector,'REVIEWED_ADAPTIVE_DRIVER_SHA',None):
            self.assertEqual(collector.attempt_receipt(self.f,self.f.entry)['status'],'UNRESOLVED')
    def test_user_supplied_fake_utility_path_cannot_self_authorize(self):
        fake=self.f.folder/'fake_assessor.py';shutil.copyfile(ROOT/'scripts/assess_height_family_stage3_v1.py',fake)
        self.f.request['primary_assessment_utility_file']=str(fake);self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_primary_report_must_be_an_actual_versioned_stage3_assessment(self):
        self.f.request['primary_assessment_file']=str(self.f.selection/'outside_assessment_tree.json');self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_missing_primary_source_record_inventory_fails(self):
        self.f.assessment_document['NN_heights'][0]['Meep']['record_sha256']='wrong';self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_all_required_primary_field_inventory_and_full_input_inventory_are_verified(self):
        field=next(iter(self.f.entries['stage3',66]['required_file_sha256']))
        saved=self.f.assessment_document['input_sha256'].pop(field);self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.assessment_document['input_sha256'][field]=saved;self.f.sign()
        Path(field).write_bytes(b'changed-native-field')
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_exact_primary_job_proof_and_unchanged_bytecode_flag_are_required(self):
        jobs=self.f.assessment_document['primary_execution']['primary_jobs'];removed=jobs.pop();self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
        jobs.append(removed);self.f.assessment_document['scientific_function_bytecode_changed']=True;self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_cross_host_inventory_aliases_are_normalized_and_duplicates_rejected(self):
        def alias(name):
            name=str(name)
            if name.startswith('/mnt/'):
                return name[5].upper()+':/'+name[7:]
            return '/mnt/'+name[0].lower()+'/'+name[3:].replace('\\','/')
        for key in ('input_sha256','ledger_snapshot_sha256'):
            self.f.assessment_document[key]={alias(name):digest for name,digest in self.f.assessment_document[key].items()}
        self.f.sign();result=self.f.result();self.assertEqual(result['status'],'PASS',result)
        original=next(iter(self.f.assessment_document['input_sha256']))
        self.f.assessment_document['input_sha256'][alias(original)]=self.f.assessment_document['input_sha256'][original]
        self.f.sign();result=self.f.result();self.assertEqual(result['status'],'FAIL')
        self.assertTrue(any('Duplicate normalized path' in reason for reason in result['missing_evidence']))
    def test_primary_must_complete_before_adaptive_selection(self):
        self.f.entries['stage3',79]['analysis_status']='failed'
        self.f.snapshot['entries'][-1]['analysis_status']='failed'
        self.f.write(self.f.selection/'prior_ledger.json',self.f.snapshot);self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_two_numerical_variables_or_changed_forecast_are_rejected(self):
        for jobs in (self.f.pair['jobs'],self.f.request['pairs'][0]['jobs']):
            for job in jobs:job['padding_um']=15.
        self.f.sign();self.assertEqual(self.f.result()['status'],'FAIL')
        for jobs in (self.f.pair['jobs'],self.f.request['pairs'][0]['jobs']):
            for job in jobs:job['padding_um']=10.
        self.f.pair['forecast'][0]['measured_wrapper_s']=1.;self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_selection_must_fit_original_deadline_and_be_prospective(self):
        self.f.receipt['remaining_minus185s']=1.;self.f.sign();self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.receipt['remaining_minus185s']=100000.;self.f.receipt['created_utc']='2026-11-14T04:00:03Z';self.f.sign()
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_each_new_worker_must_still_fit_pending_forecast_at_its_UTC_start(self):
        self.f.use_NN_entry()
        self.f.entry['started_utc']=(collector.utc(collector.DEADLINE)-timedelta(seconds=200)).isoformat()
        result=self.f.result();self.assertEqual(result['status'],'FAIL')
        self.assertTrue(any('Pending measured forecast' in reason for reason in result['missing_evidence']))
    def test_reference_geometry_or_actual_worker_output_redirect_is_rejected(self):
        self.f.entry['case_id']='NN_H100';self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.entry['case_id']=None;self.f.entry['command'][6]=str(self.f.folder/'redirected.json')
        self.assertEqual(self.f.result()['status'],'FAIL')
    def test_explicit_pair_run_and_four_attempt_cap_are_required(self):
        self.f.ledger['adaptive_pair_runs']=[];self.assertEqual(self.f.result()['status'],'FAIL')
        self.f.sign();self.f.ledger['entries']+= [dict(self.f.entry,id=i) for i in (81,82,83,84)]
        self.assertEqual(self.f.result()['status'],'FAIL')


class TestAdaptiveDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=core_tests.Fixture(self.temp.name)
    def diagnostic(self,base,target):
        data=self.f.data();folder=self.f.f.stage;selection=folder/'adaptive_diag_receipt.json'
        pair={'pair_id':'diag','changed_variable':'duration','baseline':{'stage_id':'stage3','record_id':base['id']}}
        selection.write_text(json.dumps({'selected_pairs':[pair]}));digest=collector.sha(selection)
        target.update(pair_id='diag',selection_receipt_sha256=digest)
        e=SimpleNamespace(folder=folder,stages={'stage3':{'folder':folder,'stage_name':folder.name}},
            ledger={'entries':[target],'adaptive_pair_runs':[{'pair_id':'diag','selection_sha256':digest}],
                    'adaptive_selections':[{'receipt_file':selection.name,'receipt_sha256':digest}]},
            protect=collector.sha,document=collector.read)
        return collector.selected_pair_diagnostics(e,data,{},core_tests.assessment.load_science())[0]
    def test_completed_execution_does_not_hide_failed_numerical_stability(self):
        base=self.f.devices[100.,32.];target=self.f.pair(100,32,'duration',{'duration':502.5})
        self.f.f.modify_analysis(target,lambda a:a.update(normalized_curvature=.08))
        row=self.diagnostic(base,target)
        self.assertEqual(row['native_status'],'PASS');self.assertEqual(row['numerical_status'],'FAIL')
        self.assertEqual(row['target_nominal_status'],'PASS')
        result=wrapper.add_execution_diagnostics({}, {'primary_execution':{'status':'PASS'},'adaptive_execution':{'status':'PASS'},
            'selected_adaptive_diagnostics':[row]},wrapper.load_core())
        self.assertEqual(result['continuation_execution']['status'],'PASS')
        self.assertTrue(result['execution_completion_is_not_optical_acceptance'])
        self.assertNotIn('full_selected_continuation_acceptance',result)
        self.assertEqual(result['selected_adaptive_diagnostics'][0]['numerical_status'],'FAIL')
    def test_fine_temporal_Q_gate_crossing_stays_optical_failure_without_new_Q_delta_gate(self):
        base=self.f.devices[100.,32.];target=self.f.pair(100,32,'duration',{'duration':502.5})
        def change(a):a.update(angular_second_moment_ratio=1.11);a['optical_gates']['angular_second_moment']=False
        self.f.f.modify_analysis(target,change);row=self.diagnostic(base,target)
        self.assertEqual(row['numerical_status'],'PASS');self.assertEqual(row['target_nominal_status'],'FAIL')
        self.assertTrue(row['Q_gate_crossing']);self.assertTrue(row['Q_change_is_diagnostic_not_a_new_convergence_gate'])
    def test_chained_duration_diagnostic_uses_exact_receipt_baseline(self):
        base=self.f.pair(100,32,'duration',{'duration':502.5});self.f.f.modify_analysis(base,lambda a:a.update(normalized_curvature=.06))
        target=self.f.pair(100,32,'duration',{'duration':1005.});self.f.f.modify_analysis(target,lambda a:a.update(normalized_curvature=.061))
        row=self.diagnostic(base,target)
        self.assertEqual(row['baseline_settings']['duration'],502.5);self.assertEqual(row['target_settings']['duration'],1005.)
        self.assertAlmostEqual(row['comparison']['measurements']['curvature_absolute_change'],.001)
        self.assertEqual(row['numerical_status'],'PASS')


class TestImmutableReadTracking(unittest.TestCase):
    def test_repeated_authenticated_read_cannot_silently_replace_old_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder:
            path=Path(folder)/'known.txt';path.write_text('first')
            evidence=collector.Evidence.__new__(collector.Evidence);evidence.inputs={}
            first=evidence.protect(path);path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'between authenticated reads'):evidence.protect(path)
            self.assertEqual(evidence.inputs[str(path.resolve())],first)


if __name__=='__main__':unittest.main(verbosity=2)
