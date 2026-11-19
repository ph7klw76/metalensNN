"""Prospective adaptive policy/executor fixtures; no real jobs or Meep."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_adaptive_test',ROOT/'scripts/run_height_family_adaptive_v1.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
b=a.base


class FakeStage:
    def __init__(self,folder):
        self.folder=folder;self.config=b.read(b.PROPOSAL.parent/'frozen_inputs/config.json')
        self.plan=b.proposal_plan(b.read(b.PROPOSAL));self.all_prior_entries=[];self.catalog=[];self.cases={};self.active=None;self.env={}
        self.ledger=dict(entries=[],first_started_utc=b.ORIGINAL_START,absolute_deadline_utc=b.DEADLINE,
                         elapsed_s=100000.,prior_attempt_count=65)
        (folder/'records').mkdir();(folder/'logs').mkdir();(folder/'analyses').mkdir()
        for item in self.plan['primary_jobs']:
            H=item['H_um'];cid=b.case_id(H)
            if cid not in self.cases:
                path=folder/(cid+'.json');path.write_text(json.dumps(dict(H_um=H,target_diameter_um=15.)))
                self.cases[cid]=dict(case_id=cid,design_file=path.name,design_sha256=b.sha(path),physical_geometry_id='physical-'+cid)
            effective=b.u.settings_for(self.config,item);identity=item['proposed_id']
            record=dict(settings=effective,power_balance_error=.001,config_fingerprint=b.fingerprint(self.config))
            if item['method']=='reference':record['incident_power']=1.
            path=folder/f'records/{identity}.json';path.write_text(json.dumps(record))
            entry=dict(item,id=identity,status='completed',record_file=str(path.relative_to(folder)),record_sha256=b.sha(path),
                wrapper_wall_s=100. if item['method']=='reference' else 120.,analysis_status='completed',
                design_sha256=None if item['method']=='reference' else self.cases[cid]['design_sha256'])
            if item['method']=='NN':
                ap=folder/f'analyses/{identity}.json';ap.write_text(json.dumps(dict(angular_second_moment_ratio=1.099,normalized_curvature=.02)))
                entry.update(analysis_file=str(ap.relative_to(folder)),analysis_sha256=b.sha(ap))
            self.ledger['entries'].append(entry)
        self.config_fingerprint=b.fingerprint(self.config);self.save()
        # H40 is already completed in the prior catalog, outside primary14.
        first=next(e for e in self.ledger['entries'] if e['method']=='NN')
        old=copy.deepcopy(first);old.update(id=44,H_um=40.)
        self.catalog.append(dict(stage_id='stage1',stage_name='stage1-fixture',stage_folder=str(folder),entry=old))
        for name in ['preparation_manifest.json','source_manifest.json','frozen_cases.json']:(folder/name).write_text('{}')
        (folder/'config.json').write_text(json.dumps(self.config));(folder/'plan.json').write_text(json.dumps(self.plan))
    def case(self,H):return self.cases[b.case_id(H)]
    def verify(self):pass
    def elapsed(self):return 100000.
    def remaining(self):return 50000.
    def save(self):b.write(self.folder/'ledger.json',self.ledger)
    def cache(self,item):
        matches=[e for e in self.ledger['entries'] if e['status']=='completed' and b.identity(e)==b.identity(item)]
        if not matches:return None
        entry=matches[-1]
        return dict(entry=entry,path=self.folder/entry['record_file'],record=b.read(self.folder/entry['record_file']),
                    stage_id='stage3',stage_name='stage3-fixture',stage_folder=str(self.folder))
    def terminate(self):pass


class TestAdaptive(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.addCleanup(patch.stopall);patch.object(b,'FOLDER',self.folder).start()
        self.stage=FakeStage(self.folder)
    def pair(self,H=100.,variable='duration',value=None):
        entry=next(e for e in self.stage.ledger['entries'] if e['method']=='NN' and e['H_um']==H)
        if value is None:value=entry[variable]*1.5 if variable=='duration' else 48.
        settings={key:entry[key] for key in b.u.KEYS};settings[variable]=value
        return dict(pair_id=f'pair_{H:g}_{variable}',reason='Root reviewed frozen primary evidence',
            baseline=dict(stage_id='stage3',record_id=entry['id']),changed_variable=variable,
            jobs=[dict(method=method,**settings) for method in ['reference','NN']],forecast_ceiling_s=20000.)
    def request(self,pairs=None):
        utility=ROOT/'scripts/assess_height_family_stage3_v1.py'
        assessment=self.folder/'assessment'/'fixture'/'assessment.json';assessment.parent.mkdir(parents=True,exist_ok=True)
        ledger_sha=b.sha(self.folder/'ledger.json')
        inputs={str(path.resolve()):b.sha(path) for path in self.folder.rglob('*') if path.is_file()
                and path.name!='ledger.json' and (path.parent in [self.folder,self.folder/'records',self.folder/'analyses'])}
        for name in ['assess_height_family_stage3_v1.py','collect_height_family_stage3_v1.py','plot_height_family_nn_forward_v3.py']:
            path=ROOT/'scripts'/name;inputs[str(path.resolve())]=b.sha(path)
        rows=[]
        for H in a.HEIGHTS:
            if H==40.:entry=self.stage.catalog[0]['entry'];name='stage1-fixture'
            else:entry=next(e for e in self.stage.ledger['entries'] if e['method']=='NN' and e['H_um']==H);name=self.folder.name
            rows.append(dict(H_um=H,Meep=dict(identity=name+':'+str(entry['id']),
                record_sha256=entry['record_sha256'],analysis_sha256=entry['analysis_sha256'])))
        document=dict(ledger_snapshot_sha256={str((self.folder/'ledger.json').resolve()):ledger_sha},
            status='SAVED_THREE_STAGE_ASSESSMENT',scientific_source_sha256=a.SCIENCE_SHA,scientific_function_bytecode_changed=False,
            reporting_height_enumeration=list(a.HEIGHTS),NN_heights=rows,input_sha256=inputs,
            primary_execution=dict(status='PASS',required=14,completed_verified=14,
                primary_jobs=[dict(proposed_id=job['proposed_id'],completed_verified=True,status='completed') for job in self.stage.plan['primary_jobs']]),
            authorization=dict(status='PASS'),budget=dict(status='PASS',additional_attempts=len(self.stage.ledger['entries']),
                cumulative_attempts=65+len(self.stage.ledger['entries'])),assessment_utility_sha256=b.sha(utility))
        assessment.write_text(json.dumps(document))
        return dict(reason='ROOT prospective selection under approved amendment',fine_time_priority_review='Reviewed H100/H75 fine Q margins; coarse0.003522 is diagnostic, not a correction or new gate.',
            primary_assessment_file=str(assessment),primary_assessment_sha256=b.sha(assessment),
            primary_assessment_ledger_sha256=ledger_sha,primary_assessment_utility_file=str(utility),
            primary_assessment_utility_sha256=b.sha(utility),pairs=[self.pair()] if pairs is None else pairs)
    def selection_receipt(self,request,pairs):
        folder=self.folder/'adaptive-selection';folder.mkdir()
        for name,data in [('selection_request_exact.json',json.dumps(request).encode()),('prior_ledger.json',(self.folder/'ledger.json').read_bytes())]:
            (folder/name).write_bytes(data)
        shutil.copyfile(a.__file__,folder/Path(a.__file__).name);shutil.copyfile(request['primary_assessment_file'],folder/'primary_assessment.json')
        target=folder/'assessment_source_snapshots/reviewed_assessor.py';target.parent.mkdir();shutil.copyfile(request['primary_assessment_utility_file'],target)
        for name in ['preparation_manifest.json','source_manifest.json','frozen_cases.json']:(self.folder/name).write_text('{}')
        receipt=dict(status='PROSPECTIVE_STAGE3_ADAPTIVE_SELECTION_BEFORE_DATA',selected_mode=a.MODE,execution_scope=a.SCOPE,
            decision_origin='EXPLICIT_ROOT_SELECTION_REASON',driver_sha256=a.sha(a.__file__),frozen_primary_controller_sha256=a.BASE_SHA,
            decision_text_exact=request['reason'],decision_text_sha256=b.hashlib.sha256(request['reason'].encode()).hexdigest(),
            fine_time_priority_review=request['fine_time_priority_review'],no_new_Q_gate=True,no_coarse_to_fine_correction=True,no_fully_steady_claim=True,
            approval_receipt_sha256=b.AUTH_SHA,approved_proposal_sha256=b.PROPOSAL_SHA,adaptive_launch_authorized_by_this_receipt=True,
            max_started_pairs=2,adaptive_attempt_cap=4,new_attempt_cap=18,cumulative_attempt_cap=83,adaptive_autoexecution=False,
            original_first_started_utc=b.ORIGINAL_START,original_absolute_deadline_utc=b.DEADLINE,
            preparation_manifest_sha256=b.sha(self.folder/'preparation_manifest.json'),source_manifest_sha256=b.sha(self.folder/'source_manifest.json'),
            frozen_cases_sha256=b.sha(self.folder/'frozen_cases.json'),request_sha256=b.sha(folder/'selection_request_exact.json'),
            primary_assessment_sha256=request['primary_assessment_sha256'],primary_assessment_utility_sha256=request['primary_assessment_utility_sha256'],
            prior_ledger_snapshot_sha256=b.sha(folder/'prior_ledger.json'),primary_assessment_ledger_sha256=request['primary_assessment_ledger_sha256'],
            selected_pairs=pairs,selected_jobs=[job for pair in pairs for job in pair['jobs']])
        path=folder/'selection_receipt.json';path.write_text(json.dumps(receipt));link=dict(receipt_file=str(path.relative_to(self.folder)),receipt_sha256=b.sha(path))
        self.stage.ledger['adaptive_selections']=[link];self.stage.ledger['current_adaptive_selection']=link
        return path,link
    def write_report(self,request,document):
        Path(request['primary_assessment_file']).write_text(json.dumps(document))
        request['primary_assessment_sha256']=b.sha(request['primary_assessment_file'])
    def executable_stage(self):
        stage=a.AdaptiveStage.__new__(a.AdaptiveStage);stage.__dict__.update(self.stage.__dict__)
        for name in ['verify','case','cache','save','elapsed','remaining']:
            setattr(stage,name,getattr(self.stage,name))
        stage.active=None
        return stage
    def test_primary_completion_and_exact_assessment_snapshot_required(self):
        request=self.request();pairs=a.request_selection(request,self.stage);self.assertEqual(len(pairs),1)
        self.stage.ledger['entries'][-1]['analysis_status']='failed'
        with self.assertRaisesRegex(ValueError,'primary independent analysis'):a.request_selection(request,self.stage)
    def test_stale_or_unreviewed_assessment_is_rejected(self):
        request=self.request();self.stage.ledger['note']='changed';self.stage.save()
        with self.assertRaisesRegex(ValueError,'exact current stage3 ledger'):a.verify_assessment(request,self.stage)
        request=self.request();utility=self.folder/'arbitrary.py';utility.write_text('# reviewed saved-data assessment only\n')
        request.update(primary_assessment_utility_file=str(utility),primary_assessment_utility_sha256=b.sha(utility))
        document=b.read(request['primary_assessment_file']);document['assessment_utility_sha256']=b.sha(utility)
        Path(request['primary_assessment_file']).write_text(json.dumps(document));request['primary_assessment_sha256']=b.sha(request['primary_assessment_file'])
        with self.assertRaisesRegex(ValueError,'reviewed saved-analysis'):a.verify_assessment(request,self.stage)
    def test_reviewed_v2_wrapper_and_fixed_source_inventory_are_accepted(self):
        request=self.request();utility=ROOT/'scripts/assess_height_family_stage3_v2.py'
        request.update(primary_assessment_utility_file=str(utility),primary_assessment_utility_sha256=b.sha(utility))
        document=b.read(request['primary_assessment_file']);document.update(assessment_schema='height_family_stage3_v2',
            assessment_core_sha256=a.ASSESSOR_CORE_SHA,assessment_utility_sha256=b.sha(utility),
            analysis_source_sha256={str((ROOT/'scripts'/name).resolve()):b.sha(ROOT/'scripts'/name) for name in
                ['assess_height_family_stage3_v2.py','assess_height_family_stage3_v1.py','collect_height_family_stage3_v2.py']})
        self.write_report(request,document);a.verify_assessment(request,self.stage)
        document['analysis_source_sha256'][str(utility.resolve())]='0'*64;self.write_report(request,document)
        with self.assertRaisesRegex(ValueError,'analysis source inventory changed'):a.verify_assessment(request,self.stage)
    def test_report_location_schema_coverage_progress_budget_and_record_lineage_fail_closed(self):
        for mutate in [lambda d:d.update(scientific_source_sha256='0'*64),
                       lambda d:d['reporting_height_enumeration'].pop(),
                       lambda d:d['primary_execution'].update(completed_verified=13),
                       lambda d:d['primary_execution']['primary_jobs'][0].update(proposed_id=1),
                       lambda d:d['budget'].update(cumulative_attempts=65),
                       lambda d:d['NN_heights'][0]['Meep'].update(record_sha256='0'*64),
                       lambda d:d['authorization'].update(status='FAIL')]:
            request=self.request();document=b.read(request['primary_assessment_file']);mutate(document);self.write_report(request,document)
            with self.assertRaises(ValueError):a.verify_assessment(request,self.stage)
        request=self.request();outside=self.folder/'assessment.json';shutil.copyfile(request['primary_assessment_file'],outside)
        request['primary_assessment_file']=str(outside)
        with self.assertRaisesRegex(ValueError,'contained stage3'):a.verify_assessment(request,self.stage)
    def test_required_input_inventory_cannot_be_missing_or_stale(self):
        request=self.request();document=b.read(request['primary_assessment_file'])
        del document['input_sha256'][str((self.folder/'config.json').resolve())];self.write_report(request,document)
        with self.assertRaisesRegex(ValueError,'required immutable'):a.verify_assessment(request,self.stage)
        request=self.request();document=b.read(request['primary_assessment_file'])
        document['input_sha256'][str((self.folder/'config.json').resolve())]='0'*64;self.write_report(request,document)
        with self.assertRaisesRegex(ValueError,'immutable input inventory changed'):a.verify_assessment(request,self.stage)
    def test_windows_wsl_path_keys_normalize_without_duplicate_aliases(self):
        path=(ROOT/'scripts/assess_height_family_stage3_v1.py').resolve();native=str(path)
        if sys.platform=='win32':alternate='/mnt/'+native[0].lower()+'/'+native[3:].replace('\\','/')
        else:
            self.assertTrue(native.startswith('/mnt/'));alternate=native[5].upper()+':\\'+native[7:].replace('/','\\')
        self.assertEqual(a.local_path(alternate),path)
        self.assertEqual(a.normalized_inventory({alternate:b.sha(path)})[path],b.sha(path))
        with self.assertRaisesRegex(ValueError,'Duplicate normalized'):a.normalized_inventory({native:b.sha(path),alternate:b.sha(path)})
        request=self.request();document=b.read(request['primary_assessment_file'])
        document['input_sha256'][alternate]=document['input_sha256'].pop(native);self.write_report(request,document)
        a.verify_assessment(request,self.stage)
    def test_exact_pair_isolation_and_boundaries(self):
        valid=a.isolated_jobs(self.pair(),self.stage)
        self.assertEqual([job['method'] for job in valid['jobs']],['reference','NN'])
        self.assertTrue(all(job['phase']=='ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC' and job['job_type']=='duration' for job in valid['jobs']))
        for variable,value in [('resolution',64.),('pml_um',4.),('padding_um',25.),('duration',1000.)]:
            with self.assertRaises(ValueError):a.isolated_jobs(self.pair(variable=variable,value=value),self.stage)
        bad=self.pair();bad['jobs'][1]['padding_um']=15.
        with self.assertRaises(ValueError):a.isolated_jobs(bad,self.stage)
        bad=self.pair();bad['jobs'][1]['method']='direct'
        with self.assertRaises(ValueError):a.isolated_jobs(bad,self.stage)
        bad=self.pair();bad['jobs'][0]['extra']='hidden'
        with self.assertRaises(ValueError):a.isolated_jobs(bad,self.stage)
    def test_mesh_and_padding_labels_are_honest(self):
        mesh=a.isolated_jobs(self.pair(variable='resolution',value=40.),self.stage)
        self.assertEqual(mesh['jobs'][1]['job_type'],'refinement')
        padding=a.isolated_jobs(self.pair(variable='padding_um',value=15.),self.stage)
        self.assertIn('collection-radius',padding['padding_scope']);self.assertIn('not fixed-aperture',padding['padding_scope'])
    def test_measured_forecast_ceiling_deadline_and6h_checks(self):
        pair=self.pair();result=a.isolated_jobs(pair,self.stage)
        self.assertAlmostEqual(result['forecast_s'],(100+120)*1.5*1.15+120)
        pair['forecast_ceiling_s']=1.
        with self.assertRaises(ValueError):a.isolated_jobs(pair,self.stage)
        request=self.request();self.stage.remaining=lambda:10.
        with self.assertRaisesRegex(RuntimeError,'original remaining deadline'):a.request_selection(request,self.stage)
    def test_two_pair_four_attempt_global_caps_and_priority_review(self):
        request=self.request([self.pair(100.),self.pair(75.)]);self.assertEqual(len(a.request_selection(request,self.stage)),2)
        request['pairs'].append(self.pair(70.,'resolution',48.))
        with self.assertRaises(ValueError):a.request_selection(request,self.stage)
        request=self.request();request['fine_time_priority_review']=''
        with self.assertRaises(ValueError):a.request_selection(request,self.stage)
        self.stage.ledger['adaptive_pair_runs']=[dict(pair_id='a'),dict(pair_id='b')]
        self.stage.save();request=self.request()
        with self.assertRaises(ValueError):a.request_selection(request,self.stage)
    def test_selection_receipt_hash_source_sidecars_and_reserved_slots(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        a.verify_selection_receipt(self.stage,path,link)
        original=b.read(path);original['adaptive_launch_authorized_by_this_receipt']=False;path.write_text(json.dumps(original));link['receipt_sha256']=b.sha(path)
        with self.assertRaisesRegex(ValueError,'authority/scope'):a.verify_selection_receipt(self.stage,path,link)
    def test_root_request_sidecar_cannot_change_after_prospective_selection(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        (path.parent/'selection_request_exact.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'exact adaptive request changed'):a.verify_selection_receipt(self.stage,path,link)
    def test_unstarted_selection_requires_explicit_supersession(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        self.stage.save();request=self.request([self.pair(75.)])
        with self.assertRaisesRegex(ValueError,'Explicit supersession'):a.request_selection(request,self.stage)
        request['supersedes_unstarted_selection_sha256']=link['receipt_sha256']
        self.assertEqual(len(a.request_selection(request,self.stage)),1)
    def test_duplicate_target_numerics_rejected_across_distinct_pair_ids_and_prior_allocations(self):
        first=self.pair();second=copy.deepcopy(first);second['pair_id']='different_label_same_target'
        with self.assertRaisesRegex(ValueError,'Duplicate adaptive target'):a.request_selection(self.request([first,second]),self.stage)
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        self.stage.ledger['adaptive_pair_runs']=[dict(pair_id=pairs[0]['pair_id'],selection_sha256=link['receipt_sha256'],status='completed')]
        self.stage.save();request=self.request([second])
        with self.assertRaisesRegex(ValueError,'Duplicate adaptive target'):a.request_selection(request,self.stage)
    def test_per_worker_forecast_rechecks_remaining_time_after_reference_overrun(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        stage=self.executable_stage();before=len(stage.ledger['entries']);stage.remaining=lambda:10.
        with patch.object(a.subprocess,'Popen',side_effect=AssertionError('No worker after reference leaves insufficient forecast')):
            with self.assertRaisesRegex(RuntimeError,'forecast no longer fits'):
                stage.execute_adaptive_job(pairs[0]['jobs'][1],path,link['receipt_sha256'],pairs[0]['forecast'][1])
        self.assertEqual(before,len(stage.ledger['entries']))
    def test_final_prePopen_forecast_guard_handles_time_spent_verifying_sources(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        stage=self.executable_stage();state=dict(left=50000.,verifies=0)
        stage.remaining=lambda:state['left']
        def verify():
            state['verifies']+=1
            if state['verifies']>=2:state['left']=10.
        stage.verify=verify
        with patch.object(a.subprocess,'Popen',side_effect=AssertionError('No worker when final forecast check fails')):
            with self.assertRaisesRegex(RuntimeError,'forecast no longer fits'):
                stage.execute_adaptive_job(pairs[0]['jobs'][0],path,link['receipt_sha256'],pairs[0]['forecast'][0])
        self.assertEqual(stage.ledger['entries'][-1]['status'],'failed')
        self.assertFalse(stage.ledger['entries'][-1]['solver_launched'])
    def test_worker_cannot_substitute_a_smaller_scheduling_forecast(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        stage=self.executable_stage();wrong=dict(pairs[0]['forecast'][0],worker_forecast_s=1.)
        with self.assertRaisesRegex(ValueError,'differs from prospective measured forecast'):
            stage.execute_adaptive_job(pairs[0]['jobs'][0],path,link['receipt_sha256'],wrong)
    def test_analysis_only_cache_recovery_does_not_allocate_another_attempt(self):
        stage=a.AdaptiveStage.__new__(a.AdaptiveStage);stage.verify=lambda:None
        entry=dict(method='NN',status='completed',analysis_status='failed');stage.ledger=dict(entries=[entry]);calls=[]
        stage.cache=lambda item:dict(stage_id='stage3',entry=entry)
        stage.analyze=lambda e:(calls.append('analysis'),e.update(analysis_status='completed'))
        with patch.object(a.subprocess,'Popen',side_effect=AssertionError('No new worker for cached analysis')):
            stage.execute_adaptive_job(dict(method='NN'),Path('not-used'),'',{})
        self.assertEqual(len(stage.ledger['entries']),1);self.assertEqual(calls,['analysis'])
    def test_adaptive_receipt_precedes_Popen_and_failed_attempt_is_charged(self):
        request=self.request();pairs=a.request_selection(request,self.stage);path,link=self.selection_receipt(request,pairs)
        stage=a.AdaptiveStage.__new__(a.AdaptiveStage);stage.__dict__.update(self.stage.__dict__)
        for name in ['verify','case','cache','save','elapsed','remaining']:
            setattr(stage,name,getattr(self.stage,name))
        stage.active=None
        def fail(args,**kwargs):
            entry=stage.ledger['entries'][-1];self.assertEqual(entry['id'],80)
            self.assertEqual(entry['phase'],'ADAPTIVE_ISOLATED_NUMERICAL_DIAGNOSTIC')
            self.assertEqual(entry['selection_receipt_sha256'],link['receipt_sha256'])
            self.assertIsNone(entry['design_sha256']);self.assertEqual(entry['paired_NN_case_id'],'NN_H100')
            raise OSError('Synthetic adaptive Popen failure')
        with patch.object(a.subprocess,'Popen',side_effect=fail):
            with self.assertRaisesRegex(OSError,'Synthetic adaptive'):stage.execute_adaptive_job(pairs[0]['jobs'][0],path,link['receipt_sha256'],pairs[0]['forecast'][0])
        self.assertEqual(len(stage.ledger['entries']),15);self.assertEqual(stage.ledger['entries'][-1]['status'],'failed')
    def test_default_inspection_creates_no_selection_or_worker(self):
        out=io.StringIO();before=set(self.folder.rglob('*'))
        with patch.object(sys,'argv',['adaptive']),patch.object(a,'AdaptiveStage',return_value=self.stage),\
             patch.object(a,'prepare_selection',side_effect=AssertionError('No implicit selection')),\
             patch.object(a,'run_next_pair',side_effect=AssertionError('No implicit pair')),patch('sys.stdout',out):a.main()
        self.assertEqual(json.loads(out.getvalue())['worker_launches'],0);self.assertEqual(before,set(self.folder.rglob('*')))
        self.assertEqual(a.sha(a.BASE_PATH),a.BASE_SHA)


if __name__=='__main__':unittest.main(verbosity=2)
