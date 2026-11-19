"""Pure saved-evidence packaging guards; never use real stage/seal/solver."""
import copy
import base64
import csv
from datetime import datetime,timedelta,timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_seal_test',ROOT/'scripts/seal_height_family_stage3_v1.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class TestStage3Seal(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)/'campaign/maxwell_validation_v3';self.folder.mkdir(parents=True)
        self.at=s.utc(s.START)+timedelta(hours=2)
    def clean_windows(self):
        return dict(status='VERIFIED_FILTERED_WINDOWS_PYTHON_PROCESS_SCAN',native_windows_host_queried=True,
            all_candidate_command_lines_readable=True,raw_command_lines_returned=False,matches=[])
    def ledger(self,count=14):
        entries=[]
        for i in range(count):
            started=(s.utc(s.START)+timedelta(seconds=10+i*2)).isoformat()
            finished=(s.utc(s.START)+timedelta(seconds=11+i*2)).isoformat()
            entry=dict(id=66+i,method='reference' if i%2==0 else 'NN',status='completed',started_utc=started,finished_utc=finished)
            if entry['method']=='NN':entry.update(analysis_status='completed',analysis_attempts=[dict(status='completed',started_utc=started,finished_utc=finished)])
            entries.append(entry)
        return dict(entries=entries,status='PRIMARY_EXECUTED_REQUIRES_ASSESSMENT_ADAPTIVE_UNSELECTED',
            prior_attempt_count=65,first_started_utc=s.START,original_first_started_utc=s.START,
            absolute_deadline_utc=s.DEADLINE,elapsed_s=100.)
    def decision(self,count=14):
        return dict(unused_additional_solve_slots=18-count,closing_with_unused_slots=count<18,
            unused_slot_reason='ROOT reviewed saved results and remaining original time; no further selected work',no_more_optical_calls_under_this_stage=True)
    def final_report(self,ledger=None):
        ledger=self.ledger() if ledger is None else ledger
        (self.folder/'ledger.json').write_text(json.dumps(ledger))
        inputs={}
        for name in ['config.json','plan.json','source_manifest.json','preparation_manifest.json','frozen_cases.json']:
            path=self.folder/name;path.write_text('{}');inputs[str(path.resolve())]=s.sha(path)
        for entry in ledger['entries']:
            record=self.folder/f"record-{entry['id']}.json";record.write_text('{}')
            native=self.folder/f"native-{entry['id']}.bin";native.write_bytes(b'native fields fixture')
            entry.update(record_file=record.name,record_sha256=s.sha(record),required_file_sha256={str(native.resolve()):s.sha(native)})
            inputs[str(record.resolve())]=s.sha(record);inputs[str(native.resolve())]=s.sha(native)
            if entry['method']=='NN':
                analysis=self.folder/f"analysis-{entry['id']}.json";analysis.write_text('{}')
                entry.update(analysis_file=analysis.name,analysis_sha256=s.sha(analysis));inputs[str(analysis.resolve())]=s.sha(analysis)
        (self.folder/'ledger.json').write_text(json.dumps(ledger))
        rows=[]
        for i,H in enumerate(s.HEIGHTS):
            entry=next(e for e in ledger['entries'] if e['method']=='NN');design=self.folder/f'design-{H:g}.json';design.write_text('{}');inputs[str(design.resolve())]=s.sha(design)
            record=dict(design_sha256=s.sha(design),eta=.9,C=.01,Q=1.01,planes=[dict(z_um=0.,diameter_x_um=15.,diameter_y_um=14.9)],
                core_half_angles_deg=[2.,2.1])
            proxy=copy.deepcopy(record);proxy.update(design_file=str(design))
            measured=copy.deepcopy(record);measured.update(identity=self.folder.name+':'+str(entry['id']),resolution=32.,
                record_file=str(self.folder/entry['record_file']),record_sha256=entry['record_sha256'],
                analysis_file=str(self.folder/entry['analysis_file']),analysis_sha256=entry['analysis_sha256'],settings=dict(output_z_um=H+4.))
            rows.append(dict(H_um=H,height_use='validation-observed' if H==75. else 'training height',proxy=proxy,Meep=measured,
                ideal_declared_Gaussian=[],selection_rule='finest verified',same_geometry=True,unfinished_finer_attempts=[],
                native=dict(status='PASS'),nominal_optical=dict(status='FAIL'),selected_mesh=dict(status='FAIL',resolutions=[24.,32.])))
        report=dict(status='SAVED_THREE_STAGE_ASSESSMENT',created_utc=(self.at-timedelta(seconds=30)).isoformat(),
            assessment_schema='height_family_stage3_v2',assessment_utility_sha256=s.ASSESSOR_SHA,
            assessment_core_sha256=s.CORE_SHA,scientific_source_sha256=s.SCIENCE_SHA,
            scientific_function_bytecode_changed=False,reporting_height_enumeration=list(s.HEIGHTS),
            execution_completion_is_not_optical_acceptance=True,authorization=dict(status='PASS'),
            budget=dict(status='PASS',additional_attempts=len(ledger['entries']),cumulative_attempts=65+len(ledger['entries']),
                actual_cumulative_cases=25,reserved_case_ceiling=28,original_first_started_utc=s.START,absolute_deadline_utc=s.DEADLINE,checks=dict(all_idle_time_counts=True)),
            primary_execution=dict(status='PASS',required=14,completed_verified=14,
                primary_jobs=[dict(proposed_id=i,completed_verified=True,status='completed') for i in range(66,80)]),
            adaptive_execution=dict(status='PASS',pending_current_selected_pairs=[],started_pair_runs=[]),
            continuation_execution=dict(status='PASS'),selected_adaptive_diagnostics=[],
            ledger_snapshot_sha256={str((self.folder/'ledger.json').resolve()):s.sha(self.folder/'ledger.json')},
            input_sha256=inputs,analysis_source_sha256={str((ROOT/name).resolve()):digest for name,digest in s.REVIEWED_SOURCES.items()
                if name.endswith(('assess_height_family_stage3_v2.py','assess_height_family_stage3_v1.py','collect_height_family_stage3_v2.py'))},
            NN_heights=rows,
            original_seven_height_family_acceptance=dict(status='FAIL'),all_eight_nominal_heights_acceptance=dict(status='FAIL'),boundary_duration_checks=[])
        path=self.folder/'assessment/final_fixture/assessment.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report))
        return ledger,path,report
    def graph_fixture(self):
        ledger,path,report=self.final_report();canonical=self.folder.parent/'training_result.json';canonical.write_text('{"training":"canonical"}')
        alias=self.folder/'training_result.json';alias.write_bytes(canonical.read_bytes());digest=s.sha(canonical)
        prior=self.folder.parent/'maxwell_validation_v1';prior.mkdir(exist_ok=True)
        proposal=prior/'maxwell_validation_proposal.json';proposal.write_text(json.dumps(dict(numerical_acceptance=dict(curvature_delta=.005))))
        linkage=self.folder/'prior_stages.json';linkage.write_text(json.dumps(dict(stages=[dict(stage_id='stage1',folder=str(prior))])))
        for p in [canonical,alias,proposal,linkage]:report['input_sha256'][str(p.resolve())]=s.sha(p)
        path.write_text(json.dumps(report))
        comparison=dict(status='DESCRIPTIVE_SAME_NN_GEOMETRY_PROXY_VS_DIRECT_MAXWELL',utility_sha256=s.SCIENCE_SHA,
            publication_layout_sha256=s.LAYOUT_SHA,stage3_publication_adapter_sha256=s.PLOT_SHA,stage3_collector_sha256=s.COLLECTOR_SHA,
            partial_report_reason=None,rows=[copy.deepcopy(r) for r in report['NN_heights'] if r['H_um'] in (40.,75.,100.)],
            ledger_snapshot_sha256=copy.deepcopy(report['ledger_snapshot_sha256']),input_sha256=copy.deepcopy(report['input_sha256']),
            config=s.read(self.folder/'config.json'),numerical_limits=s.read(proposal)['numerical_acceptance'],
            full_optical_acceptance_claimed=False,NN_superiority_claimed=False,global_diffraction_optimum_claimed=False,experimental_validation_claimed=False)
        for key in ['authorization','budget','primary_execution','adaptive_execution','selected_adaptive_diagnostics']:comparison[key]=copy.deepcopy(report[key])
        comparison['CSV_training_result_lineage']=dict(status='VERIFIED_IDENTICAL_TRAINING_RESULT_ALIASES_FOR_CSV_ONLY',
            canonical_file=str(canonical.resolve()),canonical_sha256=digest,
            authenticated_aliases=[dict(input_inventory_key=str(p.resolve()),resolved_file=str(p.resolve()),sha256=digest) for p in [canonical,alias]],
            full_comparison_and_delivery_input_inventory_preserved=True,scientific_records_and_numeric_CSV_mapping_unchanged=True)
        output=self.folder/'forward_comparison/final_fixture';output.mkdir(parents=True,exist_ok=True)
        (output/'comparison.json').write_text(json.dumps(comparison))
        columns,mapper=s.frozen_csv_mapping();projected=dict(comparison,input_sha256={str(canonical.resolve()):digest})
        science=SimpleNamespace(finite=lambda v:isinstance(v,(int,float)) and not isinstance(v,bool),close=lambda x,y:x==y)
        with (output/'proxy_vs_selected_Meep.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(mapper(projected,science))
        for name in ['REPORT.txt','beam_propagation.png','beam_propagation.svg','beam_propagation.pdf','forward_model_metrics.png','forward_model_metrics.svg','forward_model_metrics.pdf']:
            (output/name).write_bytes(b'completed graph fixture')
        manifest=dict(status='STAGE3_DESCRIPTIVE_PUBLICATION_NO_OPTICAL_RUN',scientific_source_sha256=s.SCIENCE_SHA,
            publication_layout_sha256=s.LAYOUT_SHA,stage3_adapter_sha256=s.PLOT_SHA,scientific_logic_changed=False,
            optical_solves_launched=0,training_runs_launched=0,input_sha256=copy.deepcopy(comparison['input_sha256']),
            CSV_training_result_lineage=copy.deepcopy(comparison['CSV_training_result_lineage']))
        self.refresh_graph(output,manifest)
        return ledger,path,report,output,comparison,manifest
    def refresh_graph(self,output,manifest):
        manifest['files']={p.name:dict(sha256=s.sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file() and p.name!='delivery_manifest.json'}
        (output/'delivery_manifest.json').write_text(json.dumps(manifest))
    def test_actual_attempts_unused_slots_caps_and_interruption_are_truthful(self):
        for count in (14,16,18):
            result=s.closed_ledger(self.ledger(count),self.decision(count),at=self.at)
            self.assertEqual(result['actual_new_attempts'],count);self.assertEqual(result['cumulative_attempts'],65+count)
            self.assertEqual(result['cumulative_completed'],64+count);self.assertEqual(result['unused_additional_solve_slots'],18-count)
            self.assertEqual(result['historical_interrupted_attempt_id'],41)
        with self.assertRaises(ValueError):s.closed_ledger(self.ledger(19),self.decision(19),at=self.at)
        with self.assertRaises(ValueError):s.closed_ledger(self.ledger(),self.decision(18),at=self.at)
    def test_absolute_idle_counts_and_late_packaging_does_not_reset_experiment_clock(self):
        result=s.closed_ledger(self.ledger(),self.decision(),at=self.at)
        self.assertEqual(result['absolute_elapsed_s_including_idle'],7200.)
        late=s.utc(s.DEADLINE)+timedelta(hours=1)
        result=s.closed_ledger(self.ledger(),self.decision(),at=late)
        self.assertEqual(result['absolute_elapsed_s_including_idle'],176400.)
        self.assertTrue(result['packaging_after_optical_deadline'])
        ledger=self.ledger();ledger['entries'][-1]['finished_utc']=late.isoformat()
        with self.assertRaisesRegex(ValueError,'attempt exceeded'):s.closed_ledger(ledger,self.decision(),at=late)
    def test_clock_reset_noncontiguous_failed_primary_and_started_analyzer_are_rejected(self):
        for mutate in [lambda l:l.update(first_started_utc=self.at.isoformat()),
                       lambda l:l['entries'][0].update(id=65),lambda l:l['entries'][0].update(status='failed'),
                       lambda l:l['entries'][-1].update(analysis_status='started'),
                       lambda l:l['entries'][-1]['analysis_attempts'][0].update(status='started'),
                       lambda l:l.update(status='RUNNING_APPROVED_PRIMARY_ONLY')]:
            ledger=self.ledger();mutate(ledger)
            with self.assertRaises(ValueError):s.closed_ledger(ledger,self.decision(),at=self.at)
    def test_naive_nonUTC_and_future_completion_timestamps_are_rejected(self):
        for timestamp in ['2026-11-13T09:01:00','2026-11-13T09:01:00+08:00']:
            ledger=self.ledger();ledger['entries'][0]['started_utc']=timestamp
            with self.assertRaisesRegex(ValueError,'aware UTC'):s.closed_ledger(ledger,self.decision(),at=self.at)
        ledger=self.ledger();ledger['entries'][-1]['finished_utc']=(self.at+timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):s.closed_ledger(ledger,self.decision(),at=self.at)
        ledger=self.ledger();ledger['entries'][-1]['analysis_attempts'][0]['finished_utc']=(self.at+timedelta(seconds=1)).isoformat()
        with self.assertRaises(ValueError):s.closed_ledger(ledger,self.decision(),at=self.at)
    def test_owned_process_identity_is_command_based_not_reused_PID(self):
        unrelated=dict(pid=123,cwd='/tmp',argv=['python','/tmp/height_family_maxwell_worker_v2.py'])
        s.require_no_owned_processes([unrelated],self.clean_windows())
        for script in ['run_height_family_maxwell_v3.py','run_height_family_adaptive_v1.py','height_family_maxwell_worker_v2.py','height_family_maxwell_analysis_v2.py','plot_height_family_stage3_v3.py']:
            with self.assertRaisesRegex(RuntimeError,'still active'):
                s.require_no_owned_processes([dict(pid=123,cwd=str(ROOT),argv=['python','-B',str(ROOT/'scripts'/script)])],self.clean_windows())
        with self.assertRaises(RuntimeError):s.require_no_owned_processes([dict(pid=124,cwd=str(ROOT),argv=['python','scripts/run_height_family_maxwell_v3.py'])],self.clean_windows())
        for module in ['scripts.plot_height_family_stage3_v3','plot_height_family_stage3_v3']:
            with self.assertRaisesRegex(RuntimeError,'module remains active'):
                s.require_no_owned_processes([dict(pid=125,cwd=str(ROOT),argv=['python','-m',module])],self.clean_windows())
    def test_native_Windows_plot_and_unavailable_proof_block_without_private_output(self):
        proof=self.clean_windows();proof['matches']=[dict(pid=321,script_names=['plot_height_family_stage3_v3.py'],workspace_match=True)]
        with self.assertRaisesRegex(RuntimeError,'Native Windows project Python process remains active'):
            s.require_no_owned_processes([],proof)
        proof['matches'][0]['workspace_match']=False
        with self.assertRaisesRegex(RuntimeError,'Native Windows project Python process remains active'):
            s.require_no_owned_processes([],proof)
        proof=self.clean_windows();proof['all_candidate_command_lines_readable']=False
        with self.assertRaisesRegex(RuntimeError,'unavailable or malformed'):s.require_no_owned_processes([],proof)
        with patch.object(s,'windows_processes',side_effect=RuntimeError('Native Windows process proof unavailable')):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):s.require_no_owned_processes([])
    def test_native_process_query_is_filtered_mocked_and_no_raw_commands_escape(self):
        proof=self.clean_windows();calls=[]
        def fake_run(args,**kwargs):
            calls.append((args,kwargs));return SimpleNamespace(returncode=0,stdout=json.dumps(proof),stderr='')
        with patch.object(s.subprocess,'run',side_effect=fake_run):result=s.windows_processes(s.PROCESS_SCRIPTS)
        self.assertEqual(result,proof);self.assertEqual(len(calls),1)
        script=base64.b64decode(calls[0][0][-1]).decode('utf-16-le')
        self.assertIn('Get-CimInstance Win32_Process -Filter',script)
        self.assertIn('plot_height_family_stage3_v3.py',script)
        self.assertIn('raw_command_lines_returned=$false',script)
        with patch.object(s.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='{}',stderr='UNRELATED_PRIVATE_COMMAND')):
            with self.assertRaises(RuntimeError) as caught:s.windows_processes(s.PROCESS_SCRIPTS)
        self.assertNotIn('UNRELATED_PRIVATE_COMMAND',str(caught.exception))
    def test_safe_paths_member_duplicates_and_environment_exclusions(self):
        with self.assertRaises(ValueError):s.contained(self.folder,'../outside')
        for name in ['/absolute','../escape','stage/../escape','stage\\wrong','stage//duplicate','C:/absolute','stage/null\0']:
            with self.assertRaises(ValueError):s.membersafe(name)
        members={};path=self.folder/'x';path.write_text('x');s.add_member(members,'stage/x',path,s.sha(path))
        with self.assertRaisesRegex(ValueError,'Duplicate'):s.add_member(members,'stage/x',path,s.sha(path))
        bad=self.folder/'.venv-meep';bad.mkdir();(bad/'x').write_text('x')
        with self.assertRaisesRegex(ValueError,'Environment/cache'):s.file_inventory(self.folder)
    def test_exact_zip_members_crc_and_each_SHA_are_verified_and_no_overwrite(self):
        payload=self.folder/'evidence.json';payload.write_text('{"value":1}')
        members={'stage/evidence.json':(payload,s.sha(payload))};archive=self.folder/'test.zip'
        result=s.package_archive(archive,members);self.assertEqual(result['members_verified'],1)
        with self.assertRaises(FileExistsError):s.package_archive(archive,members)
        other=self.folder/'extra.zip'
        with zipfile.ZipFile(other,'x') as bundle:
            bundle.writestr('stage/evidence.json',payload.read_bytes());bundle.writestr('extra','unrelated')
        with self.assertRaisesRegex(ValueError,'membership/CRC'):s.verify_archive(other,members)
        duplicate=self.folder/'duplicate.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(duplicate,'x') as bundle:
                bundle.writestr('stage/evidence.json',payload.read_bytes());bundle.writestr('stage/evidence.json',payload.read_bytes())
        with self.assertRaisesRegex(ValueError,'membership/CRC'):s.verify_archive(duplicate,members)
        wrong=self.folder/'wrong.zip'
        with zipfile.ZipFile(wrong,'x') as bundle:bundle.writestr('stage/evidence.json','same name, different data')
        with self.assertRaisesRegex(ValueError,'payload SHA differs'):s.verify_archive(wrong,members)
    def test_prearchive_input_change_is_detected(self):
        payload=self.folder/'data';payload.write_text('first');members={'stage/data':(payload,s.sha(payload))};payload.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed before archive'):s.package_archive(self.folder/'changed.zip',members)
    def test_prior_manifest_every_payload_archive_and_interrupted41_are_preserved(self):
        stages=[];pins={}
        for sid,count in [('stage1',45),('stage2',20)]:
            origin=self.folder/sid;origin.mkdir();payload=origin/'raw.bin';payload.write_bytes(b'preserved optical evidence')
            entries=[dict(id=i,status='interrupted' if sid=='stage1' and i==41 else 'completed') for i in range(1,count+1)]
            ledger=origin/'ledger.json';ledger.write_text(json.dumps(dict(entries=entries)))
            manifest=origin/'delivery_manifest.json';manifest.write_text(json.dumps(dict(files={p.name:dict(sha256=s.sha(p),bytes=p.stat().st_size) for p in [ledger,payload]})))
            archive=self.folder/(sid+'.zip');archive.write_bytes(b'prior sealed archive fixture')
            pins[sid]=dict(manifest=s.sha(manifest),ledger=s.sha(ledger),archive=s.sha(archive),attempts=count)
            stages.append(dict(stage_id=sid,folder=str(origin),archive_file=str(archive),ledger_sha256=s.sha(ledger),
                delivery_manifest_sha256=s.sha(manifest),archive_sha256=s.sha(archive)))
        (self.folder/'prior_stages.json').write_text(json.dumps(dict(prior_attempt_count=65,actual_prior_physical_cases=25,reserved_case_ceiling=28,stages=stages)))
        with patch.object(s,'PRIOR_PINS',pins):
            result=s.verify_priors(self.folder);self.assertEqual(sum(row['payload_files'] for row in result),4)
            payload=self.folder/'stage1/raw.bin';payload.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'delivered payload changed'):s.verify_priors(self.folder)
            payload.write_bytes(b'preserved optical evidence');(self.folder/'stage2.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'seal/archive changed'):s.verify_priors(self.folder)
    def test_additive_metadata_cannot_overwrite_an_existing_final_manifest(self):
        path=self.folder/'delivery_manifest.json';s.write_new(path,dict(status='preserve'))
        with self.assertRaises(FileExistsError):s.write_new(path,dict(status='replace'))
        self.assertEqual(s.read(path)['status'],'preserve')
    def test_optical_failures_remain_explicit_despite_completed_execution(self):
        ledger,path,report=self.final_report();s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        summary=s.optical_summary(report)
        self.assertEqual(summary['original_seven_height_family'],'FAIL')
        self.assertFalse(summary['execution_completion_is_optical_acceptance'])
        self.assertTrue(all(row['native']=='PASS' and row['nominal']=='FAIL' and row['mesh']=='FAIL' for row in summary['nominal_native_mesh']))
        report['NN_heights'][0]['selected_mesh']['status']='UNRESOLVED';path.write_text(json.dumps(report))
        s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        self.assertEqual(s.optical_summary(report)['nominal_native_mesh'][0]['mesh'],'UNRESOLVED')
    def test_stale_assessment_and_missing_source_input_inventory_are_rejected(self):
        ledger,path,report=self.final_report();ledger['note']='changed';(self.folder/'ledger.json').write_text(json.dumps(ledger))
        with self.assertRaisesRegex(ValueError,'ledger snapshot is stale'):s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        ledger,path,report=self.final_report();report['analysis_source_sha256'].pop(next(iter(report['analysis_source_sha256'])))
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'source inventory differs'):s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        ledger,path,report=self.final_report();del report['input_sha256'][str((self.folder/'config.json').resolve())];path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'omits immutable'):s.verify_assessment(self.folder,path,ledger,{},at=self.at)
    def test_latest_reviewed_partial_report_cannot_be_bypassed_with_earlier_complete_report(self):
        ledger,path,report=self.final_report();later=self.folder/'assessment/later/assessment.json';later.parent.mkdir()
        other=copy.deepcopy(report);other['created_utc']=(self.at-timedelta(seconds=10)).isoformat();other['primary_execution']['status']='UNRESOLVED';later.write_text(json.dumps(other))
        with self.assertRaisesRegex(ValueError,'latest reviewed'):s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        with self.assertRaisesRegex(ValueError,'incomplete: primary_execution'):s.verify_assessment(self.folder,later,ledger,{},at=self.at)
    def test_final_graph_metadata_delivery_full_inventory_and_CSV_are_authenticated(self):
        ledger,path,report,output,comparison,manifest=self.graph_fixture()
        for name in ['comparison.json','delivery_manifest.json']:
            result=s.verify_graph_artifact(self.folder,output/name,ledger,report)
            self.assertEqual(result['verified_export_files'],9)
        with self.assertRaisesRegex(ValueError,'comparison.json or'):s.verify_graph_artifact(self.folder,output/'beam_propagation.png',ledger,report)
    def test_stale_partial_graph_cannot_be_sealed_with_a_fresh_final_assessment(self):
        ledger,path,report,output,comparison,manifest=self.graph_fixture()
        comparison['primary_execution'].update(status='UNRESOLVED',completed_verified=4)
        comparison['ledger_snapshot_sha256'][str((self.folder/'ledger.json').resolve())]='0'*64
        (output/'comparison.json').write_text(json.dumps(comparison));self.refresh_graph(output,manifest)
        with self.assertRaisesRegex(ValueError,'incomplete current primary_execution'):s.verify_graph_artifact(self.folder,output/'comparison.json',ledger,report)
        comparison['primary_execution']=copy.deepcopy(report['primary_execution'])
        (output/'comparison.json').write_text(json.dumps(comparison));self.refresh_graph(output,manifest)
        with self.assertRaisesRegex(ValueError,'ledger snapshot differs'):s.verify_graph_artifact(self.folder,output/'delivery_manifest.json',ledger,report)
    def test_graph_missing_row_fields_wrong_status_config_thresholds_and_claims_are_rejected(self):
        for mutate in [lambda c:c['rows'][0].pop('native'),
                       lambda c:c['rows'][0]['selected_mesh'].update(status='PASS'),
                       lambda c:c['rows'][0]['Meep'].update(record_sha256='0'*64),
                       lambda c:c.update(config=dict(changed_Q_gate=2.)),
                       lambda c:c.update(numerical_limits=dict(curvature_delta=1.)),
                       lambda c:c.update(full_optical_acceptance_claimed=True),
                       lambda c:c['input_sha256'].pop(next(k for k in c['input_sha256'] if k.endswith('training_result.json')))]:
            ledger,path,report,output,comparison,manifest=self.graph_fixture();mutate(comparison)
            (output/'comparison.json').write_text(json.dumps(comparison));self.refresh_graph(output,manifest)
            with self.assertRaises(ValueError):s.verify_graph_artifact(self.folder,output/'comparison.json',ledger,report)
    def test_graph_training_alias_proof_must_cover_all_authenticated_aliases(self):
        ledger,path,report,output,comparison,manifest=self.graph_fixture()
        comparison['CSV_training_result_lineage']['authenticated_aliases'].pop()
        manifest['CSV_training_result_lineage']=copy.deepcopy(comparison['CSV_training_result_lineage'])
        (output/'comparison.json').write_text(json.dumps(comparison));self.refresh_graph(output,manifest)
        with self.assertRaisesRegex(ValueError,'omits an authenticated training alias'):
            s.verify_graph_artifact(self.folder,output/'comparison.json',ledger,report)
    def test_graph_CSV_numeric_status_units_and_source_changes_fail_even_after_manifest_rehash(self):
        for column,value in [('source_training_record_file',str(self.folder/'wrong-training.json')),
                             ('Gaussian_core_half_angle_x_deg','0.034906585'),
                             ('useful_efficiency_percent','0.9'),('nominal_optical_status','PASS')]:
            ledger,path,report,output,comparison,manifest=self.graph_fixture();csv_path=output/'proxy_vs_selected_Meep.csv'
            with csv_path.open(newline='',encoding='utf-8') as stream:reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
            rows[1][column]=value
            with csv_path.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
            self.refresh_graph(output,manifest)
            with self.assertRaisesRegex(ValueError,'CSV source/numeric/status/unit mapping differs'):
                s.verify_graph_artifact(self.folder,output/'comparison.json',ledger,report)
    def test_graph_CSV_path_columns_accept_equivalent_Windows_WSL_paths(self):
        ledger,path,report,output,comparison,manifest=self.graph_fixture();canonical=self.folder.parent/'training_result.json'
        native=str((ROOT/'scripts/plot_height_family_stage3_v3.py').resolve())
        alternate='/mnt/'+native[0].lower()+'/'+native[3:].replace('\\','/') if os.name=='nt' else native[5].upper()+':\\'+native[7:].replace('/','\\')
        comparison['rows'][0]['proxy']['design_file']=native
        columns,mapper=s.frozen_csv_mapping();science=SimpleNamespace(finite=lambda v:isinstance(v,(int,float)),close=lambda x,y:x==y)
        rows=mapper(dict(comparison,input_sha256={str(canonical.resolve()):s.sha(canonical)}),science);rows[0]['design_file']=alternate
        csv_path=output/'proxy_vs_selected_Meep.csv'
        with csv_path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        s.verify_graph_csv(csv_path,comparison,s.normalized_inventory(comparison['input_sha256']),canonical.resolve(),s.sha(canonical))
    def test_selected_but_unstarted_pair_cannot_be_discarded_by_root_closure(self):
        ledger,path,report=self.final_report();receipt=self.folder/'selected.json';receipt.write_text(json.dumps(dict(selected_pairs=[dict(pair_id='must-finish')])))
        ledger['current_adaptive_selection']=dict(receipt_file=receipt.name,receipt_sha256=s.sha(receipt));(self.folder/'ledger.json').write_text(json.dumps(ledger))
        report['ledger_snapshot_sha256'][str((self.folder/'ledger.json').resolve())]=s.sha(self.folder/'ledger.json');path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'Unstarted selected pair'):s.verify_assessment(self.folder,path,ledger,{},at=self.at)
    def test_selected_diagnostic_failures_can_be_delivered_without_becoming_optical_pass(self):
        ledger=self.ledger(16)
        for entry in ledger['entries'][14:]:entry.update(pair_id='done',selection_receipt_sha256='0'*64)
        ledger,path,report=self.final_report(ledger);run=dict(pair_id='done',selection_sha256='0'*64,status='completed');ledger['adaptive_pair_runs']=[run]
        (self.folder/'ledger.json').write_text(json.dumps(ledger));report['ledger_snapshot_sha256'][str((self.folder/'ledger.json').resolve())]=s.sha(self.folder/'ledger.json')
        report['adaptive_execution']['started_pair_runs']=[dict(pair_id='done',selection_sha256='0'*64,status='PASS')]
        report['selected_adaptive_diagnostics']=[dict(pair_id='done',receipt_sha256='0'*64,target_identity=self.folder.name+':81',
            native_status='PASS',numerical_status='FAIL',target_nominal_status='FAIL')]
        path.write_text(json.dumps(report));s.verify_assessment(self.folder,path,ledger,{},at=self.at)
        self.assertEqual(s.optical_summary(report)['selected_adaptive_diagnostics'][0]['numerical_status'],'FAIL')
    def test_default_cli_is_inert_and_never_inspects_or_seals(self):
        out=io.StringIO()
        with (patch.object(sys,'argv',['seal-tool']),patch.object(s,'seal',side_effect=AssertionError('No implicit sealing')),
              patch.object(s,'inspect_decision',side_effect=AssertionError('No implicit preparation')),patch('sys.stdout',out)):s.main()
        result=json.loads(out.getvalue());self.assertFalse(result['seal_created']);self.assertFalse(result['archive_created'])
        self.assertEqual(result['optical_solves_launched'],0)
    @unittest.skipUnless(os.name=='posix','Shared-lock integration uses temporary WSL fixtures only')
    def test_temporary_package_manifest_is_final_tree_write_and_report_selfSHA_is_external(self):
        # Exercise output ordering using a synthetic proof; never inspect actual
        # campaign data. Guard semantics are covered separately above.
        fake_root=self.folder/'project';fake_root.mkdir();campaign=fake_root/'campaign';campaign.mkdir()
        folder=campaign/'maxwell_validation_v3';folder.mkdir();prior=campaign/'maxwell_validation_v1';prior.mkdir()
        (prior/'runner.lock').write_text('')
        ledger=folder/'ledger.json';ledger.write_text('{"status":"closed fixture"}')
        payload=folder/'raw.bin';payload.write_bytes(b'closed evidence fixture')
        (fake_root/'scripts').mkdir();(fake_root/'tests').mkdir()
        script=fake_root/'scripts'/Path(s.__file__).name;script.write_bytes(Path(s.__file__).read_bytes())
        testsource=fake_root/'tests/test_height_family_stage3_seal_v1.py';testsource.write_text('# fixture test snapshot\n')
        decision=campaign/'root_decision.json';decision.write_text(json.dumps(dict(seal_utility_sha256=s.sha(s.__file__))))
        _,_,report=self.final_report()
        proof=dict(initial_files=s.file_inventory(folder),source_pins={'scripts/'+script.name:s.sha(script)},
            decision_sha256=s.sha(decision),decision=s.read(decision),ledger_sha256=s.sha(ledger),ledger=dict(status='closed fixture'),
            assessment_file='fixture saved report',assessment_sha256='1'*64,
            accounting=dict(actual_new_attempts=14,cumulative_attempts=79,unused_additional_solve_slots=4),report=report,priors=[])
        writes=[];original_write=s.write_new
        def observed_write(path,data):
            writes.append(Path(path));return original_write(path,data)
        with (patch.object(s,'ROOT',fake_root),patch.object(s,'CAMPAIGN',campaign),patch.object(s,'FOLDER',folder),
              patch.object(s,'inspect_decision',return_value=proof),patch.object(s,'require_no_owned_processes',return_value=None),
              patch.object(s,'write_new',side_effect=observed_write),patch('sys.stdout',io.StringIO())):
            s.seal(decision)
        manifest=folder/'delivery_manifest.json';index=writes.index(manifest)
        self.assertTrue(all(not path.is_relative_to(folder) for path in writes[index+1:]))
        self.assertEqual(ledger.read_text(),'{"status":"closed fixture"}');self.assertEqual(payload.read_bytes(),b'closed evidence fixture')
        verification=campaign/'height_specific_Maxwell_continuation_v3_evidence.verification.json'
        self.assertEqual(Path(str(verification)+'.sha256').read_text().split()[0],s.sha(verification))
        self.assertFalse(s.read(verification)['optical_outcomes']['execution_completion_is_optical_acceptance'])
        self.assertEqual(s.read(verification)['stage_tree_writes_after_manifest'],0)
        expected=set(s.read(manifest)['files'])|{'delivery_manifest.json'}
        self.assertEqual(set(s.file_inventory(folder)),expected)

if __name__=='__main__':unittest.main(verbosity=2)
