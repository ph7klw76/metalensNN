"""Pure metadata/CRC/fake-analyzer fixtures; no actual campaign recovery."""
import copy
from datetime import datetime,timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('attempt83_postprocess_test',ROOT/'scripts/recover_height_family_attempt83_postprocess_v1.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class TestPostprocess83(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name);self.stage=self.folder/'stage3';self.stage.mkdir()
    def bundle(self):
        cfg=dict(materials=dict(substrate=dict(n=1.45),polymer=dict(n=1.5263),ambient=dict(n=1.)),wavelength_um=1.55)
        settings=dict(H_um=100.,resolution=40.,pml_um=2.,padding_um=10.,duration=335.)
        case=dict(case_id='NN_H100',design_file='frozen_designs/NN_H100.json',design_sha256='a'*64,physical_geometry_id='b'*64)
        entries=[dict(id=i,status='completed',method='reference' if i%2==0 else 'NN',analysis_status='completed') for i in range(66,84)]
        ref=entries[-2];ref.update(settings,effective_settings=settings,record_file='records/82.json')
        entry=dict(id=83,status='started',method='NN',solver_launched=True,reference_id=82,reference_stage_id='stage3',case_id='NN_H100',
            design_sha256=case['design_sha256'],physical_geometry_id=case['physical_geometry_id'],effective_settings=settings,record_file='records/83.json',**settings)
        entry['command']=[str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
            '--config',str(self.stage/'config.json'),'--output',str(self.stage/'records/83.json'),'--H','100.0','--resolution','40.0',
            '--pml','2.0','--padding','10.0','--duration','335.0','--design',str(self.stage/case['design_file']),'--reference',str(self.stage/'records/82.json')]
        entries[-1]=entry
        common=dict(settings=settings,worker_sha256=r.WORKER_SHA,solver='CPU Meep 1.35.0',config_fingerprint=r.canonical_sha(cfg),
            requested_config=str(self.stage/'config.json'),input_index=1.45,polymer_index=1.5263,ambient_index=1.,wavelength_um=1.55)
        raw=dict(common,design_file=str(self.stage/case['design_file']),reference_file=str(self.stage/'records/82.json'))
        reference=dict(common,incident_power=800.,power_balance_error=.001)
        jobs=[dict(method=m,**settings) for m in ['reference','NN']]
        selection=dict(driver_sha256=r.DRIVER_SHA,adaptive_autoexecution=False,selected_pairs=[dict(jobs=jobs)])
        ledger=dict(entries=entries,prior_attempt_count=65,first_started_utc=r.START,absolute_deadline_utc=r.DEADLINE)
        return ledger,cfg,case,raw,reference,selection
    def test_exact_saved_bundle_and_real_executable_canonicalization_pass(self):
        data=self.bundle();before=copy.deepcopy(data[0]);entry,ref=r.validate_bundle(self.stage,*data)
        self.assertEqual(entry['id'],83);self.assertEqual(ref['id'],82);self.assertEqual(data[0],before)
    @unittest.skipUnless(os.name=='posix','Actual existing WSL venv symlink check')
    def test_actual_WSL_venv_python_symlink_resolves_both_sides(self):
        executable=ROOT/'.venv-meep/bin/python';self.assertTrue(executable.is_symlink())
        self.assertNotEqual(executable,r.local(executable))
        r.validate_bundle(self.stage,*self.bundle())
    def test_wrong_configuration_design_reference_source_or_numerics_are_rejected(self):
        for mutate in [lambda d:d[3].update(requested_config=str(self.stage/'wrong.json')),
                       lambda d:d[3].update(reference_file=str(self.stage/'records/other.json')),
                       lambda d:d[3].update(design_file=str(self.stage/'other.json')),
                       lambda d:d[3].update(worker_sha256='0'*64),lambda d:d[3].update(polymer_index=1.5),
                       lambda d:d[4].update(incident_power=0.),lambda d:d[0]['entries'][-1].update(resolution=48.),
                       lambda d:d[0]['entries'][-1]['command'].__setitem__(0,str(self.folder/'unrelated_python'))]:
            data=list(copy.deepcopy(self.bundle()));mutate(data)
            with self.assertRaises(ValueError):r.validate_bundle(self.stage,*data)
    def test_known_completion_or_existing_analysis_cannot_be_overwritten(self):
        for field,value in [('solver_exit_code',0),('finished_utc',r.DEADLINE),('analysis_file','exists.json'),('analysis_status','failed')]:
            data=list(self.bundle());data[0]['entries'][-1][field]=value
            with self.assertRaisesRegex(ValueError,'unfinalized83'):r.validate_bundle(self.stage,*data)
    def test_npz_allCRC_and_member_contract_without_arrays(self):
        path=self.folder/'fields.npz'
        with zipfile.ZipFile(path,'x') as z:z.writestr('r_um.npy',b'synthetic bytes, not loaded')
        proof=r.verify_npz(path);self.assertFalse(proof['arrays_loaded']);self.assertEqual(proof['members'],['r_um.npy'])
        bad=self.folder/'bad.npz'
        with zipfile.ZipFile(bad,'x') as z:z.writestr('../escape.npy',b'bad')
        with self.assertRaises(ValueError):r.verify_npz(bad)
    def test_only_exact_frozen_analyzer_command_is_allowed_no_worker(self):
        proof=dict(raw_path=self.stage/'records/83.json',target_diameter_um=16.22759437561035)
        command=r.analysis_command(proof,self.folder/'analysis.json')
        self.assertEqual(r.local(command[2]),r.local(r.ANALYZER));self.assertEqual(r.flag(command,'--count'),'512');self.assertEqual(r.flag(command,'--window'),'128.0')
        wrong=list(command);wrong[2]=str(ROOT/'scripts/height_family_maxwell_worker_v2.py')
        with patch.object(r.subprocess,'Popen',side_effect=AssertionError('No worker may launch')):
            with self.assertRaises(ValueError):r.run_analysis(wrong,wrong,self.folder/'log',{},600)
    def test_invalid_request_is_inert_before_any_process_or_campaign_access(self):
        with (patch.object(r,'read',side_effect=AssertionError('No actual campaign read')),
              patch.object(r.subprocess,'Popen',side_effect=AssertionError('No launch'))):
            with self.assertRaisesRegex(ValueError,'solver-free'):r.collect_proof({})
    def test_default_CLI_is_inert_no_recovery_or_process_queries(self):
        text=io.StringIO()
        with patch.object(sys,'argv',['postprocess']),patch.object(r,'recover',side_effect=AssertionError('No implicit recovery')),patch('sys.stdout',text):r.main()
        self.assertFalse(json.loads(text.getvalue())['analyzer_launched'])
    @unittest.skipUnless(os.name=='posix','Temporary shared-lock integration only in WSL')
    def test_fake_analysis_overlay_keeps_stale_ledger_unknown_exit_and_real_postprocess_time(self):
        campaign=self.folder/'campaign';stage=campaign/'maxwell_validation_v3';stage.mkdir(parents=True)
        old=campaign/'maxwell_validation_v1';old.mkdir();(old/'runner.lock').write_text('')
        output=campaign/'attempt83_postprocess_recovery_v1';ledger=stage/'ledger.json';ledger.write_text('{"entries":[{"id":83,"status":"started"}]}')
        raw=stage/'83.json';raw.write_text('{}');request=self.folder/'request.json';request.write_text('{}')
        proof=dict(ledger_sha256=r.sha(ledger),entry=dict(id=83,status='started',effective_settings=dict(H_um=100.)),
            raw_path=raw,raw_sha256=r.sha(raw),required={},context={},evidence=dict(raw_worker_wall_s=8470.77),
            target_diameter_um=16.22759437561035,protected={str(ledger):r.sha(ledger),str(raw):r.sha(raw)},raw_audit_file=str(request))
        request.write_text(json.dumps(dict(raw_completion_audit_sha256=r.sha(request))))
        def fake_analysis(command,expected,log,env,timeout):
            self.assertEqual(command,expected);self.assertEqual(r.local(command[2]),r.local(r.ANALYZER))
            r.write_new(Path(r.flag(command,'--output')),dict(record=str(raw),settings=proof['entry']['effective_settings'],declared_target_diameter_um=proof['target_diameter_um'],
                reconstruction_pass=False,synthetic_optical_pass=False));log.write_text('fake analyzer only')
            return dict(exit_code=0,elapsed_s=.01)
        before=ledger.read_bytes()
        with (patch.object(r,'STAGE',stage),patch.object(r,'CAMPAIGN',campaign),patch.object(r,'OUTPUT',output),
              patch.object(r,'collect_proof',return_value=proof),patch.object(r,'no_owned_processes',return_value=dict(status='PASS_SYNTHETIC_ZERO_PROCESS')),
              patch.object(r,'run_analysis',side_effect=fake_analysis),patch('sys.stdout',io.StringIO())):r.recover(request)
        overlay=r.read(output/'recovery_overlay.json');self.assertEqual(ledger.read_bytes(),before)
        self.assertEqual(overlay['original_entry83']['status'],'started')
        for field in ['solver_exit_code','worker_finished_utc','controller_finished_utc']:self.assertIsNone(overlay['attempt83_result'][field])
        self.assertTrue(overlay['postprocessing']['after_original_deadline']);self.assertEqual(overlay['postprocessing']['new_optical_solves'],0)
        self.assertFalse(overlay['attempt83_result']['nominal_optical_pass']);self.assertEqual(overlay['status'],'COMPLETE_FROM_SAVED_ARTIFACTS_POSTPROCESS_ONLY')
        manifest=r.read(output/'recovery_manifest.json')
        self.assertTrue(all(r.sha(output/name)==value['sha256'] for name,value in manifest['files'].items()))
if __name__=='__main__':unittest.main(verbosity=2)
