"""Pure bounded-controller fixtures. No Meep/JAX or real subprocess calls."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest import mock
import copy
import importlib.util
import json
import tempfile
import types
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_frozen_NN_500nm_meep_v2.py'
spec=importlib.util.spec_from_file_location('slc500_runner',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def approval():
    return {'approved_utc':'2026-11-17T00:00:00+00:00','absolute_deadline_utc':'2026-11-17T08:00:00+00:00'}


class ControllerTests(unittest.TestCase):
    def test_exact_six_devices_no_reference_fallback(self):
        jobs=m.plan();self.assertEqual([j['id'] for j in jobs],list(range(84,90)))
        self.assertEqual([(j['H_um'],j['resolution'],j['duration']) for j in jobs],
                         [(40,24,160),(40,32,160),(75,24,259),(75,32,259),(100,24,335),(100,32,335)])
        self.assertTrue(all(j['method']=='NN' for j in jobs))
        bad=copy.deepcopy(jobs[0]);bad['resolution']=40
        with self.assertRaises(ValueError):m.exact_job(bad)
        bad=copy.deepcopy(jobs[0]);bad['method']='reference'
        with self.assertRaises(ValueError):m.exact_job(bad)

    def test_absolute_clock_counts_idle_and_cleanup_separately(self):
        auth=approval()
        self.assertEqual(m.remaining(auth,'2026-11-17T04:00:00+00:00'),4*3600-185)
        self.assertEqual(m.capacity(5,auth,'2026-11-17T00:00:00+00:00'),21600)
        with self.assertRaises(RuntimeError):m.capacity(6,auth,'2026-11-17T00:00:00+00:00')
        with self.assertRaises(RuntimeError):m.capacity(0,auth,'2026-11-17T08:00:00+00:00')
        changed=copy.deepcopy(auth);changed['absolute_deadline_utc']='2026-11-18T08:00:00+00:00'
        with self.assertRaises(ValueError):m.remaining(changed,'2026-11-17T01:00:00+00:00')

    def test_unapproved_run_cannot_launch(self):
        with mock.patch.object(m,'verify',return_value={'approved':False}),mock.patch.object(m.subprocess,'Popen') as process:
            with self.assertRaises(RuntimeError):m.Runner()
            process.assert_not_called()

    def test_exact_current_question_rejects_old_generic_question(self):
        m.exact_budget_question(m.PENDING_QUESTION.read_bytes())
        with self.assertRaises(ValueError):m.exact_budget_question(b'Approve the old18solve continuation?')

    def test_expired_immediately_before_popen_launches_nothing(self):
        runner=m.Runner.__new__(m.Runner);runner.active=None;runner.approval=approval()
        runner.save=mock.Mock()
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'remaining',return_value=-1),\
             mock.patch.object(m.subprocess,'Popen') as process:
            with self.assertRaises(RuntimeError):runner.call(['fake-worker'],Path(tmp)/'log.txt',600,{},False)
            process.assert_not_called()

    def test_post_wait_utc_jump_is_timeout(self):
        runner=m.Runner.__new__(m.Runner);runner.approval=approval()
        runner.active=mock.Mock();runner.active.wait.return_value=0
        with mock.patch.object(m,'remaining',side_effect=[100,-1]),mock.patch.object(runner,'terminate') as terminate:
            with self.assertRaises(m.subprocess.TimeoutExpired):runner.wait(60,['fake-analyzer'])
            terminate.assert_called_once()

    def test_failed_or_stale_case_is_not_retried(self):
        runner=m.Runner.__new__(m.Runner);runner.ledger={'entries':[{'id':84,'status':'failed'}]}
        with self.assertRaises(RuntimeError):runner.cached(m.plan()[0])

    def test_swapped_cached_job_cannot_skip_resolution(self):
        runner=m.Runner.__new__(m.Runner)
        runner.ledger={'entries':[{'id':84,'status':'completed','job':m.plan()[1]}]}
        with mock.patch.object(runner,'checked_device') as check:
            with self.assertRaises(ValueError):runner.cached(m.plan()[0])
            check.assert_not_called()

    def test_cached_solver_recovers_analysis_without_launch(self):
        runner=m.Runner.__new__(m.Runner);runner.ledger={'entries':[]}
        entry={'status':'completed','native_pass':True}
        with mock.patch.object(m,'verify'),mock.patch.object(m,'authorization'),\
             mock.patch.object(runner,'cached',return_value=entry),mock.patch.object(runner,'analyze') as analyze,\
             mock.patch.object(m.subprocess,'Popen') as process:
            runner.execute(m.plan()[0]);analyze.assert_called_once_with(entry);process.assert_not_called()

    def test_nominal_failure_does_not_block_valid_native_measurement(self):
        runner=m.Runner.__new__(m.Runner);runner.ledger={'entries':[]}
        entry={'status':'completed','native_pass':True,'nominal_optical_pass':False}
        with mock.patch.object(m,'verify'),mock.patch.object(m,'authorization'),\
             mock.patch.object(runner,'cached',return_value=entry),mock.patch.object(runner,'analyze'):
            runner.execute(m.plan()[0])
        entry['native_pass']=False
        with mock.patch.object(m,'verify'),mock.patch.object(m,'authorization'),\
             mock.patch.object(runner,'cached',return_value=entry),mock.patch.object(runner,'analyze'):
            with self.assertRaises(RuntimeError):runner.execute(m.plan()[0])

    def test_os_launch_failure_remains_charged_and_receipt_precedes_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'ledger.json').write_text('{}')
            (folder/'preparation_manifest.json').write_text('{}');(folder/'source_manifest.json').write_text('{}')
            (folder/'approval_receipt.json').write_text('{}')
            job=m.plan()[0];case=dict(case_id=job['case_id'],design_file='design.json',design_sha256='x',physical_geometry_id='p')
            ref=dict(H_um=40,resolution=24,settings={'same':'settings'},record_file='ref.json',record_sha256='y')
            runner=m.Runner.__new__(m.Runner);runner.ledger={'entries':[]};runner.approval=approval();runner.active=None
            runner.refs=[ref];runner.cases=[case];runner.config={}
            runner.base=types.SimpleNamespace(u=types.SimpleNamespace(settings_for=lambda *_:{'same':'settings'}))
            runner.save=mock.Mock();runner.terminate=mock.Mock()
            def fail(*args,**kwargs):
                self.assertEqual(len(runner.ledger['entries']),1)
                self.assertTrue((folder/'prelaunch/84/receipt.json').is_file())
                self.assertEqual(runner.ledger['entries'][0]['status'],'started')
                raise OSError('synthetic OS launch failure')
            with mock.patch.object(m,'FOLDER',folder),mock.patch.object(m,'verify'),mock.patch.object(m,'authorization'),\
                 mock.patch.object(m,'capacity',return_value=60),mock.patch.object(m,'remaining',return_value=28800),\
                 mock.patch.object(m,'no_owned_worker'),mock.patch.object(runner,'cached',return_value=None),\
                 mock.patch.object(runner,'call',side_effect=fail):
                with self.assertRaises(OSError):runner.execute(job)
            self.assertEqual(len(runner.ledger['entries']),1)
            self.assertEqual(runner.ledger['entries'][0]['status'],'failed')
            self.assertFalse(runner.ledger['entries'][0]['solver_launched'])

    def test_default_inspection_has_zero_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'FOLDER',Path(tmp)/'absent'),\
             mock.patch.object(m.sys,'argv',[str(SCRIPT)]),mock.patch.object(m.subprocess,'Popen') as process:
            m.main();process.assert_not_called()


if __name__=='__main__':unittest.main()
