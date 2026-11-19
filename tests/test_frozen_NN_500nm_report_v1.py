"""Offline pure report fixtures: no solver, training, reconstruction or GPU."""
import csv
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/report_frozen_NN_500nm_meep_v1.py'
spec=importlib.util.spec_from_file_location('saved_report',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def proxies():
    return [dict(H_um=H,arm=m.PROXY_ARM,metrics={'declared_useful_efficiency':.9},proxy_gates={'status':'FAIL'}) for H in m.HEIGHTS]
def analysis(eta=.88,native=True,nominal=False):
    return dict(conservative_useful_efficiency=eta,reconstruction_pass=native,synthetic_optical_pass=nominal,
                numerical_checks={'native':native},optical_gates={'efficiency':nominal})


class ReportTests(unittest.TestCase):
    def test_pending_stays_missing_in_json_csv_and_prose(self):
        rows,meshes=m.build_rows(proxies(),{}, {},{},mock.Mock())
        self.assertTrue(all(r['Meep_useful_efficiency_percent'] is None for r in rows))
        self.assertTrue(all(r['mesh_r24_to_r32_status']=='UNRESOLVED' for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'out.csv';m.write_csv(path,rows)
            with path.open() as stream:parsed=list(csv.DictReader(stream))
            self.assertTrue(all(r['Meep_useful_efficiency_percent']=='' for r in parsed))
        data=dict(status='PARTIAL_SAVED_COMPARISON',verified_analysis_count=0,attempted_new_solves=1,cumulative_attempts=84,
                  rows=rows,geometry_scope='synthetic fixture',proxy_definition='proxy',Meep_definition='Meep',
                  postprocessing_utc='now',approved_utc='earlier',absolute_deadline_utc='later')
        text=m.report_text(data);self.assertIn('NA',text);self.assertNotIn('Meep(%)  0',text)

    def test_nominal_native_and_mesh_findings_remain_separate(self):
        analyses={(40,24):analysis(),(40,32):analysis(.879)}
        compare=mock.Mock(return_value={'status':'FAIL','checks':{'curvature_delta':False,'useful_efficiency':True},
                                        'measurements':{'useful_efficiency_relative_change':.001}})
        rows,meshes=m.build_rows(proxies(),analyses,{}, {},compare)
        row=rows[1]
        self.assertEqual(row['native_status'],'PASS');self.assertEqual(row['nominal_optical_status'],'FAIL')
        self.assertEqual(row['mesh_r24_to_r32_status'],'FAIL');self.assertAlmostEqual(row['Meep_useful_efficiency_percent'],87.9)
        self.assertEqual(row['efficiency_mesh_check_status'],'PASS');self.assertEqual(row['useful_efficiency_relative_mesh_change_percent'],.1)
        compare.assert_called_once()

    def test_incomplete_pair_is_never_fitted_or_assessed(self):
        compare=mock.Mock();rows,_=m.build_rows(proxies(),{(75,24):analysis()},{}, {},compare)
        compare.assert_not_called();self.assertEqual(rows[2]['Meep_useful_efficiency_percent'],88)
        self.assertIsNone(rows[3]['Meep_useful_efficiency_percent'])

    def test_direct_weight_route_is_not_selected_as_existing_slc(self):
        data=proxies()+[dict(H_um=H,arm='direct_NN500_nearest_even',metrics={'declared_useful_efficiency':.1},proxy_gates={'status':'FAIL'}) for H in m.HEIGHTS]
        rows,_=m.build_rows(data,{}, {},{},mock.Mock())
        self.assertEqual({r['proxy_useful_efficiency_percent'] for r in rows},{90})

    def test_contradictory_headlines_or_nonfinite_metric_rejected(self):
        wrong=analysis();wrong['reconstruction_pass']=False
        with self.assertRaises(ValueError):m.build_rows(proxies(),{(40,24):wrong},{}, {},mock.Mock())
        wrong=analysis(float('nan'))
        with self.assertRaises(ValueError):m.build_rows(proxies(),{(40,24):wrong},{}, {},mock.Mock())

    def test_changed_file_sha_or_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'file';path.write_text('changed')
            with self.assertRaises(ValueError):m.require_sha(path,'wrong',{})
            with self.assertRaises(ValueError):m.contained(Path(tmp),'../escape')

    def test_frozen_scientific_comparator_loads_without_jax_or_meep(self):
        import sys
        before=set(sys.modules);compare=m.frozen_comparator()
        self.assertTrue(callable(compare))
        self.assertNotIn('jax',set(sys.modules)-before);self.assertNotIn('meep',set(sys.modules)-before)

    def test_redirected_worker_and_wrong_analysis_grid_rejected(self):
        job=dict(H_um=40.0,resolution=24.0,duration=160.0)
        case=dict(design_file='frozen_designs/test.json',target_diameter_um=12.)
        ref=dict(record_file='artifacts/reference.json')
        raw=m.FOLDER/'records/test.json';analysis=m.FOLDER/'analyses/test.json'
        worker=m.worker_command(job,case,ref,raw)
        m.require_command(worker,worker)
        redirected=worker.copy();redirected[redirected.index('--reference')+1]='another-reference.json'
        with self.assertRaises(ValueError):m.require_command(redirected,worker)
        command=m.analysis_command(case,raw,analysis)
        entry={'analysis_attempts':[dict(status='completed',exit_code=0,command=command,
                   output_file=str(analysis.relative_to(m.ROOT)))]}
        m.require_analysis_attempt(entry,case,raw,analysis)
        entry['analysis_attempts'][0]['command']=command.copy()
        entry['analysis_attempts'][0]['command'][command.index('--count')+1]='256'
        with self.assertRaises(ValueError):m.require_analysis_attempt(entry,case,raw,analysis)
        entry['analysis_attempts'][0]['command']=command
        entry['analysis_attempts'][0]['exit_code']=1
        with self.assertRaises(ValueError):m.require_analysis_attempt(entry,case,raw,analysis)


if __name__=='__main__':unittest.main()
