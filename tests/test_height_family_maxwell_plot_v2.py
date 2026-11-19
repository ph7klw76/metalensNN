"""Known-record plotting/provenance checks; no live campaign or solver use."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
plot=module('height_family_plot',ROOT/'scripts/plot_height_family_maxwell_v2.py')
fixtures=module('height_family_report_fixtures',ROOT/'tests/test_height_family_maxwell_report_v2.py')


class TestHeightMaxwellPlot(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=fixtures.Fixture(self.temp.name)
        self.f.config.update(wavelength_um=1.55,mfd_um=10.4,materials={'polymer':{'n':1.5263}})
        self.f.config['natural_training']['investigation_gates']={
            'max_abs_normalized_curvature':.1,'max_angular_second_moment_ratio':1.1,'useful_efficiency_min':.9}
        fingerprint=hashlib.sha256(json.dumps(self.f.config,sort_keys=True).encode()).hexdigest()
        self.f.fingerprint=fingerprint;self.ledger_extra={}
        for entry in self.f.entries:
            raw=self.f.read(entry['record_file']);raw['config_fingerprint']=fingerprint
            self.f.write(entry['record_file'],raw)
            entry['record_sha256']=plot.sha256(self.f.folder/entry['record_file'])
            if 'analysis_file' in entry:entry['analysis_sha256']=plot.sha256(self.f.folder/entry['analysis_file'])
        self.f.write('config.json',self.f.config);self.flush()
        self.proxy=self.f.folder/'proxy.csv'
        self.proxy.write_text('H_um,useful_efficiency\n'+''.join(f'{H},0.95\n' for H in range(40,101,10)),encoding='utf-8')
    def flush(self):self.f.write('ledger.json',{'entries':self.f.entries,'elapsed_s':1000,**self.ledger_extra})
    def collect(self):self.flush();return plot.collect(self.f.folder,self.proxy)
    def modify(self,entry,callback):
        self.f.modify_analysis(entry,callback)
        entry['analysis_sha256']=plot.sha256(self.f.folder/entry['analysis_file'])
    def approve_reserve(self):
        path=self.f.write('frozen_designs/NN_H40.json',{'H_um':40.,'heights_um':[1.]})
        geometry_hash=plot.sha256(path)
        for entry in self.f.entries:
            if entry['method']=='NN' and entry['H_um']==40:entry['design_sha256']=geometry_hash
        self.f.write('cumulative_amendment.json',{'cases':[{'method':'NN','H_um':40.,
            'design_file':'frozen_designs/NN_H40.json','design_sha256':geometry_hash}]})
        receipt=self.f.write('approval_receipt.json',{'status':'APPROVED_BY_USER',
            'proposal_sha256':plot.sha256(self.f.folder/'maxwell_validation_proposal.json')})
        self.ledger_extra['approval_sha256']=plot.sha256(receipt)
    def reserve(self,approved=True,job_type='reserve_mesh',status='completed',duration=300):
        prior=len(self.f.entries)
        entry=self.f.device('NN',40,32,job_type=job_type,duration=duration)
        for new in self.f.entries[prior:]:
            new['record_sha256']=plot.sha256(self.f.folder/new['record_file'])
            if 'analysis_file' in new:new['analysis_sha256']=plot.sha256(self.f.folder/new['analysis_file'])
        entry['status']=status
        if approved:self.approve_reserve()
        return entry
    def test_nominal_only_with_actual_hashes_and_angles(self):
        data=self.collect();self.assertEqual(data['plotted_nominal_points'],21)
        self.assertTrue(data['complete_nominal_grid'])
        self.assertEqual(len(data['proxy_efficiency']),7)
        row=data['records'][0]
        self.assertAlmostEqual(row['core_half_angles_deg'][0],.07*180/3.141592653589793)
        self.assertEqual(row['analysis_sha256'],plot.sha256(self.f.folder/row['analysis_file']))
    def test_absent_analysis_is_a_gap(self):
        entry=self.f.find();(self.f.folder/entry['analysis_file']).unlink()
        data=self.collect();self.assertEqual(data['plotted_nominal_points'],20)
        self.assertIn({'H_um':40.,'resolution':24.},data['absent_nominal_points'])
        self.assertFalse(data['complete_nominal_grid'])
    def test_hash_mismatch_is_not_plotted(self):
        self.f.find()['analysis_sha256']='b'*64
        data=self.collect();self.assertEqual(data['plotted_nominal_points'],20)
        self.assertTrue(any('analysis_sha256' in row['reason'] for row in data['skipped_entries']))
    def test_nonfinite_curvature_preserves_other_data(self):
        entry=self.f.find();self.modify(entry,lambda a:a.update(normalized_curvature=float('nan')))
        data=self.collect();row=next(x for x in data['records'] if x['id']==entry['id'])
        self.assertIsNone(row['C']);self.assertEqual(row['Q'],1.02)
        self.assertFalse(data['complete_nominal_grid'])
        json.dumps(data,allow_nan=False)
    def test_invalid_fit_does_not_produce_angles(self):
        entry=self.f.find();self.modify(entry,lambda a:a['metrics']['fitted_core_divergence'].update(valid=False))
        row=next(x for x in self.collect()['records'] if x['id']==entry['id'])
        self.assertEqual(row['core_half_angles_deg'],[None,None])
    def test_render_fixture_and_refuse_overwrite(self):
        output=self.f.folder/'fixture.png';data=self.collect();plot.render(data,output)
        self.assertEqual(output.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        original=output.read_bytes()
        with self.assertRaises(FileExistsError):plot.render(data,output)
        self.assertEqual(output.read_bytes(),original)
    def test_cli_writes_hash_sidecar(self):
        output=self.f.folder/'figure.png'
        argv=['plot','--folder',str(self.f.folder),'--output',str(output),'--proxy-summary',str(self.proxy)]
        with patch.object(sys,'argv',argv):self.assertEqual(plot.main(),0)
        sidecar=json.loads(output.with_suffix('.data.json').read_text())
        self.assertEqual(sidecar['plotted_nominal_points'],21)
        self.assertTrue(sidecar['input_sha256'])
        with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):plot.main()
    def test_completed_approved_reserve32_is_distinct_and_preserves_nominal_failures(self):
        old=self.f.find();self.modify(old,lambda a:a.update(normalized_curvature=.2,synthetic_optical_pass=False))
        entry=self.reserve();data=self.collect()
        self.assertEqual(data['plotted_nominal_points'],21)
        self.assertEqual(data['plotted_reserve_mesh_points'],1)
        self.assertEqual(data['plotted_total_points'],22)
        self.assertIn(32.,data['meshes_cells_per_um'])
        self.assertEqual(len(data['absent_reserve_mesh_points']),6)
        reserved=next(r for r in data['records'] if r['id']==entry['id'])
        self.assertEqual(reserved['plot_marker'],'D')
        self.assertEqual(reserved['series_type'],'approved_reserve_mesh')
        preserved=next(r for r in data['records'] if r['id']==old['id'])
        self.assertEqual(preserved['C'],.2);self.assertIs(preserved['synthetic_optical_pass'],False)
        plot.render(data,self.f.folder/'reserve_fixture.png')
    def test_missing_reserve32_is_a_gap_not_a_required_nominal_point(self):
        self.approve_reserve();data=self.collect()
        self.assertEqual(data['plotted_reserve_mesh_points'],0)
        self.assertEqual(len(data['absent_reserve_mesh_points']),7)
        self.assertEqual(data['plotted_nominal_points'],21);self.assertTrue(data['complete_nominal_grid'])
    def test_failed_reserve32_is_preserved_but_not_plotted(self):
        entry=self.reserve(status='failed');data=self.collect()
        self.assertEqual(data['plotted_reserve_mesh_points'],0)
        skipped=next(row for row in data['skipped_entries'] if row['id']==entry['id'])
        self.assertEqual(skipped['status'],'failed')
        self.assertEqual(len(data['absent_reserve_mesh_points']),7)
    def test_reserve32_with_missing_analysis_is_not_fabricated(self):
        entry=self.reserve();(self.f.folder/entry['analysis_file']).unlink()
        self.assertEqual(self.collect()['plotted_reserve_mesh_points'],0)
    def test_completed_reserve32_requires_saved_approval(self):
        entry=self.reserve(approved=False);data=self.collect()
        self.assertEqual(data['plotted_reserve_mesh_points'],0)
        self.assertTrue(any(row['id']==entry['id'] and 'approval' in row['reason'] for row in data['skipped_entries']))
    def test_reserve32_geometry_hash_mismatch_is_rejected(self):
        entry=self.reserve();entry['design_sha256']='b'*64
        self.assertEqual(self.collect()['plotted_reserve_mesh_points'],0)
    def test_reserve32_different_duration_and_duration_job_are_excluded(self):
        self.reserve(duration=450)
        self.assertEqual(self.collect()['plotted_reserve_mesh_points'],0)
        entry=self.reserve(job_type='reserve_duration',duration=450)
        data=self.collect();self.assertNotIn(entry['id'],[row['id'] for row in data['records']])


if __name__=='__main__':unittest.main(verbosity=2)
