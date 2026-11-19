"""Pure saved-record/selection/figure fixtures; no Meep, JAX or training."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

plot=load('nn_forward_plot_test',ROOT/'scripts/plot_height_family_nn_forward_v3.py')
fixtures=load('nn_forward_saved_fixture',ROOT/'tests/test_height_family_nn_direct_comparison_v2.py')


class TestNNForwardComparison(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.f=fixtures.Fixture(self.temp.name)
    def build(self,data=None):return plot.build_comparison(self.f.collect() if data is None else data,{})
    def finer(self,res=40):
        ref=self.f.reference(self.f.stage,40,res)
        return self.f.device(self.f.stage,'NN',40,res,ref)
    def test_known_paraxial_rayleigh_distance_grows_diameter_by_sqrt_two(self):
        D=10.;wavelength=1.55;n=1.;zr=math.pi*n*(D/2)**2/wavelength
        values=plot.gaussian_diameters(D,[0.,zr],wavelength,n)
        self.assertEqual(values[0],D);self.assertAlmostEqual(values[1],D*math.sqrt(2),places=12)
    def test_invalid_gaussian_contract_is_rejected(self):
        for args in [(-1,[0],1.55,1),(10,[-1],1.55,1),(10,[0],float('nan'),1),(10,[0],1.55,0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):plot.gaussian_diameters(*args)
    def test_same_nn_geometry_does_not_require_direct_optimizer_pair(self):
        data=self.f.collect();self.assertFalse(data['matched_Meep_pairs'])
        result=self.build(data);row=result['rows'][0]
        self.assertTrue(row['same_geometry']);self.assertEqual(row['Meep']['resolution'],32.)
        self.assertEqual(len(result['NN_nominal_histories']),2)
        self.assertFalse(any(r['method']=='direct' for r in result['NN_nominal_histories']))
        self.assertEqual(row['selected_mesh']['status'],'PASS')
        self.assertFalse(result['full_optical_acceptance_claimed'])
    def test_missing_h75_has_no_meep_curve_and_retains_validation_label(self):
        row=self.build()['rows'][1]
        self.assertEqual(row['H_um'],75.);self.assertIsNone(row['Meep'])
        self.assertIsNotNone(row['proxy']);self.assertEqual(row['native']['status'],'UNRESOLVED')
        self.assertIn('validation-observed',row['height_use'])
    def test_finest_native_failure_cannot_fall_back_to_older_pass(self):
        entry=self.finer()
        self.f.modify_analysis(entry,lambda a:a['numerical_checks'].update(crosscheck=False))
        row=self.build()['rows'][0]
        self.assertEqual(row['Meep']['resolution'],40.);self.assertEqual(row['native']['status'],'FAIL')
        self.assertEqual(row['selected_mesh']['resolutions'],[32.,40.])
        self.assertEqual(row['selected_mesh']['status'],'FAIL')
    def test_full_precision_angular_gate_failure_is_not_rounded_to_pass(self):
        entry=self.finer()
        def change(a):
            a['angular_second_moment_ratio']=1.103364771550854
            a['optical_gates']['angular_second_moment']=False
        self.f.modify_analysis(entry,change)
        row=self.build()['rows'][0]
        self.assertEqual(row['Meep']['Q'],1.103364771550854)
        self.assertEqual(row['nominal_optical']['status'],'FAIL')
        self.assertIn('angular_second_moment',row['nominal_optical']['failed_checks'])
    def test_stored_true_flags_cannot_conceal_missing_nominal_scalar(self):
        entry=self.finer()
        self.f.modify_analysis(entry,lambda a:a.update(conservative_T=None))
        row=self.build()['rows'][0]
        self.assertEqual(row['nominal_optical']['status'],'UNRESOLVED')
        self.assertIn('Finite nominal observable transmission',row['nominal_optical']['missing_evidence'])
    def test_missing_processing_width_stays_gap_and_blocks_nominal_pass(self):
        entry=self.finer()
        self.f.modify_analysis(entry,lambda a:a['metrics']['planes'].pop(1))
        row=self.build()['rows'][0]
        self.assertIsNone(row['Meep']['planes'][1]['diameter_x_um'])
        self.assertEqual(row['nominal_optical']['status'],'UNRESOLVED')
        self.assertEqual(row['selected_mesh']['status'],'UNRESOLVED')
    def test_proxy_maxwell_geometry_mismatch_is_rejected(self):
        data=self.f.collect();data['Meep_records'][-1]['design_sha256']='different'
        with self.assertRaisesRegex(ValueError,'identical frozen NN geometry'):self.build(data)
    def test_numerical_variant_is_not_selected_as_nominal_geometry_curve(self):
        data=self.f.collect();variant=copy.deepcopy(data['Meep_records'][-1])
        variant.update(resolution=48.,job_type='pml',identity='variant:48');data['Meep_records'].append(variant)
        self.assertEqual(self.build(data)['rows'][0]['Meep']['resolution'],32.)
    def test_unfinished_finer_attempt_keeps_latest_curve_but_mesh_unresolved(self):
        data=self.f.collect();latest=data['Meep_records'][-1]
        data['all_attempts'].append({'identity':'maxwell_validation_v2:999','method':'NN','H_um':40.,
            'resolution':40.,'job_type':'refinement','status':'started',
            **{key:latest['settings'][key] for key in ('pml_um','padding_um','duration')}})
        row=self.build(data)['rows'][0]
        self.assertEqual(row['Meep']['resolution'],32.)
        self.assertEqual(row['selected_mesh']['status'],'UNRESOLVED')
        self.assertEqual(len(row['unfinished_finer_attempts']),1)
    def test_passing_coarse_pair_does_not_replace_declared_fine_mesh(self):
        for res in (8,16):
            ref=self.f.reference(self.f.prior,100,res)
            self.f.device(self.f.prior,'NN',100,res,ref)
        self.f.flush(self.f.prior);self.f.seal_prior()
        row=self.build()['rows'][2]
        self.assertEqual(row['Meep']['resolution'],16.)
        self.assertEqual(row['selected_mesh']['comparison']['status'],'PASS')
        self.assertEqual(row['selected_mesh']['status'],'UNRESOLVED')
        self.assertIn('Declared NN fine mesh >=24 cells/um is missing',row['selected_mesh']['missing_evidence'])
    def test_only_same_duration_mesh_pair_is_compared(self):
        data=self.f.collect();fine=copy.deepcopy(data['Meep_records'][-1])
        fine.update(resolution=40.,identity='fixture:fine')
        fine['settings']['duration']+=1;data['Meep_records'].append(fine)
        row=self.build(data)['rows'][0]
        self.assertEqual(row['Meep']['resolution'],40.)
        self.assertEqual(row['selected_mesh']['status'],'UNRESOLVED')
        self.assertNotIn('resolutions',row['selected_mesh'])
    def test_changed_saved_analysis_is_rejected(self):
        data=self.f.collect();path=Path(data['Meep_records'][-1]['analysis_file'])
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'analysis changed'):self.build(data)
    def test_fresh_output_render_and_precise_labels(self):
        output=self.f.stage/'forward_comparison/fixture_v1'
        argv=['plot','--folder',str(self.f.stage),'--output',str(output)]
        self.f.flush(self.f.stage)
        with patch.object(sys,'argv',argv):self.assertEqual(plot.main(),0)
        self.assertEqual((output/'nn_proxy_vs_direct_maxwell.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        report=(output/'REPORT.txt').read_text()
        self.assertIn('Proxy forward prediction of physics-seeded NN geometry',report)
        self.assertIn('Direct Maxwell simulation of same geometry',report)
        self.assertIn('not the separately named non-neural direct-table optimizer',report)
        manifest=plot.read(output/'delivery_manifest.json')
        for name,evidence in manifest['files'].items():self.assertEqual(plot.sha(output/name),evidence['sha256'])
        with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):plot.main()
    def test_output_outside_version_directory_is_rejected(self):
        output=self.f.base/'outside'
        with patch.object(sys,'argv',['plot','--folder',str(self.f.stage),'--output',str(output)]),self.assertRaises(ValueError):plot.main()
        self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
