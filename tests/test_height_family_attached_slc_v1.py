"""Synthetic connected solids only; never export actual frozen family files."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('family_attached_export_test',ROOT/'scripts/export_height_family_attached_slc_v1.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
audit_spec=importlib.util.spec_from_file_location('family_attached_independent_auditor',ROOT/'scripts/audit_metalens_slc.py')
auditor=importlib.util.module_from_spec(audit_spec);audit_spec.loader.exec_module(auditor)

class TestFamilyAttachedSLC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core,cls.assembly=e.load_helpers();cls.builder=cls.core.load_builder(e.BUILDER)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.design=dict(units='um',H_um=2.,cylinder_diameter_um=6.,heights_um=[1.,.5,0.])
        self.source=self.folder/'source.json';self.source.write_text(json.dumps(self.design))
    def export_fixture(self,filename='C01.slc'):
        with patch('sys.stdout',io.StringIO()):
            return e.export_component(self.design,self.source,self.folder/'export',self.builder,self.core,self.assembly,filename,.25,False)
    def test_current_reviewed_builder_is_native_minimum_Z(self):
        self.assertEqual(self.builder.source_sha256,e.BUILDER_SHA)
        planes,samples,top=self.builder.slc_layer_schedule(0.,.003,.00025,.5,100000)
        self.assertEqual(planes[0],0.);self.assertGreater(samples[0],planes[0]);self.assertEqual(len(planes),12)
        self.assertLess(samples[-1],top)
    def test_source_derived_solid_retains_full_cylinder_where_relief_is_zero(self):
        h,H,D=e.validate_profile(self.design,.25);v,f=self.core.relief_mesh(H+h,D/2,128)
        check=self.assembly.check_mesh(v,f)
        self.assertTrue(check['single_connected_component']);np.testing.assert_allclose(np.ptp(v,axis=0),[6.,6.,3.])
        self.assertTrue(np.any(np.isclose(np.hypot(v[:,0],v[:,1]),3.)&np.isclose(v[:,2],2.)))
    def test_native_copy_matches_independent_assembly_and_spacing_audit(self):
        before=self.source.read_bytes();manifest=self.export_fixture();final=Path(manifest['final_slc'])
        self.assertEqual(self.source.read_bytes(),before);self.assertFalse(manifest['legacy_min_z_relabeling_applied'])
        self.assertEqual(e.sha(final),e.sha(manifest['actual_builder_output']))
        independent=auditor.audit(final.read_bytes(),before,.25,128,'assembly')
        self.assertEqual(independent['status'],'PASS',independent['failed_checks'])
        self.assertEqual(manifest['layer_count'],12);self.assertLess(abs(manifest['terminal_top_um']-3.),1e-5)
        self.assertLess(abs(manifest['actual_slab_interval_min_um']-.25),1e-5)
        self.assertLess(abs(manifest['actual_slab_interval_max_um']-.25),1e-5)
        self.assertLess(abs(manifest['terminal_slab_interval_um']-.25),1e-5)
        self.assertEqual(manifest['dimensions_um'],[6.,6.,3.]);self.assertEqual(manifest['nonzero_top_relief_diameter_um'],4.)
    def test_builder_returned_path_is_consumed_even_when_requested_stem_is_shortened(self):
        manifest=self.export_fixture('H040.slc')
        self.assertNotEqual(Path(manifest['actual_builder_output']).name,'H040.slc')
        self.assertEqual(Path(manifest['final_slc']).name,'H040.slc')
        self.assertEqual(e.sha(manifest['final_slc']),e.sha(manifest['actual_builder_output']))
    def test_midpoint_output_cannot_be_mistaken_for_native_minimum_Z(self):
        parsed=dict(layers=[((i+.5)*.25*.001,[]) for i in range(12)],sampling=[.000125,.00025,0,0],top_mm=.003)
        with self.assertRaisesRegex(ValueError,'native minimum-Z'):e.native_min_z_check(parsed,self.design,.25)
    def test_sampling_table_spacing_must_match_actual_lower_bound_grid(self):
        parsed=dict(layers=[(i*.25*.001,[]) for i in range(12)],sampling=[0.,.0005,0,0],top_mm=.003)
        with self.assertRaisesRegex(ValueError,'native minimum-Z'):e.native_min_z_check(parsed,self.design,.25)
    def test_units_nonmonotone_profile_and_inexact_layers_are_rejected(self):
        for mutate in [lambda d:d.update(units='mm'),lambda d:d.update(H_um=0.),lambda d:d.update(heights_um=[1.,0.,1.]),lambda d:d.update(heights_um=[1.,-.5,0.])]:
            d=copy.deepcopy(self.design);mutate(d)
            with self.assertRaises(ValueError):e.validate_profile(d,.25)
        with self.assertRaises(ValueError):e.validate_profile(self.design,.3)
    def test_source_bytes_must_match_passed_geometry_and_fresh_destination(self):
        mismatch=copy.deepcopy(self.design);mismatch['H_um']=3.
        with self.assertRaisesRegex(ValueError,'source bytes differ'):
            e.export_component(mismatch,self.source,self.folder/'bad',self.builder,self.core,self.assembly,'C01.slc',.25,False)
        self.export_fixture()
        with self.assertRaises(FileExistsError):self.export_fixture()
    def test_batch_CLI_refuses_existing_folder_before_any_export(self):
        with patch.object(sys,'argv',['family_export','--output',str(self.folder)]),patch.object(e,'authenticate_sources',side_effect=AssertionError('No existing-output rewrite')):
            with self.assertRaises(FileExistsError):e.main()
if __name__=='__main__':unittest.main(verbosity=2)
