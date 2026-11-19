"""Pure quantization/contract tests; no JAX, optics, training or subprocess."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/evaluate_frozen_NN_500nm_proxy_v1.py'
spec=importlib.util.spec_from_file_location('coarse_proxy',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class QuantizationTests(unittest.TestCase):
    def test_two_routes_are_not_conflated(self):
        cont=np.asarray([.23,.25,.75,1.25,1.75,2.25],np.float32)
        fine=np.rint(cont/np.float32(.05))*np.float32(.05)
        np.testing.assert_array_equal(m.exported_slc_quantization(fine),[.5,.5,1,1.5,2,2.5])
        np.testing.assert_array_equal(m.direct_nn_quantization_numpy(cont),[0,0,1,1,2,2])

    def test_nominal_float32_ties_up(self):
        h=np.asarray([.25,.75,1.25,1.75,2.25,2.75],np.float32)
        np.testing.assert_array_equal(m.exported_slc_quantization(h),[.5,1,1.5,2,2.5,3])

    def test_reject_off_grid_nonfinite_and_outside(self):
        for h in ([.23],[float('nan')],[-.05],[3.05]):
            with self.assertRaises(ValueError):m.exported_slc_quantization(h)

    def test_real_fabrication_arrays_authenticate_without_optics(self):
        replay,contract,protected=m.authenticate()
        self.assertEqual(contract['config']['fabrication']['height_step_um'],.05)
        self.assertGreater(len(protected),15)
        self.assertEqual(contract['selection']['selection']['NN']['seed'],47)

    def test_radial_scope_escape_and_changed_coarse_rejected(self):
        cfg=dict(annuli=50,cylinder_diameter_um=50,fabrication={'height_step_um':.05},materials={},mfd_um=10.4)
        fine=dict(H_um=40,cylinder_diameter_um=50,units='um',heights_um=[.25]*50,materials={},mfd_um=10.4)
        coarse=copy.deepcopy(fine);coarse['heights_um']=[.5]*50
        m.validate_geometry(cfg,fine,coarse,40)
        bad=copy.deepcopy(cfg);bad['annuli']=100
        with self.assertRaises(ValueError):m.validate_geometry(bad,fine,coarse,40)
        coarse['heights_um'][0]=0
        with self.assertRaises(ValueError):m.validate_geometry(cfg,fine,coarse,40)

    def test_changed_target_or_seed_rejected(self):
        frozen=dict(declared_target_diameter_um=12,target_vector_widths_um=[[12,12]],
                    seed_continuous_um=[1,2],seed_quantized_um=[1,2],H_um=40,
                    target_output_z_um=44,processing_planes_from_output_um=[0],materials={},input_mode={})
        m.verify_bundle_metadata(copy.deepcopy(frozen),frozen)
        for key in ('declared_target_diameter_um','seed_continuous_um','target_output_z_um'):
            bad=copy.deepcopy(frozen);bad[key]=14 if key!='seed_continuous_um' else [1,2.1]
            with self.assertRaises(ValueError):m.verify_bundle_metadata(bad,frozen)

    def test_coarse_failure_is_reported_not_exception(self):
        metrics=dict(transmission=.95,declared_overlap=.8,declared_useful_efficiency=.76,
                     widths_um=[[12,12]],target_widths_um=[[12,12]],max_sidelobe_fraction=.01,
                     normalized_curvature=.01,angular_second_moment_ratio=1.05,valid_crossings=[[True,True]])
        criteria=dict(transmission_min=.9,declared_Gaussian_overlap_min=.98,useful_efficiency_min=.9,
                      output_diameter_relative_tolerance=.1,max_sidelobe_fraction=.02,
                      max_abs_normalized_curvature=.1,max_angular_second_moment_ratio=1.1)
        gate=m.gates(metrics,criteria)
        self.assertEqual(gate['status'],'FAIL');self.assertFalse(gate['checks']['useful_efficiency'])


if __name__=='__main__':unittest.main()
