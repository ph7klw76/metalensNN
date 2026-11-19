"""Focused primary-array guard fixtures; no fields or optical analysis calls."""
import builtins
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/diagnose_processing_window_sensitivity_v2.py'


def load_pure_append_module():
    original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in {'meep','jax','jaxlib'}:
            raise AssertionError('Pure append guard must not import '+name)
        return original_import(name,*args,**kwargs)
    spec=importlib.util.spec_from_file_location('processing_window_append_fixture',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    with patch('builtins.__import__',side_effect=guard):spec.loader.exec_module(module)
    return module


class TestPrimaryArrayGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.module=load_pure_append_module()
    def setUp(self):
        self.planes=[0.,25.,100.]
        self.saved={'metrics':{'planes':[dict(z_um=z,diameter_x_um=x,diameter_y_um=y,
            valid_crossings=[True,True]) for z,x,y in [(0.,10.,10.2),(25.,11.,11.3),(100.,15.,16.)]]}}
        self.case={'samplings':[{'processing_planes':copy.deepcopy(self.saved['metrics']['planes'])}]}
    def test_identical_primary_arrays_have_zero_difference(self):
        with patch.object(self.module.analysis,'analyze',side_effect=AssertionError('No field analysis permitted')):
            result=self.module.verify_primary_array(self.case,self.saved,self.planes)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(result['maximum_relative_change'],0.)
        self.assertEqual(len(result['rows']),6)
        self.assertTrue(all(row['relative_change']==0. for row in result['rows']))
    def test_changed_primary_width_is_rejected(self):
        self.case['samplings'][0]['processing_planes'][1]['diameter_y_um']+=.001
        with self.assertRaises(ValueError):self.module.verify_primary_array(self.case,self.saved,self.planes)
    def test_missing_primary_plane_or_width_is_rejected(self):
        missing_plane=copy.deepcopy(self.case);missing_plane['samplings'][0]['processing_planes'].pop()
        with self.assertRaises(ValueError):self.module.verify_primary_array(missing_plane,self.saved,self.planes)
        missing_width=copy.deepcopy(self.case);del missing_width['samplings'][0]['processing_planes'][0]['diameter_x_um']
        with self.assertRaises(ValueError):self.module.verify_primary_array(missing_width,self.saved,self.planes)
    def test_absent_or_empty_primary_structure_raises_value_error(self):
        for case in ({},{'samplings':[]},{'samplings':[{}]},
                {'samplings':[{'processing_planes':[]}]}):
            with self.subTest(case=case),self.assertRaises(ValueError):
                self.module.verify_primary_array(case,self.saved,self.planes)


if __name__=='__main__':unittest.main(verbosity=2)
