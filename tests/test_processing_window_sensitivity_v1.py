"""Pure processing-window sensitivity fixtures; no saved fields or optics."""
import builtins
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/diagnose_processing_window_sensitivity_v1.py'
FORBIDDEN_IMPORTS={'jax','jaxlib','meep'}


def import_pure_diagnostic():
    original_import=builtins.__import__
    before=set(sys.modules)
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0] in FORBIDDEN_IMPORTS:
            raise AssertionError('Pure diagnosis must not import '+name)
        return original_import(name,*args,**kwargs)
    spec=importlib.util.spec_from_file_location('processing_window_sensitivity_fixture',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    with patch('builtins.__import__',side_effect=guarded_import):spec.loader.exec_module(module)
    introduced={name for name in set(sys.modules)-before if name.split('.')[0] in FORBIDDEN_IMPORTS}
    if introduced:raise AssertionError('Optical/training runtime introduced: '+repr(introduced))
    return module


def analysis(rows):
    return {'metrics':{'planes':[dict(z_um=z,diameter_x_um=x,diameter_y_um=y,
        valid_crossings=[True,True]) for z,x,y in rows]}}


class TestProcessingWindowComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.diagnostic=import_pure_diagnostic()
    def compare(self,primary,wide,planes):
        result=self.diagnostic.compare_processing_planes(primary,wide,planes)
        self.assertIn(result['status'],('COMPLETE','UNRESOLVED'))
        self.assertNotIn(result['status'],('PASS','FAIL'))
        return result
    def row(self,result,z,axis):
        matches=[row for row in result['rows'] if row['z_um']==z and row['axis']==axis]
        self.assertEqual(len(matches),1)
        return matches[0]
    def assert_missing_axis(self,result,z,axis):
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertTrue(result['missing_evidence'])
        for row in result['rows']:
            if row['z_um']==z and row['axis']==axis:
                self.assertIsNone(row.get('relative_change'))
                self.assertIsNone(row.get('absolute_change_um'))
    def test_import_avoids_meep_and_jax(self):
        self.assertTrue(callable(import_pure_diagnostic().compare_processing_planes))
    def test_asymmetric_plane_axis_maximum_and_primary_normalization(self):
        primary=analysis([(0.,10.,20.),(25.,12.,24.),(100.,20.,30.)])
        wide=analysis([(100.,20.4,31.8),(0.,10.1,20.2),(25.,11.4,24.48)])
        result=self.compare(primary,wide,[0.,25.,100.])
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['missing_evidence'],[])
        self.assertEqual(len(result['rows']),6)
        shrink=self.row(result,25.,'x')
        self.assertEqual(shrink['primary_diameter_um'],12.)
        self.assertEqual(shrink['wide_window_diameter_um'],11.4)
        self.assertAlmostEqual(shrink['absolute_change_um'],.6,places=13)
        self.assertAlmostEqual(shrink['relative_change'],abs(11.4/12.-1),places=14)
        self.assertAlmostEqual(result['maximum_relative_change'],abs(31.8/30.-1),places=14)
        self.assertEqual(result['maximum_location'],{'z_um':100.,'axis':'y'})
    def test_two_percent_and_tiny_excess_are_preserved_without_acceptance_gate(self):
        primary=analysis([(0.,10.,30.)])
        wide=analysis([(0.,10.2,30.600000012)])
        result=self.compare(primary,wide,[0.])
        self.assertEqual(result['status'],'COMPLETE')
        self.assertAlmostEqual(self.row(result,0.,'x')['relative_change'],abs(10.2/10.-1),places=14)
        expected=abs(30.600000012/30.-1)
        self.assertAlmostEqual(result['maximum_relative_change'],expected,places=14)
        self.assertGreater(result['maximum_relative_change'],.0200000001)
        self.assertEqual(result['maximum_location'],{'z_um':0.,'axis':'y'})
    def test_missing_requested_plane_is_unresolved_without_zero_substitution(self):
        primary=analysis([(0.,10.,20.),(25.,12.,24.)])
        wide=analysis([(0.,10.1,20.2)])
        result=self.compare(primary,wide,[0.,25.])
        self.assert_missing_axis(result,25.,'x');self.assert_missing_axis(result,25.,'y')
        self.assertAlmostEqual(self.row(result,0.,'x')['relative_change'],.01,places=14)
    def test_invalid_widths_and_crossings_are_missing_for_either_input(self):
        original=analysis([(0.,10.,20.)])
        for side in ('primary','wide'):
            for value in (None,0.,-1.,math.nan,math.inf):
                with self.subTest(side=side,width=value):
                    primary=copy.deepcopy(original);wide=copy.deepcopy(original)
                    target=primary if side=='primary' else wide
                    target['metrics']['planes'][0]['diameter_x_um']=value
                    self.assert_missing_axis(self.compare(primary,wide,[0.]),0.,'x')
            with self.subTest(side=side,crossing=False):
                primary=copy.deepcopy(original);wide=copy.deepcopy(original)
                target=primary if side=='primary' else wide
                target['metrics']['planes'][0]['valid_crossings'][0]=False
                self.assert_missing_axis(self.compare(primary,wide,[0.]),0.,'x')
    def test_absent_width_and_crossing_metadata_are_not_assumed_valid(self):
        primary=analysis([(0.,10.,20.)]);wide=analysis([(0.,10.,20.)])
        del wide['metrics']['planes'][0]['diameter_x_um']
        self.assert_missing_axis(self.compare(primary,wide,[0.]),0.,'x')
        wide=analysis([(0.,10.,20.)]);del wide['metrics']['planes'][0]['valid_crossings']
        self.assert_missing_axis(self.compare(primary,wide,[0.]),0.,'x')
        self.assert_missing_axis(self.compare(primary,wide,[0.]),0.,'y')
    def test_no_valid_measurements_leave_maximum_unknown(self):
        result=self.compare(analysis([]),analysis([]),[0.,25.])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertTrue(result['missing_evidence'])
        self.assertIsNone(result.get('maximum_relative_change'))
        self.assertIsNone(result.get('maximum_location'))
        for row in result['rows']:
            self.assertIsNone(row.get('relative_change'))


if __name__=='__main__':unittest.main(verbosity=2)
