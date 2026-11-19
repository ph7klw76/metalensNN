"""Reuse the audited analytic math fixtures for the new H80 wrapper."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
wrapper=load('h80_sampling_wrapper',ROOT/'scripts/diagnose_height_family_h80_sampling_v2.py')
sampling_tests=load('reused_h100_sampling_fixtures',ROOT/'tests/test_h100_saved_field_sampling_diagnostic_v2.py')
angular_tests=load('reused_h100_angular_fixtures',ROOT/'tests/test_h100_angular_contributions_v2.py')
sampling_tests.diag=wrapper;angular_tests.diag=wrapper


class TestH80SamplingMath(sampling_tests.TestSavedFieldSamplingMath):
    pass


class TestH80AngularMath(angular_tests.TestAngularContributions):
    pass


if __name__=='__main__':unittest.main(verbosity=2)
