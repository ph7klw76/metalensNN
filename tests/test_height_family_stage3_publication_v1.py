"""Stage3 export readiness and source pins; existing layout/physics unchanged."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_publication_policy_test',ROOT/'scripts/plot_height_family_stage3_v1.py')
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)


class TestStage3Publication(unittest.TestCase):
    def fixture(self):return {'authorization':{'status':'PASS'},'budget':{'status':'PASS'},'primary_execution':{'status':'PASS'}}
    def test_existing_publication_and_scientific_sources_remain_pinned(self):
        layout=publication.load_layout();science=layout.load_science()
        self.assertEqual(layout.sha(publication.LAYOUT_PATH),publication.LAYOUT_SHA)
        self.assertEqual(science.sha(layout.SCIENTIFIC_SOURCE),layout.EXPECTED_SCIENTIFIC_SHA)
        with patch.object(publication,'sha',return_value='changed'),self.assertRaises(ValueError):publication.load_layout()
    def test_incomplete_primary_needs_explicit_root_partial_reason(self):
        data=self.fixture();data['primary_execution']['status']='UNRESOLVED'
        with self.assertRaisesRegex(RuntimeError,'Primary is incomplete'):publication.require_ready(data)
        with self.assertRaises(RuntimeError):publication.require_ready(data,'  ')
        publication.require_ready(data,'Root requested a partial saved-field comparison')
    def test_reason_cannot_bypass_bad_authorization_or_exhausted_deadline(self):
        for key in ('authorization','budget'):
            data=self.fixture();data[key]['status']='FAIL'
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'authorization'):
                publication.require_ready(data,'Root requested a partial comparison')
    def test_completed_primary_passes_without_changing_input(self):
        data=self.fixture();before=copy.deepcopy(data);publication.require_ready(data)
        self.assertEqual(data,before)


if __name__=='__main__':unittest.main(verbosity=2)
