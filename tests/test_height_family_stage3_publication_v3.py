"""Training-source alias provenance only; no optics, rendering or metric edits."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage3_plot_alias_adapter',ROOT/'scripts/plot_height_family_stage3_v3.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


class TestTrainingAliasProjection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.root=Path(self.temp.name);self.stage=self.root/'maxwell_validation_v3';self.stage.mkdir()
        self.canonical=self.root/'training_result.json';self.canonical.write_bytes(b'one frozen training result')
        self.alias=self.root/'maxwell_validation_v2/final_training_inputs/training_result.json'
        self.alias.parent.mkdir(parents=True);self.alias.write_bytes(self.canonical.read_bytes())
        self.sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.science=SimpleNamespace(local_path=Path,sha=self.sha)
        self.data={'input_sha256':{str(self.canonical):self.sha(self.canonical),str(self.alias):self.sha(self.alias),'other_input':'other_digest'},
                   'rows':[{'scientific_scalar':1.2345678901234567}], 'all_attempts':[{'id':69,'status':'completed'}]}
    def tearDown(self):self.temp.cleanup()
    def project(self):return plot.csv_training_projection(self.data,self.stage,self.science)
    def test_original_failure_two_identical_authenticated_copies_is_supported(self):
        before=copy.deepcopy(self.data);projected,proof=self.project()
        training=[p for p in projected['input_sha256'] if Path(p).name=='training_result.json']
        self.assertEqual(training,[str(self.canonical.resolve())])
        self.assertEqual(len(proof['authenticated_aliases']),2)
        self.assertEqual(projected['input_sha256']['other_input'],'other_digest')
        self.assertIs(projected['rows'],self.data['rows'])
        self.assertIs(projected['all_attempts'],self.data['all_attempts'])
        self.assertEqual(self.data,before)
    def test_single_canonical_record_preserves_previous_behavior(self):
        del self.data['input_sha256'][str(self.alias)]
        projected,proof=self.project()
        self.assertEqual(projected,self.data)
        self.assertEqual(len(proof['authenticated_aliases']),1)
    def test_missing_actual_canonical_path_is_rejected_even_with_valid_copy(self):
        del self.data['input_sha256'][str(self.canonical)]
        with self.assertRaisesRegex(ValueError,'Canonical'):self.project()
    def test_different_authenticated_training_result_cannot_be_silently_selected(self):
        self.alias.write_bytes(b'different training result')
        self.data['input_sha256'][str(self.alias)]=self.sha(self.alias)
        with self.assertRaisesRegex(ValueError,'canonical source SHA'):self.project()
    def test_stale_hash_or_conflicting_normalized_alias_is_rejected(self):
        self.alias.write_bytes(b'changed after inventory')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.project()
        self.alias.write_bytes(self.canonical.read_bytes())
        spelling=self.root/'maxwell_validation_v3/../training_result.json'
        self.data['input_sha256'][str(spelling)]='0'*64
        with self.assertRaisesRegex(ValueError,'Conflicting normalized'):self.project()


if __name__=='__main__':unittest.main(verbosity=2)
