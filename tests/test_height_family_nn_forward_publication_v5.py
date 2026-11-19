"""CSV mapping/units and new publication labels; no new optical definitions."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

layout=load('height_specific_publication_v5_test',ROOT/'scripts/plot_height_family_nn_forward_publication_v5.py')
fixtures=load('height_specific_csv_fixture',ROOT/'tests/test_height_family_nn_direct_comparison_v2.py')


class TestHeightSpecificPublication(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.temp.cleanup)
        self.fixture=fixtures.Fixture(self.temp.name)
        for H in (75,100):
            ref=self.fixture.reference(self.fixture.stage,H,24)
            self.fixture.device(self.fixture.stage,'NN',H,24,ref)
        self.fixture.flush(self.fixture.stage);self.science=layout.load_science()
    def test_csv_copies_xy_fraction_percent_and_existing_degree_half_angles(self):
        data=layout.prepare_data(self.fixture.stage,self.science)
        row=data['rows'][1];m=row['Meep']
        m['planes'][0].update(diameter_x_um=14.41808170,diameter_y_um=14.33853755)
        m.update(eta=.954177014425,C=-.04091735698455,Q=1.095015760854,
                 core_half_angles_deg=[2.345678901234,3.456789012345])
        rows=layout.csv_rows(data,self.science)
        self.assertEqual([(r['H_um'],r['forward_model']) for r in rows],
            [(40.,'proxy_forward_prediction'),(40.,'direct_Meep_forward_simulation'),
             (75.,'proxy_forward_prediction'),(75.,'direct_Meep_forward_simulation'),
             (100.,'proxy_forward_prediction'),(100.,'direct_Meep_forward_simulation')])
        measured=rows[3]
        self.assertEqual(measured['diameter_x_um'],14.41808170)
        self.assertEqual(measured['diameter_y_um'],14.33853755)
        self.assertEqual(measured['useful_efficiency_fraction'],.954177014425)
        self.assertEqual(measured['useful_efficiency_percent'],100*.954177014425)
        self.assertEqual(measured['Gaussian_core_half_angle_x_deg'],2.345678901234)
        self.assertEqual(measured['Gaussian_core_half_angle_y_deg'],3.456789012345)
        self.assertEqual(measured['retained_band_angular_ratio_Q'],1.095015760854)
        self.assertEqual(measured['downstream_distance_um'],0.)
        self.assertEqual(measured['output_reference_z_um'],79.)
        self.assertEqual(measured['resolution_cells_per_um'],24.)
        self.assertEqual(measured['native_status'],'PASS')
        self.assertEqual(measured['selected_mesh_status'],'UNRESOLVED')
        self.assertEqual(rows[2]['native_status'],'NOT_APPLICABLE_PROXY')
    def test_csv_requires_unique_anchored_training_source_and_keeps_missing_rows(self):
        data=layout.prepare_data(self.fixture.stage,self.science)
        data['rows'][1]['Meep']=None
        rows=layout.csv_rows(data,self.science)
        self.assertEqual(rows[3]['native_status'],'UNRESOLVED_MISSING')
        self.assertEqual(rows[3]['useful_efficiency_fraction'],'')
        data['input_sha256']={p:d for p,d in data['input_sha256'].items() if Path(p).name!='training_result.json'}
        with self.assertRaisesRegex(ValueError,'training record'):layout.csv_rows(data,self.science)
    def test_csv_roundtrip_new_labels_and_manifest(self):
        output=self.fixture.stage/'forward_comparison/v5_fixture'
        argv=['publication','--folder',str(self.fixture.stage),'--output',str(output)]
        with patch.object(sys,'argv',argv):self.assertEqual(layout.main(),0)
        with (output/'proxy_vs_selected_Meep.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),6);self.assertEqual(tuple(rows[0]),layout.CSV_COLUMNS)
        self.assertEqual(rows[3]['output_reference_z_um'],'79.0')
        self.assertEqual(rows[3]['selected_mesh_status'],'UNRESOLVED')
        report=(output/'REPORT.txt').read_text()
        self.assertIn('Patterns differ between heights',report)
        self.assertIn('WITHIN each height',report)
        manifest=self.science.read(output/'delivery_manifest.json')
        self.assertEqual(len(manifest['files']),9)
        self.assertEqual(manifest['predecessor_layout_sha256'],layout.PREDECESSOR_LAYOUT_SHA)
        for name,row in manifest['files'].items():self.assertEqual(layout.sha(output/name),row['sha256'])
        self.assertEqual(layout.sha(ROOT/'scripts/plot_height_family_nn_forward_publication_v4.py'),layout.PREDECESSOR_LAYOUT_SHA)
        with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):layout.main()


if __name__=='__main__':unittest.main(verbosity=2)
