"""Layout/export checks only; scientific tests remain in pinned v3."""
import importlib.util
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

layout=load('publication_layout_test',ROOT/'scripts/plot_height_family_nn_forward_publication_v4.py')
fixtures=load('publication_record_fixture',ROOT/'tests/test_height_family_nn_direct_comparison_v2.py')


class TestPublicationLayout(unittest.TestCase):
    def test_exact_scientific_source_is_required(self):
        self.assertEqual(layout.sha(layout.SCIENTIFIC_SOURCE),layout.EXPECTED_SCIENTIFIC_SHA)
        with patch.object(layout,'sha',return_value='changed'),self.assertRaisesRegex(ValueError,'Pinned scientific'):
            layout.load_science()
    def test_compact_raster_and_vector_exports_and_fresh_version(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder:
            fixture=fixtures.Fixture(folder);fixture.flush(fixture.stage)
            output=fixture.stage/'forward_comparison/publication_fixture'
            argv=['publication','--folder',str(fixture.stage),'--output',str(output)]
            with patch.object(sys,'argv',argv):self.assertEqual(layout.main(),0)
            science=layout.load_science();manifest=science.read(output/'delivery_manifest.json')
            self.assertEqual(len(manifest['files']),8)
            self.assertEqual(manifest['scientific_source_sha256'],layout.EXPECTED_SCIENTIFIC_SHA)
            self.assertFalse(manifest['scientific_logic_changed'])
            for stem in ('beam_propagation','forward_model_metrics'):
                png=(output/f'{stem}.png').read_bytes()
                self.assertEqual(png[:8],b'\x89PNG\r\n\x1a\n')
                self.assertEqual(struct.unpack('>II',png[16:24]),(2160,1080))
                svg=ET.parse(output/f'{stem}.svg').getroot()
                self.assertEqual(svg.attrib['width'],'518.4pt')
                self.assertEqual(svg.attrib['height'],'259.2pt')
                pdf=(output/f'{stem}.pdf').read_bytes();self.assertTrue(pdf.startswith(b'%PDF-'))
                self.assertRegex(pdf,rb'/MediaBox\s*\[\s*0\s+0\s+518\.4\s+259\.2\s*\]')
            for name,row in manifest['files'].items():self.assertEqual(layout.sha(output/name),row['sha256'])
            data=science.read(output/'comparison.json')
            self.assertEqual(data['publication_layout']['minimum_font_pt'],8.5)
            self.assertEqual(data['utility_sha256'],layout.EXPECTED_SCIENTIFIC_SHA)
            self.assertIsNone(data['rows'][1]['Meep'])
            with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):layout.main()
    def test_output_location_cannot_escape_fresh_version_tree(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder:
            stage=Path(folder)/'stage';output=Path(folder)/'outside'
            with patch.object(sys,'argv',['publication','--folder',str(stage),'--output',str(output)]),self.assertRaises(ValueError):layout.main()
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
