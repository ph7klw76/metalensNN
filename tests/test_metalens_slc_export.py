"""Independent fixtures and physical-scale regressions for the SLC component."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/export_metalens_slc.py'
spec=importlib.util.spec_from_file_location('metalens_slc_export',SCRIPT)
export=importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def fixture(gaps=0,closed=True,units='MM'):
    """A specified 2 µm square with a 1 µm square hole, not writer output."""
    outer=[[-.001,-.001],[.001,-.001],[.001,.001],[-.001,.001],[-.001,-.001]]
    hole=[[-.0005,-.0005],[-.0005,.0005],[.0005,.0005],[.0005,-.0005],[-.0005,-.0005]]
    if not closed:
        outer[-1]=[0.,0.]
    header=f'-SLCVER 2.0 -UNIT {units} -TYPE PART -PACKAGE TEST\r\n\x1a'.encode()
    payload=header+bytes(256)+struct.pack('<B4f',1,.000025,.00005,0,0)
    payload+=struct.pack('<fI',.000025,2)
    for xy in [outer,hole]:
        payload+=struct.pack('<II',5,gaps)+struct.pack('<10f',*np.array(xy).ravel())
    return payload+struct.pack('<fI',.00005,0xffffffff)


class TestMetalensSLC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder=export.load_builder(export.DEFAULT_BUILDER)

    def use_fixture(self,payload,callback):
        # Keep all test artifacts under the task's writable workspace.
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            p=Path(temp)/'test.slc'
            p.write_bytes(payload)
            return callback(p)

    def test_known_square_with_hole_in_physical_mm(self):
        def check(p):
            actual=export.independent_read(p)
            loops=actual['layers'][0][1]
            self.assertAlmostEqual(sum(export.area(x) for x in loops)*1e6,3.,places=6)
            self.assertGreater(export.area(loops[0]),0)
            self.assertLess(export.area(loops[1]),0)
            self.assertAlmostEqual(actual['sampling'][1]*1000,.05,places=8)
            self.assertEqual(len(self.builder.slc_readback(p)['layers']),1)
        self.use_fixture(fixture(),check)

    def test_truncated_file_rejected_by_both_readers(self):
        def check(p):
            for read in [export.independent_read,self.builder.slc_readback]:
                with self.assertRaises(ValueError): read(p)
        self.use_fixture(fixture()[:-1],check)

    def test_open_contour_rejected_by_both_readers(self):
        def check(p):
            for read in [export.independent_read,self.builder.slc_readback]:
                with self.assertRaises(ValueError): read(p)
        self.use_fixture(fixture(closed=False),check)

    def test_gap_record_rejected_by_both_readers(self):
        def check(p):
            for read in [export.independent_read,self.builder.slc_readback]:
                with self.assertRaises(ValueError): read(p)
        self.use_fixture(fixture(gaps=1),check)

    def test_wrong_units_rejected_by_both_readers(self):
        def check(p):
            for read in [export.independent_read,self.builder.slc_readback]:
                with self.assertRaises(ValueError): read(p)
        self.use_fixture(fixture(units='UM'),check)

    def test_relief_has_no_cylinder_or_zero_height_annuli(self):
        v,f=export.relief_mesh([1.,1.,0.],3.,segments=32)
        np.testing.assert_allclose(np.ptp(v,axis=0),[4.,4.,1.],atol=1e-12)
        self.assertEqual(v[:,2].min(),0)
        mesh=self.builder.slc_mesh_from_arrays(v*.001,f)
        self.assertAlmostEqual(mesh['volume_mm3']*1e9,16*np.sin(2*np.pi/32)*4,places=10)

    def test_disconnected_relief_is_not_silently_bridged(self):
        with self.assertRaises(ValueError):
            export.relief_mesh([1.,0.,1.],3.)

    def test_scale_override_prevents_builder_enlargement(self):
        v,f=export.relief_mesh([1.],17.)
        mesh=self.builder.slc_mesh_from_arrays(v*.001,f)
        options=self.builder.slc_options(layer_um=.05,xy_mm_per_um=.001,z_mm_per_um=.001,
                                          xy_origin='preserve',z_origin='preserve')
        converted,factors,translation=self.builder.slc_transform_mesh(mesh,options)
        np.testing.assert_array_equal(factors,[1.,1.,1.])
        np.testing.assert_array_equal(translation,[0.,0.,0.])
        np.testing.assert_allclose(np.ptp(converted['vertices'],axis=0),[.034,.034,.001])
        # Demonstrate the unrelated GUI defaults would alter the physical design.
        _,wrong,_=self.builder.slc_transform_mesh(mesh,self.builder.slc_options())
        np.testing.assert_array_equal(wrong,[240.,240.,100.])

    def test_minimum_z_serialization_fixes_known_solid_volume(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT.parent) as temp:
            source=Path(temp)/'midpoint.slc'
            corrected=Path(temp)/'minimum.slc'
            source.write_bytes(fixture())
            before=export.independent_read(source)
            after=export.write_min_z_slc(source,corrected,.05)
            self.assertEqual(after['sampling'][0],0)
            self.assertEqual(after['layers'][0][0],0)
            self.assertEqual(after['top_mm'],before['top_mm'])
            for old,new in zip(before['layers'][0][1],after['layers'][0][1]):
                np.testing.assert_array_equal(old,new)
            # 3 µm² net square area over a 0.05 µm slab =0.15 µm³.
            vol=sum(export.area(p) for p in after['layers'][0][1])*after['top_mm']*1e9
            self.assertAlmostEqual(vol,.15,places=7)
            self.builder.slc_readback(corrected)

    def test_old_M01_cannot_pass_standard_minimum_z_verification(self):
        old=SCRIPT.parents[1]/'exports/metalens_only_slc_v1/M01.slc'
        design=__import__('json').loads(export.DEFAULT_DESIGN.read_text())
        v,f=export.relief_mesh(design['heights_um'],design['cylinder_diameter_um']/2)
        with self.assertRaises((ValueError,AssertionError)):
            export.verify(old,design,v,f,128,.05)


if __name__=='__main__':
    unittest.main(verbosity=2)
