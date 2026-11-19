"""Independent byte fixtures test attached-cylinder minimum-Z solid auditing.

No exporter, builder or mesh generator is imported. A small analytic solid is
encoded directly so agreement cannot come from sharing the writer's geometry.
"""
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'independent_cylinder_slc_audit', ROOT / 'scripts/audit_metalens_slc.py')
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)

SOURCE = json.dumps({'units': 'um', 'H_um': 2., 'cylinder_diameter_um': 6.,
                     'heights_um': [1., .5, 0.]}).encode()


def analytic_fixture(*, stored_offset=0., physical_support_height=2.,
                     radius_scale=1., missing_index=None, duplicated_index=None,
                     units='MM', reversed_index=None):
    """D6 cylinder H2, radius2 shelf .5, radius1 shelf .5, dz=.25.

    The radius law and binary packing below are deliberately independent of
    auditor helpers. Defects change geometry or encoding rather than expected
    assertions. Every coordinate is serialized in physical millimetres.
    """
    dz = .25
    top = physical_support_height + 1.
    count = round(top / dz)
    header = (f'-SLCVER 2.0 -UNIT {units} -TYPE PART -PACKAGE ANALYTIC_TEST '
              f'-EXTENTS {-3*radius_scale/1000:.12g},{3*radius_scale/1000:.12g} '
              f'{-3*radius_scale/1000:.12g},{3*radius_scale/1000:.12g} '
              f'{stored_offset/1000:.12g},{top/1000:.12g}\r\n\x1a').encode()
    raw = bytearray(header + bytes(256))
    raw += struct.pack('<B4f', 1, stored_offset / 1000, dz / 1000, 0., 0.)
    for i in range(count):
        if i == missing_index:
            continue
        sample = (i + .5) * dz
        if sample < physical_support_height:
            radius = 3.
        elif sample < physical_support_height + .5:
            radius = 2.
        else:
            radius = 1.
        angles = [2 * math.pi * j / 16 for j in range(16)]
        points = [(radius * radius_scale * math.cos(a) / 1000,
                   radius * radius_scale * math.sin(a) / 1000) for a in angles]
        if i == reversed_index:
            points.reverse()
        points.append(points[0])
        copies = 2 if i == duplicated_index else 1
        raw += struct.pack('<fI', (i * dz + stored_offset) / 1000, copies)
        for _ in range(copies):
            raw += struct.pack('<II', len(points), 0)
            raw += struct.pack('<' + 'f' * (2 * len(points)),
                               *(x for point in points for x in point))
    raw += struct.pack('<fI', top / 1000, 0xffffffff)
    return bytes(raw)


class TestCylinderMetalensSLCAudit(unittest.TestCase):
    def run_audit(self, raw, **kwargs):
        return audit_module.audit(raw, SOURCE, layer_um=.25, segments=16,
                                  component='assembly', **kwargs)

    def assert_rejected(self, raw):
        try:
            result = self.run_audit(raw)
        except ValueError:
            return
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(result['failed_checks'])

    def test_independently_constructed_connected_assembly_passes(self):
        result = self.run_audit(analytic_fixture())
        self.assertEqual(result['status'], 'PASS', result['failed_checks'])
        measurements = result['measurements']
        self.assertEqual(measurements['layer_count'], 12)
        self.assertEqual(measurements['expected_bounds_um'],
                         [[-3., 3.], [-3., 3.], [0., 3.]])
        self.assertEqual(measurements['source_support_height_included_um'], 2.)
        self.assertEqual(measurements['source_nonzero_relief_footprint_um'], 4.)
        # H2*R3² + h1*R1² + h.5*(R2²-R1²), independently analytic.
        known_volume = 8 * math.sin(math.pi / 8) * (18 + 1 + 1.5)
        self.assertAlmostEqual(measurements['standard_minimum_z_volume_um3'],
                               known_volume, places=5)
        self.assertLess(measurements['symmetric_difference_relative_to_regular_polygon_mesh'],
                        1e-6)
        self.assertTrue(result['checks']['explicit_cylinder_relief_interface_record'])
        self.assertTrue(result['checks']['one_connected_solid_no_internal_material_boundaries'])

    def test_midpoint_z_labels_fail_even_with_syntactically_valid_table(self):
        result = self.run_audit(analytic_fixture(stored_offset=.125))
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['checks']['minimum_z_record_grid_starts_at_zero'])
        self.assertFalse(result['checks']['standard_minimum_z_symmetric_difference'])

    def test_missing_cylinder_fails(self):
        self.assert_rejected(analytic_fixture(physical_support_height=0.))

    def test_cylinder_radius_wrong_by_10_percent_fails(self):
        result = self.run_audit(analytic_fixture(radius_scale=.9))
        self.assertFalse(result['checks']['full_cylinder_cross_sections_below_relief_base'])
        self.assertEqual(result['status'], 'FAIL')

    def test_shifted_support_relief_interface_fails(self):
        result = self.run_audit(analytic_fixture(physical_support_height=2.25))
        self.assertFalse(result['checks']['all_contour_boundaries_match_source_at_slab_midpoints'])
        self.assertFalse(result['checks']['standard_minimum_z_step_transitions'])
        self.assertEqual(result['status'], 'FAIL')

    def test_missing_layer_at_interface_fails(self):
        result = self.run_audit(analytic_fixture(missing_index=8))
        self.assertFalse(result['checks']['explicit_cylinder_relief_interface_record'])
        self.assertFalse(result['checks']['explicit_fixed_layer_count'])
        self.assertFalse(result['checks']['one_connected_solid_no_internal_material_boundaries'])

    def test_duplicate_internal_exterior_is_not_accepted_as_one_solid(self):
        self.assert_rejected(analytic_fixture(duplicated_index=8))

    def test_clockwise_exterior_fails(self):
        result = self.run_audit(analytic_fixture(reversed_index=8))
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['checks']['counterclockwise_exteriors'])

    def test_truncated_terminal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit(analytic_fixture()[:-1])

    def test_wrong_units_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit(analytic_fixture(units='UM'))

    def test_assembly_requires_defined_positive_support_height(self):
        for height in [0., -1., float('nan')]:
            invalid = json.loads(SOURCE)
            invalid['H_um'] = height
            with self.assertRaises(ValueError):
                audit_module.audit(analytic_fixture(), json.dumps(invalid).encode(),
                                   .25, 16, 'assembly')

    def test_original_relief_regressions_keep_m01_failed_and_m02_passed(self):
        source = (ROOT / 'artifacts/1550nm_natural_nn_v1/designs/NN.json').read_bytes()
        for relative, expected in [
                ('exports/metalens_only_slc_v1/M01.slc', 'FAIL'),
                ('exports/metalens_only_slc_v2/M02.slc', 'PASS')]:
            with self.subTest(file=relative):
                result = audit_module.audit((ROOT / relative).read_bytes(), source)
                self.assertEqual(result['status'], expected, result['failed_checks'])
                self.assertEqual(result['component'], 'relief-only')

    def test_relief_only_export_cannot_pass_as_complete_assembly(self):
        source = (ROOT / 'artifacts/1550nm_natural_nn_v1/designs/NN.json').read_bytes()
        raw = (ROOT / 'exports/metalens_only_slc_v2/M02.slc').read_bytes()
        result = audit_module.audit(raw, source, component='assembly')
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['checks']['physical_bounds_and_attached_cylinder_match_assembly'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
