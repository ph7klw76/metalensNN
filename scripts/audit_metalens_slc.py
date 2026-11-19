"""Independent, read-only audit of this project's frozen metalens SLC.

No exporter/builder code is imported. SLC Z records are interpreted as minimum
Z levels: each contour is held to the next record or terminal top. This differs
from treating contour Z as the centre of a voxel. Only the current centrally
connected, monotonically decreasing annular relief, alone or on its attached
solid cylinder, is supported by this audit.
The aligned regular-polygon tessellation is checked separately from circles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np


POSITION_TOLERANCE_UM = 1e-5
SYMMETRIC_DIFFERENCE_RELATIVE_TOLERANCE = 1e-6
AREA_RELATIVE_TOLERANCE = 1e-6
FORMAT_REFERENCE = 'https://paulbourke.org/dataformats/slc/'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def signed_area(p):
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                        - p[:, 1] * np.roll(p[:, 0], -1)) / 2)


def read_slc(raw):
    """Parse records directly from the supplied bytes, without writer helpers."""
    end = raw.find(b'\r\n\x1a', 0, 2048)
    if end < 0 or end + 3 > 2048:
        raise ValueError('Missing/oversized ASCII SLC header')
    header = raw[:end].decode('ascii')
    for required in [r'-SLCVER\s+2\.0(?:\s|$)', r'-UNIT\s+MM(?:\s|$)',
                     r'-TYPE\s+PART(?:\s|$)']:
        if not re.search(required, header):
            raise ValueError('Expected SLC 2.0 PART in MM units')
    match = re.search(r'-EXTENTS\s+([^ ]+)\s+([^ ]+)\s+([^ ]+)', header)
    if not match:
        raise ValueError('Missing EXTENTS')
    extents = np.asarray([[float(x) * 1000 for x in token.split(',')]
                          for token in match.groups()])
    if extents.shape != (3, 2) or not np.isfinite(extents).all():
        raise ValueError('Invalid EXTENTS')
    cursor = end + 3
    if len(raw) < cursor + 257:
        raise ValueError('Truncated reserved/table area')
    reserved_zero = not any(raw[cursor:cursor + 256])
    cursor += 256
    count = raw[cursor]
    cursor += 1
    if count != 1:
        raise ValueError('This fixed-spacing audit expects one sampling entry')

    def take(fmt):
        nonlocal cursor
        length = struct.calcsize(fmt)
        if cursor + length > len(raw):
            raise ValueError('Truncated little-endian record')
        value = struct.unpack_from(fmt, raw, cursor)
        cursor += length
        return value

    sampling = np.asarray(take('<4f'), dtype=float)
    if not np.isfinite(sampling).all() or sampling[1] <= 0:
        raise ValueError('Invalid sampling entry')
    if sampling[2] != 0 or sampling[3] != 0:
        raise ValueError('Unexpected line-width compensation/reserved value')
    layers = []
    while True:
        z_mm, nboundaries = take('<fI')
        if not math.isfinite(z_mm):
            raise ValueError('Nonfinite Z')
        if nboundaries == 0xffffffff:
            if not layers or z_mm * 1000 <= layers[-1]['z_um']:
                raise ValueError('Invalid terminal top')
            if cursor != len(raw):
                raise ValueError('Trailing bytes after terminal record')
            top_um = z_mm * 1000
            break
        if not nboundaries or nboundaries > (len(raw) - cursor) // 8:
            raise ValueError('Empty/implausible boundary count')
        if layers and z_mm * 1000 <= layers[-1]['z_um']:
            raise ValueError('Z records are not strictly increasing')
        loops = []
        for _ in range(nboundaries):
            nvertices, ngaps = take('<II')
            if nvertices < 4 or ngaps or nvertices > (len(raw) - cursor) // 8:
                raise ValueError('Invalid PART vertex/gap count')
            xy = np.asarray(take('<' + 'f' * (2 * nvertices))).reshape(-1, 2)
            if not np.isfinite(xy).all() or not np.array_equal(xy[0], xy[-1]):
                raise ValueError('Nonfinite/open contour')
            if np.any(np.all(np.diff(xy, axis=0) == 0, axis=1)):
                raise ValueError('Zero-length segment')
            loops.append(xy[:-1] * 1000)
        layers.append({'z_um': z_mm * 1000, 'loops': loops})
    if abs(sampling[0] * 1000 - layers[0]['z_um']) > 1e-10:
        raise ValueError('Sampling start does not equal first contour Z')
    return {'header': header, 'extents_um': extents, 'sampling': sampling,
            'layers': layers, 'top_um': top_um,
            'reserved_bytes_zero': reserved_zero}


def self_intersections(p):
    """Detect intersections of non-adjacent edges, including collinear overlap."""
    q = np.roll(p, -1, axis=0)
    a, b = p[:, None, :], q[:, None, :]
    c, d = p[None, :, :], q[None, :, :]

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    ab, cd = b - a, d - c
    o1, o2 = cross(ab, c - a), cross(ab, d - a)
    o3, o4 = cross(cd, a - c), cross(cd, b - c)
    # Cross-product signs, plus closed bounding boxes, cover touching/overlap.
    eps = 1e-12
    straddle1 = ((o1 <= eps) & (o2 >= -eps)) | ((o2 <= eps) & (o1 >= -eps))
    straddle2 = ((o3 <= eps) & (o4 >= -eps)) | ((o4 <= eps) & (o3 >= -eps))
    overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                     <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + eps, axis=2)
    hit = straddle1 & straddle2 & overlap
    n = len(p)
    indices = np.arange(n)
    adjacent = ((indices[:, None] - indices[None, :]) % n)
    hit &= (adjacent != 0) & (adjacent != 1) & (adjacent != n - 1)
    return int(np.count_nonzero(np.triu(hit, 1)))


def polygon(radius, segments):
    angle = np.arange(segments) * (2 * math.pi / segments)
    return radius * np.column_stack([np.cos(angle), np.sin(angle)])


def point_boundary_distances(points, boundary):
    starts = boundary
    edges = np.roll(boundary, -1, axis=0) - starts
    delta = points[:, None, :] - starts[None, :, :]
    t = np.clip(np.sum(delta * edges[None, :, :], axis=2)
                / np.sum(edges * edges, axis=1)[None, :], 0, 1)
    separation = delta - t[:, :, None] * edges[None, :, :]
    return np.sqrt(np.min(np.sum(separation * separation, axis=2), axis=1))


def convex_contains(points, boundary, tolerance=1e-12):
    edges = np.roll(boundary, -1, axis=0) - boundary
    delta = points[:, None, :] - boundary[None, :, :]
    cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
    return bool(np.all(cross >= -tolerance))


def intersection_area(subject, clip):
    """Convex polygon clipping; expected disks are aligned regular polygons."""
    if convex_contains(subject, clip):
        return signed_area(subject)
    if convex_contains(clip, subject):
        return signed_area(clip)
    current = subject.copy()
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if not len(current):
            return 0.0
        edge = b - a
        delta = current - a
        side = edge[0] * delta[:, 1] - edge[1] * delta[:, 0]
        inside = side >= -1e-13
        output = []
        for i in range(len(current)):
            j = (i + 1) % len(current)
            if inside[i]:
                output.append(current[i])
            if inside[i] != inside[j]:
                output.append(current[i] + (current[j] - current[i])
                              * (side[i] / (side[i] - side[j])))
        current = np.asarray(output, dtype=float).reshape(-1, 2)
    return max(0.0, signed_area(current)) if len(current) >= 3 else 0.0


def audit(raw, source_raw, layer_um=.05, segments=128, component='relief-only'):
    """Compare minimum-Z contour occupancy with source JSON, never writer output.

    ``relief-only`` retains the original API and zero-based relief coordinates.
    ``assembly`` includes the full source cylinder from zero through H_um and
    its top relief through H_um + max(heights_um), as one material solid.
    """
    if component not in ('relief-only', 'assembly'):
        raise ValueError('Component must be relief-only or assembly')
    parsed = read_slc(raw)
    source = json.loads(source_raw.decode('utf-8-sig'))
    if source.get('units') != 'um':
        raise ValueError('Source design must explicitly use um')
    heights = np.asarray(source['heights_um'], dtype=float)
    if (heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all()
            or np.any(heights < 0) or np.any(np.diff(heights) > POSITION_TOLERANCE_UM)
            or heights[0] <= 0):
        raise ValueError('Audit supports the current nonincreasing connected radial profile only')
    aperture_radius = float(source['cylinder_diameter_um']) / 2
    if not math.isfinite(aperture_radius) or aperture_radius <= 0:
        raise ValueError('Invalid source aperture')
    support_height = float(source['H_um']) if component == 'assembly' else 0.0
    if (not math.isfinite(support_height) or support_height < 0
            or (component == 'assembly' and support_height <= 0)):
        raise ValueError('Assembly requires a finite positive source H_um')
    if layer_um <= 0 or not math.isfinite(layer_um) or segments < 16 or segments % 4:
        raise ValueError('Invalid layer spacing or regular polygon segment count')
    edges = np.linspace(0, aperture_radius, len(heights) + 1)
    relief_footprint = float(edges[1:][heights > 0][-1])
    footprint = aperture_radius if component == 'assembly' else relief_footprint
    absolute_heights = heights + support_height
    expected_top = float(absolute_heights.max())

    def source_radius(z):
        if z < 0:
            return 0.0
        active = absolute_heights > z
        return float(edges[1:][active][-1]) if np.any(active) else 0.0

    factor = segments * math.sin(2 * math.pi / segments) / 2
    source_volume = float(factor * np.sum(np.diff(edges ** 2) * absolute_heights))
    circular_volume = float(math.pi * np.sum(np.diff(edges ** 2) * absolute_heights))
    layers = parsed['layers']
    zs = np.asarray([v['z_um'] for v in layers])
    tops = np.r_[zs[1:], parsed['top_um']]
    areas, radii = [], []
    failures = []
    shape_errors, area_errors = [], []
    intersection_count = 0
    winding_pass = True
    convexity_pass = True
    for i, item in enumerate(layers):
        if len(item['loops']) != 1:
            raise ValueError('Current frozen profile requires exactly one exterior per slab')
        p = item['loops'][0]
        area = signed_area(p)
        areas.append(area)
        radii.append(float(np.sqrt(np.sum(p * p, axis=1)).max()))
        winding_pass &= area > 0
        intersection_count += self_intersections(p)
        # Collinear extra vertices produced by triangle slicing are acceptable.
        a = np.roll(p, -1, axis=0) - p
        b = np.roll(a, -1, axis=0)
        turn = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        # Float32 quantization perturbs the collinear triangle-edge vertices.
        # Bound the turn by a distance tolerance, rather than fixed area units.
        turn_allowance = POSITION_TOLERANCE_UM * (np.linalg.norm(a, axis=1)
                                                 + np.linalg.norm(b, axis=1))
        convexity_pass &= bool(np.all(turn >= -turn_allowance))
        midpoint = (zs[i] + tops[i]) / 2
        expected_radius = source_radius(midpoint)
        if not expected_radius:
            shape_errors.append(float('inf'))
            area_errors.append(float('inf'))
            continue
        expected = polygon(expected_radius, segments)
        shape_errors.append(max(float(point_boundary_distances(p, expected).max()),
                                float(point_boundary_distances(expected, p).max())))
        area_errors.append(abs(area / (factor * expected_radius ** 2) - 1))
    areas = np.asarray(areas)
    radii = np.asarray(radii)
    all_xy = np.concatenate([item['loops'][0] for item in layers])
    xy_bounds = np.column_stack([all_xy.min(axis=0), all_xy.max(axis=0)])
    expected_bounds = np.array([[-footprint, footprint], [-footprint, footprint],
                               [0, expected_top]])
    actual_bounds = np.vstack([xy_bounds, [zs[0], parsed['top_um']]])
    max_extent_error = float(np.max(np.abs(parsed['extents_um'] - expected_bounds)))
    max_dimension_error = float(np.max(np.abs(actual_bounds - expected_bounds)))
    expected_count = round(expected_top / layer_um)
    grid_error = (float(np.max(abs(zs - np.arange(len(zs)) * layer_um)))
                  if len(zs) else float('inf'))

    # Genuine minimum-Z reconstruction, including an empty region below first Z.
    integration_levels = np.unique(np.r_[0, absolute_heights, zs, parsed['top_um']])
    reconstructed_volume = 0.0
    symmetric_difference = 0.0
    cache = {}
    for lower, upper in zip(integration_levels, integration_levels[1:]):
        middle = (lower + upper) / 2
        radius = source_radius(middle)
        expected_area = factor * radius ** 2
        index = int(np.searchsorted(zs, middle, side='right') - 1)
        if middle >= parsed['top_um']:
            index = -1
        actual_area = float(areas[index]) if index >= 0 else 0.0
        reconstructed_volume += actual_area * (upper - lower)
        if index < 0 or not radius:
            difference_area = expected_area + actual_area
        elif not winding_pass or not convexity_pass or intersection_count:
            difference_area = float('inf')
        else:
            key = (index, radius)
            if key not in cache:
                overlap = intersection_area(layers[index]['loops'][0], polygon(radius, segments))
                cache[key] = max(0.0, actual_area + expected_area - 2 * overlap)
            difference_area = cache[key]
        symmetric_difference += difference_area * (upper - lower)
    transition_errors = []
    intervals_contiguous = True
    for i, height in enumerate(absolute_heights):
        if height <= 0:
            continue
        # The annular-bin interior is far from contour edges; no tolerance rescue.
        active_indices = np.flatnonzero(radii > (edges[i] + edges[i + 1]) / 2)
        if not len(active_indices):
            transition_errors.append(float('inf'))
            continue
        intervals_contiguous &= bool(np.all(np.diff(active_indices) == 1))
        transition_errors.append(abs(float(tops[active_indices[-1]]) - float(height)))
    max_transition_error = max(transition_errors)
    relative_difference = symmetric_difference / source_volume

    bounds_label = 'assembly' if component == 'assembly' else 'relief'
    checks = {
        'header_sampling_binary_full_length': True,
        'little_endian_float32_uint32': True,
        'finite_closed_gap_free_contours': True,
        'counterclockwise_exteriors': bool(winding_pass),
        'no_self_intersections': intersection_count == 0,
        'convex_current_profile_contours': bool(convexity_pass),
        'all_contour_boundaries_match_source_at_slab_midpoints': max(shape_errors) <= POSITION_TOLERANCE_UM,
        'all_contour_areas_match_source_at_slab_midpoints': max(area_errors) <= AREA_RELATIVE_TOLERANCE,
        f'header_extents_match_{bounds_label}': max_extent_error <= POSITION_TOLERANCE_UM,
        ('physical_bounds_and_attached_cylinder_match_assembly'
         if component == 'assembly' else 'physical_bounds_and_no_cylinder_match_relief'):
            max_dimension_error <= POSITION_TOLERANCE_UM,
        'explicit_fixed_layer_count': len(zs) == expected_count,
        'minimum_z_record_grid_starts_at_zero': grid_error <= POSITION_TOLERANCE_UM,
        'sampling_spacing_matches_requested': abs(parsed['sampling'][1] * 1000 - layer_um) <= POSITION_TOLERANCE_UM,
        'standard_minimum_z_step_transitions': max_transition_error <= POSITION_TOLERANCE_UM,
        'radial_bins_have_contiguous_z_occupancy': bool(intervals_contiguous),
        'standard_minimum_z_symmetric_difference': relative_difference <= SYMMETRIC_DIFFERENCE_RELATIVE_TOLERANCE,
    }
    if component == 'assembly':
        cylinder_slabs = np.flatnonzero((zs + tops) / 2 < support_height)
        # The source's expected cross section below H is the full cylinder;
        # reuse those already independently calculated boundary discrepancies.
        cylinder_errors = [shape_errors[i] for i in cylinder_slabs]
        interface_record_error = float(np.min(np.abs(zs - support_height)))
        checks.update({
            'full_cylinder_cross_sections_below_relief_base':
                bool(cylinder_errors) and max(cylinder_errors) <= POSITION_TOLERANCE_UM,
            'explicit_cylinder_relief_interface_record':
                interface_record_error <= POSITION_TOLERANCE_UM,
            'one_connected_solid_no_internal_material_boundaries':
                bool(winding_pass and convexity_pass and not intersection_count
                     and np.all(areas > 0) and intervals_contiguous
                     and max(shape_errors) <= POSITION_TOLERANCE_UM
                     and grid_error <= POSITION_TOLERANCE_UM
                     and len(zs) == expected_count),
        })
    failures = [key for key, value in checks.items() if not value]
    return {
        'status': 'PASS' if not failures else 'FAIL',
        'scope': ('Independent contour/geometry audit of frozen attached cylinder and top relief'
                  if component == 'assembly' else
                  'Independent contour/geometry audit of frozen metalens relief only'),
        'component': component,
        'format_reference': FORMAT_REFERENCE,
        'z_semantics': 'Each contour applies from its minimum Z to the next record or terminal top; no midpoint reinterpretation',
        'checks': checks, 'failed_checks': failures,
        'tolerances': {'position_um': POSITION_TOLERANCE_UM,
                       'relative_cross_section_area': AREA_RELATIVE_TOLERANCE,
                       'relative_symmetric_difference_to_regular_polygon_mesh': SYMMETRIC_DIFFERENCE_RELATIVE_TOLERANCE},
        'measurements': {
            'bytes': len(raw), 'layer_count': len(zs),
            'sampling_start_um': float(parsed['sampling'][0] * 1000),
            'sampling_spacing_um': float(parsed['sampling'][1] * 1000),
            'first_minimum_z_um': float(zs[0]), 'last_minimum_z_um': float(zs[-1]),
            'terminal_top_um': parsed['top_um'],
            'actual_bounds_um': actual_bounds.tolist(), 'expected_bounds_um': expected_bounds.tolist(),
            'maximum_contour_boundary_distance_um': max(shape_errors),
            'maximum_cross_section_area_relative_error': max(area_errors),
            'nonadjacent_edge_intersections': intersection_count,
            'maximum_minimum_z_grid_error_um': grid_error,
            'maximum_step_transition_error_um': max_transition_error,
            'source_regular_polygon_volume_um3': source_volume,
            'standard_minimum_z_volume_um3': reconstructed_volume,
            'signed_volume_error_um3': reconstructed_volume - source_volume,
            'signed_volume_relative_error': reconstructed_volume / source_volume - 1,
            'symmetric_difference_volume_um3': symmetric_difference,
            'symmetric_difference_relative_to_regular_polygon_mesh': relative_difference,
            'source_circular_annuli_volume_um3': circular_volume,
            'expected_tessellation_volume_relative_error_to_circles': abs(source_volume / circular_volume - 1),
            'expected_tessellation_maximum_sagitta_um': footprint * (1 - math.cos(math.pi / segments)),
            'azimuthal_segments': segments,
            ('source_support_height_included_um' if component == 'assembly'
             else 'source_support_height_excluded_um'): source.get('H_um'),
            'source_declared_aperture_um': 2 * aperture_radius,
            'source_nonzero_relief_footprint_um': 2 * relief_footprint,
            'reserved_bytes_zero': parsed['reserved_bytes_zero'],
        },
        'printer_import_tested': False, 'exposure_calibration_verified': False,
        'optical_performance_verified_by_this_audit': False,
        'limitations': ['This audit supports the current nonincreasing centrally connected annular profile and its solid cylindrical base.',
                        f'Regular-{segments}-gon fidelity and its deviation from circles are separate measurements.',
                        'No receiving-printer import, scan paths, exposure, shrinkage or optical validation is performed.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slc', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layer-um', type=float, default=.05)
    parser.add_argument('--segments', type=int, default=128)
    parser.add_argument('--component', choices=['relief-only', 'assembly'], default='relief-only')
    args = parser.parse_args()
    raw = args.slc.read_bytes()
    source_raw = args.design.read_bytes()
    original_hash, design_hash = sha256(raw), sha256(source_raw)
    try:
        result = audit(raw, source_raw, args.layer_um, args.segments, args.component)
    except (ValueError, IndexError, KeyError, OverflowError, struct.error) as exc:
        result = {'status': 'FAIL', 'error': str(exc), 'failed_checks': ['parse_or_supported_source_shape']}
    unchanged = (sha256(args.slc.read_bytes()) == original_hash
                 and sha256(args.design.read_bytes()) == design_hash)
    result['provenance'] = {'slc': str(args.slc.resolve()), 'slc_sha256': original_hash,
                            'source_design': str(args.design.resolve()),
                            'source_design_sha256': design_hash,
                            'slc_and_source_unchanged_during_audit': unchanged}
    if not unchanged:
        result['status'] = 'FAIL'
        result.setdefault('failed_checks', []).append('input_hashes_unchanged')
    # JSON must remain machine-readable even when a malformed shape yields infinity.
    def sanitize(value):
        if isinstance(value, np.generic):
            return sanitize(value.item())
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    content = json.dumps(sanitize(result), indent=2, allow_nan=False)
    if args.output:
        output = args.output.resolve()
        if output in [args.slc.resolve(), args.design.resolve()]:
            raise ValueError('Audit output cannot overwrite input SLC/design')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content + '\n', encoding='utf-8')
    print(content)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
