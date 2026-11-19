"""Read a desktop export directly to check layer and radial geometry."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


POSITION_TOLERANCE_UM = 1e-5
LAYER_UM = 0.5
SEGMENTS = 128


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_export(path, expected_height):
    raw = path.read_bytes()
    manifest = json.loads(path.with_suffix('.manifest.json').read_bytes())
    audit_bytes = path.with_suffix('.audit.json').read_bytes()
    audit = json.loads(audit_bytes)
    source_bytes = manifest['source_design_exact_json_utf8'].encode('utf-8')
    design = json.loads(source_bytes)
    require(design == manifest['source_design'], 'Manifest source disagrees with its exact JSON')

    height = design['H_um']
    relief = design['heights_um']
    radius = design['cylinder_diameter_um'] / 2
    require(height == expected_height and height * 2 == round(height * 2), 'Unexpected cylinder height')
    require(radius == 25 and design['units'] == 'um', 'Unexpected cylinder footprint or units')
    require(len(relief) == 50 and all(0 <= h <= 3 and h * 2 == round(h * 2) for h in relief),
            'Unexpected 500 nm relief profile')
    require(all(a >= b for a, b in zip(relief, relief[1:])), 'Expected one connected cross-section at every layer')
    slc_sha = hashlib.sha256(raw).hexdigest()
    require(audit['status'] == 'PASS' and all(audit['checks'].values()), 'Export-side audit did not pass')
    require(audit['provenance']['slc_sha256'] == slc_sha, 'SLC SHA mismatch')
    require(audit['provenance']['source_design_sha256'] == hashlib.sha256(source_bytes).hexdigest(),
            'Source SHA mismatch')
    require(manifest['independent_audit_sha256'] == hashlib.sha256(audit_bytes).hexdigest(), 'Audit SHA mismatch')
    profile_sha = hashlib.sha256(struct.pack('<50f', *relief)).hexdigest()
    provenance = design.get('inference_provenance') or {}
    if 'export_profile_sha256' in provenance:
        require(provenance['export_profile_sha256'] == profile_sha, 'Inferred profile SHA mismatch')

    header_end = raw.index(b'\r\n\x1a')
    header = raw[:header_end].decode('ascii')
    require('-SLCVER 2.0' in header and '-UNIT MM' in header and '-TYPE PART' in header, 'Wrong header')
    extents = re.search(r'-EXTENTS\s+(\S+)\s+(\S+)\s+(\S+)', header).groups()
    absolute = [height + h for h in relief]
    total = max(absolute)
    for token, expected in zip(extents, ((-25, 25), (-25, 25), (0, total))):
        require(all(abs(float(v) * 1000 - w) <= POSITION_TOLERANCE_UM
                    for v, w in zip(token.split(','), expected, strict=True)), 'Incorrect header dimensions')
    position = header_end + 3
    require(raw[position:position + 256] == bytes(256), 'Nonzero reserved bytes')
    position += 256
    require(raw[position] == 1, 'SLC must have one fixed sampling interval')
    position += 1
    sampling_start, sampling_spacing, compensation, reserved = struct.unpack_from('<4f', raw, position)
    position += 16
    require(sampling_start == compensation == reserved == 0, 'Unexpected sampling table values')
    require(abs(sampling_spacing * 1000 - LAYER_UM) <= POSITION_TOLERANCE_UM, 'Incorrect sampling spacing')

    radial_step = radius / len(absolute)
    expected_count = round(total / LAYER_UM)
    recovered_tops = [0.] * len(absolute)
    z_layers = []
    max_radius_error = 0.
    count_full_radius = 0
    polygon_area_coefficient = SEGMENTS * math.sin(2 * math.pi / SEGMENTS) / 2
    for index in range(expected_count):
        z_mm, contours, points, gaps = struct.unpack_from('<fIII', raw, position)
        position += 16
        z_um = z_mm * 1000
        z_layers.append(z_um)
        require(abs(z_um - index * LAYER_UM) <= POSITION_TOLERANCE_UM, f'Incorrect lower bound at layer {index}')
        require(contours == 1 and gaps == 0 and points >= SEGMENTS + 1, f'Incorrect topology at layer {index}')
        xy = [(x * 1000, y * 1000) for x, y in struct.iter_unpack('<ff', raw[position:position + points * 8])]
        position += points * 8
        require(len(xy) == points and all(math.isfinite(x) and math.isfinite(y) for x, y in xy), 'Incomplete contour')
        require(xy[0] == xy[-1] and all(a != b for a, b in zip(xy, xy[1:])), 'Open or degenerate contour')
        sample_z = (index + 0.5) * LAYER_UM
        expected_radius = sum(t > sample_z for t in absolute) * radial_step
        observed_radius = max(math.hypot(x, y) for x, y in xy)
        max_radius_error = max(max_radius_error, abs(observed_radius - expected_radius))
        require(abs(observed_radius - expected_radius) <= POSITION_TOLERANCE_UM, f'Wrong radius at layer {index}')
        count_full_radius += abs(observed_radius - radius) <= POSITION_TOLERANCE_UM
        area = math.fsum(x * v - y * u for (x, y), (u, v) in zip(xy, xy[1:])) / 2
        require(abs(area / (expected_radius ** 2 * polygon_area_coefficient) - 1) < 2e-6,
                f'Wrong 128-gon area or winding at layer {index}')
        for band in range(len(absolute)):
            if observed_radius > (band + 0.5) * radial_step:
                recovered_tops[band] = (index + 1) * LAYER_UM

    terminal_z, sentinel = struct.unpack_from('<fI', raw, position)
    terminal_z *= 1000
    require(sentinel == 0xffffffff and position + 8 == len(raw), 'Missing or nonterminal end record')
    require(abs(terminal_z - total) <= POSITION_TOLERANCE_UM, 'Incorrect top Z')
    require(recovered_tops == absolute, 'Binary annular transitions differ from the source')
    require(count_full_radius >= round(height / LAYER_UM), 'No full-length solid cylinder')
    return {
        'status': 'PASS', 'slc': str(path), 'slc_sha256': slc_sha,
        'profile_sha256_float32': profile_sha, 'cylinder_height_um': height,
        'cylinder_diameter_um': radius * 2, 'observed_terminal_height_um': terminal_z,
        'predicted_useful_modal_efficiency': provenance.get('predicted_useful_modal_efficiency'),
        'layer_count': expected_count, 'full_radius_layer_count': count_full_radius,
        'checked_annular_transitions': len(absolute), 'max_radius_error_um': max_radius_error,
        'max_observed_layer_interval_error_um': max(abs(b - a - LAYER_UM)
                                                    for a, b in zip(z_layers, z_layers[1:] + [terminal_z])),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slc', type=Path)
    parser.add_argument('height_um', type=float)
    args = parser.parse_args()
    print(json.dumps(inspect_export(args.slc, args.height_um), indent=2, allow_nan=False))
