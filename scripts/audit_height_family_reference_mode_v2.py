"""Read-only source-mode diagnostics from saved homogeneous Meep references.

No Meep/JAX import, optical solve, fitted amplitude or acceptance threshold is
introduced. Reconstruct the saved incident E/H plane using the frozen power
convention, propagate to the fiber facet, and compare with the declared
vector-projected Gaussian on the identical Cartesian grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from fiber_metalens_nn.collimation_dft_power import (
    centered_lp_to_cartesian, native_flux_quadrature,
    paraxial_power_equivalent_fields,
)
from fiber_metalens_nn.collimation_optics import (
    fingerprint, flux, forward_spectrum, kernels, overlap, propagate,
    vector_spectrum,
)
from fiber_metalens_nn.metrics import full_width


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ('H_um', 'resolution', 'pml_um', 'padding_um', 'duration')


def local_path(value):
    text = str(value)
    if os.name == 'nt' and re.match(r'^/mnt/[a-z]/', text):
        return Path(text[5].upper()+':/'+text[7:])
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', text):
        return Path('/mnt/'+text[0].lower()+'/'+text[3:].replace('\\', '/'))
    return Path(value)


def read(path):
    return json.loads(local_path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(local_path(path).read_bytes()).hexdigest()


def identity(entry):
    return tuple(float(entry[key]) for key in IDENTITY_KEYS)


def widths(field, x):
    intensity = np.sum(np.abs(field[:3])**2, axis=0)
    middle = len(x)//2
    return [full_width(x, intensity[middle-1:middle+1].mean(axis=0), np.exp(-2)),
            full_width(x, intensity[:, middle-1:middle+1].mean(axis=1), np.exp(-2))]


def curvature(spectrum, dx, wavelength, diameter, index):
    count = spectrum.shape[-1]
    x = (np.arange(count)-count/2+.5)*dx
    xx, yy = np.meshgrid(x, x, indexing='xy')
    r2 = xx*xx+yy*yy
    kx, ky, _, _, _, k0 = kernels(count, dx, wavelength, n=index)
    ex = propagate(spectrum, dx, wavelength, 0, n=index)[0]
    gx = np.fft.ifft2(1j*kx*spectrum[0])
    gy = np.fft.ifft2(1j*ky*spectrum[0])
    core = r2 < (.6*diameter)**2
    denominator = index*k0*np.sum(core*np.abs(ex)**2*r2)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Nonzero finite phase-current denominator required')
    inverse = float(np.sum(core*(np.imag(np.conj(ex)*gx)*xx+
                               np.imag(np.conj(ex)*gy)*yy))/denominator)
    return dict(inverse_R_per_um=inverse,
                normalized_curvature=float(np.pi*index*(diameter/2)**2/wavelength*inverse),
                definition='Radial Ex phase-current inside r<0.6*declared MFD; medium-specific Rayleigh normalization')


def diagnose_reference(record_path, config, count=512, window=128.):
    record_path = local_path(record_path)
    record = read(record_path)
    if record['config_fingerprint'] != fingerprint(config):
        raise ValueError('Reference/configuration fingerprint differs')
    settings = record['settings']
    if settings['m'] != 1 or 'incident_power' not in record:
        raise ValueError('Saved m=+1 homogeneous reference required')
    index = float(config['materials']['substrate']['n'])
    wavelength, mfd = float(config['wavelength_um']), float(config['mfd_um'])
    if (record['input_index'] != index or record['wavelength_um'] != wavelength
            or settings['mfd_um'] != mfd):
        raise ValueError('Saved source physical inputs differ from configuration')
    resolution, courant = float(settings['resolution']), float(settings['Courant'])
    fields_path = local_path(record['fields_file'])
    audit_path = local_path(record['power_audit_fields_file'])
    with np.load(fields_path, allow_pickle=False) as saved, np.load(audit_path, allow_pickle=False) as audit:
        pair_weights = np.asarray(audit['pair_weights'])
        if (pair_weights.shape != (len(audit['plane_z_um']), 2)
                or not np.isfinite(pair_weights).all()
                or np.max(np.abs(pair_weights-.5)) > 1e-8):
            raise ValueError('Every saved reference plane requires symmetric actual pair weights')
        reference_z = float(saved['z_um'][0])
        if abs(reference_z-float(settings['reflection_z_um'])) > 1e-9:
            raise ValueError('First saved field is not at the incident monitor')
        if abs(float(audit['Courant'])-courant) > 1e-12:
            raise ValueError('Saved field Courant differs from record')
        x, dx, field = centered_lp_to_cartesian(saved['r_um'], saved['fields'][0],
                                               1/resolution, count, window)
        native = native_flux_quadrature(audit['flux_E_0'], audit['flux_H_0'])
        native_error = abs(native/float(audit['monitor_fluxes'][0])-1)
    corrected = paraxial_power_equivalent_fields(
        field, index, wavelength, resolution, courant, symmetric_axial_collapse=True)
    spectrum = forward_spectrum(corrected, dx, wavelength, n=index)
    kz = kernels(count, dx, wavelength, n=index)[2]
    facet_z = 0.
    facet_spectrum = spectrum*np.exp(1j*kz*(facet_z-reference_z))[None]
    facet = propagate(facet_spectrum, dx, wavelength, 0, n=index)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    target_spectrum = vector_spectrum(
        np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(mfd/2)**2)), dx, wavelength, n=index)
    target = propagate(target_spectrum, dx, wavelength, 0, n=index)
    measured = widths(facet, x)
    target_widths = widths(target, x)
    if any(value is None for value in measured+target_widths):
        raise ValueError('Missing central 1/e^2 width crossing')
    measured, target_widths = np.asarray(measured), np.asarray(target_widths)
    field_power = float(flux(facet).sum()*dx**2)
    return dict(
        record=str(record_path), settings=settings,
        record_sha256=sha(record_path), fields_sha256=sha(fields_path),
        power_audit_sha256=sha(audit_path), config_fingerprint=record['config_fingerprint'],
        saved_worker_sha256=record['worker_sha256'],
        source_reference_z_um=reference_z, facet_z_um=facet_z,
        propagation_distance_um=facet_z-reference_z,
        input_phase_index=index, wavelength_um=wavelength, scalar_MFD_parameter_um=mfd,
        measured_facet_widths_um=measured.tolist(),
        vector_projected_target_facet_widths_um=target_widths.tolist(),
        relative_width_deviation_from_vector_target=(measured/target_widths-1).tolist(),
        relative_width_deviation_from_scalar_MFD=(measured/mfd-1).tolist(),
        declared_vector_Gaussian_mode_overlap=overlap(facet, target, dx),
        measured_facet_phase_current=curvature(facet_spectrum, dx, wavelength, mfd, index),
        target_facet_phase_current=curvature(target_spectrum, dx, wavelength, mfd, index),
        native_incident_power=float(record['incident_power']),
        native_incident_quadrature_relative_error=native_error,
        reconstructed_forward_LP_power=field_power,
        reconstructed_over_native_quarter_power=field_power/(float(record['incident_power'])/4),
        power_convention='Native cylindrical sector and LP real-phasor powers differ by the common factor4; no amplitude fit',
        all_saved_reference_pair_weights=pair_weights.tolist(),
        processing=dict(cartesian_count=count, window_um=window, dx_um=dx,
                        width_definition='Total-E intensity central two-line mean 1/e^2 crossings; identical vector-target grid'),
        acceptance_thresholds_added=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--window', type=float, default=128.)
    args = parser.parse_args()
    folder, output = local_path(args.folder), local_path(args.output)
    if output.exists():
        raise FileExistsError('Preserve the previous diagnostic; choose a fresh output')
    if args.count < 32 or args.count % 2 or not np.isfinite(args.window) or args.window <= 0:
        raise ValueError('Even Cartesian count>=32 and positive finite window required')
    config = read(folder/'config.json')
    # The live runner may append while these saved-field diagnostics execute.
    # Hash the exact ledger snapshot parsed here rather than a later reread.
    ledger_bytes = (folder/'ledger.json').read_bytes()
    ledger = json.loads(ledger_bytes.decode('utf-8-sig'))
    ledger_digest = hashlib.sha256(ledger_bytes).hexdigest()
    manifest = read(folder/'source_manifest.json')
    for name, digest in manifest.items():
        if sha(ROOT/name) != digest or sha(folder/'source_snapshots'/name) != digest:
            raise ValueError('Frozen source or snapshot changed: '+name)
    completed, unresolved = [], []
    reference_entries = [entry for entry in ledger['entries'] if entry['method'] == 'reference']
    for entry in reference_entries:
        if entry['status'] != 'completed':
            unresolved.append(dict(entry_id=entry['id'], status=entry['status'],
                                   settings={key:entry[key] for key in IDENTITY_KEYS}))
            continue
        path = folder/entry['record_file']
        try:
            if sha(path) != entry['record_sha256']:
                raise ValueError('Completed reference record hash changed')
            item = diagnose_reference(path, config, args.count, args.window)
            if item['saved_worker_sha256'] != manifest['scripts/height_family_maxwell_worker_v2.py']:
                raise ValueError('Saved worker hash differs from frozen source')
            completed.append(dict(entry_id=entry['id'], **item))
        except (OSError, KeyError, ValueError) as error:
            unresolved.append(dict(entry_id=entry['id'], status='DIAGNOSTIC_UNRESOLVED',
                                   reason=str(error), settings={key:entry[key] for key in IDENTITY_KEYS}))
    attempted = {identity(entry) for entry in reference_entries}
    for job in read(folder/'plan.json')['jobs']:
        if job['method'] == 'reference' and identity(job) not in attempted:
            unresolved.append(dict(status='NOT_STARTED_IN_SNAPSHOTTED_LEDGER',
                                   settings={key:job[key] for key in IDENTITY_KEYS}))
    report = dict(
        status='SAVED_REFERENCE_DIAGNOSTICS_ONLY', created_utc=datetime.now(timezone.utc).isoformat(),
        folder=str(folder), ledger_status=ledger['status'],
        ledger_sha256=ledger_digest, configuration_sha256=sha(folder/'config.json'),
        diagnostic_script_sha256=sha(__file__), frozen_source_manifest_sha256=sha(folder/'source_manifest.json'),
        completed_reference_count=len(completed), unresolved_reference_count=len(unresolved),
        source_mode_diagnostics=completed, unresolved_references=unresolved,
        acceptance_thresholds_added=False, optical_solves_launched=0,
        scope='Saved homogeneous-reference source-mode fidelity; no new optical acceptance or mesh-convergence claim',
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), completed_references=len(completed),
                          unresolved_references=len(unresolved), optical_solves_launched=0)))


if __name__ == '__main__':
    main()
