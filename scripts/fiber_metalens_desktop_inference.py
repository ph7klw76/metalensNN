"""Predict a height-specific metalens with the frozen NN and optical proxy.

The desktop app invokes this in the bundled Windows JAX runtime or the project's
WSL environment. Standard output is a single JSON object. The trusted checkpoint
and original optical configuration are authenticated before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/1550nm_height_family_n15263_v2'
CHECKPOINT = CAMPAIGN / 'NN_seed47/checkpoint.pkl'
HEIGHT_RANGE_UM = (40.0, 100.0)
V5_PROXY_HELPER_SHA = '9dbe44ff8716da100d6c0b9a661a29c61c31b931756081c3ac7ce41bd1ab9877'


def valid_height(height_um):
    height_um = float(height_um)
    if not math.isfinite(height_um) or not HEIGHT_RANGE_UM[0] <= height_um <= HEIGHT_RANGE_UM[1]:
        raise ValueError('Cylinder height must be between 40 and 100 µm.')
    return height_um


def authenticate_model():
    for name, digest in proxy_helpers.PINS.items():
        if name.startswith(('artifacts/', 'scripts/')):
            frozen_replay.require_hash(ROOT / name, digest)
    contract = frozen_replay.load_frozen_contract(CAMPAIGN)
    if contract['checkpoints']['NN'] != CHECKPOINT:
        raise ValueError('Unexpected selected NN checkpoint path.')
    checkpoint_path = CHECKPOINT.relative_to(ROOT).as_posix()
    checkpoint_bytes = CHECKPOINT.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if checkpoint_sha != proxy_helpers.PINS[checkpoint_path]:
        raise ValueError('Frozen NN checkpoint failed its checksum.')
    # Deserialize only the trusted byte sequence matched to the original SHA.
    checkpoint = pickle.loads(checkpoint_bytes)
    selected = contract['selection']['selection']['NN']
    if (checkpoint['signature'] != contract['signature']
            or checkpoint['step'] != contract['config']['natural_training']['steps']
            or checkpoint['calls'] != selected['objective_calls']
            or checkpoint['best'] is None
            or checkpoint['best_key'] != selected['validation_key']
            or min(row['key'] for row in checkpoint['validations']) != checkpoint['best_key']):
        raise ValueError('Selected validation checkpoint does not match the frozen training record.')
    return contract, checkpoint['best'], checkpoint_sha


def infer(height_um, backend='cpu'):
    """Return the 50 and 500 nm predictions for one requested cylinder height."""
    height_um = valid_height(height_um)
    if backend not in ('cpu', 'cuda'):
        raise ValueError('Backend must be cpu or cuda.')
    os.environ.update(
        JAX_PLATFORMS=backend, JAX_ENABLE_X64='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_ENABLE_COMPILATION_CACHE='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    )
    if backend == 'cuda':
        gpu = proxy_helpers.gpu_inventory()
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(min(.75, 6144 / gpu['total_mib']))

    sys.path.insert(0, str(ROOT / 'src'))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import linen as nn
    from fiber_metalens_nn.height_conditioned_loss_v2 import HeightConditionedLoss

    device = jax.devices()[0]
    if device.platform != {'cpu': 'cpu', 'cuda': 'gpu'}[backend]:
        raise RuntimeError(f'JAX did not select the requested {backend} backend.')
    contract, best_params, checkpoint_sha = authenticate_model()
    config = contract['config']
    training = config['natural_training']
    original_source = ROOT / frozen_replay.DEPENDENCIES[0]
    model_code, decoder_code, _ = frozen_replay.extract_decoder_nodes(original_source.read_text())
    environment = dict(nn=nn, cfg=config, jnp=jnp, jax=jax, tc=training,
                       hmax=config['height_max_um'])
    exec(compile(model_code, str(original_source), 'exec'), environment)
    environment['model'] = environment['Model']()
    exec(compile(decoder_code, str(original_source), 'exec'), environment)
    decode = jax.jit(lambda params, height, seed: environment['phenotype'](
        params, height, (None, None, None, seed, None, None), 'NN'))

    loss = HeightConditionedLoss(config)
    height_key = str(height_um)
    frozen = contract['all_targets'].get(height_key)
    if frozen is None:
        bundle = loss.bundle(height_um)
        metadata = loss.metadata[height_key]
    else:
        # Retain the original target and decoder seed at all archived heights.
        # Only the forward incident field at the top of the cylinder is rebuilt.
        top = loss.p.pedestal(height_um)
        target = loss.p.target(frozen['declared_target_diameter_um'])
        target_flux = .5 * jnp.real(target[0] * jnp.conj(target[4]) - target[1] * jnp.conj(target[3]))
        target_moment = jnp.sum(target_flux * (loss.kx**2 + loss.ky**2)) / jnp.sum(target_flux)
        bundle = (top, target, jnp.asarray(frozen['target_vector_widths_um'], jnp.float32),
                  jnp.asarray(frozen['seed_continuous_um'], jnp.float32),
                  jnp.asarray(frozen['declared_target_diameter_um'], jnp.float32), target_moment)
        metadata = frozen

    continuous = decode(jax.tree.map(jnp.asarray, best_params), jnp.float32(height_um), bundle[3])
    fine_step = config['fabrication']['height_step_um']
    fine = np.asarray(jnp.clip(jnp.round(continuous / fine_step) * fine_step, 0, config['height_max_um']))
    coarse = proxy_helpers.exported_slc_quantization(fine)
    diameter = float(config['cylinder_diameter_um'])
    annuli = int(config['annuli'])
    radial_width = diameter / (2 * annuli)
    target_diameter = float(metadata['declared_target_diameter_um'])
    target_z = float(metadata['target_output_z_um'])
    if frozen is not None:
        original_record, = (row for row in contract['result']['records']
                            if row['method'] == 'NN' and row['H_um'] == height_um)
        original_design = frozen_replay.read(CAMPAIGN / f'designs/H{height_um:g}/NN.json')
        match = frozen_replay.profile_comparison(
            fine, original_design['heights_um'], original_record['profile_sha256'])
        if not (match['exact_float32_profile_match'] and match['profile_byte_hash_match']):
            raise ValueError('The 50 nm decoder profile differs from the frozen height-family result.')

    score = jax.jit(loss.terms)
    results = []
    for heights, step in ((fine, fine_step), (coarse, .5)):
        metrics = proxy_helpers.plain(score(jnp.asarray(heights), bundle))
        efficiency = metrics['declared_useful_efficiency']
        if not math.isfinite(efficiency) or not 0 <= efficiency <= 1:
            raise ValueError('Optical proxy returned an invalid useful modal efficiency.')
        gates = proxy_helpers.gates(metrics, training['investigation_gates'])
        results.append(dict(
            kind='height_specific_NN', H_um=height_um, units='um',
            cylinder_diameter_um=diameter, annuli=annuli, radial_band_width_um=radial_width,
            height_step_um=step, heights_um=heights.tolist(), max_relief_um=float(heights.max()),
            assembly_top_um=height_um + float(heights.max()),
            profile_sha256=hashlib.sha256(heights.astype('<f4').tobytes()).hexdigest(),
            useful_modal_efficiency=efficiency, proxy_checks_pass=gates['status'] == 'PASS',
            proxy_checks=gates['checks'], materials=config['materials'], mfd_um=config['mfd_um'],
            target_diameter_um=target_diameter, output_reference_z_um=target_z,
        ))
    if backend == 'cuda':
        gpu_after = proxy_helpers.gpu_inventory()
        peak = device.memory_stats().get('peak_bytes_in_use')
        if (peak is not None and peak > 6144 * 1024**2) or gpu_after['used_mib'] - gpu['used_mib'] > 6144:
            raise RuntimeError('Prediction exceeded the original 6 GiB GPU budget.')

    domain = next((split for split in ('train', 'validation', 'heldout')
                   if height_um in training[f'{split}_H_um']), 'unseen_in_range')
    return dict(
        schema_version=1, height_um=height_um, cylinder_diameter_um=diameter,
        radial_edges_um=[radial_width * index for index in range(annuli + 1)],
        wavelength_um=config['wavelength_um'], fiber_mfd_um=config['mfd_um'],
        polymer_index=config['materials']['polymer']['n'], target_diameter_um=target_diameter,
        output_reference_z_um=target_z, backend=device.platform,
        provenance=dict(checkpoint_sha256=checkpoint_sha, checkpoint_parameter_source='NN_seed47/checkpoint.best',
                        source_signature=contract['signature'], height_domain=domain,
                        efficiency_method='One-way finite-cylinder optical proxy with vector Gaussian overlap',
                        profile_500nm_method='Round nominal 50 nm profile to nearest 500 nm, ties upward'),
        design_50nm=results[0], design_500nm=results[1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height-um', required=True, type=float)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        result = infer(args.height_um, args.backend)
        print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Prediction failed: {error}\n')


# These helpers own the original checksum map, replay and quantization details.
# Check the source before import and before any pickle is read.
helper_path = ROOT / 'scripts/evaluate_frozen_NN_500nm_proxy_v1.py'
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != V5_PROXY_HELPER_SHA:
    raise ValueError('Frozen proxy helper source changed.')
import evaluate_frozen_NN_500nm_proxy_v1 as proxy_helpers

frozen_replay_path = ROOT / 'scripts/verify_height_family_frozen_inference_v3.py'
if proxy_helpers.sha(frozen_replay_path) != proxy_helpers.PINS[frozen_replay_path.relative_to(ROOT).as_posix()]:
    raise ValueError('Frozen NN replay helper source changed.')
import verify_height_family_frozen_inference_v3 as frozen_replay


if __name__ == '__main__':
    main()
