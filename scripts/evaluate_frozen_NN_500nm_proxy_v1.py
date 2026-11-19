"""Nine bounded frozen-weight proxy scores; no training or Maxwell solver.

The exported-SLC route is Q500_half_up(Q50_even(decoder)); the direct-weight
route is Q500_even(decoder). Neither changes the frozen optical configuration.
Only explicit --run evaluates optics. Imports and quantization tests are inert.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT/'artifacts/1550nm_height_family_n15263_v2'
EXPORT = ROOT/'exports/height_family_cylinder_attached_slc_500nm_v5'
DEFAULT_OUTPUT = CAMPAIGN/'frozen_NN_500nm_proxy_v1'
HEIGHTS = (40, 75, 100)
PINS = {
    'scripts/verify_height_family_frozen_inference_v3.py': 'ad4f442f27678a07c7143a5298e5e908f5aa40bff3084457ae12e4fa2f4ffb3b',
    'artifacts/1550nm_height_family_n15263_v2/config.json': 'ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734',
    'artifacts/1550nm_height_family_n15263_v2/NN_seed47/checkpoint.pkl': '42e86dfcc5840170cf5372bd5553a893bd5a71baef8cb3c2f63d18ced8cb6ebc',
    'artifacts/1550nm_height_family_n15263_v2/train_frozen_train_validation_targets.json': '2b5f1c36abb49ad9f43c20df99d3710141148060961cddcfb56be9a1669e601c',
    'artifacts/1550nm_height_family_n15263_v2/train_source_manifest.json': '5c00fefe8483fe9e645c9f84c112e5cee9a00363550635d4069e384d0b441aff',
    'artifacts/1550nm_height_family_n15263_v2/training_result.json': '872f8b08173b41a11f3c1e3a78513dbe292d46fcbc35fda1647d8d2ecbbaadea',
    'artifacts/1550nm_height_family_n15263_v2/selection_before_heldout.json': 'e0b5aa82b871940356748b41d3796b92f4c2aa0e21833a209403776bfe987591',
    'exports/height_family_cylinder_attached_slc_500nm_v5/exporter_500nm.py.txt': 'd1fcf2172b6165b25b2d21ee12edd46f7e2e3811e20465c116f8864944501399',
}
FINE_SHA = {
    40:'16a765787ce3dd50ad34bc6e8f3dac89e0fd0e6c3723ec6460cfa8bbe1b0b362',
    75:'7861a7d4a8409026b1d57ed20bb5fcb9e30d2dbe7b4a68b8214ae1ee9a458c69',
    100:'0e83c24f7978f764c937b74f15b2d6e6f71da96d3ce320677e941b8cc94b3772',
}
COARSE_SHA = {
    40:'06b642955e9c19a3c7bc6865501640dfe2b57db4cc1754fc8354216e317ac6a1',
    75:'cf782ba2091d29a67022678e7ce1bd7a8060e3084d3038e617e46d2b4ff0145d',
    100:'9a69d7bffce92dc858186ba190536d83b8319a7d27be405f236e3f769ea2aeaa',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc(): return datetime.now(timezone.utc).isoformat()
def plain(value):
    if isinstance(value, dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    if hasattr(value, 'dtype'): return plain(np.asarray(value).tolist())
    return value
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(plain(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def exported_slc_quantization(fine):
    """Exact v5 fabrication rule on nominal integer 50-nm grid, ties upward."""
    fine = np.asarray(fine, dtype=float)
    if fine.ndim != 1 or not np.isfinite(fine).all():
        raise ValueError('Finite one-dimensional source profile required')
    indices = np.rint(fine/.05).astype(np.int64)
    if np.max(np.abs(indices*.05-fine)) > 2e-6:
        raise ValueError('Source is not on the frozen nominal 50-nm grid')
    if np.any((indices < 0) | (indices > 60)):
        raise ValueError('Source height outside 0..3 um')
    return (((indices+5)//10)*.5).astype(np.float32)


def direct_nn_quantization_numpy(continuous):
    """Pure test counterpart of actual JAX float32 round-to-nearest-even."""
    continuous = np.asarray(continuous, np.float32)
    if not np.isfinite(continuous).all(): raise ValueError('Nonfinite decoder output')
    return np.clip(np.rint(continuous/np.float32(.5))*np.float32(.5), 0, 3)


def validate_geometry(config, fine, coarse, H):
    if config['annuli'] != 50 or config['cylinder_diameter_um'] != 50:
        raise ValueError('Frozen 50-ring / 50-um diameter geometry required')
    if config['fabrication']['height_step_um'] != .05:
        raise ValueError('The frozen 50-nm optical configuration must remain unchanged')
    if config['cylinder_diameter_um']/2/config['annuli'] != .5:
        raise ValueError('Radial bands must remain 500 nm')
    for design in (fine, coarse):
        if (design['H_um'] != H or design['cylinder_diameter_um'] != 50
                or design['units'] != 'um' or len(design['heights_um']) != 50
                or design['materials'] != config['materials']
                or design['mfd_um'] != config['mfd_um']):
            raise ValueError('Fabrication/source geometry or material mismatch')
    expected = exported_slc_quantization(fine['heights_um'])
    if not np.array_equal(expected, np.asarray(coarse['heights_um'], np.float32)):
        raise ValueError('Existing SLC fabrication profile differs from original v5 quantization')


def verify_bundle_metadata(actual, frozen):
    """Numerical replay tolerance only; no target/seed fitting is permitted."""
    fields = ('declared_target_diameter_um','target_vector_widths_um',
              'seed_continuous_um','seed_quantized_um')
    evidence = {}
    for name in fields:
        a,b = np.asarray(actual[name]),np.asarray(frozen[name])
        if a.shape != b.shape or not np.allclose(a,b,rtol=2e-6,atol=2e-5):
            raise ValueError('Computed target/seed disagrees with frozen metadata: '+name)
        evidence[name] = {'max_abs_difference':float(np.max(np.abs(a-b))),
                          'rtol':2e-6,'atol':2e-5}
    for name in ('H_um','target_output_z_um','processing_planes_from_output_um','materials','input_mode'):
        if actual[name] != frozen[name]:
            raise ValueError('Frozen target/source interpretation differs: '+name)
    return evidence


def replay_metric_check(actual, frozen):
    evidence = {}
    for name in ('declared_useful_efficiency','declared_overlap','transmission',
                 'normalized_curvature','angular_second_moment_ratio',
                 'max_sidelobe_fraction','widths_um','target_widths_um'):
        a,b = np.asarray(actual[name]),np.asarray(frozen[name])
        atol = 5e-4 if 'widths' in name else 2e-5
        if a.shape != b.shape or not np.allclose(a,b,rtol=2e-5,atol=atol):
            raise ValueError('Fine-profile proxy replay differs from saved result: '+name)
        evidence[name] = float(np.max(np.abs(a-b)))
    if actual['valid_crossings'] != frozen['valid_crossings']:
        raise ValueError('Fine-profile crossing replay differs')
    return evidence


def gates(metrics, criteria):
    width = float(np.max(np.abs(np.asarray(metrics['widths_um'])[0]/
                               np.asarray(metrics['target_widths_um'])[0]-1)))
    checks = {
        'transmission':metrics['transmission'] >= criteria['transmission_min'],
        'mode_overlap':metrics['declared_overlap'] >= criteria['declared_Gaussian_overlap_min'],
        'useful_efficiency':metrics['declared_useful_efficiency'] >= criteria['useful_efficiency_min'],
        'output_width':width <= criteria['output_diameter_relative_tolerance'],
        'sidelobes':metrics['max_sidelobe_fraction'] <= criteria['max_sidelobe_fraction'],
        'curvature':abs(metrics['normalized_curvature']) <= criteria['max_abs_normalized_curvature'],
        'angular_second_moment':metrics['angular_second_moment_ratio'] <= criteria['max_angular_second_moment_ratio'],
        'valid_crossings':bool(np.asarray(metrics['valid_crossings']).all()),
    }
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'output_width_relative_error':width}


def authenticate():
    protected = {}
    pins = dict(PINS)
    for H in HEIGHTS:
        pins[str((CAMPAIGN/f'designs/H{H}/NN.json').relative_to(ROOT))] = FINE_SHA[H]
        pins[str((EXPORT/f'fabrication_sources/NN_H{H}_500nm_fabrication.json').relative_to(ROOT))] = COARSE_SHA[H]
    for name, expected in pins.items():
        path = ROOT/name
        if sha(path) != expected: raise ValueError('Trusted input changed: '+name)
        protected[str(path)] = expected
    spec = importlib.util.spec_from_file_location('frozen_inference_contract_v3',ROOT/'scripts/verify_height_family_frozen_inference_v3.py')
    replay = importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
    contract = replay.load_frozen_contract(CAMPAIGN)
    for name,digest in contract['source_hashes'].items(): protected[str(ROOT/name)] = digest
    for name in ('all_target_metadata.json','training_result.json','selection_before_heldout.json','delivery_manifest.json'):
        protected[str(CAMPAIGN/name)] = sha(CAMPAIGN/name)
    for H in HEIGHTS:
        validate_geometry(contract['config'],read(CAMPAIGN/f'designs/H{H}/NN.json'),
                          read(EXPORT/f'fabrication_sources/NN_H{H}_500nm_fabrication.json'),H)
    return replay, contract, protected


def gpu_inventory():
    line = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'],text=True).splitlines()[0]
    name,total,used,free = [s.strip() for s in line.split(',')]
    return dict(name=name,total_mib=float(total),used_mib=float(used),free_mib=float(free))


def run(output):
    begin = time.monotonic(); started = utc()
    replay, contract, protected = authenticate()
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Fresh output directory required')
    gpu_before = gpu_inventory()
    if '4060' not in gpu_before['name']: raise RuntimeError('RTX 4060 required')
    os.environ.update(JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_ENABLE_COMPILATION_CACHE='false',
                      XLA_PYTHON_CLIENT_MEM_FRACTION=str(min(.75,6144/gpu_before['total_mib'])),
                      OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path.insert(0,str(ROOT/'src'))
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from fiber_metalens_nn.height_conditioned_loss_v2 import HeightConditionedLoss
    if jax.devices()[0].platform != 'gpu': raise RuntimeError('GPU backend required')
    cfg=contract['config'];tc=cfg['natural_training']
    cls,fn,segments = replay.extract_decoder_nodes((ROOT/replay.DEPENDENCIES[0]).read_text())
    namespace=dict(nn=nn,cfg=cfg,jnp=jnp,jax=jax,tc=tc,
                   hgrid=jnp.asarray(tc['train_H_um'],jnp.float32),hmax=cfg['height_max_um'])
    exec(compile(cls,replay.DEPENDENCIES[0],'exec'),namespace)
    namespace['model']=namespace['Model']()
    exec(compile(fn,replay.DEPENDENCIES[0],'exec'),namespace)
    checkpoint = pickle.loads(contract['checkpoints']['NN'].read_bytes())
    chosen = contract['selection']['selection']['NN']
    if (checkpoint['signature'] != contract['signature'] or checkpoint['step'] != tc['steps']
            or checkpoint['calls'] != chosen['objective_calls'] or checkpoint['best'] is None
            or checkpoint['best_key'] != chosen['validation_key']
            or min(v['key'] for v in checkpoint['validations']) != checkpoint['best_key']):
        raise ValueError('Frozen selected checkpoint.best state mismatch')
    params=jax.tree.map(jnp.asarray,checkpoint['best'])
    predict=jax.jit(lambda H,seed:namespace['phenotype'](params,H,(None,None,None,seed,None,None),'NN'))
    loss=HeightConditionedLoss(cfg); score=jax.jit(loss.terms)
    output.mkdir(parents=True)
    write(output/'receipt_before_evaluation.json',dict(status='BOUNDED_PROXY_ONLY',started_utc=started,
          checkpoint_parameter_source='NN_seed47/checkpoint.best',input_sha256=protected,
          evaluator_sha256=sha(__file__),maximum_scored_terms_calls=9,target_bundle_constructions=3,
          no_training=True,no_Maxwell=True,no_RAM_gates=True,GPU_budget_mib=6144,gpu_before=gpu_before))
    rows=[]; arrays={}; scored=0; bundle_checks={}; replay_checks={}
    for H in HEIGHTS:
        frozen=contract['pretraining'][str(float(H))]
        computed=loss.bundle(H)
        bundle_checks[str(H)]=verify_bundle_metadata(loss.metadata[str(float(H))],frozen)
        # Freeze original recorded seed, target size and vector widths; never fit a new target.
        seed=jnp.asarray(frozen['seed_continuous_um'],jnp.float32)
        target=loss.p.target(frozen['declared_target_diameter_um'])
        tp=.5*jnp.real(target[0]*jnp.conj(target[4])-target[1]*jnp.conj(target[3]))
        moment=jnp.sum(tp*(loss.kx**2+loss.ky**2))/jnp.sum(tp)
        bundle=(computed[0],target,jnp.asarray(frozen['target_vector_widths_um'],jnp.float32),seed,
                jnp.asarray(frozen['declared_target_diameter_um'],jnp.float32),moment)
        continuous=predict(jnp.asarray(H,jnp.float32),seed)
        fine=np.asarray(jnp.clip(jnp.round(continuous/.05)*.05,0,3).block_until_ready())
        design=read(CAMPAIGN/f'designs/H{H}/NN.json')
        saved=[r for r in contract['result']['records'] if r['method']=='NN' and r['H_um']==H]
        if len(saved)!=1: raise ValueError('One frozen fine NN metric row required')
        match=replay.profile_comparison(fine,design['heights_um'],saved[0]['profile_sha256'])
        if not match['exact_float32_profile_match'] or not match['profile_byte_hash_match']:
            raise ValueError('50-nm checkpoint replay is not bit-exact')
        slc=np.asarray(read(EXPORT/f'fabrication_sources/NN_H{H}_500nm_fabrication.json')['heights_um'],np.float32)
        direct=np.asarray(jnp.clip(jnp.round(continuous/.5)*.5,0,3).block_until_ready())
        if not np.array_equal(direct,direct_nn_quantization_numpy(np.asarray(continuous))):
            raise ValueError('JAX/NumPy nearest-even direct-500 mapping differs')
        arrays.update({f'H{H}_continuous_um':np.asarray(continuous),f'H{H}_fine50_um':fine,
                       f'H{H}_SLC500_um':slc,f'H{H}_directNN500_um':direct})
        cache={}
        for arm,profile in (('fine50',fine),('existing_SLC500_half_up',slc),('direct_NN500_nearest_even',direct)):
            digest=hashlib.sha256(profile.tobytes()).hexdigest(); cached=digest in cache
            if not cached:
                if scored>=9: raise RuntimeError('Scored terms-call bound exceeded')
                metrics=plain(jax.tree.map(lambda a:a.block_until_ready(),score(jnp.asarray(profile),bundle)))
                scored+=1;cache[digest]=metrics
            metrics=cache[digest]
            if arm=='fine50': replay_checks[str(H)]=dict(profile=match,metrics=replay_metric_check(metrics,saved[0]['metrics']))
            rows.append(dict(H_um=H,arm=arm,metrics=metrics,proxy_gates=gates(metrics,tc['investigation_gates']),
                         profile_sha256=digest,deduplicated_terms_call=cached,radial_width_um=.5,
                         changed_rings_from_fine50=int(np.count_nonzero(profile!=fine)),
                         direct_vs_SLC_changed_ring_indices=np.flatnonzero(direct!=slc).tolist(),
                         useful_efficiency_percent=100*metrics['declared_useful_efficiency'],
                         useful_efficiency_change_from_fine_percentage_points=100*(metrics['declared_useful_efficiency']-cache[hashlib.sha256(fine.tobytes()).hexdigest()]['declared_useful_efficiency'])))
        print(json.dumps({'H_um':H,'scored_terms_calls':scored,'eta_percent':{r['arm']:r['useful_efficiency_percent'] for r in rows if r['H_um']==H}}),flush=True)
    gpu_after=gpu_inventory()
    peak=jax.devices()[0].memory_stats().get('peak_bytes_in_use')
    if (peak is not None and peak>6144*1024**2) or gpu_after['used_mib']-gpu_before['used_mib']>6144:
        raise RuntimeError('6-GiB GPU budget exceeded')
    for name,digest in protected.items():
        if sha(name)!=digest: raise ValueError('Protected input changed during evaluation: '+name)
    np.savez_compressed(output/'evaluated_profiles.npz',**arrays)
    report=dict(status='COMPLETED_FROZEN_WEIGHT_PROXY_ONLY',rows=rows,started_utc=started,finished_utc=utc(),
        wall_s=time.monotonic()-begin,scored_terms_calls=scored,maximum_scored_terms_calls=9,
        target_bundle_constructions=3,bundle_replay=bundle_checks,fine_replay=replay_checks,
        original_config_unchanged=True,input_sha256=protected,evaluator_sha256=sha(__file__),
        checkpoint_sha256=sha(contract['checkpoints']['NN']),source_signature=contract['signature'],
        decoded_source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in segments.items()},
        criteria=tc['investigation_gates'],GPU_allocator_peak_bytes=peak,gpu_before=gpu_before,gpu_after=gpu_after,
        definition='eta=abs(EH_inner(actual,target))^2 / (target_power * same incident_power), exactly HeightConditionedLoss.terms',
        routes={'existing_SLC500':'nominal50nm integer-grid ties-up after frozen Q50 nearest-even',
                'direct_NN500':'JAX float32 nearest-even Q500 of continuous checkpoint.best decoder output'},
        limitations=['One-way finite-cylinder proxy estimate, not independent Maxwell, measured printing, or experimental efficiency.',
                     '500-nm radial bands are unchanged; no claim that these are meta-atom pitch or minimum laser linewidth.',
                     'Coarse proxy gate failures are reported without aborting or optimizing.',
                     'H75 was validation-observed, not held out. No new optimizer updates, physical solves or target fitting.'])
    write(output/'results.json',report)
    with (output/'efficiencies.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['H_um','arm','useful_modal_efficiency_percent','change_from_fine_percentage_points','proxy_gates'])
        for row in rows: writer.writerow([row['H_um'],row['arm'],row['useful_efficiency_percent'],row['useful_efficiency_change_from_fine_percentage_points'],row['proxy_gates']['status']])
    write(output/'manifest.json',dict(status=report['status'],files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps({'status':report['status'],'output':str(output),'scored_terms_calls':scored}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if not args.run:
        print(json.dumps({'status':'INSPECTION_ONLY_NO_EVALUATION','scored_terms_calls':0,'output':str(args.output)}));return
    run(args.output)


if __name__=='__main__': main()
