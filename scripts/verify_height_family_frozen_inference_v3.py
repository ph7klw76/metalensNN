"""Replay selected frozen NN/direct decoders without training or optics.

Only trusted local checkpoints from the hashed height-family study are loaded.
The original Model/phenotype AST is compiled in isolation; the trainer's child,
loss, optimizer, proxy propagation and Meep are never invoked. Pre-training
continuous phase seeds are the original decoder inputs, not fitted new seeds.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
DEPENDENCIES=(
    'scripts/run_height_family_n15263_v2.py',
    'src/fiber_metalens_nn/height_conditioned_loss_v2.py',
    'src/fiber_metalens_nn/collimation_proxy.py',
    'src/fiber_metalens_nn/collimation_optics.py',
    'src/fiber_metalens_nn/collimation_loss_v2.py',
    'src/fiber_metalens_nn/natural_collimation.py',
    'src/fiber_metalens_nn/natural_nn_loss_v1.py',
)
HEIGHTS=(40.,75.,100.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path,expected):
    actual=sha(path)
    if actual!=expected:raise ValueError(f'Frozen file hash differs: {path}')
    return actual


def extract_decoder_nodes(source):
    """Extract only the exact original nested decoder definitions."""
    tree=ast.parse(source)
    child=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='child']
    if len(child)!=1:raise ValueError('One original child function required')
    classes=[node for node in child[0].body if isinstance(node,ast.ClassDef) and node.name=='Model']
    functions=[node for node in child[0].body if isinstance(node,ast.FunctionDef) and node.name=='phenotype']
    if len(classes)!=1 or len(functions)!=1:
        raise ValueError('One original Model class and phenotype function required')
    if [a.arg for a in functions[0].args.args]!=['parameters','H','bundle','method']:
        raise ValueError('Unexpected original phenotype signature')
    class_module=ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[]))
    phenotype_module=ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[]))
    segments={
        'Model':ast.get_source_segment(source,classes[0]),
        'phenotype':ast.get_source_segment(source,functions[0]),
    }
    return class_module,phenotype_module,segments


def profile_comparison(actual,expected,expected_profile_sha256):
    """Exact float32 export equality; no tolerance can move a quantized ring."""
    actual=np.asarray(actual,dtype=np.float32)
    expected=np.asarray(expected,dtype=np.float32)
    if actual.shape!=expected.shape or actual.ndim!=1:
        raise ValueError('Matching one-dimensional profile shapes required')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Finite actual and frozen profiles required')
    digest=hashlib.sha256(actual.tobytes()).hexdigest()
    differences=np.flatnonzero(actual!=expected).tolist()
    return dict(exact_float32_profile_match=not differences,
                replay_profile_sha256=digest,
                frozen_profile_sha256=expected_profile_sha256,
                profile_byte_hash_match=digest==expected_profile_sha256,
                changed_ring_indices_zero_based=differences,
                maximum_height_difference_um=float(np.max(np.abs(actual-expected))))


def interpolation_evidence(H,grid):
    grid=np.asarray(grid,float);H=float(H)
    if grid.ndim!=1 or len(grid)<2 or np.any(np.diff(grid)<=0):
        raise ValueError('Strictly increasing training-height grid required')
    if H in grid:
        return dict(kind='training_table_row',height_um=H,row_index=int(np.flatnonzero(grid==H)[0]))
    if not grid[0]<H<grid[-1]:
        raise ValueError('Replay heights must be within the declared table range')
    upper=int(np.searchsorted(grid,H));lower=upper-1
    weight=float((H-grid[lower])/(grid[upper]-grid[lower]))
    return dict(kind='linear_interpolation_of_trained_residuals',height_um=H,
                lower_height_um=float(grid[lower]),upper_height_um=float(grid[upper]),
                lower_row_index=lower,upper_row_index=upper,
                lower_weight=1-weight,upper_weight=weight,
                no_height_specific_reoptimization=True)


def load_frozen_contract(campaign=CAMPAIGN):
    campaign=Path(campaign)
    manifest=read(campaign/'train_source_manifest.json')
    selection=read(campaign/'selection_before_heldout.json')
    result=read(campaign/'training_result.json')
    config=read(campaign/'config.json')
    if set(manifest['sources'])!=set(DEPENDENCIES):
        raise ValueError('Unexpected original training source set')
    sources={name:require_hash(ROOT/name,manifest['sources'][name]) for name in DEPENDENCIES}
    require_hash(campaign/'config.json',manifest['config_sha256'])
    signature=hashlib.sha256((campaign/'config.json').read_bytes()+
                            b''.join((ROOT/name).read_bytes() for name in DEPENDENCIES)).hexdigest()
    if signature!=manifest['signature'] or signature!=selection['signature']:
        raise ValueError('Original source/configuration signature mismatch')
    if selection.get('heldout_used') is not False or result.get('heldout_created_after_selection') is not True:
        raise ValueError('Original validation-before-withheld selection evidence is absent')
    delivery=read(campaign/'delivery_manifest.json')
    for name in ('training_result.json','selection_before_heldout.json','all_target_metadata.json','config.json'):
        require_hash(campaign/name,delivery['files'][name]['sha256'])
    pretraining=read(campaign/'train_frozen_train_validation_targets.json')
    all_targets=read(campaign/'all_target_metadata.json')
    for H in HEIGHTS:
        key=str(float(H))
        for field in ('seed_continuous_um','declared_target_diameter_um','target_output_z_um'):
            if pretraining[key][field]!=all_targets[key][field]:
                raise ValueError('Decoder seed/target changed after its pre-training freeze')
    expected_seeds={'NN':47,'direct':11}
    checkpoints={}
    for method in ('NN','direct'):
        chosen=selection['selection'][method]
        if chosen['seed']!=expected_seeds[method] or chosen!=result['selection'][method]:
            raise ValueError('Unexpected selected arm or training-result mismatch')
        reports=[read(campaign/f'{method}_seed{seed}/training_report.json')
                 for seed in config['natural_training']['seeds']]
        independently_selected=min(reports,key=lambda row:row['validation_key'])
        if chosen!=independently_selected:
            raise ValueError('Selected seed differs from original validation-only rule')
        path=campaign/f'{method}_seed{chosen["seed"]}/checkpoint.pkl'
        require_hash(path,selection['checkpoint_sha256'][method])
        checkpoints[method]=path
    return dict(config=config,selection=selection,result=result,pretraining=pretraining,
                all_targets=all_targets,signature=signature,source_hashes=sources,checkpoints=checkpoints)


def replay(campaign=CAMPAIGN):
    """Six CPU decoder predictions from actual selected best parameters."""
    contract=load_frozen_contract(campaign)
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['JAX_ENABLE_X64']='false'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['JAX_ENABLE_COMPILATION_CACHE']='false'
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    if any(device.platform!='cpu' for device in jax.devices()):
        raise RuntimeError('This bounded replay permits CPU inference only')
    config=contract['config'];tc=config['natural_training']
    source=(ROOT/DEPENDENCIES[0]).read_text()
    cls,fn,segments=extract_decoder_nodes(source)
    namespace=dict(nn=nn,cfg=config,jnp=jnp,jax=jax,tc=tc,
                   hgrid=jnp.asarray(tc['train_H_um'],jnp.float32),hmax=config['height_max_um'])
    exec(compile(cls,str(ROOT/DEPENDENCIES[0]),'exec'),namespace)
    namespace['model']=namespace['Model']()
    exec(compile(fn,str(ROOT/DEPENDENCIES[0]),'exec'),namespace)
    phenotype=namespace['phenotype'];step=config['fabrication']['height_step_um'];hmax=config['height_max_um']
    rows=[];checkpoint_evidence={}
    for method in ('NN','direct'):
        path=contract['checkpoints'][method]
        # Trusted local pickle is verified against pre-heldout selected SHA above.
        checkpoint=pickle.loads(path.read_bytes())
        chosen=contract['selection']['selection'][method]
        if (checkpoint['signature']!=contract['signature'] or checkpoint['step']!=tc['steps']
                or checkpoint['calls']!=chosen['objective_calls'] or checkpoint['best'] is None
                or checkpoint['best_key']!=chosen['validation_key']
                or min(v['key'] for v in checkpoint['validations'])!=checkpoint['best_key']):
            raise ValueError('Actual trained checkpoint/selected best state is inconsistent')
        params=jax.tree.map(jnp.asarray,checkpoint['best'])
        checkpoint_evidence[method]=dict(path=str(path),sha256=sha(path),seed=chosen['seed'],
            actual_optimizer_step=checkpoint['step'],objective_calls=checkpoint['calls'],
            parameter_source='checkpoint.best selected by validation, not checkpoint.parameters final update',
            parameter_count=sum(int(np.asarray(a).size) for a in jax.tree.leaves(params)),
            nonvacuous_trained_recovery=checkpoint['recovery'])
        predict=jax.jit(lambda par,H,b,method=method:phenotype(par,H,b,method))
        for H in HEIGHTS:
            metadata=contract['pretraining'][str(float(H))]
            seed=jnp.asarray(metadata['seed_continuous_um'],jnp.float32)
            bundle=(None,None,None,seed,None,None)
            continuous=predict(params,jnp.asarray(H,jnp.float32),bundle)
            quantized=np.asarray(jnp.clip(jnp.round(continuous/step)*step,0,hmax).block_until_ready())
            design_path=Path(campaign)/f'designs/H{H:g}/{method}.json';design=read(design_path)
            records=[row for row in contract['result']['records'] if row['method']==method and row['H_um']==H]
            if len(records)!=1:raise ValueError('One original training-result profile per method/height required')
            if (design['source_signature']!=contract['signature'] or design['height_domain']!=records[0]['height_domain']
                    or design['target_diameter_um']!=records[0]['declared_target_diameter_um']):
                raise ValueError('Frozen export provenance differs from original record')
            comparison=profile_comparison(quantized,design['heights_um'],records[0]['profile_sha256'])
            baseline=read(Path(campaign)/f'designs/H{H:g}/baseline.json')
            baseline_q=np.asarray(jnp.clip(jnp.round(seed/step)*step,0,hmax))
            seed_matches_baseline=np.array_equal(baseline_q,np.asarray(baseline['heights_um'],np.float32))
            row=dict(method=method,H_um=H,height_domain=records[0]['height_domain'],
                     design_file=str(design_path),design_sha256=sha(design_path),
                     decoder_seed_source='original pre-training continuous seed, matched to frozen all-target metadata',
                     seed_quantization_matches_frozen_baseline=bool(seed_matches_baseline),
                     target_diameter_um=design['target_diameter_um'],**comparison)
            if method=='direct':row['residual_table_evaluation']=interpolation_evidence(H,tc['train_H_um'])
            rows.append(row)
    passed=all(row['exact_float32_profile_match'] and row['profile_byte_hash_match']
               and row['seed_quantization_matches_frozen_baseline'] for row in rows)
    return dict(status='PASS_FROZEN_DECODER_REPLAY' if passed else 'FAIL_CPU_REPLAY_REQUIRES_DIAGNOSIS',
        created_utc=datetime.now(timezone.utc).isoformat(),backend=str(jax.devices()),JAX_version=jax.__version__,
        original_signature=contract['signature'],source_hashes=contract['source_hashes'],
        extracted_decoder_source_sha256={name:hashlib.sha256(segment.encode()).hexdigest() for name,segment in segments.items()},
        checkpoints=checkpoint_evidence,rows=rows,decoded_evaluations=len(rows),
        source_integrity_preserved=True,optimizer_updates=0,training_started=False,
        new_proxy_forward_calculations=0,Meep_solves=0,GPU_used=False,
        scope='Reproduces frozen geometry from selected learned parameters; does not establish optical acceptance.',
        H75_selection_status='Validation-observed, including seed/checkpoint selection; direct residuals interpolate trained H70/H80 rows. Not a held-out test.',
        limitations='Pre-training decoder seeds are replayed as frozen inputs; this does not rerun pedestal optics or independently validate the seed physics. A CPU mismatch is not excused or rounded away.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=CAMPAIGN)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing replay output')
    if not args.output.parent.is_dir():raise FileNotFoundError('Prepared output parent required')
    result=replay(args.campaign)
    result['replay_script_sha256']=sha(__file__)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],output=str(args.output),decoded_evaluations=result['decoded_evaluations'],
                          optimizer_updates=0,Meep_solves=0,GPU_used=False),indent=2))
    if result['status']!='PASS_FROZEN_DECODER_REPLAY':raise SystemExit(2)


if __name__=='__main__':main()
