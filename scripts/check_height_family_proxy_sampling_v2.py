"""Frozen-profile sampling and datasheet-MFD sensitivity, with no optimization."""
from pathlib import Path
import copy
import hashlib
import json
import os
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_PLATFORMS']='cuda'
os.environ['OMP_NUM_THREADS']='1'
from fiber_metalens_nn.runtime import configure
configure()
import jax
import jax.numpy as jnp
import numpy as np
from fiber_metalens_nn.height_conditioned_loss_v2 import HeightConditionedLoss
from fiber_metalens_nn.collimation_loss_v2 import plane_widths
from fiber_metalens_nn.collimation_metrics import fitted_divergence

FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2'
def read(p):return json.loads(Path(p).read_text())
def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'dtype') and hasattr(x,'shape'):return np.asarray(x).tolist()
    return x
def write(p,x):Path(p).write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def ready(x):return jax.tree.map(lambda a:a.block_until_ready() if hasattr(a,'block_until_ready') else a,x)
def angles(fit):return [a['half_angle_rad'] for a in fit['axes']] if fit['valid'] else None

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Sampling stage300s wall bound')))
    signal.alarm(300)
    started=time.monotonic();cfg=read(FOLDER/'config.json');result=read(FOLDER/'training_result.json')
    if jax.devices()[0].platform!='gpu':raise RuntimeError('RTX GPU backend required')
    nominal={r['H_um']:r for r in result['records'] if r['method']=='NN' and r['height_domain']=='training'}
    paths={H:FOLDER/f'designs/H{H:g}/NN.json' for H in nominal}
    before={H:hashlib.sha256(p.read_bytes()).hexdigest() for H,p in paths.items()}
    rows=[]
    for label,settings in [('grid256',{'grid_n':256}),('relief48',{'relief_slices':48}),('pedestal_dz025',{'pedestal_dz_um':.25})]:
        candidate=copy.deepcopy(cfg);candidate['proxy'].update(settings)
        loss=HeightConditionedLoss(candidate);terms=jax.jit(loss.terms)
        for H,original in nominal.items():
            design=read(paths[H]);h=jnp.asarray(design['heights_um'],jnp.float32);bundle=list(loss.bundle(H))
            # Keep the same declared physical Gaussian as the selection record.
            # Re-sample it on the refined grid, never fit it to the new output.
            D=original['declared_target_diameter_um'];target=loss.p.target(D)
            widths=[]
            for z in cfg['evaluation']['processing_planes_um']:
                field=jnp.fft.ifft2(target*(jnp.exp(1j*loss.kz*z)*loss.valid)[None],axes=(-2,-1))
                widths.append(plane_widths(field,loss.x)[0])
            power=.5*jnp.real(target[0]*jnp.conj(target[4])-target[1]*jnp.conj(target[3]))
            moment=jnp.sum(power*(loss.kx**2+loss.ky**2))/jnp.sum(power)
            bundle[1]=target;bundle[2]=jnp.stack(widths);bundle[4]=jnp.asarray(D,jnp.float32);bundle[5]=moment
            m=plain(ready(terms(h,tuple(bundle))));spectrum=np.asarray(loss.p.spectrum(h,bundle[0]))
            fit=fitted_divergence(spectrum,loss.p.dx,cfg['wavelength_um']);a=angles(fit);b=angles(original['fitted_core_divergence']);old=original['metrics']
            dw=np.abs(np.asarray(m['widths_um'][0])/np.asarray(old['widths_um'][0])-1)
            eta=abs(m['declared_useful_efficiency']/old['declared_useful_efficiency']-1)
            dc=abs(m['normalized_curvature']-old['normalized_curvature']);dq=abs(m['angular_second_moment_ratio']-old['angular_second_moment_ratio'])
            fit_ok=a is not None and b is not None and all(abs(x-y)<=max(.02*abs(y),.001) for x,y in zip(a,b))
            checks={'output_diameter_change_below2percent':bool(np.max(dw)<.02),'useful_efficiency_change_below5percent':eta<.05,'core_divergence_change':fit_ok,'curvature_change_below005':dc<=.005,'retained_band_angular_ratio_change_below02':dq<=.02}
            rows.append(dict(setting=label,H_um=H,checks=checks,all_pass=all(checks.values()),diameter_relative_changes=dw.tolist(),eta_relative_change=eta,curvature_absolute_change=dc,retained_angular_ratio_absolute_change=dq,metrics=m,fitted_core_divergence=fit,target_held_fixed=True))
            print(json.dumps(dict(setting=label,H_um=H,all_pass=all(checks.values()),deltaC=dc)),flush=True)
            write(FOLDER/'proxy_sampling_progress.json',rows)
        loss.p.pedestal_cache.clear();loss.p.target_cache.clear();del loss,terms;jax.clear_caches()
    tolerance=[]
    for mfd in [9.9,10.9]:
        varied=copy.deepcopy(cfg);varied['mfd_um']=mfd;loss=HeightConditionedLoss(varied);terms=jax.jit(loss.terms)
        for H in [40.,70.,100.]:
            h=jnp.asarray(read(paths[H])['heights_um'],jnp.float32);bundle=loss.bundle(H);m=plain(ready(terms(h,bundle)))
            spectrum=np.asarray(loss.p.spectrum(h,bundle[0]));fit=fitted_divergence(spectrum,loss.p.dx,cfg['wavelength_um'])
            tolerance.append(dict(H_um=H,MFD_um=mfd,metrics=m,fitted_core_divergence=fit,scope='unretuned nominal design; changed source, no corrective training'))
        del loss,terms;jax.clear_caches()
    if not all(hashlib.sha256(p.read_bytes()).hexdigest()==before[H] for H,p in paths.items()):raise RuntimeError('Frozen designs changed during checks')
    final=dict(status='PASS_PROXY_SAMPLING' if all(r['all_pass'] for r in rows) else 'UNRESOLVED_PROXY_SAMPLING',checks=rows,MFD_tolerance_diagnostics=tolerance,wall_s=time.monotonic()-started,profiles_unchanged=True,independent_Maxwell_launched=False,limitations='One-way scalar model and retained spectral band only; no full all-angle M2 or Maxwell acceptance. Nominal frozen target held fixed in numerical refinements.')
    write(FOLDER/'proxy_sampling.json',final);print(json.dumps({'status':final['status'],'wall_s':final['wall_s'],'checks':len(rows),'MFD_diagnostics':len(tolerance)}),flush=True)

if __name__=='__main__':main()
