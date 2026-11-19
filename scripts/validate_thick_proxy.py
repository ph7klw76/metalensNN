"""Freeze and validate the finite-thickness repair before neural training."""
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import copy
import gc
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from fiber_metalens_nn.geometry import load_config, baselines
from fiber_metalens_nn.thick_proxy import ThickProxy
from fiber_metalens_nn.trial import write_json

root = Path('artifacts/acceptance-repair')
c = load_config()
c['proxy'].update(model='split_step',thickness_slices=25,split_step_reference_index=1.26)
c['maxwell']['profile_prism'] = True
c['training']['fresnel_seed_offset_um'] = c['height_max_um']/2
c['training']['heldout_targets_um'] = [7.6,7.8,8.0,8.6]
c['repair_provenance'] = {
    'seed_rule':'phase seed at the geometric mid-height, offset hmax/2; this changes heights, never measurement coordinates',
    'regression_targets_already_inspected_um':[7.6,7.8,8.6],
    'new_prospective_target_um':8.0,
    'full_wave_backpropagation':False,
}
write_json('configs/finite_thickness_repair.json',c)
proxy = ThickProxy(c)
# Unprojected, off-layer-boundary heights for smooth derivative comparisons.
h = np.asarray(baselines(7.4,c)['fresnel'])+.013
h = np.clip(h,.013,2.487)
v,g = proxy.value_gradient(jnp.asarray(h,jnp.float32),7.4)
gradient = []
for i in [3,12,25]:
    step = .002
    plus,minus = h.copy(),h.copy()
    plus[i] += step
    minus[i] -= step
    fd = float((proxy.objective(plus,7.4)-proxy.objective(minus,7.4))/(2*step))
    ad = float(g[i])
    gradient.append(dict(index=i,autodiff=ad,finite_difference=fd,relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-4)))
baseline = baselines(7.4,c)['fresnel']
metrics = {'256x25':proxy.evaluate(baseline,7.4)}
del proxy,g,v
jax.clear_caches();gc.collect()
for name,n,slices in [('256x50',256,50),('512x50',512,50)]:
    test = copy.deepcopy(c)
    test['proxy']['thickness_slices'] = slices
    p = ThickProxy(test,grid_n=n)
    metrics[name] = p.evaluate(baseline,7.4)
    print(json.dumps({'sampling':name,**metrics[name]}),flush=True)
    del p
    jax.clear_caches();gc.collect()
comparisons = {}
for a,b in [('256x25','256x50'),('256x50','512x50')]:
    x,y = metrics[a],metrics[b]
    comparisons[a+' -> '+b] = dict(focus_change=abs(x['focal_um']/y['focal_um']-1),eta_change=abs(x['eta_R']/y['eta_R']-1))
passed = all(g['relative_error']<.03 for g in gradient) and all(v['focus_change']<.02 and v['eta_change']<.05 for v in comparisons.values())
report = dict(status='PASS' if passed else 'FAIL_PROXY_VALIDATION',gradient=gradient,metrics=metrics,comparisons=comparisons)
write_json(root/'proxy-validation.json',report)
print(json.dumps(report),flush=True)
