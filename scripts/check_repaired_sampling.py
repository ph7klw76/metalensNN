"""Post-training sampling audit; never select or modify a learned geometry."""
import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import copy
import gc
import json
from pathlib import Path
import jax
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.thick_proxy import ThickProxy
from fiber_metalens_nn.trial import write_json

root = Path('artifacts/acceptance-repair')
c = load_config('configs/finite_thickness_repair.json')
design = json.loads((root/'phase5/selected-NN-f7.8.json').read_text())
metrics = {}
for name,n,padding,layers in [('production',256,2,25),('refined',512,2,50),('larger_window',256,3,25)]:
    cfg = copy.deepcopy(c)
    cfg['proxy']['thickness_slices'] = layers
    proxy = ThickProxy(cfg,grid_n=n,padding=padding)
    metrics[name] = proxy.evaluate(design['heights_um'],7.8)
    print(json.dumps({name:metrics[name]}),flush=True)
    del proxy
    jax.clear_caches();gc.collect()
checks = {}
for name in ['refined','larger_window']:
    a,b = metrics['production'],metrics[name]
    focus = abs(a['focal_um']/b['focal_um']-1)
    eta = abs(a['eta_R']/b['eta_R']-1)
    checks[name] = dict(focus_change=focus,eta_change=eta,pass_=focus<.02 and eta<.05)
report = dict(metrics=metrics,checks=checks,pass_=all(c['pass_'] for c in checks.values()),geometry_changed=False)
write_json(root/'trained-design-sampling.json',report)
gate_path = root/'phase6-7/gate.json'
gate = json.loads(gate_path.read_text())
gate['trained_design_sampling'] = report
resume = json.loads((root/'phase5/trained-state-resume.json').read_text())
gate['trained_state_resume_pass'] = resume['pass']
if not report['pass_'] or not resume['pass']:
    gate['status'] = 'FAIL_FINAL_MODEL_CHECKS'
    gate['engineering_gate'] = 'FAIL'
write_json(gate_path,gate)
raise SystemExit(0 if report['pass_'] and resume['pass'] else 2)
