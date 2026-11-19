"""Independent, sequential acceptance rerun with bounded adaptive mesh refinement."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from fiber_metalens_nn.geometry import load_config, baselines, project, save_design
from fiber_metalens_nn.maxwell_batch import MaxwellBatch, relative_checks
from fiber_metalens_nn.trial import write_json

root = Path('artifacts/acceptance-repair')
c = load_config('configs/finite_thickness_repair.json')
folder = root/'phase6-7'
folder.mkdir(exist_ok=True)
training = root/'phase5'
assert json.loads((root/'proxy-validation.json').read_text())['status'] == 'PASS'
assert json.loads((training/'gate.json').read_text())['status'] == 'PASS'
selected = json.loads((training/'selection-before-heldout.json').read_text())['selected_seed']
write_json(folder/'frozen-selection.json',dict(selected_seed=selected,
    config_sha256=hashlib.sha256(Path('configs/finite_thickness_repair.json').read_bytes()).hexdigest(),
    rule='proxy validation only; fixed before new Maxwell evaluation',
    regression_target_previously_inspected_um=7.8,prospective_target_um=8.0))
batch = MaxwellBatch(c,root/'final-maxwell')
reference = batch.job()
reverse = batch.job(m=-1)
reference_check = dict(m_plus=reference,m_minus=reverse,
    pass_=reference['pass'] and reverse['pass'] and abs(reference['incident_aperture_power']/reverse['incident_aperture_power']-1)<.02)
write_json(folder/'reference-check.json',reference_check)
assert reference_check['pass_']

# Training-only bounded feedback is kept separate from untouched NN outputs.
train_h = np.array(json.loads((training/'selected-NN-f7.4.json').read_text())['heights_um'])
feedback_paths = {'untouched':str(training/'selected-NN-f7.4.json')}
for sign in [-1,1]:
    h = train_h.copy()
    h[16:] += sign*c['fabrication']['height_step_um']
    path = folder/f'training-feedback-{sign:+d}.json'
    save_design(path,project(h,c),7.4,'training_only_Maxwell_feedback',c)
    feedback_paths[str(sign)] = str(path)
feedback = {name:batch.job(path,resolution=20) for name,path in feedback_paths.items()}
best = max(feedback,key=lambda name:feedback[name]['eta_R']-.1*abs(feedback[name]['focal_um']/7.4-1))
write_json(folder/'feedback-selection.json',dict(target_um=7.4,selected=best,reports=feedback,
    heldout_used=False,score='eta_R - 0.1*abs(f/f_target-1)'))
feedback_mesh = [feedback[best]]
for res in [30,40]:
    feedback_mesh.append(batch.job(feedback_paths[best],resolution=res))
write_json(folder/'feedback-mesh.json',feedback_mesh)

cases = {}
for name,h in baselines(7.8,c).items():
    path = folder/f'{name}-f7.8.json'
    save_design(path,h,7.8,'finite_thickness_seed_'+name,c)
    cases[name] = str(path)
cases['original_fresnel'] = 'artifacts/phase6/fresnel-f7.8.json'
cases['direct'] = str(training/f'seed-{selected}/direct-f7.8.json')
cases['NN'] = str(training/'selected-NN-f7.8.json')
cases['NN_prospective_8.0'] = str(training/f'seed-{selected}/nn-f8.0.json')
results = {}
for resolution in [20,30,40]:
    for name,path in cases.items():
        results.setdefault(name,[]).append(batch.job(path,resolution=resolution))
        write_json(folder/'mesh-results.json',results)
checks = {}
for name,series in results.items():
    focus = name not in ['no_lens','flat']
    checks[name] = relative_checks(series[-2],series[-1],c,focus)
    for res in [60,80]:
        if checks[name]['pass']:
            break
        series.append(batch.job(cases[name],resolution=res))
        checks[name] = relative_checks(series[-2],series[-1],c,focus)
        write_json(folder/'mesh-results.json',results)
    write_json(folder/'mesh-convergence.json',checks)

boundary = {}
for name in ['NN','fresnel']:
    standard = results[name][-1]
    boundary[name] = {}
    fine = standard['settings']['resolution']
    for label,options in [('spacing',{'padding':4.5}),('PML',{'pml':2.0}),('duration',{'duration':90.0})]:
        report = batch.job(cases[name],resolution=fine,**options)
        boundary[name][label] = dict(metrics=report,comparison=relative_checks(standard,report,c))
        write_json(folder/'boundary-tests.json',boundary)
nn,fr = results['NN'][-1],results['fresnel'][-1]
feedback_check = relative_checks(feedback_mesh[-2],feedback_mesh[-1],c)
engineering = all(v['pass'] for v in checks.values()) and feedback_check['pass'] and all(v['comparison']['pass'] for group in boundary.values() for v in group.values())
optical = abs(nn['focal_um']/7.8-1)<=c['acceptance']['focal_error_relative_max'] and nn['eta_R']/fr['eta_R']>=c['acceptance']['eta_relative_to_fresnel_min']
prospective = results['NN_prospective_8.0'][-1]
report = dict(status='PASS' if engineering and optical else 'FAIL_ACCEPTANCE',
    result_label=c['result_label'],engineering_gate='PASS' if engineering else 'FAIL',
    optical_gate='PASS' if optical else 'FAIL',mesh_convergence=checks,feedback_convergence=feedback_check,
    NN=nn,matched_fresnel=fr,original_fresnel=results['original_fresnel'][-1],
    focal_relative_error=abs(nn['focal_um']/7.8-1),eta_relative_to_fresnel=nn['eta_R']/fr['eta_R'],
    prospective_8_0=prospective,prospective_focal_pass=abs(prospective['focal_um']/8.-1)<=.1,
    unique_geometries=len(batch.ledger),full_wave_backpropagation=False,
    regression_target_is_fresh_heldout=False,original_gates_preserved=True)
write_json(folder/'gate.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ['NN','matched_fresnel','original_fresnel','prospective_8_0']}),flush=True)
raise SystemExit(0 if report['status']=='PASS' else 2)
