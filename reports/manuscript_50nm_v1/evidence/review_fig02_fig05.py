"""Bounded CSV/metadata spot-check; no fields or historical hash collection."""
import csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];C=ROOT/'artifacts/1550nm_height_family_n15263_v2';F=OUT.parent/'figures/final_v3';checks=[];hashes={}
def local(p):
    p=str(p)
    if len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+'/'+p[3:].replace('\\','/')
    return Path(p).resolve()
def sha(p):
    p=local(p);h=hashlib.sha256(p.read_bytes()).hexdigest();hashes[str(p)]=h;return h
def read(p):return json.loads(local(p).read_text(encoding='utf-8-sig'))
def rows(name):
    with (F/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)
def same(a,b):return math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-13)
facts=read(OUT/'manuscript_facts.json');check('Stable independently extracted facts',sha(OUT/'manuscript_facts.json')=='0faecc0940a7d3330de108fff2afbc998869037699770851105c74e1b0cdc299')
manifest=read(F/'fig02_fig05_manifest.json')
for name in ('fig02_requested_heights_proxy.csv','fig02_seed_selection.csv','fig05_nominal_mesh_records.csv','fig05_successive_mesh_checks.csv','fig05_original_coarse_controls.csv','fig02_fig05_captions.txt'):check('ManifestCSV/captionSHA '+name,sha(F/name)==manifest['files'][name]['sha256'])
script=local(manifest['script_file']);check('Frozenplot source',sha(script)==manifest['script_sha256']=='59717d91d0a04af2caebf7d89ff0686affe5020066303f79c0863f9b951d8e72')
requested=rows('fig02_requested_heights_proxy.csv');check('Nine requestedproxy points',len(requested)==9)
for r in requested:
    truth=next(x for x in facts['learning']['saved_proxy_comparison_requested_heights'] if x['H_um']==float(r['H_um']) and x['method']==r['method'])
    for k,t in [('proxy_useful_eta_fraction',truth['useful_efficiency_percent']/100),('proxy_objective_L',truth['loss']),('proxy_C',truth['normalized_curvature_C']),('proxy_retained_band_Q',truth['angular_second_moment_Q']),('eta_gain_over_physics_seed_percentage_points',truth['eta_gain_over_quantized_field_seed_percentage_points']),('loss_change_from_physics_seed',truth['loss_change_over_quantized_field_seed'])]:check('Proxy savedscalar '+r['method']+r['H_um']+' '+k,same(r[k],t))
    check('Observedheight domain '+r['H_um'],r['height_domain']==truth['height_domain'])
seeds=rows('fig02_seed_selection.csv');best_steps={}
for r in seeds:
    method=r['method'];seed=int(r['seed']);arm=next(a for a in facts['learning']['arms'] if a['method']==method and a['seed']==seed);vpath=C/f'{method}_seed{seed}/validation.json';values=read(vpath);best=min(values,key=lambda v:v['key']);best_steps[f'{method}_seed{seed}']=best['step']
    check('Bestlexicographic checkpoint step '+str((method,seed)),int(r['selected_checkpoint_step'])==best['step'])
    check('Best checkpointvalues '+str((method,seed)),same(r['worst_validation_eta_fraction'],-arm['validation_key'][1]) and same(r['mean_validation_L'],arm['validation_key'][2]) and int(r['objective_calls'])==162)
    check('Selectedseed arm '+str((method,seed)),(r['selected_arm']=='True')==(facts['learning']['selection']['selection'][method]['seed']==seed))
nominal=rows('fig05_nominal_mesh_records.csv');check('Eleven nominalmesh points',len(nominal)==11)
for r in nominal:
    raw=read(r['record_file']);ref=read(raw['reference_file']);a=read(r['analysis_file'])
    check('Nominalmesh scalars '+r['identity'],all(same(r[k],t) for k,t in [('useful_eta_fraction',a['conservative_useful_efficiency']),('C',a['normalized_curvature']),('retained_band_Q',a['angular_second_moment_ratio']),('device_power_closure_fraction',raw['power_balance_error']),('reference_power_closure_fraction',ref['power_balance_error']),('maximum_native_closure_fraction',max(raw['power_balance_error'],ref['power_balance_error']))]))
    if r['selected_latest']=='True':
        truth=next(x for x in facts['all_eight_Maxwell_performance'] if x['H_um']==float(r['H_um']));check('Latestselected identity '+r['identity'],r['identity']==truth['selected_identity'])
check('Exactlylatest43/69/83',set(r['identity'] for r in nominal if r['selected_latest']=='True')=={'maxwell_validation_v1:43','maxwell_validation_v3:69','maxwell_validation_v3:83'})
pairs=rows('fig05_successive_mesh_checks.csv');check('Eight successive pairs',len(pairs)==8)
for p in pairs:
    match=next(x for x in facts['old_mesh_history'] if x['prior_identity']==p['prior_identity'] and x['fine_identity']==p['fine_identity']);m=match['comparison']['measurements']
    check('Pair status preserved '+p['prior_identity']+'->'+p['fine_identity'],p['declared_mesh_check_status']==match['comparison']['status'])
    for k,v in [('maximum_processing_D_relative_change',m['maximum_diameter_relative_change']),('eta_relative_change',m['useful_efficiency_relative_change']),('absolute_C_change',m['curvature_absolute_change']),('retained_band_Q_absolute_change_diagnostic',m['retained_band_angular_ratio_absolute_change'])]:check('Savedpair comparison '+k+p['prior_identity'],same(p[k],v))
check('Four earlierfailed pairs retained',sum(p['declared_mesh_check_status']=='FAIL' for p in pairs)==4)
controls=rows('fig05_original_coarse_controls.csv');check('Four separately sampledcoarse controls',len(controls)==4)
for c in controls:
    t=next(x for x in facts['original_sampled_controls'] if x['kind']==c['kind'] and x['H_um']==float(c['H_um']))
    check('Original16 control metadata '+c['kind']+c['H_um'],float(c['resolution_cells_per_um'])==16 and c['nominal_identity']==t['nominal']['identity'] and c['variation_identity']==t['variation']['identity'] and c['numerical_comparison_status']==t['status'])
    check('Originalcontrol Q '+c['kind']+c['H_um'],same(c['Q_nominal'],t['Q_prior']) and same(c['Q_variant'],t['Q_variant']))
source=script.read_text();caption=(F/'fig02_fig05_captions.txt').read_text()
check('Closure percent sourceconversion',"100*r['maximum_native_closure_fraction']" in source and "Width change / closure (%)" in source)
check('Caption retains scientific limits','not a fixed-H learning curve' in caption and 'full lexicographic validation key' in caption and 'not an untouched test' in caption and 'sampled separately at16' in caption and 'not at each finest mesh' in caption and 'Q-change itself is diagnostic' in caption and 'original process exit/finish was unrecorded' in caption)
result=dict(status='PASS_FIG02_FIG05_BOUNDED_CSV_AND_LABEL_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,passed_assertions=len(checks),file_sha256=hashes,best_checkpoint_steps=best_steps,source_data_counts=dict(requested_proxy_points=9,seed_arms=6,nominal_mesh_records=11,successive_mesh_pairs=8,separate_coarse_controls=4),material_findings=[],full_historical_hash_collection=False,field_analysis_or_optical_or_training_runs=0)
with (OUT/'fig02_fig05_independent_review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status='PASS',checks=len(checks),review_sha256=sha(OUT/'fig02_fig05_independent_review.json'),best_steps=best_steps)))
