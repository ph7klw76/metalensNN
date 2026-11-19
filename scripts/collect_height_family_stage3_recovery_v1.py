"""Additive saved-field recovery assessment; original execution ledger unchanged.

Uses frozen scientific selectors/gates. Never backdates the current clock,
constructs a controller, loads field arrays, or launches an optical worker.
"""
from __future__ import annotations
import argparse
import copy
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COLLECTOR_SHA='081f45029b9db2543d1c5f0f3a347f4f274981954df2ffe5089a3b908a89278a'
CORE_SHA='77dae64d7a81f78ec9e338e2d3e1c576756a0aa7c91977ecbba89cc87db2d08f'
REVIEWED_RECOVERY_HELPER_SHA='8d1d2c3a79955e5f57e7e359b6f58f317155b669150ddce54d64c5d8494e4f9a'
REVIEWED_RAW_COMPLETION_AUDIT_SHA='72cbeb744b3024b483c303e9b6194a437f73d23bf0b7e0dccd88bf2038018822'
SCHEMA='height_family_attempt83_postprocess_overlay_v1'
ANALYZER_SHA='4d1c2cd04c539660928f29d66bce3e701ef41a115512931070dd0f11a6dfc55b'
FINALIZED_OUTPUT_SHA={'recovery_overlay.json':'eeec761f674aecf735ae923a2d0b7acd1632cc6735611adb6f805055f60c5fea',
    'analysis83.json':'79167ce0ce1a479c9a086a500ef985975afbe9938812e19d6564fd14dc1fa8da',
    'receipt_before_analysis.json':'c23bda4735848242e3633b0bbdffe7c876cd78e391f81fdba6ebd388ce0dbce7',
    'root_request_exact.json':'5f2ee2288063ce5e2c3681dd98a9ffedff250cacd8a4a8d72778a080e9e79f06'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path,expected):
    if sha(path)!=expected:raise ValueError('Reviewed source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def modules():
    return (load('recovery_frozen_collector',ROOT/'scripts/collect_height_family_stage3_v2.py',COLLECTOR_SHA),
            load('recovery_frozen_core',ROOT/'scripts/assess_height_family_stage3_v1.py',CORE_SHA))


def finalized_output_binding(folder,protect):
    """Root-confirmed finalized outputs, independently anchored by15c5 review."""
    for name,expected in FINALIZED_OUTPUT_SHA.items():
        if protect(Path(folder)/name)!=expected:raise ValueError('Finalized recovery output trust anchor differs: '+name)


def validate_recovery_semantics(overlay,original_entry,*,deadline):
    if overlay.get('schema')!=SCHEMA or overlay.get('status')!='COMPLETE_FROM_SAVED_ARTIFACTS_POSTPROCESS_ONLY':
        raise ValueError('Completed explicit postprocess-only overlay required')
    if overlay.get('original_entry83')!=original_entry or original_entry.get('id')!=83 or original_entry.get('status')!='started':
        raise ValueError('Exact unmodified stale original83 entry required')
    if overlay.get('original_entry83_sha256')!=hashlib.sha256(json.dumps(original_entry,sort_keys=True).encode()).hexdigest():
        raise ValueError('Original entry83 canonical hash differs')
    result=overlay.get('attempt83_result',{})
    if result.get('id')!=83 or result.get('verification_status')!='complete_from_artifacts' or result.get('analysis_status')!='completed':
        raise ValueError('Saved raw/analysis completion must be explicit')
    if any(result.get(k,'missing') is not None for k in ('solver_exit_code','worker_finished_utc','controller_finished_utc')):
        raise ValueError('Unknown original execution exit/finish must remain null')
    evidence=overlay.get('raw_completion_evidence',{})
    if evidence.get('exact_worker_exit_and_completion_clock_unknown') is not True:
        raise ValueError('Artifact timing is not an exact worker exit/finish record')
    if evidence.get('all_artifact_mtimes_before_original_deadline') is not True:
        raise ValueError('Independent evidence of pre-deadline saved artifacts required')
    post=overlay.get('postprocessing',{})
    def utc(value):
        time=datetime.fromisoformat(value.replace('Z','+00:00'))
        if time.tzinfo is None:raise ValueError('UTC-aware postprocessing dates required')
        return time.astimezone(timezone.utc)
    if (post.get('after_original_deadline') is not True or post.get('old_clock_reset') is not False
            or post.get('new_optical_solves')!=0 or post.get('analysis_exit_code')!=0
            or not utc(deadline)<utc(post['started_utc'])<=utc(post['finished_utc'])):
        raise ValueError('Late postprocessing must not be backdated or reset optical time')


def analysis_argv_binding(command,*,root,record,config,output,target,c):
    expected=[str(root/'.venv-meep/bin/python'),'-B',str(root/'scripts/height_family_maxwell_analysis_v2.py'),
        '--record',str(record),'--config',str(config),'--target-diameter',str(target),'--output',str(output),
        '--count','512','--window','128.0']
    def normalized(args):
        if not isinstance(args,list) or len(args)!=len(expected):raise ValueError('Exact frozen analyzer argv required')
        result=list(args)
        for i in (0,2):result[i]=str(c.local_path(args[i]).resolve())
        for flag in ('--record','--config','--output','--target-diameter','--count','--window'):
            if args.count(flag)!=1:raise ValueError('One complete analyzer argument required')
            i=args.index(flag)
            if i+1>=len(args):raise ValueError('Incomplete analyzer argument')
            result[i+1]=(str(c.local_path(args[i+1]).resolve()) if flag in ('--record','--config','--output') else str(float(args[i+1])))
        return result
    if normalized(command)!=normalized(expected):raise ValueError('Postprocess argv differs from frozen512/window128 analyzer')


def recovered_record(e,overlay,c):
    """Project authenticated saved scalars, keeping ledger83 status untouched."""
    entry=e.entries['stage3',83];result=overlay['attempt83_result'];ctx=overlay['context']
    rp=c.local_path(result['raw_record_file']).resolve();raw=e.document(rp)
    if e.protect(rp)!=result['raw_record_sha256'] or rp!=c.contained(e.folder,entry['record_file']):
        raise ValueError('Recovered raw path/SHA differs from original83 output')
    if raw['settings']!=entry['effective_settings'] or raw['config_fingerprint']!=c.fingerprint(e.config):
        raise ValueError('Recovered raw physical/numerical settings differ')
    required={str(c.local_path(raw[k]).resolve()) for k in ('fields_file','power_audit_fields_file')}
    supplied={str(c.local_path(p).resolve()):v for p,v in result['required_file_sha256'].items()}
    if set(supplied)!=required:raise ValueError('Exactly the two actual83 native payloads required')
    for p,digest in supplied.items():
        if e.protect(p)!=digest:raise ValueError('Recovered raw native payload changed')
    if raw['worker_sha256']!=e.sources['scripts/height_family_maxwell_worker_v2.py']:
        raise ValueError('Actual83 worker differs from frozen source')
    case=next(v for v in e.cases if v['case_id']==entry['case_id'])
    if any(entry[k]!=case[k] for k in ('design_sha256','physical_geometry_id')):
        raise ValueError('Recovered83 is not the frozen original NN geometry')
    design=e.document(c.contained(e.folder,case['design_file']))
    if (e.protect(raw['design_file'])!=case['design_sha256'] or ctx['physical_geometry_id']!=case['physical_geometry_id']
            or c.local_path(ctx['design_file']).resolve()!=c.local_path(raw['design_file']).resolve()
            or c.local_path(ctx['config_file']).resolve()!=e.folder/'config.json'
            or c.local_path(ctx['selection_receipt_file']).resolve()!=c.contained(e.folder,entry['selection_receipt_file'])):
        raise ValueError('Recovered actual design identity differs')
    ref=e.entries['stage3',82];refpath,reference=e.raw('stage3',ref)
    if (entry.get('reference_id')!=82 or entry.get('reference_stage_id')!='stage3' or ctx.get('reference_id')!=82
            or c.local_path(raw['reference_file']).resolve()!=refpath or raw['settings']!=reference['settings']
            or c.local_path(ctx['reference_record_file']).resolve()!=refpath or ctx['reference_record_sha256']!=ref['record_sha256']):
        raise ValueError('Recovered83 requires exact completed reference82/settings')
    if {str(c.local_path(p).resolve()):v for p,v in ctx['reference_required_file_sha256'].items()}!={str(c.local_path(p).resolve()):v for p,v in ref['required_file_sha256'].items()}:
        raise ValueError('Recovered reference82 native payload inventory differs')
    ap=c.local_path(result['analysis_file']).resolve();saved=e.document(ap)
    if e.protect(ap)!=result['analysis_sha256'] or saved['settings']!=raw['settings'] or c.local_path(saved['record']).resolve()!=rp:
        raise ValueError('Late saved analysis83 path/hash/settings/raw identity differ')
    if not c.close(saved['declared_target_diameter_um'],design['target_diameter_um']):
        raise ValueError('Late analysis changes declared target')
    checks={key:saved.get('numerical_checks',{}).get(key) is True for key in c.NATIVE_KEYS}
    missing=['Native analysis check '+key for key in c.NATIVE_KEYS if key not in saved.get('numerical_checks',{})]
    for label,value in [('reference',reference),('device',raw)]:
        number=value.get('power_balance_error');checks[label+'_power_closure']=c.finite(number) and 0<=number<.05
        if not c.finite(number):missing.append('Finite '+label+' closure')
    incident=reference.get('incident_power');checks['positive_incident_reference']=c.finite(incident) and incident>0
    planes=[]
    for z in e.config['evaluation']['processing_planes_um']:
        values=[p for p in saved.get('metrics',{}).get('planes',[]) if c.close(p.get('z_um'),z)]
        p=values[0] if len(values)==1 else {};valid=p.get('valid_crossings')==[True,True]
        planes.append({'z_um':z,**{key:p.get(key) if valid and c.finite(p.get(key)) else None for key in ('diameter_x_um','diameter_y_um')}})
    return {'identity':e.folder.name+':83','method':'NN','H_um':entry['H_um'],'resolution':entry['resolution'],
        'job_type':entry['job_type'],'phase':entry['phase'],'stage_folder':str(e.folder),
        'selection_receipt_file':entry['selection_receipt_file'],'selection_receipt_sha256':entry['selection_receipt_sha256'],
        'record_file':str(rp),'record_sha256':result['raw_record_sha256'],'analysis_file':str(ap),'analysis_sha256':result['analysis_sha256'],
        'design_sha256':entry['design_sha256'],'reference_record_file':str(refpath),'reference_record_sha256':ref['record_sha256'],
        'settings':raw['settings'],'native_checks':checks,'native_all_pass':all(checks.values()),'native_missing_evidence':missing,
        'optical_gates':{key:saved.get('optical_gates',{}).get(key) for key in c.OPTICAL_KEYS},
        'eta':saved.get('conservative_useful_efficiency'),'C':saved.get('normalized_curvature'),'Q':saved.get('angular_second_moment_ratio'),
        'planes':planes,'core_half_angles_deg':c.core_angles(saved.get('metrics',{}).get('fitted_core_divergence',{})),
        'saved_field_origin':'RECOVERED83_RAW_PLUS_LATE_POSTPROCESSING','original_ledger_status':entry['status'],
        'solver_exit_code':None,'worker_finished_utc':None,'controller_finished_utc':None}


def overlay_evidence(e,path,c):
    if REVIEWED_RECOVERY_HELPER_SHA is None or REVIEWED_RAW_COMPLETION_AUDIT_SHA is None:
        raise ValueError('Recovery helper and independent raw proof await reviewed exact source pins')
    path=c.local_path(path).resolve();overlay=e.document(path)
    finalized_output_binding(path.parent,e.protect)
    if len(e.ledger['entries'])!=18 or [x['id'] for x in e.ledger['entries']]!=list(range(66,84)):
        raise ValueError('Preserve charged18/83 attempts')
    validate_recovery_semantics(overlay,e.entries['stage3',83],deadline=c.DEADLINE)
    if (c.local_path(overlay['original_ledger_source_file']).resolve()!=e.folder/'ledger.json'
            or overlay['original_ledger_sha256']!=hashlib.sha256(e.ledger_bytes).hexdigest()):
        raise ValueError('Recovery overlay does not bind this exact unchanged original ledger')
    saved_ledger=c.local_path(overlay['original_ledger_file']).resolve()
    if e.protect(saved_ledger)!=overlay['original_ledger_sha256'] or e.document(saved_ledger)!=e.ledger:
        raise ValueError('Recovery exact ledger snapshot differs')
    manifest=e.document(path.parent/'recovery_manifest.json')
    required_members={'recovery_overlay.json','analysis83.json','analysis83.log','receipt_before_analysis.json',
        'root_request_exact.json','original_ledger_exact.json','frozen_analyzer_snapshot.py',
        'independent_raw_audit_exact.json','recover_height_family_attempt83_postprocess_v1.py','postprocess_attempt_result.json'}
    if (manifest.get('status')!='ADDITIVE_POSTPROCESS_RECOVERY_FILES_ONLY' or manifest.get('schema')!=SCHEMA
            or not required_members.issubset(manifest.get('files',{}))
            or manifest.get('original_ledger_source_sha256')!=overlay['original_ledger_sha256']
            or manifest.get('original_raw_inputs_unchanged') is not True or manifest.get('new_optical_solves')!=0):
        raise ValueError('Complete additive unchanged-input recovery manifest required')
    for name,item in manifest['files'].items():
        member=c.contained(path.parent,name)
        if e.protect(member)!=item['sha256'] or member.stat().st_size!=item['bytes']:raise ValueError('Recovery output inventory changed')
    if path.name not in manifest['files'] or manifest['files'][path.name]['sha256']!=e.protect(path):raise ValueError('Overlay absent from recovery manifest')
    helper=ROOT/'scripts/recover_height_family_attempt83_postprocess_v1.py'
    if (e.protect(helper)!=REVIEWED_RECOVERY_HELPER_SHA or overlay.get('helper_source_sha256')!=REVIEWED_RECOVERY_HELPER_SHA
            or e.protect(path.parent/helper.name)!=REVIEWED_RECOVERY_HELPER_SHA
            or e.protect(path.parent/'frozen_analyzer_snapshot.py')!=ANALYZER_SHA):
        raise ValueError('Reviewed recovery helper/analyzer snapshots changed')
    audit=c.local_path(overlay['raw_completion_audit_file']).resolve()
    if e.protect(audit)!=overlay['raw_completion_audit_sha256'] or e.protect(audit)!=REVIEWED_RAW_COMPLETION_AUDIT_SHA:
        raise ValueError('Independent raw-completion proof differs')
    audit_doc=e.document(audit)
    if not str(audit_doc.get('status','')).startswith('PASS'):raise ValueError('Independent raw-completion proof is not PASS')
    audit_inputs={}
    for key in ('input_sha256','file_sha256'):
        for name,digest in audit_doc.get(key,{}).items():
            name=str(c.local_path(name).resolve())
            if name in audit_inputs and audit_inputs[name]!=digest:raise ValueError('Conflicting independent raw proof inventory')
            audit_inputs[name]=digest
    result=overlay['attempt83_result']
    for filename,digest in {result['raw_record_file']:result['raw_record_sha256'],**result['required_file_sha256'],
                            overlay['original_ledger_source_file']:overlay['original_ledger_sha256']}.items():
        if audit_inputs.get(str(c.local_path(filename).resolve()))!=digest:raise ValueError('Independent proof omits exact raw83/native/ledger hash')
    receipt=c.local_path(overlay['receipt_before_analysis_file']).resolve()
    if e.protect(receipt)!=overlay['receipt_before_analysis_sha256']:raise ValueError('Postprocess receipt changed')
    before=e.document(receipt)
    if (before.get('status')!='PROSPECTIVE_POSTPROCESS_ONLY_BEFORE_ANALYZER'
            or before.get('helper_source_sha256')!=REVIEWED_RECOVERY_HELPER_SHA or before.get('frozen_analyzer_sha256')!=ANALYZER_SHA
            or before.get('new_optical_solves')!=0 or before.get('new_training_calls')!=0
            or before.get('modify_original_ledger') is not False or before.get('old_clock_reset') is not False
            or before.get('original_entry83')!=overlay['original_entry83']
            or any(before.get(k,'missing') is not None for k in ('solver_exit_code','worker_finished_utc','controller_finished_utc'))
            or c.utc(before['created_utc'])>c.utc(overlay['postprocessing']['started_utc'])):
        raise ValueError('Prospective postprocess-only receipt differs')
    if e.protect(path.parent/'root_request_exact.json')!=before['root_request_sha256']:raise ValueError('Exact postprocess ROOT request changed')
    for name,digest in before['protected_input_sha256'].items():
        if e.protect(name)!=digest:raise ValueError('Pre-analysis protected input changed')
    ctx=overlay['context']
    expected_sources={str((ROOT/k).resolve()):v for k,v in e.sources.items()}
    actual_sources={str(c.local_path(k).resolve()):v for k,v in ctx['source_sha256'].items()}
    if any(actual_sources.get(k)!=v for k,v in expected_sources.items()):raise ValueError('Recovery source proof omits frozen sources')
    for name,digest in ctx['source_sha256'].items():
        if e.protect(c.local_path(name))!=digest:raise ValueError('Recovery source proof changed')
    for stem in ('config','design','reference_record','selection_receipt'):
        if e.protect(ctx[stem+'_file'])!=ctx[stem+'_sha256']:raise ValueError('Recovery context changed: '+stem)
    analysis_path=c.local_path(result['analysis_file']).resolve()
    if not analysis_path.is_relative_to(path.parent) or e.protect(analysis_path)!=result['analysis_sha256']:
        raise ValueError('Recovery analysis is outside authenticated recovery output')
    outcome=e.document(path.parent/'postprocess_attempt_result.json')
    if (outcome.get('status')!='POSTPROCESS_COMPLETE' or outcome.get('error') is not None
            or outcome.get('completion',{}).get('exit_code')!=0 or outcome.get('new_optical_solves')!=0
            or outcome.get('original_ledger_modified') is not False
            or outcome.get('started_utc')!=overlay['postprocessing']['started_utc']
            or c.utc(outcome['finished_utc'])<c.utc(overlay['postprocessing']['finished_utc'])):
        raise ValueError('Postprocess completion result differs from explicit overlay')
    target=e.document(ctx['design_file'])['target_diameter_um']
    analysis_argv_binding(before['analysis_command'],root=ROOT,record=c.local_path(result['raw_record_file']).resolve(),
        config=e.folder/'config.json',output=analysis_path,target=target,c=c)
    return overlay


def add_recovered_record(base,record):
    data=copy.deepcopy(base['data'])
    if any(r['identity']==record['identity'] for r in data['Meep_records']):raise ValueError('Recovery identity already present')
    data['Meep_records'].append(record)
    return data


def scoped_optical_only(assessment,core):
    controls={str(i):r for i,r in enumerate(assessment['boundary_duration_checks']) if r.get('required_original_sampled_control')}
    rows={str(r['H_um']):core.aggregate({key:r[key] for key in ('native','nominal_optical','selected_mesh')}) for r in assessment['NN_heights']}
    coverage={'height_coverage':'PASS' if len(assessment['NN_heights'])==8 and sorted(r['H_um'] for r in assessment['NN_heights'])==list(core.HEIGHTS) else 'UNRESOLVED',
        'original_control_coverage':'PASS' if len(controls)==4 and {(r['job_type'],r['H_um']) for r in controls.values()}=={('pml',40.),('pml',100.),('padding',70.),('duration',100.)} else 'UNRESOLVED'}
    return {**core.aggregate({**coverage,**rows,**{'original_control_'+k:v for k,v in controls.items()}}),
        'scope':'Saved-field native/nominal/last-mesh and original sampled controls only; excludes unknown execution closure and expired reporting-clock acceptance',
        'full_workflow_acceptance_claimed':False}


def collect_recovered(folder,overlay_path):
    c,core=modules();folder=c.local_path(folder).resolve()
    base=c.collect_stage3(folder)  # true current time, including post-deadline FAIL
    if base['authorization']['status']!='PASS':raise ValueError('Original saved/prospective evidence authorization is not verified')
    e=c.Evidence(folder)
    e.protect(__file__)
    if base['data']['ledger_snapshot_sha256'][str(folder/'ledger.json')]!=hashlib.sha256(e.ledger_bytes).hexdigest():
        raise RuntimeError('Original ledger changed between read-only evidence passes')
    overlay=overlay_evidence(e,overlay_path,c)
    for i in (82,83):
        if c.attempt_receipt(e,e.entries['stage3',i])['status']!='PASS':raise ValueError('Original prospective authorization failed')
    record=recovered_record(e,overlay,c);data=add_recovered_record(base,record)
    for p,digest in e.inputs.items():
        if p in data['input_sha256'] and data['input_sha256'][p]!=digest:raise ValueError('Conflicting original/recovery input hash')
        data['input_sha256'][p]=digest
    auth=copy.deepcopy(base['authorization']);auth['record_authorizations'][record['identity']]={'status':'PASS','basis':'Verified recovered saved-field provenance only; original process exit/finish unknown'}
    assessment=core.assess_data(data,base['numerical_limits'],auth,base['budget'],base['primary_execution'])
    recovery={'status':'PASS_AUTHENTICATED_RECOVERED_SAVED_FIELD','overlay_file':str(c.local_path(overlay_path).resolve()),
        'overlay_sha256':e.protect(overlay_path),'postprocessing':overlay['postprocessing'],
        'raw_completion_evidence':overlay['raw_completion_evidence'],'original_ledger_status83':'started',
        'solver_exit_code83':None,'worker_finished_utc83':None,'controller_finished_utc':None,
        'original_budget':base['budget'],'original_adaptive_execution':base['adaptive_execution'],
        'no_clock_reset_or_backdating':True,'new_optical_solves':0,'full_workflow_acceptance_claimed':False}
    assessment.update(status='RECOVERED_SAVED_FIELD_SCIENTIFIC_ASSESSMENT_WORKFLOW_NOT_ACCEPTED',
        assessment_schema='height_family_stage3_recovery_v1',recovery_context=recovery,
        assessment_utility_sha256=sha(__file__),frozen_scientific_core_sha256=CORE_SHA,
        adaptive_execution=base['adaptive_execution'],selected_adaptive_diagnostics=c.selected_pair_diagnostics(e,data,base['numerical_limits'],c.load_science()))
    assessment['scoped_saved_field_optical_checks']=scoped_optical_only(assessment,core)
    for p,digest in e.inputs.items():
        if p in data['input_sha256'] and data['input_sha256'][p]!=digest:raise ValueError('Late recovery input conflicts')
        data['input_sha256'][p]=digest
    assessment['input_sha256']=dict(data['input_sha256'])
    if not all(c.sha(p)==digest for p,digest in data['input_sha256'].items()):raise RuntimeError('Immutable input changed during recovery assessment')
    return {'data':data,'assessment':assessment,'numerical_limits':base['numerical_limits'],'recovery_context':recovery}


def write_assessment_outputs(result,output,c,core):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'assessment.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(c.safe_json(result['assessment']),indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(
        'RECOVERED SAVED-FIELD SCIENTIFIC CHECKS ONLY\nOriginal ledger83 stays started; worker/controller exit and exact finishUTC remain unknown.\n'
        'Postprocessing is after the original deadline. Current full-clock/workflow acceptance is not PASS. No new optical solve.\n'+
        json.dumps(result['assessment']['scoped_saved_field_optical_checks'],indent=2)+'\n\n'+core.human_report(result['assessment']))
    columns=['H_um','selected_identity','resolution_cells_per_um','eta_fraction','C_dimensionless','Q_dimensionless',
        'native_status','nominal_status','last_mesh_status','last_mesh_pair','raw_record_sha256','analysis_sha256']
    with (output/'all_eight_height_verification.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for row in result['assessment']['NN_heights']:
            m=row.get('Meep') or {};writer.writerow(dict(zip(columns,[row['H_um'],m.get('identity'),m.get('resolution'),
                m.get('eta'),m.get('C'),m.get('Q'),row['native']['status'],row['nominal_optical']['status'],
                row['selected_mesh']['status'],json.dumps(row['selected_mesh'].get('resolutions',[])),m.get('record_sha256'),m.get('analysis_sha256')])))
    manifest={'status':'RECOVERED_SAVED_FIELD_OPTICAL_CHECKS_ONLY_NO_WORKFLOW_PASS','source_sha256':sha(__file__),
        'input_sha256':result['assessment']['input_sha256'],'recovery_context':result['recovery_context'],
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}}
    with (output/'delivery_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True)
    parser.add_argument('--overlay',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    c,core=modules();folder=c.local_path(args.folder).resolve();output=c.local_path(args.output).resolve()
    if output.exists() or not output.is_relative_to(folder/'assessment'):raise ValueError('Use a fresh additive assessment version')
    result=collect_recovered(folder,args.overlay);write_assessment_outputs(result,output,c,core)
    print(json.dumps({'output':str(output),'scoped_optical':result['assessment']['scoped_saved_field_optical_checks']['status'],'workflow':'NOT_ACCEPTED','optical_solves':0}))


if __name__=='__main__':main()
