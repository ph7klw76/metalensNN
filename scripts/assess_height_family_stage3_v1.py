"""Read saved three-stage evidence; never launch optics, training or exports.

Scientific per-record selection/gates are the unchanged SHA-pinned v3 code.
Only the reporting height enumeration expands to the original seven plus75.
The collector independently supplies three-stage authorization and accounting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
SCIENTIFIC_SOURCE=ROOT/'scripts/plot_height_family_nn_forward_v3.py'
SCIENTIFIC_SHA='aafa34734fbdd6c959278d935e43bfe0ae7e81cf877041b6380ac658d3d3d043'
FAMILY=(40.,50.,60.,70.,80.,90.,100.)
HEIGHTS=(40.,50.,60.,70.,75.,80.,90.,100.)
VARIANTS={'pml','padding','duration','reserve_duration'}


def load_science():
    import hashlib
    if hashlib.sha256(SCIENTIFIC_SOURCE.read_bytes()).hexdigest()!=SCIENTIFIC_SHA:
        raise ValueError('Unchanged pinned v3 scientific selector required')
    spec=importlib.util.spec_from_file_location('stage3_pinned_scientific_v3',SCIENTIFIC_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def aggregate(components):
    """Keep a known failure visible even when other evidence is unresolved."""
    components=dict(components)
    states=[value['status'] if isinstance(value,dict) else value for value in components.values()]
    return {'status':'FAIL' if 'FAIL' in states else ('PASS' if states and all(s=='PASS' for s in states) else 'UNRESOLVED'),
        'components':{key:(value['status'] if isinstance(value,dict) else value) for key,value in components.items()}}


def scientific_rows(data,limits,science=None):
    """Use identical function bytecode; change reporting coverage, not physics."""
    science=load_science() if science is None else science
    scope=dict(science.build_comparison.__globals__)
    scope['HEIGHTS']=HEIGHTS
    report=types.FunctionType(science.build_comparison.__code__,scope,
                              science.build_comparison.__name__,science.build_comparison.__defaults__)
    return report(data,limits)


def mesh_histories(data,limits,science):
    groups=defaultdict(dict)
    for record in data['Meep_records']:
        if record['method']!='NN' or record['job_type'] in VARIANTS:continue
        key=(record['H_um'],record['design_sha256'],*(record['settings'][k] for k in ('pml_um','padding_um','duration')))
        groups[key][record['resolution']]=record
    histories=[]
    for group in groups.values():
        resolutions=sorted(group)
        for i,coarse in enumerate(resolutions):
            for j in range(i+1,len(resolutions)):
                prior,fine=group[coarse],group[resolutions[j]]
                for record in (prior,fine):
                    if science.sha(record['analysis_file'])!=record['analysis_sha256']:raise ValueError('Historical analysis hash changed')
                comparison=science.compare_analyses(science.read(prior['analysis_file']),science.read(fine['analysis_file']),data['config'],limits,require_collimation=True)
                histories.append({'H_um':fine['H_um'],'resolutions':[coarse,fine['resolution']],
                    'prior_identity':prior['identity'],'fine_identity':fine['identity'],
                    'adjacent_in_current_sequence':j==i+1,
                    'both_native_references':prior['native_all_pass'] and fine['native_all_pass'],
                    'comparison':comparison,
                    'scope':'Arithmetic replay of unchanged saved analyses; no additional solve'})
    return histories


def boundary_assessments(data,limits,science):
    records=[r for r in data['Meep_records'] if r['method']=='NN']
    expected=[('pml',40.),('pml',100.),('padding',70.),('duration',100.)]
    originals=data.get('original_boundary_definitions',[])
    variants={r['identity']:r for r in records if r['job_type'] in VARIANTS}
    selected=[];used=set()
    # Original coarse controls are anchored to the sealed old assessment and
    # their exact saved settings. A new fine variant never replaces them.
    for kind,H in expected:
        matches=[d for d in originals if d.get('job_type')==kind and science.close(d.get('H_um'),H)]
        definition=matches[0] if len(matches)==1 else None
        variation=variants.get(definition.get('variation_identity')) if definition else None
        selected.append((kind,H,variation,definition,True))
        if variation:used.add(variation['identity'])
    for variation in sorted(variants.values(),key=lambda r:(r['H_um'],r['job_type'],r['resolution'],records.index(r))):
        if variation['identity'] in used:continue
        kind='duration' if variation['job_type']=='reserve_duration' else variation['job_type']
        selected.append((kind,variation['H_um'],variation,None,False))
    output=[]
    for kind,H,variation,definition,required_original in selected:
        changed={'pml':'pml_um','padding':'padding_um','duration':'duration'}[kind]
        minimum={'pml':3.,'padding':15.,'duration':None}[kind]
        result={'job_type':kind,'H_um':H,'required_original_sampled_control':required_original,
                'original_control_definition':definition}
        if variation is None:
            output.append({**result,'status':'UNRESOLVED','missing_evidence':['Authenticated original control definition and completed variation required']});continue
        controls=('H_um','resolution','pml_um','padding_um','duration')
        bases=[r for r in records if r['job_type'] not in VARIANTS and r['design_sha256']==variation['design_sha256']
               and all(science.close(r['settings'].get(k),variation['settings'].get(k)) for k in controls if k!=changed)]
        if not bases:
            output.append({**result,'status':'UNRESOLVED','variation_identity':variation['identity'],
                           'missing_evidence':['Same-resolution nominal field with only the intended setting changed required']});continue
        base=max(bases,key=records.index)
        analyses=[]
        for record in (base,variation):
            if science.sha(record['analysis_file'])!=record['analysis_sha256']:raise ValueError('Boundary analysis hash changed')
            analyses.append(science.read(record['analysis_file']))
        comparison=science.compare_analyses(*analyses,data['config'],limits,require_collimation=True)
        required=1.5*base['settings']['duration'] if kind=='duration' else minimum
        checks={'declared_variation_reached':science.finite(variation['settings'].get(changed)) and variation['settings'][changed]>=required-1e-8,
                'native_and_references':base['native_all_pass'] and variation['native_all_pass'],
                'declared_numerical_stability':comparison['status']=='PASS'}
        if required_original:
            checks['authenticated_original_settings']=definition is not None and all(
                science.close(variation['settings'].get(k),definition.get('settings',{}).get(k)) for k in controls)
        missing=[*comparison['missing_evidence'],*base.get('native_missing_evidence',[]),*variation.get('native_missing_evidence',[])]
        state='UNRESOLVED' if missing else ('PASS' if all(checks.values()) else 'FAIL')
        scope=('Combined radial-domain/output-aperture perturbation; not a fixed-aperture boundary proof'
               if kind=='padding' else 'Declared '+kind+' perturbation; this does not establish temporal/all-angle convergence')
        output.append({**result,'status':state,'resolution':variation['resolution'],'checks':checks,
            'missing_evidence':missing,'nominal_identity':base['identity'],'variation_identity':variation['identity'],
            'comparison':comparison,'scope':scope,
            'nominal_saved_optical_flags':analyses[0].get('optical_gates',{}),
            'variation_saved_optical_flags':analyses[1].get('optical_gates',{}),
            'Q_gate_crossing':bool(analyses[0].get('optical_gates',{}).get('angular_second_moment'))!=bool(analyses[1].get('optical_gates',{}).get('angular_second_moment')),
            'Q_prior':base.get('Q'),'Q_variant':variation.get('Q'),
            'Q_change_is_diagnostic_not_a_new_convergence_gate':True})
    return output


def assess_data(data,limits,authorization,budget,execution_progress,science=None):
    science=load_science() if science is None else science
    comparison=scientific_rows(data,limits,science)
    rows=[]
    for record in comparison['rows']:
        selected=record['Meep'];identities=[]
        if selected:identities.append(selected['identity'])
        prior=record['selected_mesh'].get('prior_identity')
        if prior:identities.append(prior)
        authorizations=[authorization.get('record_authorizations',{}).get(identity,{'status':'UNRESOLVED'}) for identity in identities]
        authorization_state=aggregate({identity:state for identity,state in zip(identities,authorizations)})
        state=aggregate({'native':record['native'],'nominal_optical':record['nominal_optical'],
                         'selected_mesh':record['selected_mesh'],'authorization':authorization_state})
        rows.append({**record,**state,'authorization':authorization_state})
    boundaries=boundary_assessments(data,limits,science)
    original_rows={str(row['H_um']):row for row in rows if row['H_um'] in FAMILY}
    required_boundaries={f"{row['job_type']}_H{row['H_um']:g}":row for row in boundaries if row['required_original_sampled_control']}
    family=aggregate({'original_seven_heights':aggregate(original_rows),
                      'four_original_sampled_controls':aggregate(required_boundaries),
                      'authorization':authorization,'budget':budget,'primary_execution':execution_progress})
    all_heights=aggregate({'eight_nominal_heights':aggregate({str(row['H_um']):row for row in rows}),
                           'four_original_sampled_controls':aggregate(required_boundaries),
                           'authorization':authorization,'budget':budget,'primary_execution':execution_progress})
    inputs=data['input_sha256']
    if not all(science.sha(path)==digest for path,digest in inputs.items()):raise RuntimeError('Immutable assessment inputs changed')
    return science.safe_json({'status':'SAVED_THREE_STAGE_ASSESSMENT','created_utc':datetime.now(timezone.utc).isoformat(),
        'original_seven_height_family_acceptance':family,'all_eight_nominal_heights_acceptance':all_heights,
        'NN_heights':rows,'boundary_duration_checks':boundaries,'all_prior_mesh_pairs':mesh_histories(data,limits,science),
        'scientific_source_sha256':SCIENTIFIC_SHA,'reporting_height_enumeration':list(HEIGHTS),
        'scientific_function_bytecode_changed':False,'authorization':authorization,'budget':budget,
        'primary_execution':execution_progress,'input_sha256':inputs,'ledger_snapshot_sha256':data['ledger_snapshot_sha256'],
        'all_attempts':data['all_attempts'],'missing_or_unverified_evidence':data['missing_evidence'],
        'H75_note':'Validation-observed; separate from original seven-height aggregate, included in eight-height aggregate',
        'NN_superiority_claimed':False,'global_diffraction_optimum_claimed':False,'full_M2_claimed':False,
        'experimental_validation_claimed':False,
        'limitations':['Profiles, physics, gate definitions and Gaussian-pulse Fourier-field extraction remain unchanged.',
            'Nominal/refinement selection excludes PML, padding and duration variants; finer failures cannot fall back to older passes.',
            'Every old failed mesh pair and the original immutable stage records remain available.',
            'Sampled coarse controls do not establish finest-grid boundary/temporal or full-angle convergence.',
            'Padding changes native output/control aperture with domain size; retain that combined-perturbation interpretation.',
            'Q is the retained-band nominal angular ratio; its last-pair changes remain diagnostic, not an added convergence threshold.',
            'Different heights use different physics-seeded NN patterns; proxy/Meep share a pattern within each height.']})


def human_report(result):
    lines=['SAVED THREE-STAGE NN ASSESSMENT',
        'Original seven-height family: '+result['original_seven_height_family_acceptance']['status'],
        'All eight nominal heights: '+result['all_eight_nominal_heights_acceptance']['status'],
        'Protocol: '+result['authorization']['status']+'; budget: '+result['budget']['status'],
        '', 'H(um) finest_mesh native nominal mesh row_status eta(%) C Q selected_pair']
    for row in result['NN_heights']:
        m=row['Meep'];pair=row['selected_mesh'].get('resolutions',[])
        if not m:lines.append(f"{row['H_um']:g} MISSING");continue
        def f(key,n):return f"{m[key]:.{n}f}" if isinstance(m.get(key),(int,float)) else 'NA'
        lines.append(f"{row['H_um']:g} {m['resolution']:g} {row['native']['status']} {row['nominal_optical']['status']} "
            f"{row['selected_mesh']['status']} {row['status']} {100*m['eta']:.6f} {f('C',9)} {f('Q',9)} {pair}")
        for key in ('native','nominal_optical','selected_mesh'):
            for reason in row[key].get('missing_evidence',[]):lines.append('  '+key+' missing: '+reason)
            for reason in row[key].get('failed_checks',[]):lines.append('  '+key+' failed: '+reason)
        for reason in row['selected_mesh'].get('comparison',{}).get('failed_checks',[]):lines.append('  Mesh pair failed: '+reason)
    lines+=['','BOUNDARY / DURATION']
    for row in result['boundary_duration_checks']:
        lines.append(f"{row['job_type']} H{row['H_um']:g}: {row['status']}; mesh {row.get('resolution','missing')}")
        if row.get('scope'):lines.append('  '+row['scope'])
        if row.get('Q_gate_crossing'):lines.append('  Nominal angular gate crosses between the saved duration/boundary cases; inspect separately from numerical stability.')
    return '\n'.join(lines+['',*result['limitations']])+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--partial-reason',help='Explicit root-requested partial assessment; otherwise primary must be complete')
    args=parser.parse_args()
    from collect_height_family_stage3_v1 import collect_stage3
    science=load_science();folder=science.local_path(args.folder).resolve();output=science.local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh stage3 assessment version')
    if not output.is_relative_to(folder/'assessment') or output==folder/'assessment':raise ValueError('Output must be below stage3/assessment/version')
    collected=collect_stage3(folder)
    if collected['primary_execution']['status']!='PASS' and not (args.partial_reason or '').strip():
        raise RuntimeError('Primary is incomplete; an explicit root-requested --partial-reason is required')
    result=assess_data(collected['data'],collected['numerical_limits'],collected['authorization'],
                       collected['budget'],collected['primary_execution'],science)
    result['assessment_utility_sha256']=science.sha(__file__)
    result['partial_assessment_reason']=args.partial_reason
    output.mkdir(parents=True,exist_ok=False)
    with (output/'assessment.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(human_report(result))
    print(json.dumps({'output':str(output),'original_family':result['original_seven_height_family_acceptance']['status'],
                      'all_eight_heights':result['all_eight_nominal_heights_acceptance']['status'],'optical_solves_launched':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
