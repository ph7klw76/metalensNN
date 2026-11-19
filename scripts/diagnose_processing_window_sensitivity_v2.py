"""Append H40r32 only to the sealed saved-field window diagnostic.

Reuse all v1 cases by SHA; evaluate only record43 at512/L128 and1024/L256.
Verify current H40/75/100 primary graph arrays without rerunning75/80/100.
No Meep, JAX, new gates, halo trimming or primary reclassification.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from diagnose_processing_window_sensitivity_v1 import compare_processing_planes,SETTINGS
from compare_height_family_nn_direct_v2 import collect,local_path,read,sha,safe_json
import height_family_maxwell_analysis_v2 as analysis

V1_SHA='cb5cd1856836517455e18e75b6a2d05be03ac21abfcb2f509dbbd378adf72525'
V1_SOURCE_SHA='2d43cf51c3fe6b0358b98982d8cdfc60ec0387ed3a671e39456f70f868359e53'


def verify_primary_array(case,saved,planes):
    """Cross-check a reused diagnostic array against its sealed raw analysis."""
    try:primary=case['samplings'][0]['processing_planes']
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError('Reused primary processing array is missing') from error
    result=compare_processing_planes({'metrics':{'planes':primary}},saved,planes)
    if result['status']!='COMPLETE' or result['maximum_relative_change']>1e-12:
        raise ValueError('Reused primary processing array differs from the saved graph analysis')
    return result


def diagnose(folder):
    folder=local_path(folder).resolve();protected={}
    def protect(path,expected=None):
        path=local_path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Sealed input changed: '+str(path))
        protected[str(path)]=actual;return path
    protect(ROOT/'scripts/diagnose_processing_window_sensitivity_v1.py',V1_SOURCE_SHA)
    source=protect(folder/'diagnostics/processing_window_sensitivity_v1/diagnostic.json',V1_SHA);base=read(source)
    for path,expected in base['input_sha256'].items():protect(path,expected)
    data=collect(folder,heights=(40.,75.,100.),methods=('NN',),include_variants=False)
    for path,expected in data['input_sha256'].items():protect(path,expected)
    protect(Path(__file__));config=data['config'];planes=base['processing_planes_um']
    if planes!=[float(z) for z in config['evaluation']['processing_planes_um'] if 0<=z<=100]:
        raise ValueError('Frozen processing planes changed')
    h40=[row for row in data['Meep_records'] if row['method']=='NN' and row['H_um']==40. and row['resolution']==32.]
    if len(h40)!=1 or not h40[0]['identity'].endswith(':43'):raise ValueError('Exactly unchanged record43 H40r32 is required')
    row=h40[0];saved=read(protect(row['analysis_file'],row['analysis_sha256']))
    raw=read(protect(row['record_file'],row['record_sha256']))
    if not Path(row['reference_record_file']).name.startswith('42_reference_'):
        raise ValueError('Record43 must retain its homogeneous reference42')
    D=saved['declared_target_diameter_um'];evaluations=[]
    for count,window in SETTINGS:
        started=time.monotonic();result=analysis.analyze(row['record_file'],config,D,count,window)
        evaluations.append({'sampling':{'count':count,'window_um':window,'dx_um':window/count,
                'postprocessing_wall_s':time.monotonic()-started},
            'processing_planes':[r for r in result['metrics']['planes'] if r['z_um'] in planes],
            'C':result['normalized_curvature'],'Q':result['angular_second_moment_ratio'],
            'diagnostic_native_checks':result['numerical_checks']})
    fresh={'metrics':{'planes':evaluations[0]['processing_planes']}}
    reproduction=compare_processing_planes(saved,fresh,planes)
    if reproduction['status']!='COMPLETE' or reproduction['maximum_relative_change']>1e-12:
        raise ValueError('H40 primary widths do not reproduce the saved graph analysis')
    comparison=compare_processing_planes(fresh,{'metrics':{'planes':evaluations[1]['processing_planes']}},planes)
    new_case={'H_um':40.,'resolution':32.,'identity':row['identity'],'record_file':row['record_file'],
        'record_sha256':row['record_sha256'],'analysis_file':row['analysis_file'],'analysis_sha256':row['analysis_sha256'],
        'reference_record_file':row['reference_record_file'],'reference_record_sha256':row['reference_record_sha256'],
        'design_sha256':row['design_sha256'],'native_reference_pass':row['native_all_pass'],
        'authoritative_saved_optical_gates':saved['optical_gates'],'saved_primary_diameter_reproduction':reproduction,
        'samplings':evaluations,'window_sensitivity':comparison,
        'maximum_relative_change_over_mesh_tolerance':comparison['maximum_relative_change']/base['diameter_mesh_tolerance_for_scale_only'],
        'classification':'DIAGNOSIS_ONLY_NO_ACCEPTANCE_RECLASSIFICATION'}
    cases=[new_case,*base['cases']]
    requested=[]
    for H,res in ((40.,32.),(75.,24.),(100.,24.)):
        case=[r for r in cases if r['H_um']==H and r['resolution']==res]
        current=[r for r in data['Meep_records'] if r['method']=='NN' and r['H_um']==H and r['resolution']==res]
        if len(case)!=1 or len(current)!=1:raise ValueError('A requested graph case is missing')
        case,current=case[0],current[0]
        if (case['identity']!=current['identity'] or case['analysis_sha256']!=current['analysis_sha256']
                or case['design_sha256']!=current['design_sha256']):raise ValueError('Requested graph geometry/analysis changed')
        current_saved=read(protect(current['analysis_file'],current['analysis_sha256']))
        checked=verify_primary_array(case,current_saved,planes)
        requested.append({'H_um':H,'resolution':res,'identity':case['identity'],
            'primary_array_maximum_relative_difference':checked['maximum_relative_change'],
            'analysis_sha256':case['analysis_sha256'],'design_sha256':case['design_sha256']})
    unchanged=all(sha(path)==expected for path,expected in protected.items())
    if not unchanged:raise RuntimeError('Immutable source/raw fields changed')
    if cases[1:]!=base['cases']:raise RuntimeError('Reused v1 cases changed')
    return safe_json({'status':'PROCESSING_WINDOW_SENSITIVITY_V2_H40_APPEND_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'reused_diagnostic':str(source),'reused_diagnostic_sha256':V1_SHA,'reused_case_count':len(base['cases']),
        'reused_cases_unchanged':True,'new_saved_field_analysis_evaluations':2,'recomputed_cases':[{'H_um':40.,'resolution':32.}],
        'settings':base['settings'],'processing_planes_um':planes,
        'diameter_mesh_tolerance_for_scale_only':base['diameter_mesh_tolerance_for_scale_only'],'cases':cases,
        'requested_graph_primary_array_verification':requested,'input_sha256':protected,
        'ledger_snapshot_sha256':data['ledger_snapshot_sha256'],'all_immutable_inputs_unchanged':unchanged,
        'new_Maxwell_solves':0,'JAX_or_training_invoked':False,'primary_gate_classifications_changed':False,
        'limitations':[*base['limitations'],'Only H40r32 was newly postprocessed. All five v1 cases were reused by the exact sealed diagnostic SHA; none were recomputed.',
            'Graph-primary processing arrays at40/75/100 are verified against their unchanged saved analyses. This does not upgrade any nominal or mesh classification.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve prior diagnostic output')
    if not output.is_relative_to(folder/'diagnostics') or output==folder/'diagnostics':raise ValueError('Fresh diagnostics/version output required')
    result=diagnose(folder);output.mkdir(parents=True,exist_ok=False)
    with (output/'diagnostic.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['PROCESSING-WINDOW V2: H40R32 APPENDED; FIVE V1 CASES REUSED BY SHA',
        'Only two new saved-field analysis calls: H40 record43/reference42 at512/L128 and1024/L256. No new optical solve.','',
        'H(um) mesh max_relative_D_change(%) worst_z(um) axis source']
    for case in result['cases']:
        change=case['window_sensitivity'];where=change['maximum_location']
        origin='new H40 only' if case['H_um']==40 else 'v1 SHA reuse; not recomputed'
        lines.append(f"{case['H_um']:5g} {case['resolution']:4g} {100*change['maximum_relative_change']:.9f} {where['z_um']:6g} {where['axis']} {origin}")
        if case['H_um']==40:
            for r in change['rows']:lines.append(f"  z{r['z_um']:g} {r['axis']}: D128={r['primary_diameter_um']:.9f}um D256={r['wide_window_diameter_um']:.9f}um change={100*r['relative_change']:.9f}%")
    lines.extend(['','REQUESTED GRAPH PRIMARY ARRAY CHECKS'])
    for r in result['requested_graph_primary_array_verification']:
        lines.append(f"H{r['H_um']:g} mesh{r['resolution']:g}: maximum primary-array relative difference={r['primary_array_maximum_relative_difference']:.12g}")
    lines.extend(['',*result['limitations']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'new_saved_field_analysis_evaluations':2,
        'reused_case_count':result['reused_case_count'],'H40_maximum_relative_change':result['cases'][0]['window_sensitivity']['maximum_relative_change'],
        'requested_primary_arrays':result['requested_graph_primary_array_verification'],'immutable_inputs_unchanged':result['all_immutable_inputs_unchanged']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
