"""Saved-field processing-diameter FFT-window diagnosis; no optical solve.

Compare authoritative512/L128 with1024/L256 at the same0.25um Cartesian dx.
Both reconstruct the same saved35um-radius aperture. Alternative windows do
not reclassify the original gates or establish Maxwell/halo convergence.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compare_height_family_nn_direct_v2 import collect,local_path,read,sha,safe_json
import height_family_maxwell_analysis_v2 as analysis

ANCHORS={'H80_sampling_v1':'7e28bff309d429985fb863e8eced29b777ae6cf6a76d00741ca543bd315f63be',
    'H100_sampling_v1':'7ddf3b6c610733f7d69561fe82d247afb1a31ff1f82e87656c4a8bdae415ac72'}
SETTINGS=((512,128.),(1024,256.))
TARGETS=((75.,24.),(80.,16.),(80.,24.),(100.,16.),(100.,24.))


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def compare_processing_planes(primary,wide,planes):
    """Absolute width sensitivity with primary diameter as denominator."""
    def plane_map(result):
        return {float(row['z_um']):row for row in result.get('metrics',{}).get('planes',[])
            if finite(row.get('z_um'))}
    old,new=plane_map(primary),plane_map(wide);rows=[];missing=[]
    for z in planes:
        a,b=old.get(float(z)),new.get(float(z))
        if a is None or b is None:
            missing.append(f'Both processing planes at z={z:g} required');continue
        for i,axis in enumerate(('x','y')):
            key='diameter_'+axis+'_um';d0,d1=a.get(key),b.get(key)
            if (a.get('valid_crossings')!=[True,True] or b.get('valid_crossings')!=[True,True]
                    or not finite(d0) or not finite(d1) or d0<=0 or d1<=0):
                missing.append(f'Valid positive finite {axis} crossing at z={z:g} required');continue
            rows.append({'z_um':float(z),'axis':axis,'primary_diameter_um':d0,'wide_window_diameter_um':d1,
                'absolute_change_um':abs(d1-d0),'relative_change':abs(d1/d0-1)})
    worst=max(rows,key=lambda row:row['relative_change']) if rows else None
    return {'status':'UNRESOLVED' if missing else 'COMPLETE','rows':rows,'missing_evidence':missing,
        'maximum_relative_change':worst['relative_change'] if worst else None,
        'maximum_location':{'z_um':worst['z_um'],'axis':worst['axis']} if worst else None}


def diagnose(folder):
    folder=local_path(folder).resolve();protected={}
    def protect(path,expected=None):
        path=local_path(path).resolve();actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Sealed input changed: '+str(path))
        protected[str(path)]=actual;return path
    protect(ROOT/'scripts/assess_height_family_stage2_v3.py','09c1cee5b2c87eeda86fd7fcc0b010dbd8a9971f7d6265aabd20a9c9ae1ff244')
    for name,expected in ANCHORS.items():
        anchor=read(protect(folder/'diagnostics'/name/'diagnostic.json',expected))
        for path,value in anchor['input_sha256'].items():protect(path,value)
    data=collect(folder,heights=(75.,80.,100.),methods=('NN',),include_variants=False)
    for path,expected in data['input_sha256'].items():protect(path,expected)
    manifest=read(protect(folder/'source_manifest.json'))
    physical=('scripts/height_family_maxwell_worker_v2.py','scripts/height_family_maxwell_analysis_v2.py',
        'src/fiber_metalens_nn/collimation_optics.py','src/fiber_metalens_nn/collimation_metrics.py',
        'src/fiber_metalens_nn/collimation_dft_power.py','src/fiber_metalens_nn/metrics.py')
    for name in physical:
        protect(ROOT/name,manifest[name]);protect(folder/'source_snapshots'/name,manifest[name])
    protect(Path(__file__));config=data['config'];planes=[float(z) for z in config['evaluation']['processing_planes_um'] if 0<=z<=100]
    if not planes or planes[0]!=0:raise ValueError('Frozen0..100um processing planes required')
    prior=local_path(data['prior_stage_linkage']['folder']).resolve()
    proposal=read(protect(prior/'maxwell_validation_proposal.json'))
    diameter_scale=proposal['numerical_acceptance'].get('diameter_relative_mesh_change_max',.02)
    cases=[]
    for H,res in TARGETS:
        selected=[row for row in data['Meep_records'] if row['method']=='NN' and row['H_um']==H and row['resolution']==res]
        if len(selected)!=1:raise ValueError(f'Exactly one verified NN H{H:g}/r{res:g} record required')
        row=selected[0];saved=read(protect(row['analysis_file'],row['analysis_sha256']));D=saved['declared_target_diameter_um'];results=[]
        for count,window in SETTINGS:
            began=time.monotonic();result=analysis.analyze(row['record_file'],config,D,count,window)
            results.append(result)
            result['_diagnostic_sampling']={'count':count,'window_um':window,'dx_um':window/count,'postprocessing_wall_s':time.monotonic()-began}
        reproduction=compare_processing_planes(saved,results[0],planes)
        if reproduction['status']!='COMPLETE' or reproduction['maximum_relative_change']>1e-12:
            raise ValueError('Fresh512/L128 did not reproduce the authoritative saved diameters')
        comparison=compare_processing_planes(results[0],results[1],planes)
        cases.append({'H_um':H,'resolution':res,'identity':row['identity'],'record_file':row['record_file'],
            'record_sha256':row['record_sha256'],'analysis_file':row['analysis_file'],'analysis_sha256':row['analysis_sha256'],
            'design_sha256':row['design_sha256'],'native_reference_pass':row['native_all_pass'],
            'authoritative_saved_optical_gates':saved['optical_gates'],'saved_primary_diameter_reproduction':reproduction,
            'samplings':[{'sampling':result['_diagnostic_sampling'],'processing_planes':[
                    r for r in result['metrics']['planes'] if r['z_um'] in planes],
                'C':result['normalized_curvature'],'Q':result['angular_second_moment_ratio'],
                'diagnostic_native_checks':result['numerical_checks']} for result in results],
            'window_sensitivity':comparison,'maximum_relative_change_over_mesh_tolerance':(
                comparison['maximum_relative_change']/diameter_scale if comparison['maximum_relative_change'] is not None else None),
            'classification':'DIAGNOSIS_ONLY_NO_ACCEPTANCE_RECLASSIFICATION'})
    unchanged=all(sha(path)==value for path,value in protected.items())
    if not unchanged:raise RuntimeError('Immutable diagnostic inputs changed')
    return safe_json({'status':'PROCESSING_WINDOW_SENSITIVITY_DIAGNOSIS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'settings':[{'count':count,'window_um':window,'dx_um':window/count} for count,window in SETTINGS],
        'processing_planes_um':planes,'diameter_mesh_tolerance_for_scale_only':diameter_scale,'cases':cases,
        'input_sha256':protected,'ledger_snapshot_sha256':data['ledger_snapshot_sha256'],'all_immutable_inputs_unchanged':unchanged,
        'new_Maxwell_solves':0,'JAX_or_training_invoked':False,'primary_gate_classifications_changed':False,
        'limitations':['The primary512/L128 analysis remains authoritative; alternative sampling is diagnostic only.',
            'Both windows use the same0.25um Cartesian sampling of the same saved~35um-radius aperture; the larger window adds no measured aperture or physical padding/PML test.',
            'Finite periodic FFT propagation can move halo power back into the window. This comparison quantifies core1/e2-diameter sensitivity but does not establish complete halo or all-angle accuracy.',
            'The2% frozen mesh-diameter tolerance is shown only as a numerical scale, not a replacement gate or a new acceptance decision.',
            'H75 was validation-observed and has only one Maxwell mesh. No mesh/temporal/global-diffraction/fullM2/fabrication/experimental acceptance follows.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve prior diagnostic output')
    if not output.is_relative_to(folder/'diagnostics') or output==folder/'diagnostics':raise ValueError('Fresh diagnostics/version output required')
    result=diagnose(folder);output.mkdir(parents=True,exist_ok=False)
    with (output/'diagnostic.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['PROCESSING-DIAMETER FFT-WINDOW SENSITIVITY: DIAGNOSIS ONLY',
        'Authoritative512/L128 versus diagnostic1024/L256, identical0.25um Cartesian dx. Original classifications remain unchanged.','',
        'H(um) mesh max_relative_D_change(%) worst_z(um) axis  fraction_of2%mesh_scale']
    for case in result['cases']:
        change=case['window_sensitivity'];where=change['maximum_location']
        lines.append(f"{case['H_um']:5g} {case['resolution']:4g} {100*change['maximum_relative_change']:.9f} {where['z_um']:6g} {where['axis']} {case['maximum_relative_change_over_mesh_tolerance']:.6f}")
        for row in change['rows']:lines.append(f"  z{row['z_um']:g} {row['axis']}: D128={row['primary_diameter_um']:.9f}um D256={row['wide_window_diameter_um']:.9f}um change={100*row['relative_change']:.9f}%")
    lines.extend(['',*result['limitations']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'all_immutable_inputs_unchanged':result['all_immutable_inputs_unchanged'],
        'maxima':[{'H_um':case['H_um'],'resolution':case['resolution'],**case['window_sensitivity']['maximum_location'],
            'maximum_relative_change':case['window_sensitivity']['maximum_relative_change']} for case in result['cases']]},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
