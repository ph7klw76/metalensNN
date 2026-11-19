"""Saved-field H80 Q/C sampling diagnosis; no optical solve or new gates.

The frozen 512x512/window128 result remains authoritative. Alternative
Cartesian sampling is diagnostic only and does not establish Maxwell mesh,
duration, PML, padding, full-angle M2 or experimental convergence.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compare_height_family_nn_direct_v2 import local_path,read,sha,fingerprint,safe_json
from fiber_metalens_nn.collimation_dft_power import centered_lp_to_cartesian,paraxial_power_equivalent_fields
from fiber_metalens_nn.collimation_optics import forward_spectrum,kernels,vector_spectrum,flux
import height_family_maxwell_analysis_v2 as frozen_analysis

SAMPLINGS=((512,128.),(1024,128.),(1024,256.))


from diagnose_height_family_h100_sampling_v2 import angular_statistics,independent_current,cartesian
from diagnose_h100_angular_contributions_v2 import angular_bins


def diagnose(folder):
    folder=local_path(folder).resolve();protected={};snapshots={}
    def protect(path):
        path=local_path(path).resolve();protected[str(path)]=sha(path);return path
    def document(path):return read(protect(path))
    reused_sources={
        'scripts/diagnose_height_family_h100_sampling_v2.py':'df4216dea6d6d1acf841ea0e4b2c432600224ea013826bf808bbe90d22bf4af1',
        'scripts/diagnose_h100_angular_contributions_v2.py':'d822b1ee791c7959dc1310208e05c958905707c1d7e23c0f3fb1ee9a0267bc25'}
    for name,expected in reused_sources.items():
        if sha(protect(ROOT/name))!=expected:raise ValueError('Sealed diagnostic helper changed: '+name)
    config=document(folder/'config.json');prior_meta=document(folder/'prior_stage.json')
    prior=local_path(prior_meta['folder']).resolve()
    if sha(protect(prior/'config.json'))!=prior_meta['config_sha256'] or sha(prior/'config.json')!=sha(folder/'config.json'):
        raise ValueError('Frozen prior/stage2 configurations differ')
    prep=document(folder/'preparation_manifest.json')
    for name in ('config.json','prior_stage.json','prior_cache_catalog.json','source_manifest.json','frozen_cases.json'):
        if sha(protect(folder/name))!=prep.get(name):raise ValueError('Preparation hash changed: '+name)
    manifests=[document(stage/'source_manifest.json') for stage in (prior,folder)]
    source_checks=[]
    # These are the physical routines actually used; neither current files nor
    # their archived copies may differ from the frozen source manifests.
    for name in ('scripts/height_family_maxwell_analysis_v2.py','scripts/height_family_maxwell_worker_v2.py',
                 'src/fiber_metalens_nn/collimation_optics.py','src/fiber_metalens_nn/collimation_metrics.py',
                 'src/fiber_metalens_nn/collimation_dft_power.py','src/fiber_metalens_nn/metrics.py'):
        actual=sha(protect(ROOT/name));checks=[]
        for stage,manifest in zip((prior,folder),manifests):
            archived=sha(protect(stage/'source_snapshots'/name))
            checks.append(actual==archived==manifest.get(name))
        if not all(checks):raise ValueError('Frozen physical source mismatch: '+name)
        source_checks.append({'file':name,'sha256':actual,'both_current_and_stage_snapshots_match':True})
    protect(Path(__file__))
    catalog=document(folder/'prior_cache_catalog.json');entries={};expected_payload={}
    for stage in (prior,folder):
        data=(stage/'ledger.json').read_bytes();snapshots[str(stage/'ledger.json')]=hashlib.sha256(data).hexdigest()
        ledger=json.loads(data.decode('utf-8-sig'))
        if stage==prior and snapshots[str(stage/'ledger.json')]!=prior_meta['ledger_sha256']:raise ValueError('Prior ledger hash changed')
        for entry in ledger['entries']:
            if entry['id'] in (33,34,52,53):entries[entry['id']]=(stage,entry)
    for item in catalog['records']:
        if item['entry']['id'] in (33,34):expected_payload.update(item['required_file_sha256'])
    for identity in (52,53):expected_payload.update(entries[identity][1].get('required_file_sha256',{}))
    payload_checks=[]
    sealed_payload_paths={str(local_path(name).resolve()) for name in expected_payload}
    for name,expected in expected_payload.items():
        actual=sha(protect(name));payload_checks.append({'file':str(local_path(name)),'sha256':actual,'matches_recorded_payload_hash':actual==expected})
        if actual!=expected:raise ValueError('Sealed native field/flux payload changed: '+name)
    frozen_cases=document(folder/'frozen_cases.json')['cases']
    selected_cases=[case for case in frozen_cases if case.get('method')=='NN' and case.get('H_um')==80.]
    if len(selected_cases)!=1:raise ValueError('Exactly one frozen NN80 geometry is required')
    frozen_design=protect(folder/selected_cases[0]['design_file'])
    if sha(frozen_design)!=selected_cases[0]['design_sha256']:raise ValueError('Frozen NN80 design file changed')
    records={};analyses={};record_checks=[];geometry=[]
    for identity in (33,34,52,53):
        stage,entry=entries[identity]
        if entry['status']!='completed':raise ValueError('Only completed immutable input records may be diagnosed')
        rp=protect(stage/entry['record_file']);raw=document(rp);records[identity]=raw
        checks={'raw_hash':sha(rp)==entry['record_sha256'],'config_fingerprint':raw['config_fingerprint']==fingerprint(config),
            'actual_height80':raw['settings']['H_um']==80.,'actual_duration274':raw['settings']['duration']==274.,
            'all_native_payloads_sealed':all(str(local_path(raw[key]).resolve()) in sealed_payload_paths
                for key in ('fields_file','power_audit_fields_file','flux_data_file') if raw.get(key))}
        if identity in (34,53):
            ap=protect(stage/entry['analysis_file']);analyses[identity]=document(ap)
            checks['analysis_hash']=sha(ap)==entry['analysis_sha256']
            design=protect(raw['design_file']);geometry.append(sha(design))
            checks['design_hash']=sha(design)==entry['design_sha256']
            checks['same_prepared_frozen_NN80']=sha(design)==selected_cases[0]['design_sha256']
            refid=33 if identity==34 else 52
            checks['actual_reference_path']=local_path(raw['reference_file']).resolve()==(entries[refid][0]/entries[refid][1]['record_file']).resolve()
            checks['reference_settings']=raw['settings']==records[refid]['settings']
        if not all(checks.values()):raise ValueError(f'Record provenance mismatch: {identity}: {checks}')
        record_checks.append({'id':identity,'checks':checks})
    if len(set(geometry))!=1:raise ValueError('The two devices do not share the exact frozen geometry')
    D=analyses[34]['declared_target_diameter_um']
    if D!=analyses[53]['declared_target_diameter_um']:raise ValueError('Frozen Gaussian target changed')
    settings=[dict(records[i]['settings']) for i in (34,53)]
    for value in settings:value.pop('resolution')
    if settings[0]!=settings[1]:raise ValueError('Controlled settings differ beyond resolution')
    cases=[]
    for identity in (34,53):
        raw=records[identity];saved=analyses[identity];measurements=[]
        for count,window in SAMPLINGS:
            began=time.monotonic();result=frozen_analysis.analyze(entries[identity][0]/entries[identity][1]['record_file'],config,D,count,window)
            dx,field=cartesian(raw,count,window,config)
            spectrum=forward_spectrum(field,dx,config['wavelength_um'],n=config['materials']['ambient']['n'])
            independent=angular_statistics(spectrum,dx,config['wavelength_um'],D,config['materials']['ambient']['n'])
            C=independent_current(spectrum,dx,config['wavelength_um'],D,config['materials']['ambient']['n'])
            measurements.append({'count':count,'window_um':window,'dx_um':dx,'delta_k_per_um':2*np.pi/window,
                'is_frozen_authoritative_sampling':count==512 and window==128.,'classification':'DIAGNOSIS_ONLY',
                'Q':result['angular_second_moment_ratio'],'C':result['normalized_curvature'],
                'eta':result['conservative_useful_efficiency'],'conservative_T':result['conservative_T'],
                'declared_overlap':result['declared_overlap'],'output_diameters_um':result['metrics']['diameters_output_um'],
                'core_half_angles_rad':[a.get('half_angle_rad') for a in result['metrics']['fitted_core_divergence']['axes']],
                'independent_C':C,'independent_angular_statistics':independent,
                'C_difference_from_saved':result['normalized_curvature']-saved['normalized_curvature'],
                'Q_difference_from_saved':result['angular_second_moment_ratio']-saved['angular_second_moment_ratio'],
                'diagnostic_native_checks':result['numerical_checks'],'reconstruction':result['reconstruction'],
                'postprocessing_wall_s':time.monotonic()-began,
                'authoritative_angular_bins':angular_bins(spectrum,dx,config['wavelength_um'],D,
                    config['materials']['ambient']['n']) if count==512 and window==128. else None})
            del field,spectrum
        # At fixed dx, the larger window must preserve the original measured
        # aperture samples; this is an angular-grid test, not new field data.
        _,small=cartesian(raw,512,128,config);_,large=cartesian(raw,1024,256,config)
        crop=large[:,256:768,256:768]
        crop_error=float(np.max(abs(crop-small))/np.max(abs(small)))
        added_max=max(float(np.max(abs(large[:,:256,:]))),float(np.max(abs(large[:,768:,:]))),
                      float(np.max(abs(large[:,:,:256]))),float(np.max(abs(large[:,:,768:]))))
        cases.append({'id':identity,'resolution':raw['settings']['resolution'],'actual_duration':raw['settings']['duration'],
            'saved_authoritative':{key:saved[key] for key in ('normalized_curvature','angular_second_moment_ratio',
                'conservative_useful_efficiency','optical_gates','numerical_checks')},'samplings':measurements,
            'fixed_dx_double_window_central_crop_relative_error':crop_error,
            'fixed_dx_added_window_samples_max_relative_amplitude':added_max/float(np.max(abs(small))),
            'max_abs_Q_sampling_shift':max(abs(m['Q_difference_from_saved']) for m in measurements),
            'max_abs_C_sampling_shift':max(abs(m['C_difference_from_saved']) for m in measurements)})
        del small,large,crop
    pairs=[]
    for i,(count,window) in enumerate(SAMPLINGS):
        values=[case['samplings'][i] for case in cases]
        pairs.append({'count':count,'window_um':window,'C16':values[0]['C'],'C24':values[1]['C'],
            'absolute_C_mesh_change':abs(values[1]['C']-values[0]['C']),'Q16':values[0]['Q'],'Q24':values[1]['Q']})
    unchanged=all(sha(path)==value for path,value in protected.items())
    if not unchanged:raise RuntimeError('Immutable raw fields/source changed during diagnosis')
    return safe_json({'status':'H80_SAVED_FIELD_SAMPLING_DIAGNOSIS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_checks':source_checks,'record_checks':record_checks,'sealed_payload_checks':payload_checks,
        'config_fingerprint':fingerprint(config),'same_design_sha256':geometry[0],'declared_target_diameter_um':D,
        'frozen_gate_values':config['natural_training']['investigation_gates'],
        'saved_fine_Q_excess_above_frozen_gate':analyses[53]['angular_second_moment_ratio']-config['natural_training']['investigation_gates']['max_angular_second_moment_ratio'],
        'cases':cases,'same_sampling_mesh_comparisons':pairs,'input_sha256':protected,'ledger_snapshot_sha256':snapshots,
        'all_immutable_inputs_unchanged':unchanged,'new_Maxwell_solves':0,'JAX_or_training_invoked':False,
        'authoritative_result_reclassified':False,'new_thresholds_added':False,
        'limitations':['The original 512/window128 gate evaluation is preserved. Alternative sampling is not acceptance evidence.',
            'Larger Cartesian windows zero-pad the same saved radius~35um aperture; they add no physical aperture or Maxwell PML/padding test.',
            'Q is the uncentred retained-band transverse-wavevector moment ratio, not all-angle M2 or a global diffraction optimum.',
            'Authoritative angular bins locate contributions to Q; they do not identify a physical cause. Power fractions normalize retained forward power, not incident power.',
            'Sampling sensitivity cannot by itself separate Maxwell spatial, finite-duration, pedestal interference or proxy-model errors.',
            'Both simulations used actual duration274, output84/cross89. The requested resolved base160 is not the effective duration.',
            'No temporal, full-family, fabrication or experimental convergence is claimed.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve existing diagnosis; choose a new version')
    if not output.is_relative_to(folder/'diagnostics') or output==folder/'diagnostics':raise ValueError('Use a fresh stage2/diagnostics/version output')
    result=diagnose(folder);output.mkdir(parents=True,exist_ok=False)
    with (output/'diagnostic.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['H80 SAVED-FIELD Q/C SAMPLING DIAGNOSIS ONLY','Original acceptance and thresholds remain unchanged.','',
        'mesh count window dx       Q             C            eta']
    for case in result['cases']:
        for row in case['samplings']:
            lines.append(f"{case['resolution']:4g} {row['count']:5} {row['window_um']:6g} {row['dx_um']:.4f} {row['Q']:.10f} {row['C']:+.10f} {row['eta']:.10f}")
    lines.extend(['','SAME-SAMPLING MESH DIFFERENCES'])
    for row in result['same_sampling_mesh_comparisons']:lines.append(f"N{row['count']} L{row['window_um']:g}: |C24-C16|={row['absolute_C_mesh_change']:.10f}")
    lines.extend(['','AUTHORITATIVE512/L128 ANGULAR CONTRIBUTIONS',
        'mesh angle(deg) forward_power(%) additive_Q target_additive_Q excess_Q'])
    for case in result['cases']:
        bins=case['samplings'][0]['authoritative_angular_bins']
        for row in bins['bins']:
            lines.append(f"{case['resolution']:4g} {row['theta_min_deg']:2g}-{row['theta_max_deg']:<2g} "
                f"{100*row['field_forward_power_fraction']:14.8f} {row['field_additive_Q']:.10f} "
                f"{row['target_additive_Q']:.10f} {row['additive_Q_excess_over_target']:+.10f}")
        lines.append(f"SUM mesh{case['resolution']:g}: Q={bins['sum_field_additive_Q']:.12f}; "
            f"saved_Q={case['saved_authoritative']['angular_second_moment_ratio']:.12f}; "
            f"difference={bins['sum_field_additive_Q']-case['saved_authoritative']['angular_second_moment_ratio']:.3g}")
    lines.extend(['',*result['limitations']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'status':result['status'],'immutable_inputs_unchanged':result['all_immutable_inputs_unchanged'],
        'same_sampling_mesh_comparisons':result['same_sampling_mesh_comparisons']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
