"""Saved-field H100 Q/C sampling diagnosis; no optical solve or new gates.

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


def angular_statistics(spectrum,dx,wavelength,diameter,index=1.):
    """Independent uncentred retained-band moment and Parseval cross-check.

    The physical band is the frozen kernels() band k_perp<0.98*n*k0.
    Common FFT-amplitude and incident-power factors cancel from Q. Spatial
    derivative Poynting integrals are an independent Parseval evaluation.
    """
    count=spectrum.shape[-1];kx,ky,_,_,valid,k0=kernels(count,dx,wavelength,n=index)
    kt2=kx*kx+ky*ky
    x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x,indexing='xy')
    target=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(diameter/2)**2)),dx,wavelength,n=index)
    def distribution(s):return np.maximum(.5*np.real(s[0]*np.conj(s[4])-s[1]*np.conj(s[3])),0)*valid
    def moment(power):return float(np.sum(power*kt2)/np.sum(power))
    power=distribution(spectrum);target_power=distribution(target)
    if not np.isfinite(power).all() or not np.isfinite(target_power).all() or power.sum()<=0 or target_power.sum()<=0:
        raise ValueError('Finite positive forward angular power required')
    measured=moment(power);declared=moment(target_power)
    def spatial_moment(s):
        ids=(0,1,3,4)
        fields=[np.fft.ifft2(s[i]) for i in ids]
        def integrated(v):return float(np.sum(.5*np.real(v[0]*np.conj(v[3])-v[1]*np.conj(v[2]))))
        denominator=integrated(fields);numerator=0.
        if not np.isfinite(denominator) or denominator<=0:raise ValueError('Finite positive spatial forward power required')
        for transverse in (kx,ky):numerator+=integrated([np.fft.ifft2(1j*transverse*s[i]) for i in ids])
        return numerator/denominator
    spatial=spatial_moment(spectrum);target_spatial=spatial_moment(target)
    tails=[]
    for cut in (.25,.5,.75,.9):
        outside=kt2>(cut*index*k0)**2
        tails.append({'k_perp_over_nk0_min':cut,'power_fraction':float(power[outside].sum()/power.sum()),
            'transverse_moment_fraction':float((power*kt2)[outside].sum()/(power*kt2).sum()) if measured>0 else None})
    return {'Q_independent_spectral':measured/declared,'Q_independent_spatial_Parseval':spatial/target_spatial,
        'field_transverse_second_moment_per_um2':measured,'target_transverse_second_moment_per_um2':declared,
        'spatial_field_moment_per_um2':spatial,'spatial_target_moment_per_um2':target_spatial,
        'centroid_k_per_um':[float((power*kx).sum()/power.sum()),float((power*ky).sum()/power.sum())],
        'retained_band':'k_perp < 0.98*n*k0; unchanged from frozen analysis','tail_diagnostics':tails,
        'definition':'Ratio of uncentred transverse-wavevector second moments; not full-angle M2'}


def independent_current(spectrum,dx,wavelength,diameter,index=1.):
    count=spectrum.shape[-1];x=(np.arange(count)-count/2+.5)*dx
    xx,yy=np.meshgrid(x,x,indexing='xy');r2=xx*xx+yy*yy;core=r2<(.6*diameter)**2
    kx,ky,*_=kernels(count,dx,wavelength,n=index);ex=np.fft.ifft2(spectrum[0])
    gx=np.fft.ifft2(1j*kx*spectrum[0]);gy=np.fft.ifft2(1j*ky*spectrum[0])
    numerator=np.sum((xx*np.imag(ex.conj()*gx)+yy*np.imag(ex.conj()*gy))[core])
    denominator=2*np.pi*index/wavelength*np.sum((abs(ex)**2*r2)[core])
    return float(np.pi*index*(diameter/2)**2/wavelength*numerator/denominator)


def cartesian(raw,count,window,config):
    with np.load(local_path(raw['fields_file']),allow_pickle=False) as saved:
        _,dx,field=centered_lp_to_cartesian(saved['r_um'],saved['fields'][0],1/raw['settings']['resolution'],count,window)
    return dx,paraxial_power_equivalent_fields(field,config['materials']['ambient']['n'],config['wavelength_um'],
        raw['settings']['resolution'],raw['settings']['Courant'],symmetric_axial_collapse=True)


def diagnose(folder):
    folder=local_path(folder).resolve();protected={};snapshots={}
    def protect(path):
        path=local_path(path).resolve();protected[str(path)]=sha(path);return path
    def document(path):return read(protect(path))
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
            if entry['id'] in (26,27,46,47):entries[entry['id']]=(stage,entry)
    for item in catalog['records']:
        if item['entry']['id'] in (26,27):expected_payload.update(item['required_file_sha256'])
    for identity in (46,47):expected_payload.update(entries[identity][1].get('required_file_sha256',{}))
    payload_checks=[]
    for name,expected in expected_payload.items():
        actual=sha(protect(name));payload_checks.append({'file':str(local_path(name)),'sha256':actual,'matches_recorded_payload_hash':actual==expected})
        if actual!=expected:raise ValueError('Sealed native field/flux payload changed: '+name)
    records={};analyses={};record_checks=[];geometry=[]
    for identity in (26,27,46,47):
        stage,entry=entries[identity]
        if entry['status']!='completed':raise ValueError('Only completed immutable input records may be diagnosed')
        rp=protect(stage/entry['record_file']);raw=document(rp);records[identity]=raw
        checks={'raw_hash':sha(rp)==entry['record_sha256'],'config_fingerprint':raw['config_fingerprint']==fingerprint(config),
            'actual_height100':raw['settings']['H_um']==100.,'actual_duration335':raw['settings']['duration']==335.}
        if identity in (27,47):
            ap=protect(stage/entry['analysis_file']);analyses[identity]=document(ap)
            checks['analysis_hash']=sha(ap)==entry['analysis_sha256']
            design=protect(raw['design_file']);geometry.append(sha(design))
            checks['design_hash']=sha(design)==entry['design_sha256']
            refid=26 if identity==27 else 46
            checks['actual_reference_path']=local_path(raw['reference_file']).resolve()==(entries[refid][0]/entries[refid][1]['record_file']).resolve()
            checks['reference_settings']=raw['settings']==records[refid]['settings']
        if not all(checks.values()):raise ValueError(f'Record provenance mismatch: {identity}: {checks}')
        record_checks.append({'id':identity,'checks':checks})
    if len(set(geometry))!=1:raise ValueError('The two devices do not share the exact frozen geometry')
    D=analyses[27]['declared_target_diameter_um']
    if D!=analyses[47]['declared_target_diameter_um']:raise ValueError('Frozen Gaussian target changed')
    settings=[dict(records[i]['settings']) for i in (27,47)]
    for value in settings:value.pop('resolution')
    if settings[0]!=settings[1]:raise ValueError('Controlled settings differ beyond resolution')
    cases=[]
    for identity in (27,47):
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
                'postprocessing_wall_s':time.monotonic()-began})
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
    return safe_json({'status':'H100_SAVED_FIELD_SAMPLING_DIAGNOSIS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_checks':source_checks,'record_checks':record_checks,'sealed_payload_checks':payload_checks,
        'config_fingerprint':fingerprint(config),'same_design_sha256':geometry[0],'declared_target_diameter_um':D,
        'frozen_gate_values':config['natural_training']['investigation_gates'],
        'saved_fine_Q_excess_above_frozen_gate':analyses[47]['angular_second_moment_ratio']-config['natural_training']['investigation_gates']['max_angular_second_moment_ratio'],
        'cases':cases,'same_sampling_mesh_comparisons':pairs,'input_sha256':protected,'ledger_snapshot_sha256':snapshots,
        'all_immutable_inputs_unchanged':unchanged,'new_Maxwell_solves':0,'JAX_or_training_invoked':False,
        'authoritative_result_reclassified':False,'new_thresholds_added':False,
        'limitations':['The original 512/window128 gate evaluation is preserved. Alternative sampling is not acceptance evidence.',
            'Larger Cartesian windows zero-pad the same saved radius~35um aperture; they add no physical aperture or Maxwell PML/padding test.',
            'Q is the uncentred retained-band transverse-wavevector moment ratio, not all-angle M2 or a global diffraction optimum.',
            'Sampling sensitivity cannot by itself separate Maxwell spatial, finite-duration, pedestal interference or proxy-model errors.',
            'Both simulations used actual duration335, output104/cross109. The requested resolved base160 is not the effective duration.',
            'No temporal, full-family, fabrication or experimental convergence is claimed.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve existing diagnosis; choose a new version')
    if not output.is_relative_to(folder/'diagnostics') or output==folder/'diagnostics':raise ValueError('Use a fresh stage2/diagnostics/version output')
    result=diagnose(folder);output.mkdir(parents=True,exist_ok=False)
    with (output/'diagnostic.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['H100 SAVED-FIELD Q/C SAMPLING DIAGNOSIS ONLY','Original acceptance and thresholds remain unchanged.','',
        'mesh count window dx       Q             C            eta']
    for case in result['cases']:
        for row in case['samplings']:
            lines.append(f"{case['resolution']:4g} {row['count']:5} {row['window_um']:6g} {row['dx_um']:.4f} {row['Q']:.10f} {row['C']:+.10f} {row['eta']:.10f}")
    lines.extend(['','SAME-SAMPLING MESH DIFFERENCES'])
    for row in result['same_sampling_mesh_comparisons']:lines.append(f"N{row['count']} L{row['window_um']:g}: |C24-C16|={row['absolute_C_mesh_change']:.10f}")
    lines.extend(['',*result['limitations']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'status':result['status'],'immutable_inputs_unchanged':result['all_immutable_inputs_unchanged'],
        'same_sampling_mesh_comparisons':result['same_sampling_mesh_comparisons']},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
