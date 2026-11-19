"""Reproduce H40 curvature diagnostics from immutable saved Maxwell fields.

No Meep, JAX, optical solve or source/geometry modification is performed.
The frozen phase-current C remains authoritative. Independent quadratic and
native-radial estimates are diagnostics, not replacement acceptance metrics.
Use a fresh --output JSON filename; existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fiber_metalens_nn.collimation_dft_power import (
    centered_lp_to_cartesian, paraxial_power_equivalent_fields,
)
from fiber_metalens_nn.collimation_optics import forward_spectrum, propagate, kernels, fingerprint
from audit_height_family_reference_mode_v2 import diagnose_reference, local_path


def read(path):
    return json.loads(local_path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(local_path(path).read_bytes()).hexdigest()


def coordinates(count,dx):
    x=(np.arange(count)-count/2+.5)*dx
    xx,yy=np.meshgrid(x,x,indexing='xy')
    return xx,yy,xx*xx+yy*yy


def phase_current(spectrum,dx,wavelength,diameter,index=1.):
    """Independent NumPy evaluation of the frozen radial Ex-current formula."""
    xx,yy,r2=coordinates(spectrum.shape[-1],dx)
    ex=np.fft.ifft2(spectrum[0])
    kx,ky,*_=kernels(spectrum.shape[-1],dx,wavelength,n=index)
    gx=np.fft.ifft2(1j*kx*spectrum[0]);gy=np.fft.ifft2(1j*ky*spectrum[0])
    core=r2<(.6*diameter)**2;k=2*np.pi*index/wavelength
    denominator=k*np.sum((abs(ex)**2*r2)[core])
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('Invalid phase-current denominator')
    zr=np.pi*index*(diameter/2)**2/wavelength
    numerator=np.sum((xx*np.imag(ex.conj()*gx)+yy*np.imag(ex.conj()*gy))[core])
    gdy,gdx=np.gradient(ex,dx,edge_order=2)
    fd=np.sum((xx*np.imag(ex.conj()*gdx)+yy*np.imag(ex.conj()*gdy))[core])
    return {'C':float(zr*numerator/denominator),'spatial_finite_difference_C':float(zr*fd/denominator)}


def quadratic_phase_fit(spectrum,dx,wavelength,diameter,index=1.):
    """Intensity-weighted bright-core Ex phase fit; diagnostic only.

    Fit phi=a+bx*x+by*y+qxx*x^2+qyy*y^2+qxy*x*y inside the SAME bright
    core as the frozen estimator. A center-relative phase range below pi/2
    avoids an unwrapping ambiguity; otherwise this diagnostic is unresolved.
    This regression differs from radial phase-current weighting for
    nonquadratic fields and must not replace the predeclared definition.
    """
    ex=np.fft.ifft2(spectrum[0]);xx,yy,r2=coordinates(ex.shape[-1],dx)
    core=r2<(.6*diameter)**2
    phase=np.angle(ex*np.conj(ex[ex.shape[-1]//2,ex.shape[-1]//2]))
    span=[float(phase[core].min()),float(phase[core].max())]
    if max(abs(v) for v in span)>=np.pi/2:
        return {'status':'UNRESOLVED_PHASE_BRANCH','core_phase_range_rad':span}
    A=np.stack([np.ones(core.sum()),xx[core],yy[core],xx[core]**2,yy[core]**2,xx[core]*yy[core]],axis=1)
    weight=abs(ex[core]);coeff=np.linalg.lstsq(A*weight[:,None],phase[core]*weight,rcond=None)[0]
    residual=phase[core]-A@coeff;k=2*np.pi*index/wavelength
    zr=np.pi*index*(diameter/2)**2/wavelength
    return {'status':'DIAGNOSTIC_ONLY','C':float(zr*(coeff[3]+coeff[4])/k),
        'C_xy':[float(2*zr*coeff[3]/k),float(2*zr*coeff[4]/k)],
        'weighted_rms_phase_residual_rad':float(np.sqrt(np.sum(weight**2*residual**2)/np.sum(weight**2))),
        'core_phase_range_rad':span,'weighting':'Weighted least squares with intensity weights |Ex|^2'}


def radial_current_diagnostic(raw,wavelength,diameter,index=1.):
    """Direct centered radial-field diagnostic before forward decomposition.

    For the LP reconstruction Ex=a*cos(phi)^2+b*sin(phi)^2, with a=Er and
    b=-i*Ep. Integrate its exact azimuthal products and use native radial
    finite differences. The estimator retains backward/nonpropagating
    content and therefore is deliberately not the authoritative estimator.
    """
    with np.load(local_path(raw['fields_file']),allow_pickle=False) as saved:
        r=saved['r_um'];f=saved['fields'][0];a=f[0];b=-1j*f[1]
    da=np.gradient(a,r,edge_order=2);db=np.gradient(b,r,edge_order=2)
    current=np.imag(3/8*(a.conj()*da+b.conj()*db)+1/8*(a.conj()*db+b.conj()*da))
    intensity=3/8*(abs(a)**2+abs(b)**2)+1/4*np.real(a.conj()*b)
    dr=1/raw['settings']['resolution'];left=np.maximum(0,r-dr/2)
    right=np.maximum(left,np.minimum(.6*diameter,r+dr/2))
    zr=np.pi*index*(diameter/2)**2/wavelength;k=2*np.pi*index/wavelength
    return float(zr*np.sum(current*(right**3-left**3)/3)/(k*np.sum(intensity*(right**4-left**4)/4)))


def analytic_fixture(wavelength,diameter,index=1.):
    """Pure Gaussian fixture verifies metric code, not any optical device."""
    count=256;dx=.25;xx,yy,r2=coordinates(count,dx)
    zr=np.pi*index*(diameter/2)**2/wavelength;k=2*np.pi*index/wavelength
    expected=.04
    ex=np.exp(-r2/(diameter/2)**2+1j*k*expected/zr*r2/2)
    spectrum=np.zeros((6,count,count),complex);spectrum[0]=np.fft.fft2(ex)
    current=phase_current(spectrum,dx,wavelength,diameter,index)['C']
    fit=quadratic_phase_fit(spectrum,dx,wavelength,diameter,index)['C']
    return {'kind':'Pure analytic Gaussian, no Maxwell or training',
        'expected_C':expected,'phase_current_C':current,'quadratic_fit_C':fit,
        'phase_current_absolute_error':abs(current-expected),'quadratic_fit_absolute_error':abs(fit-expected),
        'implementation_fixture_pass':bool(abs(current-expected)<1e-8 and abs(fit-expected)<1e-8),
        'tolerance_scope':'1e-8 verifies the analytic implementation fixture only; no device acceptance gate added'}


def diagnose(folder,counts=(512,1024),window=128.):
    folder=local_path(folder).resolve();protected={}
    def protect(path):
        path=local_path(path).resolve();protected[str(path)]=sha(path);return path
    for path in [Path(__file__),ROOT/'scripts/audit_height_family_reference_mode_v2.py',
                 folder/'config.json',folder/'source_manifest.json',folder/'cumulative_amendment.json']:
        protect(path)
    cfg=read(folder/'config.json');manifest=read(folder/'source_manifest.json')
    amendment=read(folder/'cumulative_amendment.json')
    # The runner may append while this diagnostic runs. Preserve the exact
    # parsed ledger snapshot hash, without treating a later append as mutation
    # of the already completed input records.
    ledger_bytes=(folder/'ledger.json').read_bytes()
    ledger=json.loads(ledger_bytes.decode('utf-8-sig'))
    entries={entry['id']:entry for entry in ledger['entries']}
    source_checks=[]
    for name,expected in manifest.items():
        current=protect(ROOT/name);snapshot=protect(folder/'source_snapshots'/name)
        checked={'source':name,'expected_sha256':expected,'current_sha256':sha(current),
                 'snapshot_sha256':sha(snapshot)}
        checked['matches_frozen_manifest']=checked['current_sha256']==checked['snapshot_sha256']==expected
        if not checked['matches_frozen_manifest']:raise ValueError('Frozen source mismatch: '+name)
        source_checks.append(checked)
    records={};analyses={};provenance=[]
    expected_design=next(case['design_sha256'] for case in amendment['cases']
                         if case['method']=='NN' and case['H_um']==40)
    for identity in (19,20,37,38):
        entry=entries[identity]
        if entry['status']!='completed':raise ValueError(f'Entry {identity} is not completed')
        record_path=protect(folder/entry['record_file']);raw=read(record_path);records[identity]=raw
        checks={'record_matches_ledger':sha(record_path)==entry['record_sha256'],
            'config_fingerprint_matches':raw['config_fingerprint']==fingerprint(cfg),
            'worker_matches_frozen_manifest':raw['worker_sha256']==manifest['scripts/height_family_maxwell_worker_v2.py'],
            'ledger_settings_match_record':all(float(entry[key])==float(raw['settings'][key])
                for key in ('H_um','resolution','pml_um','padding_um','duration'))}
        for field in ('fields_file','power_audit_fields_file'):protect(raw[field])
        if identity in (20,38):
            ap=protect(folder/entry['analysis_file']);analysis=read(ap);analyses[identity]=analysis
            checks['analysis_matches_ledger']=sha(ap)==entry['analysis_sha256']
            design_path=protect(raw['design_file'])
            checks['design_matches_ledger_and_amendment']=sha(design_path)==entry['design_sha256']==expected_design
            checks['analysis_settings_match_record']=analysis['settings']==raw['settings']
            checks['actual_reference_names_match']=local_path(raw['reference_file']).resolve()==(folder/entries[entry['reference_id']]['record_file']).resolve()
            checks['raw_reference_settings_match']=read(raw['reference_file'])['settings']==raw['settings']
            with np.load(local_path(raw['power_audit_fields_file']),allow_pickle=False) as audit:
                checks['actual_device_axial_pair_weights_symmetric']=bool(np.max(np.abs(audit['pair_weights']-.5))<=1e-8)
        if not all(checks.values()):raise ValueError(f'Input provenance mismatch in entry {identity}: {checks}')
        provenance.append({'entry_id':identity,'checks':checks})
    D=float(analyses[20]['declared_target_diameter_um'])
    if D!=float(analyses[38]['declared_target_diameter_um']):raise ValueError('Frozen target diameters differ')
    wavelength=float(cfg['wavelength_um']);index=float(cfg['materials']['ambient']['n'])
    def construct(raw,count,plane=0,correction=True):
        with np.load(local_path(raw['fields_file']),allow_pickle=False) as saved:
            _,dx,field=centered_lp_to_cartesian(saved['r_um'],saved['fields'][plane],1/raw['settings']['resolution'],count,window)
        if correction:
            field=paraxial_power_equivalent_fields(field,index,wavelength,raw['settings']['resolution'],
                raw['settings']['Courant'],symmetric_axial_collapse=True)
        return dx,forward_spectrum(field,dx,wavelength,n=index)
    cases=[]
    for identity in (20,38):
        raw=records[identity];analysis=analyses[identity];sampling=[]
        row={'entry_id':identity,'reference_id':entries[identity]['reference_id'],
             'settings':raw['settings'],'saved_authoritative_C':analysis['normalized_curvature']}
        for count in counts:
            dx,spectrum=construct(raw,count)
            measurement=phase_current(spectrum,dx,wavelength,D,index)
            sampling.append({'count':count,'window_um':window,'dx_um':dx,**measurement,
                             'difference_from_saved_authoritative_C':measurement['C']-analysis['normalized_curvature']})
            if count==512:
                row['quadratic_phase_diagnostic']=quadratic_phase_fit(spectrum,dx,wavelength,D,index)
                row['amplitude_global_phase_scaled_C']=phase_current(spectrum*3.7*np.exp(.41j),dx,wavelength,D,index)['C']
                distance=raw['settings']['crosscheck_z_um']-raw['settings']['output_z_um']
                propagated=np.fft.fft2(propagate(spectrum,dx,wavelength,distance,n=index),axes=(-2,-1))
                _,cross=construct(raw,count,plane=1)
                predicted=phase_current(propagated,dx,wavelength,D,index)['C']
                direct=phase_current(cross,dx,wavelength,D,index)['C']
                row['crosscheck_phase_current']={'distance_um':distance,'predicted_C':predicted,'direct_saved_C':direct,
                                                'direct_minus_predicted':direct-predicted}
                _,uncorrected=construct(raw,count,correction=False)
                row['without_paraxial_Yee_correction_C']=phase_current(uncorrected,dx,wavelength,D,index)['C']
        row['Cartesian_sampling_diagnostic']=sampling
        row['native_radial_diagnostic_C']=radial_current_diagnostic(raw,wavelength,D,index)
        row['source_facet_diagnostic']=diagnose_reference(folder/entries[entries[identity]['reference_id']]['record_file'],cfg)
        row['saved_nominal_optical_pass']=analysis['synthetic_optical_pass']
        row['saved_native_reconstruction_pass']=analysis['reconstruction_pass']
        cases.append(row)
    settings=[dict(records[identity]['settings']) for identity in (20,38)]
    for setting in settings:setting.pop('resolution')
    unchanged=all(sha(path)==value for path,value in protected.items())
    if not unchanged:raise RuntimeError('Completed evidence/source changed during diagnosis')
    delta=abs(cases[1]['saved_authoritative_C']-cases[0]['saved_authoritative_C'])
    return {'status':'SAVED_FIELD_DIAGNOSTICS_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'config_fingerprint':fingerprint(cfg),'declared_target_diameter_um':D,
        'Rayleigh_normalizer_um':float(np.pi*index*(D/2)**2/wavelength),
        'ledger_snapshot_sha256':hashlib.sha256(ledger_bytes).hexdigest(),'ledger_snapshot_status':ledger['status'],
        'source_checks':source_checks,'record_design_checks':provenance,
        'device_settings_identical_except_mesh':settings[0]==settings[1],
        'same_frozen_design_sha256':expected_design,'cases':cases,
        'saved_authoritative_mesh_delta_C':delta,'analytic_implementation_fixture':analytic_fixture(wavelength,D,index),
        'input_sha256':protected,'all_completed_inputs_unchanged':unchanged,
        'new_optical_gates_added':False,'authoritative_C_replaced':False,
        'duration_convergence_claimed':False,'new_Maxwell_solves':0,'JAX_or_training_invoked':False,
        'interpretation':['Independent reconstruction reproduces saved authoritative C; no physical/code repair is justified by this diagnostic.',
            'Cartesian sampling sensitivity, independent bright-core quadratic fit and native radial estimator are diagnostic only.',
            'A changing saved full-wave wavefront is consistent with spatial or temporal numerical sensitivity; these records do not separate the two.',
            'No threshold is relaxed, and nominal optical/reconstruction PASS does not replace mesh convergence.',
            'Source/index assumptions, incomplete full-angle/M2 validation and absence of experimental validation remain.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--counts',type=int,nargs='+',default=[512,1024])
    parser.add_argument('--window',type=float,default=128.)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve the original diagnostic; choose a fresh output')
    if 512 not in args.counts or any(n<32 or n%2 for n in args.counts):raise ValueError('Even counts>=32 including512 required')
    if not np.isfinite(args.window) or args.window<=0:raise ValueError('Positive finite window required')
    if output.is_relative_to(folder) and any(part in ('records','analyses','frozen_designs','source_snapshots')
        for part in output.relative_to(folder).parts[:-1]):raise ValueError('Diagnostic cannot enter raw/provenance directories')
    result=diagnose(folder,tuple(dict.fromkeys(args.counts)),args.window)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'output':str(output),
                     'saved_authoritative_mesh_delta_C':result['saved_authoritative_mesh_delta_C'],
                     'inputs_unchanged':result['all_completed_inputs_unchanged'],
                     'analytic_fixture_pass':result['analytic_implementation_fixture']['implementation_fixture_pass']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
