"""Independent unequal-index Maxwell E/H analysis; no Meep or JAX import.

This generalizes the audited native-power path and authoritative phase-current
definition. It does not invoke the matched-index historical report template.
Nominal optical gates and numerical convergence are separate assessments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fiber_metalens_nn.collimation_dft_power import (
    paired_centered_finite_volume_power, native_flux_quadrature,
    centered_lp_to_cartesian, paraxial_power_equivalent_fields,
)
from fiber_metalens_nn.collimation_optics import (
    flux, forward_spectrum, propagate, vector_spectrum, overlap, kernels, fingerprint,
)
from fiber_metalens_nn.collimation_metrics import fitted_divergence
from fiber_metalens_nn.metrics import full_width


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def forward_field_with_power(fields, dx, wavelength, medium_index):
    """Homogeneous forward decomposition with the actual monitor-medium index."""
    spectrum = forward_spectrum(fields, dx, wavelength, n=medium_index)
    field = propagate(spectrum, dx, wavelength, 0, n=medium_index)
    power = float(flux(field).sum()*dx**2)
    return spectrum, field, power


def require_symmetric_axial_collapse(pair_weights, plane_z_um, label):
    """Reject geometries outside the retained paraxial Yee correction contract.

    The worker exports actual geometric weights, including off-grid stencils.
    The power-equivalent correction below was derived only for an equal
    two-center collapse, so every saved plane must satisfy that assumption.
    """
    weights = np.asarray(pair_weights, dtype=float)
    planes = np.asarray(plane_z_um, dtype=float)
    if (planes.ndim != 1 or not np.isfinite(planes).all()
            or weights.shape != (len(planes), 2) or not np.isfinite(weights).all()):
        raise ValueError(f'{label}: finite two-center weights per saved plane required')
    bad = np.flatnonzero(np.any(np.abs(weights-.5) > 1e-8, axis=1))
    if len(bad):
        raise ValueError(
            f'{label}: symmetric axial collapse requires [.5, .5] weights '
            f'at every saved plane; incompatible plane indices {bad.tolist()}')


def phase_current_curvature(spectrum, dx, wavelength, declared_diameter, medium_index=1.):
    """Training-defined radial phase current inside r<0.6*declared diameter."""
    count = spectrum.shape[-1]
    x = (np.arange(count)-count/2+.5)*dx
    xx, yy = np.meshgrid(x,x,indexing='xy'); r2=xx*xx+yy*yy
    kx,ky,_,_,_,k0 = kernels(count,dx,wavelength,n=medium_index)
    field = propagate(spectrum,dx,wavelength,0,n=medium_index)
    ex=field[0];gx=np.fft.ifft2(1j*kx*spectrum[0]);gy=np.fft.ifft2(1j*ky*spectrum[0])
    core=r2<(.6*declared_diameter)**2
    denominator=medium_index*k0*np.sum(core*np.abs(ex)**2*r2)
    if not np.isfinite(denominator) or denominator<=0:
        raise ValueError('Nonzero finite bright-core phase-current denominator required')
    inverse=float(np.sum(core*(np.imag(np.conj(ex)*gx)*xx+
                              np.imag(np.conj(ex)*gy)*yy))/denominator)
    zr=np.pi*medium_index*(declared_diameter/2)**2/wavelength
    return {'inverse_R_per_um':inverse,'normalized_curvature':float(zr*inverse),
            'definition':'Radial Ex phase-current regression inside r<0.6*declaredD; medium-specific Gaussian Rayleigh normalization'}


def spectrum_metrics(spectrum, dx, wavelength, target_diameter, config, incident_power):
    """Physical observables from saved forward fields and an output-defined target.

    Target diameter is the scalar Gaussian parameter frozen independently of
    trained output. Primary width comparisons use measured vector-target
    crossings; no post hoc output-size matching replaces that target.
    """
    n=float(config['materials']['ambient']['n'])
    if not all(np.isfinite(v) and v>0 for v in (incident_power,target_diameter,wavelength,n)):
        raise ValueError('Positive incident power, target size, wavelength and ambient index required')
    count=spectrum.shape[-1];x=(np.arange(count)-count/2+.5)*dx
    xx,yy=np.meshgrid(x,x,indexing='xy');rr=np.hypot(xx,yy)
    target=vector_spectrum(np.fft.fft2(np.exp(-(rr/(target_diameter/2))**2)),dx,wavelength,n=n)
    field0=propagate(spectrum,dx,wavelength,0,n=n)
    target0=propagate(target,dx,wavelength,0,n=n)
    power=float(flux(field0).sum()*dx**2)
    if not np.isfinite(power) or power<=0:
        raise ValueError('Nonzero finite forward output power required')
    mode=overlap(field0,target0,dx)
    processing=[float(z) for z in config['evaluation']['processing_planes_um']]
    diagnostic=[float(z) for z in config['evaluation'].get('diagnostic_planes_um',[])]
    if not processing or processing[0]!=0:
        raise ValueError('Processing planes must begin at the output reference')
    planes=list(dict.fromkeys(processing+diagnostic));reports=[];sides=[];target_widths=[]
    zr=np.pi*n*(target_diameter/2)**2/wavelength
    middle=count//2
    for z in planes:
        f=propagate(spectrum,dx,wavelength,z,n=n)
        tf=propagate(target,dx,wavelength,z,n=n)
        intensity=np.sum(np.abs(f[:3])**2,axis=0)
        ti=np.sum(np.abs(tf[:3])**2,axis=0)
        widths=[full_width(x,intensity[middle-1:middle+1].mean(axis=0),np.exp(-2)),
                full_width(x,intensity[:,middle-1:middle+1].mean(axis=1),np.exp(-2))]
        tw=[full_width(x,ti[middle-1:middle+1].mean(axis=0),np.exp(-2)),
            full_width(x,ti[:,middle-1:middle+1].mean(axis=1),np.exp(-2))]
        positive=np.maximum(flux(f),0);normal=float(positive.sum())
        if not np.isfinite(normal) or normal<=0:
            raise ValueError('Nonzero finite positive axial power required')
        mx,my=float((positive*xx).sum()/normal),float((positive*yy).sum()/normal)
        edge=float(positive[(np.abs(xx)>.45*count*dx)|(np.abs(yy)>.45*count*dx)].sum()/normal)
        wz=target_diameter/2*np.sqrt(1+(z/zr)**2)
        sidelobe=float(positive[rr>wz*np.sqrt(np.log(100)/2)].sum()/normal)
        if z in processing:
            sides.append(sidelobe);target_widths.append(tw)
        reports.append({'z_um':z,'diameter_x_um':widths[0],'diameter_y_um':widths[1],
            'target_diameters_um':tw,'valid_crossings':[v is not None for v in widths],
            'D4sigma_x_um':float(4*np.sqrt((positive*(xx-mx)**2).sum()/normal)),
            'D4sigma_y_um':float(4*np.sqrt((positive*(yy-my)**2).sum()/normal)),
            'edge_power_fraction':edge,'sidelobe_fraction':sidelobe,
            'power_relative_to_output':float(flux(f).sum()*dx**2/power)})
    kx,ky,kz,safe,valid,k0=kernels(count,dx,wavelength,n=n)
    angular=np.maximum(.5*np.real(spectrum[0]*np.conj(spectrum[4])-
                                  spectrum[1]*np.conj(spectrum[3])),0)
    ta=np.maximum(.5*np.real(target[0]*np.conj(target[4])-target[1]*np.conj(target[3])),0)
    moment=float((angular*(kx*kx+ky*ky)).sum()/angular.sum())
    target_moment=float((ta*(kx*kx+ky*ky)).sum()/ta.sum())
    angles=[]
    normalized=angular/angular.sum()
    for transverse in (kx,ky):
        slope=np.arctan2(transverse,safe);mean=(normalized*slope).sum()
        angles.append(float(2*np.sqrt((normalized*(slope-mean)**2).sum())))
    # The existing fit's wave-number grid is air-based; lambda0/n gives the
    # exact same homogeneous-medium k and angle/Jacobian definitions.
    fitted=fitted_divergence(spectrum,dx,wavelength/n)
    current=phase_current_curvature(spectrum,dx,wavelength,target_diameter,n)
    output=reports[0]
    return {
        'model':'Independent vector Maxwell E/H followed by homogeneous-medium forward reconstruction',
        'limitations':'Gaussian source surrogate with declared unequal indices; paraxial Yee correction checked against native power; retained propagating band only; no real-mode, ISO M2, tolerance or experimental validation',
        'transmission':power/incident_power,'declared_overlap':mode,
        'declared_useful_efficiency':mode*power/incident_power,
        'declared_target_diameter_um':float(target_diameter),
        'diameters_output_um':[output['diameter_x_um'],output['diameter_y_um']],
        'target_widths_um':target_widths,'planes':reports,'diameter_versus_distance':reports,
        'fitted_core_divergence':fitted,'second_moment_divergence_half_angle_rad':angles,
        'declared_Gaussian_diffraction_half_angle_rad':float(wavelength/(np.pi*n*target_diameter/2)),
        'phase_current_curvature':current,'wavefront':current,
        'normalized_curvature':current['normalized_curvature'],
        'angular_second_moment_ratio':moment/target_moment,
        'sidelobe_planes':sides,'max_sidelobe_fraction':max(sides),
        'all_processing_crossings_valid':all(all(row['valid_crossings']) for row in reports if row['z_um'] in processing),
        'M2':None,'physical_validation':False,
    }


def analyze(record_path, config, target_diameter, count=512, window=128):
    record_path=Path(record_path);record=read(record_path);reference=read(record['reference_file'])
    if (record['config_fingerprint']!=fingerprint(config)
            or reference['config_fingerprint']!=fingerprint(config)
            or record['settings']!=reference['settings']):
        raise ValueError('Device/reference/configuration physics or numerical settings differ')
    settings=record['settings'];res=float(settings['resolution']);dr=1/res
    if settings.get('m',1)!=1:
        raise ValueError('Current centered LP reconstruction requires the m=+1 record and reflected m=-1 symmetry')
    wavelength=float(config['wavelength_um'])
    ns=float(config['materials']['substrate']['n']);na=float(config['materials']['ambient']['n'])
    d=np.load(record['fields_file'],allow_pickle=False);r=np.load(reference['fields_file'],allow_pickle=False)
    pa=np.load(record['power_audit_fields_file'],allow_pickle=False)
    pr=np.load(reference['power_audit_fields_file'],allow_pickle=False)
    for saved, label in ((pa, 'device'), (pr, 'reference')):
        require_symmetric_axial_collapse(saved['pair_weights'],saved['plane_z_um'],label)
    radius=float(record['grid_metadata']['monitor_radius_um'])
    reflection_z=float(settings.get('reflection_z_um',config.get('maxwell',{}).get('reflection_z_um',-1.)))
    if abs(float(r['z_um'][0])-reflection_z)>1e-9:
        raise ValueError('Reference first field is not at the incident normalization monitor')
    courant=float(settings.get('Courant',pa['Courant']))
    if abs(float(pr['Courant'])-courant)>1e-12 or abs(float(pa['Courant'])-courant)>1e-12:
        raise ValueError('Saved field Courant settings differ')
    pdev=paired_centered_finite_volume_power(pa['r_um'],pa['paired_fields'][0],dr,radius,pa['pair_weights'][0])
    pref=paired_centered_finite_volume_power(pr['r_um'],pr['paired_fields'][0],dr,radius,pr['pair_weights'][0])
    if pref<=0 or not np.isfinite(pref):raise ValueError('Positive finite paired incident reference required')
    nativeT=float(record['transmission']);pairedT=pdev/pref
    if not np.isfinite(nativeT) or nativeT<=0:
        raise ValueError('Positive finite native transmission required for relative reconstruction checks')
    quadrature=max(abs(native_flux_quadrature(a[f'flux_E_{i}'],a[f'flux_H_{i}'])/a['monitor_fluxes'][i]-1)
                   for a in (pa,pr) for i in range(len(a['plane_z_um'])))
    alignment=max(float(np.max(np.abs(np.tensordot(a['pair_weights'][i],a['paired_fields'][i],axes=1)-thin['fields'][i]))/
                        np.max(np.abs(thin['fields'][i]))) for a,thin in ((pa,d),(pr,r))
                  for i in range(len(a['plane_z_um'])))
    aperture_errors=[]
    for a in (pa,pr):
        index=int(np.argmin(np.abs(a['plane_z_um']-settings['output_z_um'])))
        powers=np.array([paired_centered_finite_volume_power(a['r_um'],a['paired_fields'][index],dr,float(rad),
                        a['pair_weights'][index]) for rad in a['aperture_radii_um']])
        native=a['aperture_fluxes'];aperture_errors.extend(np.abs(powers/powers[-1]/(native/native[-1])-1).tolist())
    def convert(data,index,n):
        x,dx,f=centered_lp_to_cartesian(data['r_um'],data['fields'][index],dr,count,window)
        return x,dx,paraxial_power_equivalent_fields(f,n,wavelength,res,courant,symmetric_axial_collapse=True)
    _,dx,ref=convert(r,0,ns)
    _,_,incident=forward_field_with_power(ref,dx,wavelength,ns)
    _,dx,f=convert(d,0,na)
    spectrum,field,output_power=forward_field_with_power(f,dx,wavelength,na)
    full=spectrum_metrics(spectrum,dx,wavelength,target_diameter,config,incident)
    original_spectrum=np.fft.fft2(f,axes=(-2,-1));_,_,_,_,valid,_=kernels(count,dx,wavelength,n=na)
    unfiltered=float(flux(f).sum()*dx**2)
    filtered=np.fft.ifft2(original_spectrum*valid,axes=(-2,-1))
    lost=(unfiltered-float(flux(filtered).sum()*dx**2))/unfiltered
    _,_,cross=convert(d,1,na)
    _,cross0,_=forward_field_with_power(cross,dx,wavelength,na)
    cross_overlap=overlap(propagate(spectrum,dx,wavelength,
                                  settings['crosscheck_z_um']-settings['output_z_um'],n=na),cross0,dx)
    forwardT=output_power/incident;conservativeT=min(nativeT,forwardT)
    mode=full['declared_overlap'];eta=conservativeT*mode
    gates=config['natural_training']['investigation_gates']
    output_widths=np.asarray(full['diameters_output_um'],float)
    target_widths=np.asarray(full['target_widths_um'][0],float)
    width_valid=bool(np.isfinite(output_widths).all() and np.isfinite(target_widths).all())
    optical={
        'transmission':conservativeT>=gates['transmission_min'],
        'declared_overlap':mode>=gates['declared_Gaussian_overlap_min'],
        'useful_efficiency':eta>=gates['useful_efficiency_min'],
        'diameter':width_valid and bool(np.max(np.abs(output_widths/target_widths-1))<=gates['output_diameter_relative_tolerance']),
        'sidelobe':full['max_sidelobe_fraction']<=gates['max_sidelobe_fraction'],
        'curvature':abs(full['normalized_curvature'])<=gates['max_abs_normalized_curvature'],
        'angular_second_moment':full['angular_second_moment_ratio']<=gates['max_angular_second_moment_ratio'],
        'valid_width_crossings':full['all_processing_crossings_valid'],
    }
    reconstruction={
        'paired_vs_native_T_relative_error':abs(pairedT/nativeT-1),
        'native_quadrature_relative_error':quadrature,'thin_vs_weighted_pair_field_error':alignment,
        'max_aperture_fraction_relative_error':max(aperture_errors),
        'propagating_filter_lost_net_fraction':lost,'crosscheck_overlap':cross_overlap,
        'native_net_T':nativeT,'field_forward_T':forwardT,'paired_T':pairedT,
        'incident_field_units':incident,'input_reference_index':ns,'output_monitor_index':na,
        'wavelength_um':wavelength,'Courant':courant,
    }
    checks={'native_quadrature':quadrature<1e-12,'paired_native_ratio':abs(pairedT/nativeT-1)<.005,
            'monitor_alignment':alignment<1e-4,'aperture_fraction_disagreement':max(aperture_errors)<.01,
            'filter_net_loss':abs(lost)<.005,'crosscheck':cross_overlap>.995,
            'power_closure':record['power_balance_error']<.05}
    optical={key:bool(value) for key,value in optical.items()}
    checks={key:bool(value) for key,value in checks.items()}
    return {'record':str(record_path),'settings':settings,'result_label':config['result_label'],
        'metrics':full,'declared_target_diameter_um':float(target_diameter),
        'declared_overlap':mode,'conservative_useful_efficiency':eta,'conservative_T':conservativeT,
        'normalized_curvature':full['normalized_curvature'],'phase_current_curvature':full['phase_current_curvature'],
        'max_sidelobe_fraction':full['max_sidelobe_fraction'],'sidelobe_planes':full['sidelobe_planes'],
        'angular_second_moment_ratio':full['angular_second_moment_ratio'],
        'optical_gates':optical,'synthetic_optical_pass':all(optical.values()),
        'reconstruction':reconstruction,'numerical_checks':checks,'reconstruction_pass':all(checks.values()),
        'reflection':record['reflection'],'side_power':record['side_power'],
        'power_balance_error':record['power_balance_error'],
        'normalization':'Actual monitor-medium indices; no amplitude fit; min(native net T,reconstructed forward T)*declared overlap',
        'scope':'Synthetic Gaussian surrogate with user polymer index and assumed input phase index; nominal gates are not mesh convergence or experimental acceptance',
        'numerical_convergence_claimed':False,'physical_validation':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--target-diameter',type=float,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=512);parser.add_argument('--window',type=float,default=128.)
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve original analysis; choose a new report')
    result=analyze(args.record,read(args.config),args.target_diameter,args.count,args.window)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'reconstruction_pass':result['reconstruction_pass'],
                      'nominal_optical_pass':result['synthetic_optical_pass']},indent=2))


if __name__=='__main__':
    main()
