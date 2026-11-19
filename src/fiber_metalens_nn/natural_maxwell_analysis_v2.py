"""Independent CPU analysis: native quadrature, actual coordinates and optics."""
import json
from pathlib import Path
import numpy as np
from fiber_metalens_nn.collimation_dft_power import (paired_centered_finite_volume_power,
    native_flux_quadrature,centered_lp_to_cartesian,paraxial_power_equivalent_fields)
from fiber_metalens_nn.collimation_optics import (flux,forward_spectrum,propagate,
    vector_spectrum,overlap,kernels)
from fiber_metalens_nn.natural_collimation import measure_natural_output

def read(path):return json.loads(Path(path).read_text())

def analyze(record_path,config,target_diameter,count=512,window=128):
    record=read(record_path);reference=read(record['reference_file']);settings=record['settings'];res=settings['resolution']
    d=np.load(record['fields_file']);r=np.load(reference['fields_file']);pa=np.load(record['power_audit_fields_file']);pr=np.load(reference['power_audit_fields_file'])
    radius=record['grid_metadata']['monitor_radius_um'];dr=1/res
    # The reference's first field is at exactly the launched-flux monitor.
    assert r['z_um'][0]==config['maxwell']['reflection_z_um']
    pdev=paired_centered_finite_volume_power(pa['r_um'],pa['paired_fields'][0],dr,radius,pa['pair_weights'][0])
    pref=paired_centered_finite_volume_power(pr['r_um'],pr['paired_fields'][0],dr,radius,pr['pair_weights'][0])
    nativeT=record['transmission'];pairedT=pdev/pref
    quadrature=max(abs(native_flux_quadrature(a[f'flux_E_{i}'],a[f'flux_H_{i}'])/a['monitor_fluxes'][i]-1) for a in [pa,pr] for i in range(len(a['plane_z_um'])))
    alignment=max(float(np.max(abs(np.tensordot(a['pair_weights'][i],a['paired_fields'][i],axes=1)-thin['fields'][i]))/np.max(abs(thin['fields'][i]))) for a,thin in [(pa,d),(pr,r)] for i in range(len(a['plane_z_um'])))
    ap=[]
    for a in [pa,pr]:
        index=int(np.argmin(abs(a['plane_z_um']-settings['output_z_um'])))
        powers=np.array([paired_centered_finite_volume_power(a['r_um'],a['paired_fields'][index],dr,float(rad),a['pair_weights'][index]) for rad in a['aperture_radii_um']]);native=a['aperture_fluxes']
        ap.extend(abs(powers/powers[-1]/(native/native[-1])-1).tolist())
    def convert(data,index,n):
        x,dx,f=centered_lp_to_cartesian(data['r_um'],data['fields'][index],dr,count,window)
        f=paraxial_power_equivalent_fields(f,n,config['wavelength_um'],res,.45,symmetric_axial_collapse=True)
        return x,dx,f
    x,dx,ref=convert(r,0,config['materials']['substrate']['n']);refspec=forward_spectrum(ref,dx,1.55,n=1.5);refplus=propagate(refspec,dx,1.55,0,n=1.5)
    incident=float(flux(refplus).sum()*dx**2)
    x,dx,f=convert(d,0,1.);s=forward_spectrum(f,dx,1.55);field=propagate(s,dx,1.55,0)
    F=np.fft.fft2(f,axes=(-2,-1));kx,ky,kz,safe,valid,k0=kernels(count,dx,1.55);unfiltered=float(flux(f).sum()*dx**2)
    filtered=np.fft.ifft2(F*valid,axes=(-2,-1));lost=(unfiltered-float(flux(filtered).sum()*dx**2))/unfiltered
    xx,yy=np.meshgrid(x,x,indexing='xy');rr=np.hypot(xx,yy)
    target=vector_spectrum(np.fft.fft2(np.exp(-(rr/(target_diameter/2))**2)),dx,1.55)
    target0=propagate(target,dx,1.55,0);mode=overlap(field,target0,dx)
    full=measure_natural_output(s,dx,incident,config,settings['H_um'])
    angular=np.maximum(.5*np.real(s[0]*np.conj(s[4])-s[1]*np.conj(s[3])),0)
    ta=np.maximum(.5*np.real(target[0]*np.conj(target[4])-target[1]*np.conj(target[3])),0)
    moment=float((angular*(kx*kx+ky*ky)).sum()/angular.sum());tmoment=float((ta*(kx*kx+ky*ky)).sum()/ta.sum())
    zr=np.pi*(target_diameter/2)**2/1.55;sides=[];width_ratios=[]
    from fiber_metalens_nn.metrics import full_width
    for z in config['evaluation']['processing_planes_um']:
        g=propagate(s,dx,1.55,z);t=propagate(target,dx,1.55,z);power=np.maximum(flux(g),0);w=target_diameter/2*np.sqrt(1+(z/zr)**2)
        sides.append(float(power[rr>w*np.sqrt(np.log(100)/2)].sum()/power.sum()))
        if z==0:
            it=np.sum(abs(t[:3])**2,axis=0);middle=count//2
            targetwidths=[full_width(x,it[middle-1:middle+1].mean(axis=0),np.exp(-2)),full_width(x,it[:,middle-1:middle+1].mean(axis=1),np.exp(-2))]
            width_ratios=np.asarray(full['diameters_output_um'])/targetwidths
    _,_,cross=convert(d,1,1.);cross0=propagate(forward_spectrum(cross,dx,1.55),dx,1.55,0)
    cross_overlap=overlap(propagate(s,dx,1.55,settings['crosscheck_z_um']-settings['output_z_um']),cross0,dx)
    forwardT=full['transmission'];conservativeT=min(nativeT,forwardT);eta=conservativeT*mode
    gates=config['natural_training']['investigation_gates'];normalized_curvature=zr*full['wavefront']['inverse_R_per_um']
    optical=dict(transmission=conservativeT>=gates['transmission_min'],declared_overlap=mode>=gates['declared_Gaussian_overlap_min'],useful_efficiency=eta>=gates['useful_efficiency_min'],diameter=bool(np.max(abs(width_ratios-1))<=gates['output_diameter_relative_tolerance']),sidelobe=max(sides)<=gates['max_sidelobe_fraction'],curvature=abs(normalized_curvature)<=gates['max_abs_normalized_curvature'],angular_second_moment=moment/tmoment<=gates['max_angular_second_moment_ratio'])
    reconstruction=dict(paired_vs_native_T_relative_error=abs(pairedT/nativeT-1),native_quadrature_relative_error=quadrature,thin_vs_weighted_pair_field_error=alignment,max_aperture_fraction_relative_error=max(ap),propagating_filter_lost_net_fraction=lost,crosscheck_overlap=cross_overlap,independent_corrected_field_net_T=float(unfiltered/(flux(ref).sum()*dx**2)),native_net_T=nativeT,field_forward_T=forwardT,paired_T=pairedT,incident_field_units=incident)
    checks=dict(native_quadrature=quadrature<1e-12,paired_native_ratio=abs(pairedT/nativeT-1)<.005,monitor_alignment=alignment<1e-4,aperture_fractions=max(ap)<.01,filter_net_loss=abs(lost)<.005,crosscheck=cross_overlap>.995,power_closure=record['power_balance_error']<.05)
    optical={k:bool(v) for k,v in optical.items()};checks={k:bool(v) for k,v in checks.items()}
    return dict(record=str(record_path),settings=settings,metrics=full,declared_target_diameter_um=target_diameter,declared_overlap=mode,conservative_useful_efficiency=eta,conservative_T=conservativeT,normalized_curvature=normalized_curvature,max_sidelobe_fraction=max(sides),sidelobe_planes=sides,angular_second_moment_ratio=moment/tmoment,optical_gates=optical,synthetic_optical_pass=all(optical.values()),reconstruction=reconstruction,numerical_checks=checks,reconstruction_pass=all(checks.values()),reflection=record['reflection'],side_power=record['side_power'],power_balance_error=record['power_balance_error'],normalization='Native net T and independently reconstructed forward T reported separately; min(T_net,T_forward)*forward-mode overlap is a conservative useful estimate. No field amplitude fitted to native flux.',scope='Assumed matched materials/Gaussian source; finite-angle Yee inversion remains a paraxial approximation, checked against native weighted power for this case. No experimental/material validation.')
