"""Field-derived natural-size collimation diagnostics; no training or Maxwell.

An ideal local phase operation is a comparison reference, not a universal
efficiency bound or a realizable zero-thickness metalens.
"""
import copy
import numpy as np
from .collimation_optics import (grid,vector_spectrum,kernels,propagate,describe,
                                overlap,flux)
from .collimation_metrics import attach_divergence_fit


def require_index_matched(config):
    material=config['materials']
    if abs(material['substrate']['n']-material['polymer']['n'])>1e-12:
        raise ValueError('declared core and cylinder indices must match')
    if any(v['kappa']!=0 for v in material.values()) or material['ambient']['n']!=1:
        raise ValueError('this initial synthetic branch requires lossless media and air output')


def derive_annular_relief(top_field,x,config):
    """Encode the actual PROXY top-field phase as bounded connected relief.

    Bright-ring phase is power-weighted; dark rings use a weighted quadratic
    continuation. An unwrapped center-high profile uses an explicit piston
    hmax. It does not assume f=H or require complete 2*pi phase coverage.
    Finite-depth propagation must subsequently test this initial encoding.
    """
    f=np.asarray(top_field);x=np.asarray(x);xx,yy=np.meshgrid(x,x,indexing='xy');r=np.hypot(xx,yy)
    count=config['annuli'];radius=config['cylinder_diameter_um']/2;pitch=radius/count
    if f.shape!=r.shape or not np.isfinite(f).all():raise ValueError('finite top field on the declared Cartesian grid required')
    contrast=config['materials']['polymer']['n']-config['materials']['ambient']['n']
    if contrast<=0:raise ValueError('positive polymer-air contrast required for this relief encoding')
    phase_per_um=2*np.pi/config['wavelength_um']*contrast
    intensity=abs(f)**2;axis=f[len(x)//2-1:len(x)//2+1,len(x)//2-1:len(x)//2+1].mean()
    if abs(axis)==0:raise ValueError('nonzero central mode field required')
    ringphase=[];power=[];r2=[]
    unit=f/np.maximum(abs(f),1e-30)*np.exp(-1j*np.angle(axis))
    for j in range(count):
        mask=(r>=j*pitch)&(r<(j+1)*pitch);weight=intensity*mask;p=float(weight.sum())
        power.append(p);r2.append(float((weight*r*r).sum()/p) if p>1e-30 else ((j+.5)*pitch)**2)
        ringphase.append(float(np.angle((weight*unit).sum())))
    power=np.asarray(power);phase=np.unwrap(ringphase);r2=np.asarray(r2)
    if power.sum()<=0:raise ValueError('nonzero in-cylinder field required')
    cumulative=np.cumsum(power)/power.sum();bright=(np.r_[0.,cumulative[:-1]]<.9999)&(power>power.sum()*1e-12)
    if bright.sum()<3:raise ValueError('at least three illuminated rings required')
    a,b=np.polyfit(r2[bright],phase[bright],1,w=np.sqrt(power[bright]))
    used=np.where(bright,phase,a*r2+b)
    continuous=np.clip(config['height_max_um']-(used-b)/phase_per_um,0,config['height_max_um'])
    step=config['fabrication']['height_step_um'];quantized=np.clip(np.round(continuous/step)*step,0,config['height_max_um'])
    residual=phase+phase_per_um*quantized
    piston=np.angle(np.sum(power[bright]*np.exp(1j*residual[bright])))
    error=np.angle(np.exp(1j*(residual-piston)))
    report=dict(method='unwrapped power-weighted actual proxy top-field phase, center-high piston, bright-ring data and dark-ring quadratic continuation',ring_phase_rad=phase.tolist(),ring_power_fraction=(power/power.sum()).tolist(),bright_rings=bright.tolist(),ring_effective_r2_um2=r2.tolist(),phase_quadratic_coefficient_rad_per_um2=float(a),phase_fit_offset_rad=float(b),polymer_wavefront_radius_um=float(np.pi*config['materials']['polymer']['n']/config['wavelength_um']/a) if a else None,phase_per_height_rad_per_um=float(phase_per_um),available_phase_rad=float(phase_per_um*config['height_max_um']),full_2pi_height_um=float(config['wavelength_um']/contrast),continuous_heights_um=continuous.tolist(),quantized_heights_um=quantized.tolist(),encoded_bright_phase_RMS_rad=float(np.sqrt(np.sum(power[bright]*error[bright]**2)/power[bright].sum())),scope='Encoding error at top surface only; finite-depth annular propagation and independent vector Maxwell still required.')
    return continuous.astype(np.float32),quantized.astype(np.float32),report


def ideal_top_phase_spectrum(top_field,dx,config):
    """Exact scalar phase flattening at H, followed by air to H+hmax+clearance.

    The ideal operator has no relief thickness or fabrication constraints.
    Equal physical output plane does not make it the same material path.
    """
    field=np.asarray(top_field);flat=abs(field).astype(np.complex64)
    n=config['materials']['polymer']['n'];na=config['materials']['ambient']['n']
    scalar=flat*(2*n/(n+na))
    spectrum=vector_spectrum(np.fft.fft2(scalar),dx,config['wavelength_um'])
    _,_,kz,_,valid,_=kernels(field.shape[-1],dx,config['wavelength_um'])
    distance=config['height_max_um']+config['proxy']['output_clearance_um']
    return spectrum*(np.exp(1j*kz*distance)*valid)[None]


def curvature_fit(field,x,wavelength):
    count=len(x);middle=count//2;line=field[0,middle-1:middle+1].mean(axis=0)
    phase=np.unwrap(np.angle(line));amp=abs(line);bright=(amp>.2*amp.max())&(abs(x)<15)
    if bright.sum()<7:return dict(valid=False,reason='insufficient connected bright core')
    a,b,c=np.polyfit(x[bright],phase[bright],2,w=amp[bright])
    fitted=a*x[bright]**2+b*x[bright]+c
    rms=float(np.sqrt(np.average((phase[bright]-fitted)**2,weights=amp[bright]**2)))
    inverse=float(a*wavelength/np.pi)
    return dict(valid=True,inverse_R_per_um=inverse,R_um=None if abs(inverse)<1e-12 else 1/inverse,weighted_phase_RMS_rad=rms,definition='output Ex bright-core quadratic phase fit; not a full vector eikonal')


def fit_width_caustic(z,diameters):
    z=np.asarray(z,float);w=np.asarray(diameters,float)/2
    A,B,C=np.polyfit(z,w*w,2)
    if A<=0 or C-B*B/(4*A)<=0:return dict(valid=False,reason='nonphysical quadratic width fit')
    waist=-B/(2*A);w0=np.sqrt(C-B*B/(4*A));prediction=np.sqrt(np.maximum(A*z*z+B*z+C,0))
    error=float(np.max(abs(prediction/w-1)))
    return dict(valid=error<.02,waist_relative_to_output_um=float(waist),waist_radius_um=float(w0),Gaussian_equivalent_half_angle_rad=float(np.sqrt(A)),Gaussian_equivalent_Rayleigh_um=float(w0/np.sqrt(A)),max_width_fit_relative_error=error,scope='Fit from downstream positive-z 1/e2 widths. Negative waist coordinates are air-field back-extrapolations, not propagation through polymer. No measured or ISO M2 claim.')


def measure_natural_output(spectrum,dx,incident,config,H):
    """Report actual-size mode quality and diffraction separately from old goals."""
    legacy=copy.deepcopy(config);legacy['evaluation']={**config['historical_30um_evaluation'],**config['evaluation']}
    power=float(flux(propagate(spectrum,dx,config['wavelength_um'],0)).sum()*dx**2)
    historical=describe(spectrum,dx,config['wavelength_um'],30,legacy,power/incident)
    attach_divergence_fit(historical,spectrum,dx,config['wavelength_um'])
    planes=historical['planes'];D=np.sqrt(planes[0]['diameter_x_um']*planes[0]['diameter_y_um'])
    count=spectrum.shape[-1];x,_=grid(count,count*dx);xx,yy=np.meshgrid(x,x,indexing='xy')
    target=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(D/2)**2)),dx,config['wavelength_um'])
    target0=propagate(target,dx,config['wavelength_um'],0);field0=propagate(spectrum,dx,config['wavelength_um'],0)
    mode=overlap(field0,target0,dx);diffraction=[]
    for plane in planes:
        z=plane['z_um'];idealD=D*np.sqrt(1+(z/(np.pi*(D/2)**2/config['wavelength_um']))**2)
        diffraction.append(dict(z_from_output_um=z,measured_diameter_x_um=plane['diameter_x_um'],measured_diameter_y_um=plane['diameter_y_um'],same_output_size_flat_phase_Gaussian_diameter_um=float(idealD),D4sigma_x_um=plane['D4sigma_x_um'],D4sigma_y_um=plane['D4sigma_y_um'],edge_power_fraction=plane['edge_power_fraction']))
    theta_diff=config['wavelength_um']/(np.pi*D/2)
    fit=historical['fitted_divergence'];angles=[f['half_angle_rad'] for f in fit['axes']] if fit['valid'] else None
    original=config['historical_30um_evaluation'];lookup={p['z_um']:p for p in planes}
    oldwidth=np.array([[lookup[z]['diameter_x_um'],lookup[z]['diameter_y_um']] for z in original['processing_planes_um']])
    oldchange=float(np.max(abs(oldwidth/oldwidth[0]-1)))
    oldcriteria=dict(diameter=bool(np.all(oldwidth[0]>=original['diameter_bounds_um'][0]) and np.all(oldwidth[0]<=original['diameter_bounds_um'][1])),stability=oldchange<=original['max_diameter_change'],mode_overlap=historical['overlap']>=original['minimum_overlap'],absolute_efficiency=historical['eta_mode']>=original['minimum_eta_mode'])
    return dict(result_label=config['result_label'],model='one-way scalar finite-cylinder/relief proxy with vector air output',H_um=H,output_reference_z_um=H+config['height_max_um']+config['proxy']['output_clearance_um'],diameters_output_um=[planes[0]['diameter_x_um'],planes[0]['diameter_y_um']],natural_matched_Gaussian_diameter_um=float(D),matched_mode_definition='circular flat-phase vector Gaussian with diameter equal to geometric mean of observed output x/y 1/e2 diameters',overlap_with_actual_size_Gaussian=float(mode),useful_efficiency_actual_size=float(mode*power/incident),transmission=power/incident,wavefront=curvature_fit(field0,x,config['wavelength_um']),fitted_core_divergence=fit,second_moment_divergence_half_angle_rad=historical['second_moment_divergence_half_angle_rad'],flat_phase_same_size_diffraction_half_angle_rad=float(theta_diff),signed_core_excess_divergence_ratio=None if angles is None else [float(a/theta_diff-1) for a in angles],waist_fits=[fit_width_caustic([p['z_um'] for p in planes],[p[key] for p in planes]) for key in ['diameter_x_um','diameter_y_um']],diameter_versus_distance=diffraction,historical_30um_aspiration=dict(overlap=historical['overlap'],eta_mode=historical['eta_mode'],maximum_diameter_change_100um=oldchange,original_criteria=oldcriteria,original_optical_targets_pass=all(oldcriteria.values()),status='Separate aspiration only; these original five-plane gates are not the new natural-size objective'),M2=None,physical_validation=False,limitations='Gaussian surrogate and assumed matched index; no true fibre eigenmode/cladding, internal backward waves or exact vector interface coupling. Near-grazing/evanescent continuation excluded. Phase flattening is not zero diffraction.')
