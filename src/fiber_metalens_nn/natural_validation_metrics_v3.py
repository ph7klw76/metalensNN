"""Use the PREDECLARED phase-current criterion; retain quadratic-fit diagnostic.

v2 used the quadratic bright-core fit at the acceptance threshold while the
training/selection definition used weighted phase current. v3 reconciles
that estimator definition without changing thresholds, source or geometry.
"""
import numpy as np
from .natural_maxwell_analysis_v2 import analyze as analyze_v2
from fiber_metalens_nn.collimation_dft_power import centered_lp_to_cartesian,paraxial_power_equivalent_fields
from fiber_metalens_nn.collimation_optics import forward_spectrum,propagate,kernels

def phase_current_curvature(spectrum,dx,wavelength,declared_diameter):
    count=spectrum.shape[-1];x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x,indexing='xy');r2=xx*xx+yy*yy
    kx,ky,kz,safe,valid,k0=kernels(count,dx,wavelength)
    f=propagate(spectrum,dx,wavelength,0);Ex=f[0];I=abs(Ex)**2
    gx=np.fft.ifft2(1j*kx*spectrum[0]);gy=np.fft.ifft2(1j*ky*spectrum[0]);core=r2<(.6*declared_diameter)**2
    inverse=float(np.sum(core*(np.imag(np.conj(Ex)*gx)*xx+np.imag(np.conj(Ex)*gy)*yy))/max(k0*np.sum(core*I*r2),1e-20))
    return dict(inverse_R_per_um=inverse,normalized_curvature=float(np.pi*(declared_diameter/2)**2/wavelength*inverse),definition='Exactly the training objective: phase-current radial regression over r<0.6*declaredD, normalized by declared Gaussian Rayleigh length.')

def analyze(record_path,config,target_diameter,count=512,window=128):
    import json
    from pathlib import Path
    out=analyze_v2(record_path,config,target_diameter,count,window);record=json.loads(Path(record_path).read_text());d=np.load(record['fields_file']);res=record['settings']['resolution']
    _,dx,f=centered_lp_to_cartesian(d['r_um'],d['fields'][0],1/res,count,window)
    f=paraxial_power_equivalent_fields(f,1,1.55,res,.45,symmetric_axial_collapse=True)
    s=forward_spectrum(f,dx,1.55);current=phase_current_curvature(s,dx,1.55,target_diameter)
    out['v2_quadratic_fit_normalized_curvature']=out['normalized_curvature'];out['v2_optical_gates']=out['optical_gates'].copy()
    out['normalized_curvature']=current['normalized_curvature'];out['phase_current_curvature']=current
    # The shared measurement function is also used by the one-way proxy. Its
    # template labels do not describe the origin of these saved Maxwell fields.
    out['shared_metric_template_labels']={key:out['metrics'][key] for key in ['model','limitations']}
    out['metrics']['model']='Independent full-vector cylindrical Meep device E/H, followed by tested finite-angle output reconstruction and vector angular-spectrum continuation'
    out['metrics']['limitations']='Synthetic matched-index Gaussian source, not a measured fibre eigenmode/cladding. Maxwell includes device back-reflections and finite sidewalls; output reconstruction uses a checked paraxial Yee inversion and excludes near-grazing/evanescent continuation. No material or experimental validation; invalid waist fits and M2 are not accepted.'
    out['optical_gates']['curvature']=abs(current['normalized_curvature'])<=config['natural_training']['investigation_gates']['max_abs_normalized_curvature']
    out['synthetic_optical_pass']=all(out['optical_gates'].values())
    out['metric_definition_reconciliation']='v3 uses exact pre-training phase-current definition at unchanged0.10 threshold; v2 quadratic-fit criterion retained for comparison. Neither results nor thresholds were used to alter geometry/selection.'
    return out
