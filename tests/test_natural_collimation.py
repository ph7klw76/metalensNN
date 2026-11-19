import numpy as np
import pytest
from fiber_metalens_nn.natural_collimation import (derive_annular_relief,
    ideal_top_phase_spectrum,fit_width_caustic,require_index_matched)

def cfg():
    return dict(annuli=24,cylinder_diameter_um=48,height_max_um=3,wavelength_um=1.55,materials=dict(substrate=dict(n=1.5,kappa=0),polymer=dict(n=1.5,kappa=0),ambient=dict(n=1.,kappa=0)),fabrication=dict(height_step_um=.05),proxy=dict(output_clearance_um=1))

def test_old_mismatched_core_is_rejected_and_matched_case_allowed():
    c=cfg();require_index_matched(c);c['materials']['substrate']['n']=1.45
    with pytest.raises(ValueError,match='must match'):require_index_matched(c)

def test_field_derived_relief_cancels_known_actual_curvature_and_is_bounded():
    c=cfg();x=(np.arange(384)-191.5)/3;xx,yy=np.meshgrid(x,x);r2=xx*xx+yy*yy
    field=np.exp(-r2/5.1**2+1j*(1.7+.02*r2))
    continuous,quantized,report=derive_annular_relief(field,x,c)
    assert np.all((continuous>=0)&(continuous<=3))
    assert np.allclose(quantized/.05,np.round(quantized/.05),atol=1e-5)
    assert report['phase_quadratic_coefficient_rad_per_um2']==pytest.approx(.02,rel=.002)
    assert report['encoded_bright_phase_RMS_rad']<.06
    assert continuous[0]>continuous[5]
    assert report['polymer_wavefront_radius_um']==pytest.approx(np.pi*1.5/(1.55*.02),rel=.002)

def test_ideal_phase_reference_depends_on_amplitude_not_incoming_phase():
    c=cfg();x=(np.arange(128)-63.5)/2;xx,yy=np.meshgrid(x,x);r2=xx*xx+yy*yy;amplitude=np.exp(-r2/5**2)
    a=ideal_top_phase_spectrum(amplitude*np.exp(1j*.03*r2),.5,c)
    b=ideal_top_phase_spectrum(amplitude*np.exp(-1j*.08*r2),.5,c)
    assert np.allclose(a,b,rtol=1e-6,atol=1e-6)

def test_downstream_caustic_recovers_back_extrapolated_waist_without_zero_diffraction():
    z=np.array([0,10,25,50,75,100,200]);w0=5.;theta=.1;zw=-12.
    diam=2*np.sqrt(w0*w0+theta**2*(z-zw)**2)
    fit=fit_width_caustic(z,diam)
    assert fit['valid']
    assert fit['waist_relative_to_output_um']==pytest.approx(zw,abs=1e-10)
    assert fit['waist_radius_um']==pytest.approx(w0,abs=1e-10)
    assert fit['Gaussian_equivalent_half_angle_rad']==pytest.approx(theta,abs=1e-10)

def test_unresolved_or_nonpositive_top_field_fails_closed():
    with pytest.raises(ValueError,match='nonzero'):
        derive_annular_relief(np.zeros((32,32)),(np.arange(32)-15.5)/3,cfg())


def test_legacy_training_rejects_new_forward_only_objective_version():
    from types import SimpleNamespace
    from fiber_metalens_nn.collimation_proxy import CylinderProxy
    p=SimpleNamespace(config={'loss':{'version':'natural_output_wavefront_initial_v1'}})
    with pytest.raises(ValueError,match='Unknown'):
        CylinderProxy._objective(p,None,None,None,None)


def test_actual_size_quality_does_not_silently_pass_old_30um_gate_or_remove_diffraction():
    from fiber_metalens_nn.collimation_optics import vector_spectrum
    from fiber_metalens_nn.natural_collimation import measure_natural_output
    c=cfg();c['result_label']='TEST_ONLY';c['evaluation']=dict(processing_planes_um=[0,25,50,75,100],diagnostic_planes_um=[200])
    c['historical_30um_evaluation']=dict(processing_planes_um=[0,25,50,75,100],diagnostic_planes_um=[],diameter_bounds_um=[27,33],max_diameter_change=.05,minimum_overlap=.9,minimum_eta_mode=.7)
    x=(np.arange(256)-127.5)*.5;xx,yy=np.meshgrid(x,x)
    s=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/25)),.5,1.55)
    incident=float(.5*np.real(s[0]*np.conj(s[4])-s[1]*np.conj(s[3])).sum()*.5**2/256**2)/.96
    m=measure_natural_output(s,.5,incident,c,32)
    assert m['overlap_with_actual_size_Gaussian']>.999
    assert not m['historical_30um_aspiration']['original_optical_targets_pass']
    assert not m['historical_30um_aspiration']['original_criteria']['diameter']
    assert m['flat_phase_same_size_diffraction_half_angle_rad']>.09
    assert m['diameter_versus_distance'][-2]['measured_diameter_x_um']>20
