"""Scope regressions: centered LP cylindrical fields and bounded continuation."""
from pathlib import Path
import numpy as np
import pytest
from fiber_metalens_nn.collimation_dft_power import (centered_radial_cell_areas,
    paired_centered_finite_volume_power,linear_cartesian_on_polar,
    centered_lp_to_cartesian,native_flux_quadrature)
from fiber_metalens_nn.collimation_optics import flux,forward_spectrum,propagate


def fixture():
    return np.load(Path(__file__).parent/'data/structured_nn_h35_r16.npz',allow_pickle=False)


def test_geometric_annular_weights_clip_extra_cell_and_preserve_disk_area():
    delta=.0625;r=(np.arange(545)+.5)*delta
    for aperture in [3,8,15,34]:
        area=centered_radial_cell_areas(r,delta,aperture)
        assert area.sum()==pytest.approx(np.pi*aperture**2,rel=1e-14)
        assert np.all(area[r-delta/2>=aperture]==0)


def test_legacy_radial_linspace_and_uncovered_apertures_fail_closed():
    with pytest.raises(ValueError,match='half-cell'):
        centered_radial_cell_areas(np.linspace(0,34,545),1/16,34)
    with pytest.raises(ValueError,match='cover'):
        centered_radial_cell_areas((np.arange(10)+.5)/16,1/16,34)


def test_known_regular_gaussian_lp_power_axis_and_cartesian_conversion():
    delta=.01;r=(np.arange(2001)+.5)*delta;g=np.exp(-(r/4.5)**2);n=1.45
    f=np.array([g,1j*g,0*g,-1j*n*g,n*g,0*g])
    power=paired_centered_finite_volume_power(r,np.array([f,f]),delta,20,[.5,.5])
    expected=n*np.pi*4.5**2/4
    assert power==pytest.approx(expected,rel=2e-6)
    _,dx,cart=centered_lp_to_cartesian(r,f,delta,512,64)
    assert flux(cart).sum()*dx**2==pytest.approx(expected,rel=4e-6)


def test_azimuthal_half_sum_yields_lp_quarter_native_sector_without_fit():
    rng=np.random.default_rng(71);f=rng.normal(size=(6,55))+1j*rng.normal(size=(6,55))
    phi=np.arange(16)*2*np.pi/16;cart=linear_cartesian_on_polar(f,phi)
    sector=np.real(f[0]*np.conj(f[4])-f[1]*np.conj(f[3]))
    assert np.allclose(flux(cart).mean(axis=0),.25*sector,atol=1e-14,rtol=1e-13)


def test_actual_structured_pair_and_weighted_native_flux_agree_in_declared_scope():
    with fixture() as d:
        powers={}
        for name in ['reference','device']:
            native=native_flux_quadrature(d[name+'_flux_E'],d[name+'_flux_H'])
            assert native==pytest.approx(float(d[name+'_native_flux']),rel=1e-12)
            powers[name]=paired_centered_finite_volume_power(d['r_um'],d[name+'_pair'],float(d['cell_width_um']),float(d['aperture_radius_um']),d[name+'_weights'])
            assert powers[name]/native==pytest.approx(.25,rel=2e-4)
        assert powers['device']/powers['reference']==pytest.approx(float(d['device_native_flux']/d['reference_native_flux']),rel=.005)


def test_actual_structured_radial_aperture_fractions_match_native_flux():
    with fixture() as d:
        for name in ['reference','device']:
            p=np.array([paired_centered_finite_volume_power(d['r_um'],d[name+'_pair'],float(d['cell_width_um']),float(a),d[name+'_weights']) for a in d['aperture_radii_um']])
            native=d[name+'_aperture_native_flux']
            assert np.allclose(p/p[-1],native/native[-1],rtol=.01,atol=0)


def test_actual_quarter_cell_monitor_requires_geometric_weights():
    with fixture() as d:
        for name in ['reference','device']:
            pair=d[name+'_shifted_pair'];thin=d[name+'_shifted_thin'];w=d[name+'_shifted_weights']
            assert np.allclose(w,[.25,.75],atol=1e-14)
            assert np.max(abs(np.tensordot(w,pair,axes=1)-thin))/np.max(abs(thin))<1e-4
            # Assuming symmetric collapse at this displaced plane is detectably wrong.
            assert np.max(abs(pair.mean(axis=0)-thin))/np.max(abs(thin))>.01


def test_actual_structured_lp_cartesian_integral_matches_polar_power():
    with fixture() as d:
        for name in ['reference','device']:
            pairs=d[name+'_pair'];w=d[name+'_weights'];powers=[]
            for f in pairs:
                x,dx,cart=centered_lp_to_cartesian(d['r_um'],f,float(d['cell_width_um']),512,128)
                xx,yy=np.meshgrid(x,x,indexing='xy');disk=np.hypot(xx,yy)<float(d['aperture_radius_um'])
                powers.append(float((flux(cart)*disk).sum()*dx**2))
            polar=paired_centered_finite_volume_power(d['r_um'],pairs,float(d['cell_width_um']),float(d['aperture_radius_um']),w)
            assert float(w@powers)==pytest.approx(polar,rel=.005)


@pytest.mark.parametrize('mode',[0,32,64,80])
def test_oblique_te_forward_backward_power_and_axial_paired_products(mode):
    # Analytic continuum plane waves exercise cos(theta) and separation.
    # They do not certify a general inverse of Yee angular interpolation.
    count=256;window=128.;dx=window/count;wavelength=1.55
    s=mode*wavelength/window;c=np.sqrt(1-s*s);phase=np.exp(2j*np.pi*mode*np.arange(count)/count)
    plane=np.broadcast_to(phase,(count,count));f=np.zeros((6,count,count),complex)
    A=1.;B=.3j;f[1]=(A+B)*plane;f[3]=c*(-A+B)*plane;f[5]=s*(A+B)*plane
    plus=propagate(forward_spectrum(f,dx,wavelength),dx,wavelength,0)
    flip=f.copy();flip[3:]*=-1
    minus=propagate(forward_spectrum(flip,dx,wavelength),dx,wavelength,0)
    assert np.mean(flux(plus))==pytest.approx(.5*c*abs(A)**2,rel=1e-12)
    assert np.mean(flux(minus))==pytest.approx(.5*c*abs(B)**2,rel=1e-12)
    assert np.sum(flux(f))==pytest.approx(np.sum(flux(plus))-np.sum(flux(minus)),rel=1e-12)
    # Averages of complex fields lose cos(kz*Delta/2)^2 even at oblique incidence.
    half_phase=np.pi/wavelength*c/16
    pair=[plus*np.exp(-1j*half_phase),plus*np.exp(1j*half_phase)]
    p=np.mean([flux(v).sum() for v in pair]);wrong=flux(np.mean(pair,axis=0)).sum()
    assert wrong/p==pytest.approx(np.cos(half_phase)**2,rel=1e-12)


def test_near_grazing_cutoff_is_explicitly_outside_continuation_scope():
    count=256;dx=.5;wavelength=1.55;mode=81;s=mode*wavelength/(count*dx);c=np.sqrt(1-s*s)
    plane=np.broadcast_to(np.exp(2j*np.pi*mode*np.arange(count)/count),(count,count))
    f=np.zeros((6,count,count),complex);f[1]=plane;f[3]=-c*plane;f[5]=s*plane
    assert s>.98 and np.mean(flux(f))>0
    spectrum=forward_spectrum(f,dx,wavelength)
    assert np.max(abs(spectrum))/np.max(abs(np.fft.fft2(f,axes=(-2,-1))))<1e-12
    # An honest validity declaration must reject interpreting this as zero physical power.
