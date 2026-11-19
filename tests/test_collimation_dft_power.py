from pathlib import Path
import numpy as np
import pytest
from fiber_metalens_nn.collimation_dft_power import (circular_sector_to_linear_power,
    paired_centered_net_power,native_flux_quadrature,normal_axis_symbol,paraxial_power_equivalent_fields)
from fiber_metalens_nn.collimation_optics import cylindrical_to_cartesian,flux,forward_spectrum,propagate


def circular_gaussian(r,n=1,amplitude=1):
    g=amplitude*np.exp(-(r/4.5)**2)
    return np.array([g,1j*g,0*g,-1j*n*g,n*g,0*g])


def test_cylindrical_measure_and_lp_half_sector_factor():
    r=np.linspace(0,20,4001);f=circular_gaussian(r,1.45)
    expected=1.45*np.pi*4.5**2/4
    assert circular_sector_to_linear_power(r,f)==pytest.approx(expected,rel=1e-6)
    a=3.;mask=r<=a
    assert circular_sector_to_linear_power(r[mask],f[:,mask])/expected==pytest.approx(1-np.exp(-2*a*a/4.5**2),rel=2e-6)


def test_lp_cartesian_reconstruction_preserves_radial_power():
    r=np.linspace(0,64,5001);f=circular_gaussian(r,1.45)
    x,dx,cart=cylindrical_to_cartesian(r,f,512,64)
    assert flux(cart).sum()*dx**2==pytest.approx(circular_sector_to_linear_power(r,f),rel=1e-5)


def test_average_of_products_is_not_product_of_averages():
    r=np.linspace(0,20,2001);c=normal_axis_symbol(1.45,1.55,6,.45);delta=np.arccos(c)
    centered=circular_gaussian(r,1.45);centered[:3]*=c
    pair=np.array([centered*np.exp(-1j*delta),centered*np.exp(1j*delta)])
    true=paired_centered_net_power(r,pair)
    collapsed=circular_sector_to_linear_power(r,pair.mean(axis=0))
    assert collapsed/true==pytest.approx(c*c,rel=1e-12)


def test_two_medium_known_flux_reference_reproduces_false_gain_and_repair():
    r=np.linspace(0,20,2001);ci=normal_axis_symbol(1.45,1.55,6,.45);co=normal_axis_symbol(1,1.55,6,.45)
    def pair(n,c,amplitude):
        f=circular_gaussian(r,n,amplitude);f[:3]*=c
        return np.array([f*np.exp(-1j*np.arccos(c)),f*np.exp(1j*np.arccos(c))])
    incoming=pair(1.45,ci,1);outgoing=pair(1,co,np.sqrt(1.45*ci/co))
    assert paired_centered_net_power(r,outgoing)/paired_centered_net_power(r,incoming)==pytest.approx(1,rel=1e-12)
    wrong=circular_sector_to_linear_power(r,outgoing.mean(axis=0))/circular_sector_to_linear_power(r,incoming.mean(axis=0))
    assert wrong>1.16
    assert wrong*ci**2/co**2==pytest.approx(1,rel=1e-12)


def test_paraxial_reconstruction_removes_spurious_backward_power():
    n=1.45;c=normal_axis_symbol(n,1.55,6,.45)
    f=np.zeros((6,32,32),complex);f[0]=c*c;f[4]=n*c
    raw_minus=f.copy();raw_minus[3:]*=-1
    assert np.linalg.norm(forward_spectrum(raw_minus,.5,1.55,n=n))>0
    corrected=paraxial_power_equivalent_fields(f,n,1.55,6,.45,symmetric_axial_collapse=True)
    flipped=corrected.copy();flipped[3:]*=-1
    assert np.max(abs(forward_spectrum(flipped,.5,1.55,n=n)))<1e-10
    assert np.mean(corrected[4]/corrected[0])==pytest.approx(n)


def test_net_flux_is_forward_minus_backward_not_forward_alone():
    n=1.45;c=normal_axis_symbol(n,1.55,6,.45);A=1.;B=.3j
    f=np.zeros((6,32,32),complex);f[0]=(A+B)*c*c;f[4]=n*(A-B)*c
    f=paraxial_power_equivalent_fields(f,n,1.55,6,.45,symmetric_axial_collapse=True)
    plus=propagate(forward_spectrum(f,.5,1.55,n=n),.5,1.55,0,n=n)
    flipped=f.copy();flipped[3:]*=-1
    minus=propagate(forward_spectrum(flipped,.5,1.55,n=n),.5,1.55,0,n=n)
    assert flux(f).sum()==pytest.approx(flux(plus).sum()-flux(minus).sum(),rel=1e-12)
    assert flux(plus).sum()>flux(f).sum()


def test_unknown_alignment_and_unresolved_grid_fail_closed():
    with pytest.raises(ValueError,match="alignment"):
        paraxial_power_equivalent_fields(np.zeros((6,8,8)),1,1.55,6,.45,symmetric_axial_collapse=False)
    with pytest.raises(ValueError,match="unresolved"):
        normal_axis_symbol(3,1.55,1,.45)


def fixture():
    return np.load(Path(__file__).parent/'data/power_pair_h34_r6.npz',allow_pickle=False)


def test_actual_fluxdata_contraction_matches_meep_without_extra_factors():
    with fixture() as d:
        for label in ['reference','device']:
            actual=native_flux_quadrature(d[label+'_flux_E'],d[label+'_flux_H'])
            assert actual==pytest.approx(float(d[label+'_native_flux']),rel=1e-12)


def test_actual_centered_pair_repairs_normalized_net_transmission():
    with fixture() as d:
        ref=paired_centered_net_power(d['r_um'],d['reference_pair'])
        dev=paired_centered_net_power(d['r_um'],d['device_pair'])
        native=float(d['device_native_flux']/d['reference_native_flux'])
        assert dev/ref==pytest.approx(native,rel=.001)
        wrong=circular_sector_to_linear_power(d['r_um'],d['device_thin'])/circular_sector_to_linear_power(d['r_um'],d['reference_thin'])
        assert wrong/native>1.15
        assert np.max(abs(d['device_pair'].mean(axis=0)-d['device_thin']))/np.max(abs(d['device_thin']))<1e-4
