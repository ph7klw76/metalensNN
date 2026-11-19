import copy
import json
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from fiber_metalens_nn.collimation_loss_v2 import exp2_width,plane_widths,stability_change,objective_terms
from fiber_metalens_nn.collimation_optics import grid,vector_spectrum,propagate
from fiber_metalens_nn.metrics import full_width
from fiber_metalens_nn.collimation_proxy import CylinderProxy


def config():
    default=Path(__file__).resolve().parents[1]/"configs/1550nm_collimation_loss_v2.json"
    return json.loads(Path(os.environ.get("METALENS_V2_CONFIG",default)).read_text())


def test_gaussian_D4sigma_matches_reported_exp2_diameter():
    x,dx=grid(256,96)
    xx,yy=np.meshgrid(x,x,indexing="xy")
    spectrum=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/225)),dx,1.55)
    # Use exact Gaussian E/H fixture to isolate width definitions from vector projection.
    g=np.exp(-(xx*xx+yy*yy)/225)
    field=np.zeros((6,256,256),complex);field[0]=g;field[4]=g
    widths,valid,moments=plane_widths(jnp.asarray(field),jnp.asarray(x))
    assert np.asarray(valid).all()
    np.testing.assert_allclose(widths,[30,30],rtol=.002)
    np.testing.assert_allclose(moments,[30,30],rtol=1e-5)
    np.testing.assert_allclose(widths,moments,rtol=.002)


def test_29_percent_interior_counterexample_is_penalized():
    zr=50.;w0=np.sqrt(1.55*zr/np.pi)
    widths=np.array([[2*w0*np.sqrt(1+((z-50)/zr)**2)]*2 for z in [0,25,50,75,100]])
    assert abs(widths[-1,0]/widths[0,0]-1)<1e-12
    assert float(stability_change(jnp.asarray(widths)))==pytest.approx(1-1/np.sqrt(2),rel=1e-6)


def test_non_gaussian_halo_does_not_get_relabelled_as_exp2_width():
    x=np.linspace(-50,50,2001)
    intensity=np.exp(-2*x*x/25)+.02*np.exp(-2*(x-25)**2/4)+.02*np.exp(-2*(x+25)**2/4)
    width,valid=exp2_width(jnp.asarray(x),jnp.asarray(intensity))
    sigma=np.sqrt(np.sum(intensity*x*x)/intensity.sum())
    assert bool(valid)
    assert float(width)==pytest.approx(10,rel=.001)
    assert 4*sigma>float(width)*1.5
    assert float(width)==pytest.approx(full_width(x,intensity,np.exp(-2)),rel=1e-5)


def test_disconnected_lobes_match_existing_crossing_definition():
    x=np.linspace(-30,30,1201)
    intensity=np.exp(-2*x*x/25)+.4*np.exp(-2*(x-20)**2/4)
    width,valid=exp2_width(jnp.asarray(x),jnp.asarray(intensity))
    assert bool(valid)
    assert float(width)==pytest.approx(full_width(x,intensity,np.exp(-2)),rel=1e-5)


def test_boundary_truncated_and_zero_profiles_are_invalid():
    x=jnp.linspace(-2,2,64)
    for intensity in [jnp.ones(64),jnp.zeros(64),jnp.exp(-x*x/100)]:
        width,valid=exp2_width(x,intensity)
        assert not bool(valid)
        assert np.isfinite(float(width))


def test_crossing_derivative_matches_finite_difference():
    # Resolve interpolation sufficiently for the continuum derivative check;
    # the separate finite-difference assertion tests the discrete derivative.
    x=jnp.linspace(-30,30,4001)
    def width(w):return exp2_width(x,jnp.exp(-2*x*x/w**2))[0]
    w=5.07;eps=.001
    ad=float(jax.grad(width)(w));fd=float((width(w+eps)-width(w-eps))/(2*eps))
    assert ad==pytest.approx(fd,rel=.005,abs=.003)
    assert ad==pytest.approx(2,rel=.01)


def test_versioned_loss_uses_five_planes_and_finite_geometry_gradient():
    cfg=config();cfg["proxy"].update(grid_n=64,padding=1,window_um=64,pedestal_dz_um=2,relief_slices=6)
    proxy=CylinderProxy(cfg)
    heights=jnp.linspace(.17,2.83,cfg["annuli"])
    pedestal=proxy.pedestal(34.);target=proxy.target(28.)
    terms=objective_terms(proxy,heights,pedestal,target,28.)
    assert terms["exp2_widths_um"].shape==(5,2)
    assert np.asarray(terms["valid_crossings"]).all()
    loss,grad=proxy.value_gradient(heights,pedestal,target,28.)
    assert np.isfinite(float(loss)) and np.isfinite(np.asarray(grad)).all()
    assert float(loss)==pytest.approx(float(terms["total"]),rel=1e-5)
    i=2;eps=.005;direction=jnp.eye(cfg["annuli"])[i]*eps
    fd=float((proxy.objective(heights+direction,pedestal,target,28.)-proxy.objective(heights-direction,pedestal,target,28.))/(2*eps))
    assert float(grad[i])==pytest.approx(fd,rel=.05,abs=.001)


def test_mismatched_training_and_acceptance_planes_fail_closed():
    cfg=config();cfg["loss"]["stability_planes_um"]=[0,100]
    cfg["proxy"].update(grid_n=32,padding=1,window_um=64,pedestal_dz_um=4,relief_slices=3)
    proxy=CylinderProxy(cfg)
    with pytest.raises(ValueError,match="exactly match acceptance"):
        objective_terms(proxy,jnp.ones(cfg["annuli"]),proxy.pedestal(34),proxy.target(28),28)


def test_acceptance_thresholds_unchanged_and_legacy_loss_preserved():
    default=Path(__file__).resolve().parents[1]/"configs/1550nm_collimation.json"
    cfg=config();old=json.loads(Path(os.environ.get("METALENS_OLD_CONFIG",default)).read_text())
    assert cfg["evaluation"]==old["evaluation"]
    assert old["loss"].get("version") is None
    old["proxy"].update(grid_n=32,padding=1,window_um=64,pedestal_dz_um=4,relief_slices=3)
    proxy=CylinderProxy(old);h=jnp.linspace(.1,2.9,old["annuli"]);p=proxy.pedestal(34);t=proxy.target(28)
    value=float(proxy.objective(h,p,t,28))
    assert np.isfinite(value)
    # Independently preserve the old endpoint/radial-moment expression.
    s=proxy.spectrum(h,p)
    def inner(a,b):return .25*jnp.sum(a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*proxy.dx**2/proxy.count**2
    useful=abs(inner(s,t))**2/(jnp.real(inner(t,t))*proxy.incident)
    widths=[]
    for z in [0,100]:
        f=propagate(np.asarray(s),proxy.dx,old["wavelength_um"],z)
        sz=np.maximum(.5*np.real(f[0]*np.conj(f[4])-f[1]*np.conj(f[3])),0)
        widths.append(2*np.sqrt(2*np.sum(sz*np.asarray(proxy.r)**2)/sz.sum()))
    expected=-useful+.15*(widths[0]/28-1)**2+.1*(widths[1]/widths[0]-1)**2+.002*jnp.mean(jnp.diff(h)**2)
    assert value==pytest.approx(float(expected),rel=1e-5,abs=1e-6)
