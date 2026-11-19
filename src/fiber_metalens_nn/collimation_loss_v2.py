"""Versioned five-plane 1/e^2 collimation loss; no optical success is implied.

For I=I0*exp(-2*x^2/w^2), D_exp2=2*w=4*sigma=D4sigma.
For non-Gaussian profiles these widths can differ substantially. The loss uses
the same connected 1/e^2 central-line crossings as acceptance, and reports
centered flux second moments separately. It does not relabel D4sigma as an
observed 1/e^2 diameter.
"""
from pathlib import Path
import jax
import jax.numpy as jnp
from .collimation_optics import kernels

VERSION = "five_plane_exp2_v2"


def source_text():
    """Include this loss implementation in checkpoint compatibility checks."""
    return Path(__file__).read_text()


def exp2_width(x, intensity):
    """Piecewise differentiable crossings of the component about global maximum.

    Crossing locations and peak index are discrete, as in the acceptance
    routine. Within a fixed crossing pair, interpolation is differentiable.
    Profiles touching a boundary return valid=False and a finite placeholder;
    callers must retain the explicit invalid-profile penalty.
    """
    count = intensity.shape[0]
    index = jnp.arange(count)
    peak = jnp.argmax(intensity)
    threshold = intensity[peak] * jnp.exp(jnp.asarray(-2., intensity.dtype))
    below = intensity < threshold
    left = jnp.max(jnp.where((index < peak) & below, index, -1))
    right = jnp.min(jnp.where((index > peak) & below, index, count))
    valid = (left >= 0) & (right < count) & (intensity[peak] > 0)
    left = jnp.clip(left, 0, count-2)
    right = jnp.clip(right, 1, count-1)
    def crossing(a,b):
        difference = intensity[b]-intensity[a]
        safe = jnp.where(jnp.abs(difference)>1e-20,difference,jnp.where(difference>=0,1e-20,-1e-20))
        return x[a]+(threshold-intensity[a])*(x[b]-x[a])/safe
    return crossing(right-1,right)-crossing(left,left+1), valid


def plane_widths(fields, x):
    intensity = jnp.sum(jnp.abs(fields[:3])**2, axis=0)
    count = fields.shape[-1]
    middle = count//2
    ix = intensity[middle-1:middle+1].mean(axis=0)
    iy = intensity[:,middle-1:middle+1].mean(axis=1)
    wx, vx = exp2_width(x,ix)
    wy, vy = exp2_width(x,iy)
    sz = jnp.maximum(.5*jnp.real(fields[0]*jnp.conj(fields[4])-fields[1]*jnp.conj(fields[3])),0)
    norm = jnp.maximum(sz.sum(),1e-20)
    xx,yy = jnp.meshgrid(x,x,indexing="xy")
    mx,my = (sz*xx).sum()/norm,(sz*yy).sum()/norm
    d4x = 4*jnp.sqrt(jnp.maximum((sz*(xx-mx)**2).sum()/norm,1e-20))
    d4y = 4*jnp.sqrt(jnp.maximum((sz*(yy-my)**2).sum()/norm,1e-20))
    return jnp.stack([wx,wy]),jnp.stack([vx,vy]),jnp.stack([d4x,d4y])


def stability_change(widths):
    """Maximum absolute x/y change over every declared plane, relative to plane 0."""
    return jnp.max(jnp.abs(widths/jnp.maximum(widths[0],1e-12)-1))


def objective_terms(proxy, heights, pedestal, target, diameter):
    cfg = proxy.config
    if cfg["loss"].get("version") != VERSION:
        raise ValueError("five-plane loss requires explicit versioned configuration")
    planes = tuple(cfg["evaluation"]["processing_planes_um"])
    if planes != tuple(cfg["loss"]["stability_planes_um"]) or not planes or planes[0] != 0:
        raise ValueError("training stability planes must exactly match acceptance, starting at output")
    if cfg["loss"].get("width_definition") != "xy_connected_1_over_e2_intensity":
        raise ValueError("width definition must be declared explicitly")
    spectrum = proxy.spectrum(heights,pedestal)
    def inner(a,b):
        return .25*jnp.sum(a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+
                jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*proxy.dx**2/proxy.count**2
    useful = jnp.abs(inner(spectrum,target))**2/(jnp.real(inner(target,target))*proxy.incident)
    _,_,kz,_,valid,_ = kernels(proxy.count,proxy.dx,cfg["wavelength_um"],xp=jnp)
    x = jnp.asarray(proxy.x,jnp.float32)
    @jax.checkpoint
    def measure(z):
        field = jnp.fft.ifft2(spectrum*(jnp.exp(1j*kz*z)*valid)[None],axes=(-2,-1))
        return plane_widths(field,x)
    data = [measure(jnp.asarray(z,jnp.float32)) for z in planes]
    widths = jnp.stack([item[0] for item in data])
    validity = jnp.stack([item[1] for item in data])
    moments = jnp.stack([item[2] for item in data])
    change = stability_change(widths)
    loss = cfg["loss"]
    terms = jnp.stack([-loss["mode"]*useful,
        loss["diameter"]*jnp.mean((widths[0]/diameter-1)**2),
        loss["stability"]*change**2,
        loss["roughness"]*jnp.mean(jnp.diff(heights)**2),
        loss["invalid_profile"]*jnp.mean((~validity).astype(jnp.float32))])
    return dict(total=terms.sum(),weighted_terms=terms,useful_mode_power=useful,
                exp2_widths_um=widths,valid_crossings=validity,D4sigma_widths_um=moments,
                maximum_interval_change=change)


def objective(proxy,heights,pedestal,target,diameter):
    return objective_terms(proxy,heights,pedestal,target,diameter)["total"]
