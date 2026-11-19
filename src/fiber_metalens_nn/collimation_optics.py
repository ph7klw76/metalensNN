"""Power-normalized vector angular spectra and collimation metrics (NumPy).

E and Z0-scaled H use exp(-i omega t). No GPU imports: also used by Meep.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .metrics import full_width


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def grid(count, window):
    dx = window/count
    x = (np.arange(count)-count/2+0.5)*dx
    return x, dx


def kernels(count, dx, wavelength, n=1.0, xp=np):
    k0 = 2*xp.pi/wavelength
    k = n*k0
    f = 2*xp.pi*xp.fft.fftfreq(count, d=dx)
    kx, ky = xp.meshgrid(f, f, indexing="xy")
    valid = kx*kx+ky*ky < (0.98*k)**2
    kz = xp.sqrt(xp.maximum(k*k-kx*kx-ky*ky, 0))
    safe = xp.where(valid, kz, k)
    return kx, ky, kz, safe, valid, k0


def vector_spectrum(scalar, dx, wavelength, n=1.0, xp=np):
    """Project x polarization transverse to k, then derive H=khat cross E*n."""
    kx, ky, kz, safe, valid, k0 = kernels(scalar.shape[-1], dx, wavelength, n, xp)
    direction = xp.stack([kx, ky, kz])/(n*k0)
    electric = xp.stack([1-direction[0]**2, -direction[0]*direction[1],
                         -direction[0]*direction[2]])*scalar[None]*valid[None]
    magnetic = n*xp.cross(direction, electric, axisa=0, axisb=0, axisc=0)
    return xp.concatenate([electric, magnetic], axis=0)


def forward_spectrum(fields, dx, wavelength, n=1.0):
    """Separate forward waves using both transverse E and H and admittance."""
    spectrum = np.fft.fft2(fields, axes=(-2,-1))
    kx, ky, kz, safe, valid, k0 = kernels(fields.shape[-1], dx, wavelength, n)
    # inv(Y) = (I - kt kt^T/k^2) / (kz/k0), with JH=(Hy,-Hx).
    jx, jy = spectrum[4], -spectrum[3]
    dot = (kx*jx+ky*jy)/(n*k0)**2
    ex = .5*(spectrum[0]+k0/safe*(jx-kx*dot))*valid
    ey = .5*(spectrum[1]+k0/safe*(jy-ky*dot))*valid
    ez = -(kx*ex+ky*ey)/safe
    electric = np.stack([ex,ey,ez])*valid[None]
    direction = np.stack([kx,ky,kz])/(n*k0)
    magnetic = n*np.cross(direction,electric,axisa=0,axisb=0,axisc=0)
    return np.concatenate([electric,magnetic],axis=0)


def propagate(spectrum, dx, wavelength, z, n=1.0):
    _, _, kz, _, valid, _ = kernels(spectrum.shape[-1],dx,wavelength,n)
    return np.fft.ifft2(spectrum*(np.exp(1j*kz*z)*valid)[None],axes=(-2,-1))


def flux(fields):
    return .5*np.real(fields[0]*np.conj(fields[4])-fields[1]*np.conj(fields[3]))


def inner(a,b,dx):
    return .25*np.sum(a[0]*np.conj(b[4])-a[1]*np.conj(b[3])+
                      np.conj(b[0])*a[4]-np.conj(b[1])*a[3])*dx**2


def overlap(a,b,dx):
    pa, pb = float(inner(a,a,dx).real), float(inner(b,b,dx).real)
    return float(abs(inner(a,b,dx))**2/max(pa*pb,1e-30))


def cylindrical_to_cartesian(r, cylindrical, count=512, window=128):
    """m=+1 and reflected m=-1, half-sum reconstructs x-linear fields.

    Even components: Er, Ez, Hp; odd components: Ep, Hr, Hz.
    Circular-sector / linear power factor is 2, and cancels in flux ratios.
    """
    x, dx = grid(count,window)
    xx,yy = np.meshgrid(x,x,indexing="xy")
    rr = np.hypot(xx,yy)
    c, s = xx/np.maximum(rr,1e-30), yy/np.maximum(rr,1e-30)
    values = [np.interp(rr,r,row.real,left=row.real[0],right=0)+
              1j*np.interp(rr,r,row.imag,left=row.imag[0],right=0) for row in cylindrical]
    er,ep,ez,hr,hp,hz = values
    er,ep,ez,hr,hp,hz = er*c,1j*ep*s,ez*c,1j*hr*s,hp*c,1j*hz*s
    return x, dx, np.stack([er*c-ep*s,er*s+ep*c,ez,hr*c-hp*s,hr*s+hp*c,hz])


def describe(spectrum, dx, wavelength, diameter, config, transmission=1.0, save=None):
    count = spectrum.shape[-1]
    x,_ = grid(count,count*dx)
    xx,yy = np.meshgrid(x,x,indexing="xy")
    target = vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(diameter/2)**2)),dx,wavelength)
    target_field = propagate(target,dx,wavelength,0)
    planes = config["evaluation"]["processing_planes_um"]+config["evaluation"]["diagnostic_planes_um"]
    reports, snapshots = [], []
    p0 = None
    for z in planes:
        field = propagate(spectrum,dx,wavelength,z)
        intensity = np.sum(abs(field[:3])**2,axis=0)
        power = float(np.sum(flux(field))*dx**2)
        if p0 is None:
            p0 = power
        middle=count//2
        ix=intensity[middle-1:middle+1].mean(axis=0)
        iy=intensity[:,middle-1:middle+1].mean(axis=1)
        weight=np.maximum(flux(field),0)
        norm=max(weight.sum(),1e-30)
        mx,my=(weight*xx).sum()/norm,(weight*yy).sum()/norm
        d4x=4*np.sqrt((weight*(xx-mx)**2).sum()/norm)
        d4y=4*np.sqrt((weight*(yy-my)**2).sum()/norm)
        edge=float(weight[(abs(xx)>.45*count*dx)|(abs(yy)>.45*count*dx)].sum()/norm)
        reports.append(dict(z_um=z,diameter_x_um=full_width(x,ix,np.exp(-2)),
                            diameter_y_um=full_width(x,iy,np.exp(-2)),D4sigma_x_um=float(d4x),
                            D4sigma_y_um=float(d4y),power_relative_to_output=power/max(p0,1e-30),edge_power_fraction=edge))
        if z in (0,100):
            snapshots.append(intensity.astype(np.float32))
    f0=propagate(spectrum,dx,wavelength,0)
    mode=overlap(f0,target_field,dx)
    interval=reports[:len(config["evaluation"]["processing_planes_um"])]
    widths=np.asarray([[p["diameter_x_um"] or np.nan,p["diameter_y_um"] or np.nan] for p in interval])
    stability=float(np.nanmax(abs(widths/widths[0]-1))) if np.isfinite(widths).all() else None
    # Far-field second-moment angle about its centroid; width here is 1/e2
    # equivalent 2 sigma per Cartesian axis, not a claimed Gaussian fit.
    kx,ky,kz,safe,valid,k0=kernels(count,dx,wavelength)
    angular=.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3]))
    angular=np.maximum(angular,0)*valid
    angular/=max(angular.sum(),1e-30)
    theta=[]
    for transverse in (kx,ky):
        slope=np.arctan2(transverse,safe)
        mean=(angular*slope).sum()
        theta.append(float(2*np.sqrt((angular*(slope-mean)**2).sum())))
    # E_x phase curvature fit only on the connected bright central region.
    phase=np.unwrap(np.angle(f0[0,count//2]))
    bright=(abs(x)<diameter/2)&(abs(f0[0,count//2])>.15*abs(f0[0]).max())
    curvature=None
    if bright.sum()>5:
        coef=np.polyfit(x[bright],phase[bright],2)
        curvature=float(2*coef[0]/k0)
    criterion=config["evaluation"]
    gates={"diameter":bool(np.isfinite(widths[0]).all() and np.all(widths[0]>=criterion["diameter_bounds_um"][0]) and np.all(widths[0]<=criterion["diameter_bounds_um"][1])),
           "stability":bool(stability is not None and stability<=criterion["max_diameter_change"]),
           "mode_overlap":mode>=criterion["minimum_overlap"],
           "absolute_efficiency":transmission*mode>=criterion["minimum_eta_mode"]}
    # Downstream periodic-window pollution is surfaced, never hidden by normalization.
    result=dict(planes=reports,overlap=mode,transmission=float(transmission),eta_mode=float(transmission*mode),
                maximum_diameter_change_100um=stability,divergence_half_angle_rad=theta,
                inverse_wavefront_radius_per_um=curvature,criteria=gates,optical_targets_pass=all(gates.values()),
                M2=None,M2_note="Not claimed: the long-range caustic may exceed the saved finite window; use convergence-tested second-moment fit before reporting M2.",
                width_definition="1/e^2 central-line intensity crossings; D4sigma uses positive axial flux",
                normalization="transmitted / launched fiber-surrogate power; vector power overlap via E/H reciprocal inner product")
    if save:
        np.savez_compressed(save,x_um=x,intensity_output=snapshots[0],intensity_100um=snapshots[1],
                            output_Ex=f0[0].astype(np.complex64))
    return result
