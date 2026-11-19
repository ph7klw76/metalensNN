"""Explicit power conventions for centered/collapsed cylindrical DFT exports.

Primary correction: retain neighboring centered planes and average their
Poynting products, rather than multiply already collapsed E/H averages.
Native FluxData contraction is a separate exact monitor-quadrature audit.

The optional normal-axis formula is a paraxial reconstruction diagnostic;
it does not replace full-angle field/grid convergence or missing metadata.
"""
import math
import numpy as np


def circular_sector_to_linear_power(r_um,fields):
    """Net LP power for centered x-linear m=+1/reflected-m=-1 reconstruction.

    Fourier convention is common to numerator/reference. A real-phasor
    Poynting factor 1/2 and a circular-sector to LP factor 1/2 give 1/4.
    Cylindrical area measure is 2*pi*r*dr. Output is DFT power units, not W
    without a physical source calibration. Flux ratios cancel that scale.
    """
    r=np.asarray(r_um,dtype=float);f=np.asarray(fields)
    if r.ndim!=1 or f.shape[-2:]!=(6,len(r)) or np.any(np.diff(r)<=0) or r[0]<0:
        raise ValueError("six cylindrical components on increasing nonnegative radial coordinates required")
    integrand=np.real(f[...,0,:]*np.conj(f[...,4,:])-f[...,1,:]*np.conj(f[...,3,:]))
    return .25*2*np.pi*np.trapezoid(integrand*r,r,axis=-1)


def paired_centered_net_power(r_um,paired_fields,weights=(.5,.5)):
    """Multiply before axial collapse. Weights must come from monitor geometry."""
    f=np.asarray(paired_fields);w=np.asarray(weights,float)
    if f.ndim!=3 or f.shape[0]!=len(w) or np.any(w<0) or not np.isclose(w.sum(),1):
        raise ValueError("one nonnegative normalized axial weight per centered plane required")
    return float(np.dot(w,circular_sector_to_linear_power(r_um,f)))


def centered_radial_cell_areas(r_um,cell_width_um,aperture_radius_um):
    """Finite-volume disk weights for verified axis-origin centered samples.

    Coordinates must be (j+1/2)*Delta, obtained from Meep grid metadata or
    position quadrature, not inferred by linspace over the requested extent.
    The export may contain an extra cell beyond the aperture. Clip its cell
    boundaries geometrically; do not shift or renormalize field amplitudes.
    """
    r=np.asarray(r_um,float);delta=float(cell_width_um);a=float(aperture_radius_um)
    if (r.ndim!=1 or len(r)<2 or not np.isfinite(r).all() or
        not math.isfinite(delta) or delta<=0 or not math.isfinite(a) or a<=0):
        raise ValueError("finite centered radial coordinates, positive cell width and aperture required")
    if not np.allclose(r,(np.arange(len(r))+.5)*delta,rtol=1e-10,atol=1e-12):
        raise ValueError("verified axis-origin half-cell coordinates required; legacy linspace labels are invalid")
    if r[-1]+delta/2<a-1e-12:
        raise ValueError("radial samples do not cover the aperture")
    lo=np.maximum(r-delta/2,0);hi=np.minimum(r+delta/2,a)
    return np.pi*np.maximum(hi**2-lo**2,0)


def paired_centered_finite_volume_power(r_um,paired_fields,cell_width_um,aperture_radius_um,weights):
    """LP net power: collocate, form Poynting, then average axial planes.

    A continuum finite-volume approximation to native Yee flux; regression
    against native signed quadrature is required for each new grid/geometry
    class. This is not an exact inverse of spatial Yee interpolation.
    """
    f=np.asarray(paired_fields);w=np.asarray(weights,float)
    areas=centered_radial_cell_areas(r_um,cell_width_um,aperture_radius_um)
    if (f.shape!=(len(w),6,len(areas)) or not np.isfinite(f).all() or
        np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-12)):
        raise ValueError("finite six-component planes and normalized nonnegative geometric weights required")
    density=np.real(f[:,0]*np.conj(f[:,4])-f[:,1]*np.conj(f[:,3]))
    return float(.25*np.dot(w,density@areas))


def linear_cartesian_on_polar(cylindrical,azimuth_rad):
    """Half-sum m=+1/reflected m=-1 for the centered x-linear source only.

    Component order Er,Ep,Ez,Hr,Hp,Hz; output shape (6,azimuth,radius).
    Other source offsets, azimuthal orders, or symmetry breaking require
    their own modal reconstruction, not this half-sum.
    """
    f=np.asarray(cylindrical);phi=np.asarray(azimuth_rad,float)
    if f.ndim!=2 or f.shape[0]!=6 or phi.ndim!=1:
        raise ValueError("six radial component rows and a one-dimensional azimuth array required")
    c=np.cos(phi)[:,None];s=np.sin(phi)[:,None]
    er,ep,ez,hr,hp,hz=f
    er=er*c;ep=1j*ep*s;ez=ez*c;hr=1j*hr*s;hp=hp*c;hz=1j*hz*s
    return np.stack([er*c-ep*s,er*s+ep*c,ez,hr*c-hp*s,hr*s+hp*c,hz])


def centered_lp_to_cartesian(r_um,cylindrical,cell_width_um,count=512,window=128):
    """Use actual centered radii and a regular m=+/-1 axis limit.

    Interpolation is a continuum approximation. Its power must be checked
    against polar integration and repeated at the chosen Cartesian grid.
    The first centered sample estimates the regular axis limit to O(Delta^2).
    """
    from .collimation_optics import cylindrical_to_cartesian
    r=np.asarray(r_um,float);f=np.asarray(cylindrical)
    centered_radial_cell_areas(r,cell_width_um,r[-1]+cell_width_um/2)
    if f.shape!=(6,len(r)) or not np.isfinite(f).all():
        raise ValueError("finite six-component centered radial fields required")
    axis=np.zeros(6,complex)
    axis[0]=.5*(f[0,0]-1j*f[1,0]);axis[1]=1j*axis[0]
    axis[4]=.5*(f[4,0]+1j*f[3,0]);axis[3]=-1j*axis[4]
    return cylindrical_to_cartesian(np.r_[0,r],np.column_stack([axis,f]),count,window)


def native_flux_quadrature(E_weighted,H_unweighted):
    """Meep FluxData E already contains signed spatial weights once.

    No additional 2*pi*r, sign, phasor 1/2 or amplitude fit belongs here.
    This checks the native API power convention, separately from LP fields.
    """
    E,H=np.asarray(E_weighted),np.asarray(H_unweighted)
    if E.shape!=H.shape or E.size==0:raise ValueError("matching nonempty FluxData arrays required")
    return float(np.real(np.sum(E*np.conj(H))))


def normal_axis_symbol(n,wavelength_um,resolution_per_um,courant):
    """cos(k_z,FD*Delta/2) from the 1D Yee dispersion relation.

    Delta=1/resolution and dt=Courant*Delta. The relation is
    sin(k_z,FD*Delta/2)=n*sin(omega*dt/2)/Courant.
    Its value is determined by physical/numerical inputs, never fitted to flux.
    """
    if not all(math.isfinite(v) and v>0 for v in (n,wavelength_um,resolution_per_um,courant)):
        raise ValueError("positive finite numerical and material parameters required")
    argument=n/courant*math.sin(math.pi*courant/(resolution_per_um*wavelength_um))
    if not 0<=argument<1:raise ValueError("normal wave unresolved on this Yee grid")
    return math.sqrt(1-argument**2)


def paraxial_power_equivalent_fields(fields,n,wavelength_um,resolution_per_um,courant,*,symmetric_axial_collapse):
    """Power/impedance-consistent normal-axis approximation for legacy thin slices.

    Requires a homogeneous monitor medium and explicitly known symmetric
    collapse between adjacent centered planes. Transverse E has one native
    Yee averaging factor plus the collapse; transverse H has the collapse.
    Scale E_t,H_z by C^(-3/2), and E_z,H_t by C^(-1/2). The corrected net
    product equals the pre-collapse discrete energy flux, and a normal wave
    has H/E=n. Finite-angle/radial interpolation remains an approximation.
    """
    if not symmetric_axial_collapse:
        raise ValueError("legacy monitor alignment metadata is required; do not guess a correction")
    f=np.array(fields,dtype=complex,copy=True)
    if f.shape[-3]!=6:raise ValueError("component axis -3 must contain Ex,Ey,Ez,Hx,Hy,Hz")
    c=normal_axis_symbol(n,wavelength_um,resolution_per_um,courant)
    f[...,0,:,:]/=c**1.5;f[...,1,:,:]/=c**1.5;f[...,5,:,:]/=c**1.5
    f[...,2,:,:]/=c**.5;f[...,3,:,:]/=c**.5;f[...,4,:,:]/=c**.5
    return f
