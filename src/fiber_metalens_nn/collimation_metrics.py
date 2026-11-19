"""Validated Gaussian-core divergence fit, distinct from all-angle moments."""
import numpy as np
from .collimation_optics import kernels


def fit_angular_gaussian(theta,intensity):
    theta=np.asarray(theta)
    intensity=np.asarray(intensity)
    peak=int(np.argmax(intensity))
    if intensity[peak]<=0:
        return dict(valid=False,reason="nonpositive angular intensity")
    threshold=intensity[peak]*np.exp(-2)
    left=right=peak
    while left>0 and intensity[left-1]>=threshold: left-=1
    while right<len(intensity)-1 and intensity[right+1]>=threshold: right+=1
    # Include the crossing neighbors; the fit interval is fixed by the core's
    # 1/e^2 contour, never adjusted to force mesh convergence.
    left=max(left-1,0);right=min(right+1,len(intensity)-1)
    t=theta[left:right+1]
    y=np.log(np.maximum(intensity[left:right+1]/intensity[peak],1e-30))
    if len(t)<5 or len(np.unique(t))<5:
        return dict(valid=False,reason="fewer than five angular samples in core; increase propagation window")
    a,b,c=np.polyfit(t,y,2)
    residual=float(np.sqrt(np.mean((np.polyval([a,b,c],t)-y)**2)))
    if a>=0 or residual>.05:
        return dict(valid=False,reason="core is not Gaussian to declared 0.05 log-intensity RMS",log_intensity_RMS=residual)
    return dict(valid=True,half_angle_rad=float(np.sqrt(-2/a)),centroid_angle_rad=float(-b/(2*a)),
                log_intensity_RMS=residual,sample_count=len(t),fit_definition="quadratic log-intensity over contiguous 1/e^2 core plus crossing neighbors")


def fitted_divergence(spectrum,dx,wavelength):
    count=spectrum.shape[-1]
    kx,ky,kz,safe,valid,k0=kernels(count,dx,wavelength)
    power=np.maximum(.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])),0)
    # d^2 kt = k0^2 cos(theta) dOmega, so multiply by cos(theta).
    angular=np.fft.fftshift(power*kz/k0)
    k=np.fft.fftshift(2*np.pi*np.fft.fftfreq(count,d=dx))
    keep=abs(k)<.98*k0
    theta=np.arcsin(k[keep]/k0)
    middle=count//2
    fits=[fit_angular_gaussian(theta,angular[middle,keep]),fit_angular_gaussian(theta,angular[keep,middle])]
    return dict(valid=all(f["valid"] for f in fits),axes=fits)


def attach_divergence_fit(metric,spectrum,dx,wavelength):
    fit=fitted_divergence(spectrum,dx,wavelength)
    metric["second_moment_divergence_half_angle_rad"]=metric["divergence_half_angle_rad"]
    metric["fitted_divergence"]=fit
    if fit["valid"]:
        metric["divergence_half_angle_rad"]=[axis["half_angle_rad"] for axis in fit["axes"]]
    metric["divergence_definition"]="Gaussian core fitted over its contiguous 1/e^2 contour plus crossing neighbors; all-angle second moments retained separately"
    return metric
