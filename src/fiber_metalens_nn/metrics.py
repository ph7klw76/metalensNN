"""Fixed definitions shared by proxy and independent Maxwell analysis."""
import numpy as np


def full_width(x, intensity, fraction):
    x, intensity = np.asarray(x), np.asarray(intensity)
    center = int(np.argmax(intensity))
    threshold = intensity[center]*fraction
    left, right = center, center
    while left > 0 and intensity[left] >= threshold:
        left -= 1
    while right < len(x)-1 and intensity[right] >= threshold:
        right += 1
    if left == 0 and intensity[left] >= threshold or right == len(x)-1 and intensity[right] >= threshold:
        return None
    xl = np.interp(threshold, intensity[left:left+2], x[left:left+2])
    xr = np.interp(threshold, intensity[right-1:right+1][::-1], x[right-1:right+1][::-1])
    return float(xr-xl)


def peak_position(z, intensity):
    z, intensity = np.asarray(z), np.asarray(intensity)
    i = int(np.argmax(intensity))
    if 0 < i < len(z)-1:
        coef = np.polyfit(z[i-1:i+2], intensity[i-1:i+2], 2)
        value = -coef[1]/(2*coef[0])
        if coef[0] < 0 and z[i-1] <= value <= z[i+1]:
            return float(value), False
    return float(z[i]), i in (0, len(z)-1)


def width_metrics(x, ix, iy):
    return {"fwhm_x_um":full_width(x,ix,0.5), "fwhm_y_um":full_width(x,iy,0.5),
            "exp_minus_2_width_x_um":full_width(x,ix,np.exp(-2)),
            "exp_minus_2_width_y_um":full_width(x,iy,np.exp(-2))}
