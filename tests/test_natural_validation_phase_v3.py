"""Estimator definition regressions; analytic fields, no solver/training."""
import numpy as np
import pytest
from fiber_metalens_nn.natural_validation_metrics_v3 import phase_current_curvature

@pytest.mark.parametrize('inverse_R',[0.,.001,-.002])
def test_phase_current_recovers_known_quadratic_wavefront(inverse_R):
    count=256;dx=.5;wavelength=1.55;D=10.;x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x)
    field=np.exp(-(xx*xx+yy*yy)/25+1j*np.pi/wavelength*inverse_R*(xx*xx+yy*yy))
    s=np.zeros((6,count,count),complex);s[0]=np.fft.fft2(field)
    m=phase_current_curvature(s,dx,wavelength,D)
    assert m['inverse_R_per_um']==pytest.approx(inverse_R,abs=1e-12)

def test_phase_current_is_invariant_to_global_phase_and_amplitude():
    count=128;dx=.5;x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x)
    field=np.exp(-(xx*xx+yy*yy)/25+1j*.01*(xx*xx+yy*yy));s=np.zeros((6,count,count),complex);s[0]=np.fft.fft2(field)
    a=phase_current_curvature(s,dx,1.55,10)
    b=phase_current_curvature(s*.1*np.exp(1.3j),dx,1.55,10)
    assert a['inverse_R_per_um']==pytest.approx(b['inverse_R_per_um'],abs=1e-12)
