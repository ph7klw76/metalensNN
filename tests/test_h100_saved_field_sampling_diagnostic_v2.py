"""Pure analytic Q/C diagnostics; no Meep/JAX or actual saved-field run."""
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h100_sampling_diagnostic',ROOT/'scripts/diagnose_height_family_h100_sampling_v2.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class TestSavedFieldSamplingMath(unittest.TestCase):
    def gaussian(self,count=256,window=128.,diameter=20.):
        dx=window/count;x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x)
        s=diag.vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(diameter/2)**2)),dx,1.55)
        return s,dx
    def test_gaussian_Q_and_absolute_paraxial_limit_across_sampling(self):
        for count,window in [(128,64.),(256,128.),(512,128.)]:
            with self.subTest(count=count,window=window):
                s,dx=self.gaussian(count,window);m=diag.angular_statistics(s,dx,1.55,20.)
                self.assertAlmostEqual(m['Q_independent_spectral'],1.,places=13)
                self.assertAlmostEqual(m['Q_independent_spatial_Parseval'],1.,places=12)
                self.assertLess(abs(m['field_transverse_second_moment_per_um2']/(8/20**2)-1),.01)
    def test_known_two_mode_power_weighted_moment_and_Parseval(self):
        count=64;dx=1.;scalar=np.zeros((count,count),complex);scalar[0,0]=1;scalar[0,12]=.1
        spectrum=diag.vector_spectrum(scalar,dx,1.55)
        m=diag.angular_statistics(spectrum,dx,1.55,20.)
        transverse=2*math.pi*12/(count*dx);k=2*math.pi/1.55;cosine=math.sqrt(1-(transverse/k)**2)
        p0=.5;p1=.5*.1**2*cosine**3
        expected=p1*transverse**2/(p0+p1)
        self.assertAlmostEqual(m['field_transverse_second_moment_per_um2'],expected,places=13)
        self.assertAlmostEqual(m['spatial_field_moment_per_um2'],expected,places=12)
    def test_Q_invariant_under_common_phase_amplitude_and_free_propagation(self):
        s,dx=self.gaussian();a=diag.angular_statistics(s,dx,1.55,20.)
        kz=diag.kernels(s.shape[-1],dx,1.55)[2]
        changed=s*3.2*np.exp(.7j)*np.exp(1j*kz*40)[None]
        b=diag.angular_statistics(changed,dx,1.55,20.)
        self.assertAlmostEqual(a['Q_independent_spectral'],b['Q_independent_spectral'],places=13)
        self.assertAlmostEqual(a['Q_independent_spatial_Parseval'],b['Q_independent_spatial_Parseval'],places=12)
    def test_phase_current_known_quadratic_scalar_gaussian(self):
        count=256;dx=.5;D=20.;wave=1.55;x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x);r2=xx*xx+yy*yy
        zr=np.pi*(D/2)**2/wave;k=2*np.pi/wave;known=.04
        ex=np.exp(-r2/(D/2)**2+1j*k*known/zr*r2/2)
        spectrum=np.zeros((6,count,count),complex);spectrum[0]=np.fft.fft2(ex)
        self.assertAlmostEqual(diag.independent_current(spectrum,dx,wave,D),known,places=12)
    def test_zero_field_is_undefined_and_existing_output_is_protected(self):
        with self.assertRaises(ValueError):diag.angular_statistics(np.zeros((6,64,64),complex),1.,1.55,20.)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            folder=Path(temp);output=folder/'diagnostics/prior';output.mkdir(parents=True)
            with patch.object(sys,'argv',['diag','--folder',str(folder),'--output',str(output)]),patch.object(diag,'diagnose',side_effect=AssertionError('must not run')),self.assertRaises(FileExistsError):diag.main()


if __name__=='__main__':unittest.main(verbosity=2)
