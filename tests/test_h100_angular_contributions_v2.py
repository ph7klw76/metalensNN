"""Analytic angular-bin arithmetic; no actual fields, Meep or training."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('angular_contributions',ROOT/'scripts/diagnose_h100_angular_contributions_v2.py')
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


class TestAngularContributions(unittest.TestCase):
    def gaussian(self):
        N=256;dx=.5;x=(np.arange(N)-N/2+.5)*dx;xx,yy=np.meshgrid(x,x)
        return diag.vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/100)),dx,1.55),dx
    def test_identical_gaussian_bins_sum_one_and_zero_excess(self):
        s,dx=self.gaussian();d=diag.angular_bins(s,dx,1.55,20.)
        self.assertAlmostEqual(d['sum_field_power_fractions'],1.,places=13)
        self.assertAlmostEqual(d['sum_field_additive_Q'],1.,places=13)
        self.assertAlmostEqual(d['sum_target_additive_Q'],1.,places=13)
        self.assertAlmostEqual(d['sum_additive_Q_excess'],0.,places=13)
        for row in d['bins']:self.assertAlmostEqual(row['field_additive_Q'],row['target_additive_Q'],places=13)
    def test_known_mode_power_and_Q_occupy_the_correct_bin(self):
        N=64;dx=1.;scalar=np.zeros((N,N),complex);scalar[0,0]=1.;scalar[0,12]=.1
        s=diag.vector_spectrum(scalar,dx,1.55);d=diag.angular_bins(s,dx,1.55,20.)
        kt=2*math.pi*12/(N*dx);k=2*math.pi/1.55;cosine=math.sqrt(1-(kt/k)**2)
        p0=.5;p1=.5*.1**2*cosine**3;expected=p1/(p0+p1)
        occupied=next(row for row in d['bins'] if row['theta_min_deg']==10.)
        self.assertAlmostEqual(occupied['field_forward_power_fraction'],expected,places=13)
        self.assertAlmostEqual(occupied['field_additive_Q'],expected*kt**2/d['target_moment_per_um2'],places=13)
        self.assertAlmostEqual(d['sum_field_additive_Q'],d['Q_unbinned'],places=13)
        self.assertAlmostEqual(d['sum_additive_Q_excess'],d['Q_unbinned']-1,places=13)
    def test_common_amplitude_phase_do_not_change_bin_allocation(self):
        s,dx=self.gaussian();a=diag.angular_bins(s,dx,1.55,20.);b=diag.angular_bins(s*7*np.exp(.4j),dx,1.55,20.)
        for old,new in zip(a['bins'],b['bins']):
            self.assertAlmostEqual(old['field_forward_power_fraction'],new['field_forward_power_fraction'],places=13)
            self.assertAlmostEqual(old['field_additive_Q'],new['field_additive_Q'],places=13)
    def test_incomplete_bins_and_zero_field_are_rejected(self):
        s,dx=self.gaussian()
        for bins in [(),(0,45),(1,90),(0,20,10,90)]:
            with self.assertRaises(ValueError):diag.angular_bins(s,dx,1.55,20.,bins=bins)
        with self.assertRaises(ValueError):diag.angular_bins(s*0,dx,1.55,20.)


if __name__=='__main__':unittest.main(verbosity=2)
