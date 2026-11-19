"""Differentiable natural-size collimation loss with fixed declared targets."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from fiber_metalens_nn.collimation_proxy import CylinderProxy
from fiber_metalens_nn.collimation_optics import kernels
from fiber_metalens_nn.collimation_loss_v2 import plane_widths
from fiber_metalens_nn.natural_collimation import derive_annular_relief,require_index_matched


class NaturalLoss:
    def __init__(self,config):
        require_index_matched(config);self.config=config;self.p=CylinderProxy(config)
        p=self.p;self.x=jnp.asarray(p.x,jnp.float32);self.xx,self.yy=jnp.meshgrid(self.x,self.x,indexing='xy')
        self.kx,self.ky,self.kz,self.safe,self.valid,self.k0=kernels(p.count,p.dx,config['wavelength_um'],xp=jnp)
        self.bundle_cache={};self.metadata={}

    def bundle(self,H):
        key=float(H)
        if key not in self.bundle_cache:
            if len(self.bundle_cache)>=11:raise RuntimeError('bounded11-height cache exceeded')
            p=self.p;top=p.pedestal(H);top.block_until_ready();a=np.asarray(top)
            seed,rounded,encoding=derive_annular_relief(a,p.x,self.config)
            intensity=abs(a)**2;r=np.asarray(p.r)
            diameter=float(2*np.sqrt(2*(intensity*r*r).sum()/intensity.sum()))
            target=p.target(diameter);target.block_until_ready();width=[]
            for z in self.config['evaluation']['processing_planes_um']:
                f=jnp.fft.ifft2(target*(jnp.exp(1j*self.kz*z)*self.valid)[None],axes=(-2,-1));width.append(plane_widths(f,self.x)[0])
            target_power=.5*jnp.real(target[0]*jnp.conj(target[4])-target[1]*jnp.conj(target[3]))
            moment=jnp.sum(target_power*(self.kx**2+self.ky**2))/jnp.sum(target_power)
            self.bundle_cache[key]=(top,target,jnp.stack(width),jnp.asarray(seed),jnp.asarray(diameter,jnp.float32),moment)
            self.metadata[str(key)]=dict(H_um=key,declared_target_diameter_um=diameter,seed_continuous_um=seed.tolist(),seed_quantized_um=rounded.tolist(),encoding=encoding)
        return self.bundle_cache[key]

    def terms(self,heights,bundle):
        top,target,target_widths,seed,D,target_moment=bundle;p=self.p;cfg=self.config
        s=p.spectrum(heights,top)
        def inner(a,b):
            return .25*jnp.sum(a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*p.dx**2/p.count**2
        P=jnp.real(inner(s,s));Pt=jnp.real(inner(target,target));useful=jnp.abs(inner(s,target))**2/(Pt*p.incident)
        T=P/p.incident;overlap=useful/jnp.maximum(T,1e-20)
        f0=jnp.fft.ifft2(s,axes=(-2,-1));Ex=f0[0];I=abs(Ex)**2
        gx=jnp.fft.ifft2(1j*self.kx*s[0]);gy=jnp.fft.ifft2(1j*self.ky*s[0])
        core=p.r<.6*D
        inverse_R=jnp.sum(core*(jnp.imag(jnp.conj(Ex)*gx)*self.xx+jnp.imag(jnp.conj(Ex)*gy)*self.yy))/jnp.maximum(self.k0*jnp.sum(core*I*p.r**2),1e-20)
        zr=jnp.pi*(D/2)**2/cfg['wavelength_um'];curvature=zr*inverse_R
        angular=jnp.maximum(.5*jnp.real(s[0]*jnp.conj(s[4])-s[1]*jnp.conj(s[3])),0)
        moment=jnp.sum(angular*(self.kx**2+self.ky**2))/jnp.maximum(angular.sum(),1e-20)
        ratio=moment/jnp.maximum(target_moment,1e-20)
        widths=[];valid=[];sides=[]
        for z in cfg['evaluation']['processing_planes_um']:
            f=jnp.fft.ifft2(s*(jnp.exp(1j*self.kz*z)*self.valid)[None],axes=(-2,-1));w,v,_=plane_widths(f,self.x)
            widths.append(w);valid.append(v)
            power=jnp.maximum(.5*jnp.real(f[0]*jnp.conj(f[4])-f[1]*jnp.conj(f[3])),0)
            wz=D/2*jnp.sqrt(1+(z/zr)**2);mask=p.r>wz*jnp.sqrt(jnp.log(100.)/2)
            sides.append(jnp.sum(power*mask)/jnp.maximum(power.sum(),1e-20))
        widths=jnp.stack(widths);valid=jnp.stack(valid);sides=jnp.stack(sides);weights=cfg['loss']
        parts=jnp.stack([-weights['negative_useful_power']*useful,weights['width']*jnp.mean((widths/target_widths-1)**2),weights['phase']*curvature**2,weights['angular']*jnp.maximum(ratio-1,0)**2,weights['sidelobe']*jnp.mean(jnp.maximum(sides-.01,0)**2),weights['roughness']*jnp.mean(jnp.diff(heights)**2),weights['low_power_penalty']*jnp.maximum(weights['transmission_floor']-T,0)**2,jnp.mean((~valid).astype(jnp.float32))])
        return dict(loss=parts.sum(),weighted_terms=parts,transmission=T,declared_overlap=overlap,declared_useful_efficiency=useful,widths_um=widths,target_widths_um=target_widths,valid_crossings=valid,max_sidelobe_fraction=sides.max(),sidelobe_fraction_planes=sides,normalized_curvature=curvature,inverse_R_per_um=inverse_R,angular_second_moment_ratio=ratio)


def selection_key(metrics,criteria):
    """Quantized validation only; feasibility precedes useful efficiency."""
    violations=[];eta=[]
    for m in metrics:
        violations.extend([max(0,criteria['transmission_min']-m['transmission'])/criteria['transmission_min'],max(0,criteria['declared_Gaussian_overlap_min']-m['declared_overlap'])/criteria['declared_Gaussian_overlap_min'],max(0,criteria['useful_efficiency_min']-m['declared_useful_efficiency'])/criteria['useful_efficiency_min'],max(0,m['max_sidelobe_fraction']/criteria['max_sidelobe_fraction']-1),max(0,abs(m['normalized_curvature'])/criteria['max_abs_normalized_curvature']-1),max(0,m['angular_second_moment_ratio']/criteria['max_angular_second_moment_ratio']-1),max(0,float(np.max(abs(np.asarray(m['widths_um'])[0]/np.asarray(m['target_widths_um'])[0]-1)))/criteria['output_diameter_relative_tolerance']-1)])
        if not np.asarray(m['valid_crossings']).all():violations.append(100.)
        eta.append(m['declared_useful_efficiency'])
    return [float(sum(violations)),float(-min(eta)),float(np.mean([m['loss'] for m in metrics]))]
