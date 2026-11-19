"""Experimental one-way split-step proxy for finite-height annuli.

Includes diffraction through the thickness. It is not a vector Maxwell solver:
backscatter, vector interface coupling and nonlocal reflections remain omitted.
"""
import jax
import jax.numpy as jnp
import numpy as np
from .proxy import OpticalProxy


class ThickProxy(OpticalProxy):
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        count = config["proxy"].get("thickness_slices",25)
        self.dz = config["exit_plane_z_um"]/count
        self.layers = jnp.arange(count,dtype=self.dtype)*self.dz
        self.nref = config["proxy"].get("split_step_reference_index",1.26)
        k = 2*jnp.pi*jnp.fft.fftfreq(self.size,d=self.dx)
        kr2 = k[:,None]**2+k[None,:]**2
        kz = jnp.sqrt(((self.k0*self.nref)**2-kr2).astype(self.cdtype))
        self.half_step = jnp.exp(1j*kz*self.dz/2)

    def spectrum(self,heights):
        heights = jnp.asarray(heights,self.dtype)
        start = (self.size-self.n)//2
        pad = ((start,self.size-self.n-start),)*2
        t = self.transmission(heights)
        local_phase = jnp.exp(1j*self.k0*(self.np*heights+self.na*(self.config["exit_plane_z_um"]-heights)))
        seed = jnp.pad(((t/local_phase)@self.weights).reshape(self.n,self.n),pad)
        def step(field,z):
            fill = jnp.clip((heights-z)/self.dz,0,1)
            fraction = jnp.pad((fill@self.weights).reshape(self.n,self.n),pad)
            index = self.na+(self.np-self.na)*fraction
            field = jnp.fft.ifft2(jnp.fft.fft2(field)*self.half_step)
            field *= jnp.exp(1j*self.k0*(index-self.nref)*self.dz)
            field = jnp.fft.ifft2(jnp.fft.fft2(field)*self.half_step)
            return field,None
        seed,_ = jax.lax.scan(jax.checkpoint(step),seed,self.layers)
        spectra = jnp.fft.fft2(seed)[None,:,:]*self.vector_kernel
        transmitted = .5*self.na.real*jnp.sum((jnp.abs(t)**2)@self.weights)*self.dx**2
        raw = .5*jnp.real(spectra[0]*jnp.conj(spectra[4])-spectra[1]*jnp.conj(spectra[3])).sum()*self.dx**2/self.size**2
        return spectra*jnp.sqrt(transmitted/jnp.maximum(raw,jnp.finfo(self.dtype).tiny))

    def evaluate(self,*args,**kwargs):
        result = super().evaluate(*args,**kwargs)
        result['model'] = 'experimental_scalar_split_step_vector_output'
        result['thickness_slices'] = len(self.layers)
        return result
