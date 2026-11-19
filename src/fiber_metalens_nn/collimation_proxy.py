"""Cylinder-aware one-way split-step proxy; independent Maxwell remains required."""
from .runtime import configure
configure()
import jax
import jax.numpy as jnp
import numpy as np
from .collimation_optics import grid, kernels, vector_spectrum, describe


class CylinderProxy:
    def __init__(self,config,grid_n=None,padding=None,dz=None,slices=None):
        self.config=config
        p=config["proxy"]
        self.count=(grid_n or p["grid_n"])*(padding or p["padding"])
        self.dx=p["window_um"]/(grid_n or p["grid_n"])
        self.x,_=grid(self.count,self.count*self.dx)
        xx,yy=np.meshgrid(self.x,self.x,indexing="xy")
        self.r=jnp.asarray(np.hypot(xx,yy),jnp.float32)
        self.radius=config["cylinder_diameter_um"]/2
        self.index=jnp.minimum((self.r/(self.radius/config["annuli"])).astype(jnp.int32),config["annuli"]-1)
        self.disk=jnp.clip((self.radius-self.r)/self.dx+.5,0,1)
        self.k0=2*np.pi/config["wavelength_um"]
        self.ns=config["materials"]["substrate"]["n"]
        self.np=config["materials"]["polymer"]["n"]
        self.na=config["materials"]["ambient"]["n"]
        if self.na!=1 or any(m["kappa"] for m in config["materials"].values()):
            raise ValueError("This branch supports explicitly lossless materials and air output only")
        self.input=jnp.exp(-(self.r/(config["mfd_um"]/2))**2).astype(jnp.complex64)
        self.incident=float(.5*self.ns*np.sum(np.asarray(abs(self.input)**2))*self.dx**2)
        self.dz=dz or p["pedestal_dz_um"]
        self.slices=slices or p["relief_slices"]
        self.hmax=config["height_max_um"]
        self.relief_dz=self.hmax/self.slices
        self.pedestal_half=self._kernel(self.np,self.dz/2)
        self.relief_ref=(self.np+self.na)/2
        self.relief_half=self._kernel(self.relief_ref,self.relief_dz/2)
        self.target_cache={}
        self.pedestal_cache={}
        self.objective=jax.jit(self._objective)
        self.value_gradient=jax.jit(jax.value_and_grad(self._objective))

    def _kernel(self,n,distance):
        f=2*jnp.pi*jnp.fft.fftfreq(self.count,d=self.dx)
        kz=jnp.sqrt(((self.k0*n)**2-f[:,None]**2-f[None,:]**2).astype(jnp.complex64))
        return jnp.exp(1j*kz*distance)

    @staticmethod
    def step(field,kernel,phase):
        field=jnp.fft.ifft2(jnp.fft.fft2(field)*kernel)*phase
        return jnp.fft.ifft2(jnp.fft.fft2(field)*kernel)

    def pedestal(self,H):
        key=float(H)
        if key not in self.pedestal_cache:
            steps=int(np.ceil(H/self.dz))
            step=float(H)/steps
            kernel=self._kernel(self.np,step/2)
            index=self.na+(self.np-self.na)*self.disk
            phase=jnp.exp(1j*self.k0*(index-self.np)*step)
            seed=self.input*(2*self.ns/(self.ns+self.np))
            def body(field,_):
                return self.step(field,kernel,phase),None
            field,_=jax.jit(lambda a:jax.lax.scan(jax.checkpoint(body),a,None,length=steps))(seed)
            self.pedestal_cache[key]=field
        return self.pedestal_cache[key]

    def spectrum(self,heights,pedestal):
        local=heights[self.index]
        def body(field,z):
            fill=jnp.clip((local-z)/self.relief_dz,0,1)*self.disk
            index=self.na+(self.np-self.na)*fill
            phase=jnp.exp(1j*self.k0*(index-self.relief_ref)*self.relief_dz)
            return self.step(field,self.relief_half,phase),None
        field,_=jax.lax.scan(jax.checkpoint(body),pedestal,jnp.arange(self.slices)*self.relief_dz)
        field*=2*self.np/(self.np+self.na)
        spectra=vector_spectrum(jnp.fft.fft2(field),self.dx,self.config["wavelength_um"],xp=jnp)
        return spectra*self._kernel(self.na,self.config["proxy"]["output_clearance_um"])[None]

    def target(self,diameter):
        if float(diameter) not in self.target_cache:
            gaussian=jnp.exp(-(self.r/(diameter/2))**2).astype(jnp.complex64)
            target=vector_spectrum(jnp.fft.fft2(gaussian),self.dx,self.config["wavelength_um"],xp=jnp)
            self.target_cache[float(diameter)]=target
        return self.target_cache[float(diameter)]

    def seed(self,H):
        r=(jnp.arange(self.config["annuli"])+.5)*self.radius/self.config["annuli"]
        zr=np.pi*self.np*(self.config["mfd_um"]/2)**2/self.config["wavelength_um"]
        curvature=H/(H*H+zr*zr)
        phase=-self.k0*self.np*r*r*curvature/2
        return jnp.mod(phase,2*jnp.pi)/(self.k0*(self.np-self.na))

    def _objective(self,heights,pedestal,target,diameter):
        version=self.config["loss"].get("version")
        if version=="five_plane_exp2_v2":
            from .collimation_loss_v2 import objective
            return objective(self,heights,pedestal,target,diameter)
        if version not in (None,"legacy_endpoints_v1"):
            raise ValueError("Unknown collimation loss version")
        spectrum=self.spectrum(heights,pedestal)
        def inner(a,b):
            return .25*jnp.sum(a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+
                                jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*self.dx**2/self.count**2
        useful=jnp.abs(inner(spectrum,target))**2/(jnp.real(inner(target,target))*self.incident)
        _,_,kz,_,valid,_=kernels(self.count,self.dx,self.config["wavelength_um"],xp=jnp)
        widths=[]
        for z in (0.,100.):
            f=jnp.fft.ifft2(spectrum*(jnp.exp(1j*kz*z)*valid)[None],axes=(-2,-1))
            sz=jnp.maximum(.5*jnp.real(f[0]*jnp.conj(f[4])-f[1]*jnp.conj(f[3])),0)
            widths.append(2*jnp.sqrt(2*jnp.sum(sz*self.r**2)/jnp.maximum(sz.sum(),1e-20)))
        loss=self.config["loss"]
        # Gaussian D4sigma radial equivalent; multi-plane widths complement
        # the propagated target-mode overlap (which is propagation invariant).
        return -loss["mode"]*useful+loss["diameter"]*(widths[0]/diameter-1)**2+loss["stability"]*(widths[1]/widths[0]-1)**2+loss["roughness"]*jnp.mean(jnp.diff(heights)**2)

    def evaluate(self,heights,H,diameter,save=None):
        spectrum=np.asarray(self.spectrum(jnp.asarray(heights),self.pedestal(H)))
        power=.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])).sum()*self.dx**2/self.count**2
        result=describe(spectrum,self.dx,self.config["wavelength_um"],diameter,self.config,float(power/self.incident),save)
        result.update(model="scalar_split_step_with_vector_output",result_label=self.config["result_label"],
                      H_um=float(H),target_diameter_um=float(diameter),output_reference_z_um=float(H+self.hmax+self.config["proxy"]["output_clearance_um"]),
                      limitations="One-way scalar internal propagation and approximate normal interface transmission; no internal backward wave or full vector interface coupling.")
        return result
