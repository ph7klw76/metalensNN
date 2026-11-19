"""Vector angular-spectrum thin-element proxy, not a Maxwell solver."""
from .runtime import configure
configure()
import jax
import jax.numpy as jnp
import numpy as np

from .geometry import complex_index, edges, geometric_na
from .metrics import peak_position, width_metrics


def disk_fraction(x, radius, subpixels=8):
    xx, yy = np.meshgrid(x,x,indexing="xy")
    dx = x[1]-x[0]
    distance = np.hypot(xx,yy)
    mask = (distance <= radius-dx).astype(np.float32)
    boundary = np.abs(distance-radius) <= dx
    bx, by = xx[boundary], yy[boundary]
    count = np.zeros(len(bx),np.float32)
    for ox in (np.arange(subpixels)+0.5)/subpixels-0.5:
        for oy in (np.arange(subpixels)+0.5)/subpixels-0.5:
            count += ((bx+ox*dx)**2+(by+oy*dx)**2 < radius**2)
    mask[boundary] = count/subpixels**2
    return mask


class OpticalProxy:
    def __init__(self, config, grid_n=None, padding=None, high_precision=False):
        self.config = config
        p = config["proxy"]
        self.n = grid_n or p["grid_n"]
        self.padding = padding or p["padding"]
        self.size = self.n*self.padding
        self.dx = p["window_um"]/self.n
        self.dtype = jnp.float64 if high_precision else jnp.float32
        self.cdtype = jnp.complex128 if high_precision else jnp.complex64
        self.x = (np.arange(self.size)-self.size/2+0.5)*self.dx
        axis = (np.arange(self.n)-self.n/2+0.5)*self.dx
        xx, yy = np.meshgrid(axis,axis,indexing="xy")
        weights = np.zeros((config["annuli"]+1,self.n*self.n),np.float32)
        pixels = np.arange(self.n*self.n)
        subpixels = p["subpixels"]
        width = config["aperture_diameter_um"]/2/config["annuli"]
        for ox in (np.arange(subpixels)+0.5)/subpixels-0.5:
            for oy in (np.arange(subpixels)+0.5)/subpixels-0.5:
                index = np.floor(np.hypot(xx+ox*self.dx,yy+oy*self.dx)/width).astype(int).ravel()
                index = np.minimum(index,config["annuli"])
                weights[index,pixels] += 1/subpixels**2
        self.weights = jnp.asarray(weights[:-1],self.dtype)
        self.aperture_area = float(weights[:-1].sum()*self.dx**2)
        self.ns = complex_index(config["materials"]["substrate"])
        self.np = complex_index(config["materials"]["polymer"])
        self.na = complex_index(config["materials"]["ambient"])
        if self.na.imag != 0 or self.ns.imag != 0:
            raise ValueError("This initial proxy requires lossless ambient and substrate; no silent complex-medium substitution")
        self.k0 = 2*np.pi/config["wavelength_um"]
        k = self.k0*self.na.real
        frequencies = 2*jnp.pi*jnp.fft.fftfreq(self.size,d=self.dx).astype(self.dtype)
        kx, ky = jnp.meshgrid(frequencies,frequencies,indexing="xy")
        kz = jnp.sqrt(jnp.maximum(k*k-kx*kx-ky*ky,0))
        valid = (kx*kx+ky*ky <= (p["max_propagating_na_fraction"]*k)**2)
        self.kz = kz
        direction = jnp.stack([kx/k,ky/k,kz/k])
        polarization = jnp.stack([1-direction[0]**2, -direction[0]*direction[1], -direction[0]*direction[2]])
        magnetic = self.na.real*jnp.cross(direction,polarization,axisa=0,axisb=0,axisc=0)
        self.vector_kernel = jnp.concatenate([polarization,magnetic],axis=0).astype(self.cdtype)*valid[None,:,:]
        self.collection = jnp.asarray(disk_fraction(self.x,config["collection_radius_um"],subpixels),self.dtype)
        loss = config["loss"]
        self.sidelobe = jnp.asarray(disk_fraction(self.x,loss["sidelobe_outer_radius_um"],subpixels)-disk_fraction(self.x,loss["sidelobe_inner_radius_um"],subpixels),self.dtype)
        self.incident_power = 0.5*self.ns.real*self.aperture_area  # Z0-scaled H; common SI factor cancels
        self.objective = jax.jit(self._objective)
        self.value_gradient = jax.jit(jax.value_and_grad(self._objective))
        self.field_at = jax.jit(self._field_at)

    def transmission(self, heights):
        h = jnp.asarray(heights,self.dtype)
        r01 = (self.ns-self.np)/(self.ns+self.np)
        r12 = (self.np-self.na)/(self.np+self.na)
        factor = 2*self.ns/(self.ns+self.np)*2*self.np/(self.np+self.na)
        phase = jnp.exp((1j*self.k0*self.np*h).astype(self.cdtype))
        transmission = factor*phase/(1+r01*r12*phase**2)
        return transmission*jnp.exp((1j*self.k0*self.na*(self.config["exit_plane_z_um"]-h)).astype(self.cdtype))

    def spectrum(self, heights):
        transmission = self.transmission(heights)
        seed = (transmission @ self.weights).reshape(self.n,self.n)
        transmitted = 0.5*self.na.real*jnp.sum((jnp.abs(transmission)**2) @ self.weights)*self.dx**2
        start = (self.size-self.n)//2
        seed = jnp.pad(seed,((start,self.size-self.n-start),)*2)
        spectra = jnp.fft.fft2(seed)[None,:,:]*self.vector_kernel
        raw = 0.5*jnp.real(spectra[0]*jnp.conj(spectra[4])-spectra[1]*jnp.conj(spectra[3])).sum()*self.dx**2/self.size**2
        spectra *= jnp.sqrt(transmitted/jnp.maximum(raw,jnp.finfo(self.dtype).tiny))
        return spectra

    def fields(self, spectra, z):
        return jnp.fft.ifft2(spectra*jnp.exp(1j*self.kz*z)[None,:,:],axes=(-2,-1))

    @staticmethod
    def flux(fields):
        return 0.5*jnp.real(fields[0]*jnp.conj(fields[4])-fields[1]*jnp.conj(fields[3]))

    def _field_at(self, heights, z):
        return self.fields(self.spectrum(heights),z)

    def _objective(self, heights, target):
        spectra = self.spectrum(heights)
        offset = self.config["proxy"]["axial_offset_um"]
        fields = jax.vmap(lambda z:self.fields(spectra,z))(jnp.asarray([target-offset,target,target+offset],self.dtype))
        flux = jax.vmap(self.flux)(fields)
        eta = jnp.sum(flux*self.collection[None,:,:],axis=(-2,-1))*self.dx**2/self.incident_power
        side = jnp.sum(jnp.maximum(flux[1],0)*self.sidelobe)*self.dx**2/self.incident_power
        grid = jnp.mean(jnp.sin(jnp.pi*heights/self.config["fabrication"]["height_step_um"])**2)
        loss = self.config["loss"]
        return loss["center_eta"]*eta[1]+loss["axial_eta"]*(eta[0]+eta[2])+loss["sidelobe_power"]*side+loss["height_grid"]*grid

    def evaluate(self, heights, target, save_path=None):
        import time
        start = time.monotonic()
        p = self.config["proxy"]
        zs = np.arange(p["focus_scan_min_um"],p["focus_scan_max_um"]+p["focus_scan_step_um"]*0.1,p["focus_scan_step_um"])
        spectra = self.spectrum(jnp.asarray(heights,self.dtype))
        propagation = jax.jit(lambda z:self.fields(spectra,z))
        middle = self.size//2
        axial = []
        for z in zs:
            field = propagation(z)
            intensity = jnp.sum(jnp.abs(field[:3])**2,axis=0)
            axial.append(float(jnp.mean(intensity[middle-1:middle+1,middle-1:middle+1])))
        focal, boundary = peak_position(zs,axial)
        fields = np.asarray(propagation(focal))
        intensity = np.sum(np.abs(fields[:3])**2,axis=0)
        ix = intensity[middle-1:middle+1,:].mean(axis=0)
        iy = intensity[:,middle-1:middle+1].mean(axis=1)
        target_fields = propagation(float(target))
        sz = np.asarray(self.flux(target_fields))
        collection = np.asarray(self.collection)
        report = {"result_label":self.config["result_label"],"model":"vector_AS_thin_element_proxy",
                  "target_focal_um":float(target),"focal_um":focal,"focus_at_scan_boundary":boundary,
                  "focus_definition":"maximum on-axis electric intensity in declared axial window",
                  "eta_R":float(np.sum(sz*collection)*self.dx**2/self.incident_power),
                  "transmission":float(np.sum(sz)*self.dx**2/self.incident_power),
                  "collection_radius_um":self.config["collection_radius_um"],
                  "normalization":"Poynting power / incident aperture power",
                  "sidelobe_peak_ratio":float(np.max(intensity[(self.x[:,None]**2+self.x[None,:]**2)>self.config["loss"]["sidelobe_inner_radius_um"]**2])/intensity.max()),
                  "geometric_NA_from_end_plane":float(geometric_na(target,self.config)),
                  "geometric_NA_from_substrate_pupil":float(geometric_na(target,self.config,True)),
                  "grid_n":self.n,"padded_n":self.size,"dx_um":self.dx,"wall_s":time.monotonic()-start,
                  **width_metrics(self.x,ix,iy)}
        if save_path:
            np.savez_compressed(save_path,z_um=zs,axis_intensity=np.asarray(axial),x_um=self.x,
                                target_Sz=sz.astype(np.float32),focus_intensity=intensity.astype(np.float32))
        return report
