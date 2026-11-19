"""Height-conditioned natural-beam loss with explicitly unequal real indices.

The finite-cylinder proxy is one-way scalar internally, with vector air output.
An input-medium phase index, the supplied polymer index and the fiber's quoted
NA are distinct inputs. No measured fiber eigenmode or material absorption is
invented here. The selected profile may differ with H in this campaign.
"""
from __future__ import annotations

import math

import numpy as np
import jax.numpy as jnp

from .collimation_proxy import CylinderProxy
from .collimation_optics import kernels
from .collimation_loss_v2 import plane_widths
from .natural_collimation import derive_annular_relief
from .natural_nn_loss_v1 import selection_key


DEFAULT_WEIGHTS = {
    'negative_useful_power': 1.0,
    'width': 0.05,
    'phase': 0.05,
    'angular': 0.02,
    'sidelobe': 0.1,
    'roughness': 0.0002,
    'transmission_floor': 0.9,
    'low_power_penalty': 10.0,
    'sidelobe_loss_floor': 0.01,
}


def normal_interface_power(n_input, n_output):
    """Lossless normal-incidence Fresnel identity, not device transmission.

    E-field amplitude transmission is not a power transmission coefficient:
    the impedance factor n_output/n_input is required. This limiting-case
    check does not validate finite-angle or reflected-wave proxy physics.
    """
    ni, nt = float(n_input), float(n_output)
    if not all(math.isfinite(n) and n > 0 for n in (ni, nt)):
        raise ValueError('Finite positive real refractive indices required')
    amplitude = 2 * ni / (ni + nt)
    transmission = nt / ni * amplitude**2
    reflection = ((ni - nt) / (ni + nt))**2
    return {
        'electric_amplitude_transmission': amplitude,
        'power_transmission': transmission,
        'power_reflection': reflection,
        'power_closure_error': abs(transmission + reflection - 1),
    }


class HeightConditionedLoss:
    """Reusable per-H target bundles and differentiable objective terms.

    ``bundle(H)`` preserves the prior six-item order:
    pedestal, target spectrum, vector-target widths, continuous seed,
    scalar Gaussian-equivalent diameter, target angular second moment.
    Targets depend on the input/cylinder physics, not on trained output.
    The caller owns train/validation/test splits, finite workload bounds,
    checkpoint signatures, quantization and the permitted compute budget.
    """

    def __init__(self, config):
        self.config = config
        if int(config['annuli']) < 2:
            raise ValueError('At least two radial annuli required')
        for key in ('wavelength_um', 'mfd_um', 'cylinder_diameter_um', 'height_max_um'):
            if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
                raise ValueError(f'Finite positive {key} required')
        for material in config['materials'].values():
            if not math.isfinite(float(material['n'])) or float(material['n']) <= 0:
                raise ValueError('Finite positive real material indices required')
            if not math.isfinite(float(material['kappa'])) or float(material['kappa']) != 0:
                raise ValueError('The current proxy requires explicitly lossless materials')
        if float(config['materials']['ambient']['n']) != 1:
            raise ValueError('The current vector continuation requires air output')
        planes = tuple(float(z) for z in config['evaluation']['processing_planes_um'])
        if (not planes or planes[0] != 0 or any(not math.isfinite(z) or z < 0 for z in planes)
                or any(b <= a for a, b in zip(planes, planes[1:]))):
            raise ValueError('Processing planes must start at zero and strictly increase')
        self.planes = planes
        self.weights = {
            key: float(config.get('loss', {}).get(key, default))
            for key, default in DEFAULT_WEIGHTS.items()
        }
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError('Finite nonnegative objective weights required')
        if self.weights['transmission_floor'] > 1 or self.weights['sidelobe_loss_floor'] > 1:
            raise ValueError('Power-fraction loss floors must be between zero and one')
        self.proxy = CylinderProxy(config)
        self.p = self.proxy  # Compatibility with existing trainer conventions.
        p = self.proxy
        self.x = jnp.asarray(p.x, jnp.float32)
        self.xx, self.yy = jnp.meshgrid(self.x, self.x, indexing='xy')
        self.kx, self.ky, self.kz, self.safe, self.valid, self.k0 = kernels(
            p.count, p.dx, config['wavelength_um'], xp=jnp)
        self.bundle_cache = {}
        self.metadata = {}
        self.mode_provenance = dict(config.get('input_mode_provenance', {}))
        self.bundle_limit = config.get('campaign_limits', {}).get('bundle_cache_max')
        if self.bundle_limit is not None:
            if (isinstance(self.bundle_limit, bool)
                    or not isinstance(self.bundle_limit, int) or self.bundle_limit <= 0):
                raise ValueError('Declared bundle_cache_max must be a positive integer')

    def bundle(self, H):
        height = float(H)
        if not math.isfinite(height) or height <= 0:
            raise ValueError('Finite positive cylinder height required')
        if height not in self.bundle_cache:
            if self.bundle_limit is not None and len(self.bundle_cache) >= self.bundle_limit:
                raise RuntimeError('Explicitly declared target-bundle workload count exceeded')
            p = self.proxy
            top = p.pedestal(height)
            top.block_until_ready()
            array = np.asarray(top)
            if not np.isfinite(array).all():
                raise ValueError('Cylinder-top field is nonfinite')
            seed, quantized, encoding = derive_annular_relief(array, p.x, self.config)
            intensity = np.abs(array)**2
            radius = np.asarray(p.r)
            total = float(intensity.sum())
            if not math.isfinite(total) or total <= 0:
                raise ValueError('Nonzero finite cylinder-top intensity required')
            diameter = float(2 * np.sqrt(2 * np.sum(intensity * radius**2) / total))
            target = p.target(diameter)
            target.block_until_ready()
            widths, valid_widths = [], []
            for z in self.planes:
                field = jnp.fft.ifft2(
                    target * (jnp.exp(1j*self.kz*z)*self.valid)[None], axes=(-2, -1))
                width, is_valid, _ = plane_widths(field, self.x)
                widths.append(width)
                valid_widths.append(is_valid)
            widths = jnp.stack(widths)
            if (not np.asarray(jnp.stack(valid_widths)).all()
                    or not np.isfinite(np.asarray(widths)).all()
                    or np.any(np.asarray(widths) <= 0)):
                raise ValueError('Target crossings unresolved; change declared sampling before training')
            target_power = .5*jnp.real(
                target[0]*jnp.conj(target[4])-target[1]*jnp.conj(target[3]))
            moment = jnp.sum(target_power*(self.kx**2+self.ky**2))/jnp.sum(target_power)
            if not math.isfinite(float(moment)) or float(moment) <= 0:
                raise ValueError('Target angular second moment must be finite and positive')
            self.bundle_cache[height] = (
                top, target, widths, jnp.asarray(seed),
                jnp.asarray(diameter, jnp.float32), moment,
            )
            self.metadata[str(height)] = {
                'H_um': height,
                'declared_target_diameter_um': diameter,
                'target_scalar_diameter_definition':
                    '2*sqrt(2*intensity-weighted radial second moment of proxy cylinder-top field)',
                'target_waist_reference':
                    'Flat-phase scalar Gaussian parameter placed at the output reference, then vector projected',
                'target_output_z_um': float(height+self.config['height_max_um']+
                                            self.config['proxy']['output_clearance_um']),
                'target_vector_widths_um': np.asarray(widths).tolist(),
                'processing_planes_from_output_um': list(self.planes),
                'seed_continuous_um': seed.tolist(),
                'seed_quantized_um': quantized.tolist(),
                'encoding': encoding,
                'input_mode': {
                    'model': 'Gaussian amplitude surrogate at the facet; no core/cladding eigenmode supplied',
                    'mfd_um': float(self.config['mfd_um']),
                    'mfd_provenance': self.config.get(
                        'mfd_provenance', self.mode_provenance or
                        'Declared MFD supplied by configuration; no measured mode injected'),
                    'datasheet_nominal_mfd_um': self.mode_provenance.get('mfd_1550_um'),
                    'datasheet_mfd_tolerance_um': self.mode_provenance.get('mfd_tolerance_um'),
                    'datasheet_source_url': self.mode_provenance.get('source_url'),
                    'fiber_NA': self.config.get('fiber_nominal_NA'),
                    'fiber_NA_interpretation': self.config.get(
                        'fiber_NA_interpretation', self.mode_provenance.get('NA_measurement',
                            'Nominal descriptor, not a Gaussian hard angular cutoff')),
                    'input_phase_index': float(p.ns),
                    'input_phase_index_is_assumed': bool(self.config.get(
                        'input_phase_index_is_assumed',
                        self.mode_provenance.get('source_phase_index_is_assumed', True))),
                    'manufacturer_provenance': self.mode_provenance,
                },
                'materials': {
                    'input_n': float(p.ns), 'polymer_n': float(p.np), 'ambient_n': float(p.na),
                    'indices_are_matched': abs(p.ns-p.np) < 1e-12,
                    'extinction_coefficients': 'All explicitly zero in this model',
                    'provenance': self.config.get('material_provenance',
                        'Real indices supplied by configuration; no measured loss/dispersion model'),
                },
                'entrance_normal_incidence_limit': normal_interface_power(p.ns, p.np),
                'proxy_scope': 'One-way scalar finite-cylinder/finite-relief propagation with vector air output',
                'limitations': [
                    'Normal-incidence entrance/exit amplitude factors approximate interface coupling.',
                    'No internal backward wave or exact vector boundary coupling; independent Maxwell is required.',
                    'The retained k_perp<0.98*k0 band does not establish all-angle or ISO M2 validation.',
                    'A per-H field-derived seed permits a different pattern at each H; this is not one fixed robust pattern.',
                ],
            }
        return self.bundle_cache[height]

    def terms(self, heights, bundle):
        top, target, target_widths, _seed, D, target_moment = bundle
        p = self.proxy
        heights = jnp.asarray(heights)
        if heights.shape != (self.config['annuli'],):
            raise ValueError('One height per configured annulus required')
        spectrum = p.spectrum(heights, top)

        def inner(a, b):
            return .25*jnp.sum(
                a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+
                jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*p.dx**2/p.count**2

        power = jnp.real(inner(spectrum, spectrum))
        target_power = jnp.real(inner(target, target))
        useful = jnp.abs(inner(spectrum, target))**2/(target_power*p.incident)
        transmission = power/p.incident
        mode_overlap = useful/jnp.maximum(transmission, 1e-20)
        field0 = jnp.fft.ifft2(spectrum, axes=(-2, -1))
        ex = field0[0]
        gx = jnp.fft.ifft2(1j*self.kx*spectrum[0])
        gy = jnp.fft.ifft2(1j*self.ky*spectrum[0])
        core = p.r < .6*D
        inverse_R = jnp.sum(core*(jnp.imag(jnp.conj(ex)*gx)*self.xx+
                                 jnp.imag(jnp.conj(ex)*gy)*self.yy))/jnp.maximum(
            self.k0*jnp.sum(core*jnp.abs(ex)**2*p.r**2), 1e-20)
        zr = jnp.pi*(D/2)**2/self.config['wavelength_um']
        curvature = zr*inverse_R
        angular = jnp.maximum(.5*jnp.real(
            spectrum[0]*jnp.conj(spectrum[4])-spectrum[1]*jnp.conj(spectrum[3])), 0)
        moment = jnp.sum(angular*(self.kx**2+self.ky**2))/jnp.maximum(angular.sum(), 1e-20)
        angular_ratio = moment/jnp.maximum(target_moment, 1e-20)
        widths, valid, sides = [], [], []
        for z in self.planes:
            field = jnp.fft.ifft2(
                spectrum*(jnp.exp(1j*self.kz*z)*self.valid)[None], axes=(-2, -1))
            width, is_valid, _ = plane_widths(field, self.x)
            widths.append(width)
            valid.append(is_valid)
            flux = jnp.maximum(.5*jnp.real(
                field[0]*jnp.conj(field[4])-field[1]*jnp.conj(field[3])), 0)
            wz = D/2*jnp.sqrt(1+(z/zr)**2)
            outside = p.r > wz*jnp.sqrt(jnp.log(100.)/2)
            sides.append(jnp.sum(flux*outside)/jnp.maximum(flux.sum(), 1e-20))
        widths, valid, sides = jnp.stack(widths), jnp.stack(valid), jnp.stack(sides)
        w = self.weights
        parts = jnp.stack([
            -w['negative_useful_power']*useful,
            w['width']*jnp.mean((widths/target_widths-1)**2),
            w['phase']*curvature**2,
            w['angular']*jnp.maximum(angular_ratio-1, 0)**2,
            w['sidelobe']*jnp.mean(jnp.maximum(sides-w['sidelobe_loss_floor'], 0)**2),
            w['roughness']*jnp.mean(jnp.diff(heights)**2),
            w['low_power_penalty']*jnp.maximum(w['transmission_floor']-transmission, 0)**2,
            jnp.mean((~valid).astype(jnp.float32)),
        ])
        return {
            'loss': parts.sum(), 'weighted_terms': parts,
            'transmission': transmission, 'declared_overlap': mode_overlap,
            'declared_useful_efficiency': useful,
            'widths_um': widths, 'target_widths_um': target_widths,
            'valid_crossings': valid, 'max_sidelobe_fraction': sides.max(),
            'sidelobe_fraction_planes': sides, 'normalized_curvature': curvature,
            'inverse_R_per_um': inverse_R,
            'angular_second_moment_ratio': angular_ratio,
        }


__all__ = ['HeightConditionedLoss', 'normal_interface_power', 'selection_key']
