"""Independent CPU Maxwell adapter for the unequal-index height family.

No Meep import occurs when inspecting this module's pure setup helpers or
running --dry-run. A real solve requires the caller's separately authorized
budget, immutable configuration and matching homogeneous reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from fiber_metalens_nn.geometry import profile_polygons
from fiber_metalens_nn.collimation_optics import write, fingerprint


DEFAULT_MAXWELL = {
    'pml_um': 2.0, 'padding_um': 10.0, 'after_sources_time': 160.0,
    'source_z_um': -2.0, 'reflection_z_um': -1.0, 'bottom_um': -6.0,
    'crosscheck_distance_um': 5.0, 'top_clearance_um': 2.0, 'Courant': .45,
}


def resolved_maxwell(config):
    settings = {**DEFAULT_MAXWELL, **config.get('maxwell', {})}
    for key in DEFAULT_MAXWELL:
        value = float(settings[key])
        if not math.isfinite(value):
            raise ValueError(f'Finite Maxwell setting required: {key}')
        settings[key] = value
    if any(settings[key] <= 0 for key in (
            'pml_um', 'padding_um', 'after_sources_time', 'crosscheck_distance_um',
            'top_clearance_um', 'Courant')):
        raise ValueError('Positive Maxwell length/time/Courant settings required')
    if not 0 < settings['Courant'] <= .5:
        raise ValueError('This adapter requires Courant in (0, 0.5]')
    return settings


def effective_duration(config, H, requested=None):
    """First-round-trip lower bound; measured closure still determines success."""
    height, hmax = float(H), float(config['height_max_um'])
    index = float(config['materials']['polymer']['n'])
    requested = resolved_maxwell(config)['after_sources_time'] if requested is None else float(requested)
    if not all(math.isfinite(v) and v > 0 for v in (height, hmax, index, requested)):
        raise ValueError('Positive finite height/index/duration required')
    return float(max(requested, math.ceil(2*index*(height+hmax)+20)))


def configured_dimensions(config, H, resolution, *, pml=None, padding=None, duration=None, m=1):
    """Resolve physical geometry and actual time settings without importing Meep."""
    mw = resolved_maxwell(config)
    H, resolution = float(H), float(resolution)
    diameter = float(config['cylinder_diameter_um'])
    hmax = float(config['height_max_um'])
    clearance = float(config['proxy']['output_clearance_um'])
    wavelength, mfd = float(config['wavelength_um']), float(config['mfd_um'])
    pml = mw['pml_um'] if pml is None else float(pml)
    padding = mw['padding_um'] if padding is None else float(padding)
    if (not all(math.isfinite(v) and v > 0 for v in
                (H, resolution, diameter, hmax, clearance, pml, padding, wavelength, mfd)) or m not in (-1, 1)):
        raise ValueError('Positive finite dimensions/resolution and m=+/-1 required')
    if not mw['bottom_um']+pml < mw['source_z_um'] < mw['reflection_z_um'] < 0:
        raise ValueError('Source and reflection planes must lie outside bottom PML below facet')
    radius = diameter/2+padding
    output = H+hmax+clearance
    cross = output+mw['crosscheck_distance_um']
    top = cross+mw['top_clearance_um']+pml
    return {
        'resolution': resolution, 'pml_um': pml, 'padding_um': padding,
        'duration': effective_duration(config, H, duration), 'H_um': H,
        'output_z_um': output, 'crosscheck_z_um': cross,
        'top_um': top, 'bottom_um': mw['bottom_um'], 'm': int(m),
        'monitor_radius_um': radius, 'cell_radial_extent_um': radius+pml,
        'cell_axial_extent_um': top-mw['bottom_um'],
        'source_z_um': mw['source_z_um'], 'reflection_z_um': mw['reflection_z_um'],
        'Courant': mw['Courant'],
        'wavelength_um': wavelength, 'mfd_um': mfd,
    }


def source_envelope(radius_um, source_z_um, waist_um, wavelength_um, input_index):
    """Backward paraxial envelope for a declared facet Gaussian amplitude waist."""
    r = np.asarray(radius_um)
    z, w0, wavelength, index = map(float, (source_z_um, waist_um, wavelength_um, input_index))
    if (not math.isfinite(z) or not np.isfinite(r).all()
            or not all(math.isfinite(v) and v > 0 for v in (w0, wavelength, index))):
        raise ValueError('Finite positive waist/wavelength/input phase index required')
    zr = np.pi*index*w0*w0/wavelength
    wz = w0*np.sqrt(1+(z/zr)**2)
    inverse_R = z/(z*z+zr*zr)
    return w0/wz*np.exp(-r*r/(wz*wz)+1j*np.pi*index/wavelength*inverse_R*r*r)


def validate_design(design, config, H):
    if design.get('units') != 'um':
        raise ValueError('Design must explicitly use um')
    if abs(float(design.get('H_um', H))-float(H)) > 1e-7:
        raise ValueError('Design cylinder height differs from requested H')
    if abs(float(design.get('cylinder_diameter_um', config['cylinder_diameter_um']))-
           float(config['cylinder_diameter_um'])) > 1e-7:
        raise ValueError('Design cylinder diameter differs from configuration')
    heights = np.asarray(design['heights_um'], dtype=float)
    if (heights.shape != (int(config['annuli']),) or not np.isfinite(heights).all()
            or np.any(heights < 0) or np.any(heights > config['height_max_um']+1e-6)):
        raise ValueError('Finite bounded height per configured annulus required')
    return heights


def centered_stencil_from_probe(plane_z, sampled_z, delta):
    """Interpolation stencil from actual centered coordinates, never a guessed grid origin."""
    z, delta = float(plane_z), float(delta)
    values = np.asarray(sorted(set(float(q) for q in sampled_z)),dtype=float)
    if not math.isfinite(z) or not math.isfinite(delta) or delta<=0 or not np.isfinite(values).all():
        raise ValueError('Finite plane and positive grid spacing required')
    tolerance=max(1e-9,delta*1e-8)
    if len(values)==1 and abs(values[0]-z)<=tolerance:
        # An exactly centered plane needs only its first sample; include the
        # adjacent center with zero weight to preserve the fixed two-plane API.
        return [float(values[0]),float(values[0]+delta)],[1.,0.]
    if (len(values)!=2 or abs(values[1]-values[0]-delta)>tolerance
            or z<values[0]-tolerance or z>values[1]+tolerance):
        raise ValueError('Actual monitor probe must provide one exact center or two bracketing centers')
    alpha=float(np.clip((z-values[0])/(values[1]-values[0]),0,1))
    return values.tolist(),[1-alpha,alpha]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--design')
    parser.add_argument('--reference')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resolution', type=float, required=True)
    parser.add_argument('--H', type=float, required=True)
    parser.add_argument('--pml', type=float)
    parser.add_argument('--padding', type=float)
    parser.add_argument('--duration', type=float)
    parser.add_argument('--m', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if any(not math.isfinite(float(mat['n'])) or mat['n'] <= 0 or mat['kappa'] != 0
           for mat in cfg['materials'].values()):
        raise ValueError('Finite positive real indices and explicitly zero loss required')
    settings = configured_dimensions(cfg, args.H, args.resolution, pml=args.pml,
                                    padding=args.padding, duration=args.duration, m=args.m)
    mw = resolved_maxwell(cfg)
    design = json.loads(Path(args.design).read_text(encoding='utf-8-sig')) if args.design else None
    heights = validate_design(design, cfg, args.H) if design else None
    if design and not args.reference:
        raise ValueError('A matching homogeneous reference is required for a device')
    if args.dry_run:
        print(json.dumps({'status':'SETUP_ONLY_NO_SOLVE', 'settings':settings,
                          'resolved_maxwell':mw,'config_fingerprint':fingerprint(cfg)}, indent=2))
        return
    if Path(args.output).exists():
        raise FileExistsError('Preserve completed/failed solver records; choose a new output')
    # Import only after setup and dry-run validation. This is the first action
    # capable of executing a Maxwell calculation; caller controls allocation.
    import meep as mp
    from meep.simulation import FluxData
    start = time.monotonic()
    radius, sr = settings['monitor_radius_um'], settings['cell_radial_extent_um']
    exit_z, cross_z = settings['output_z_um'], settings['crosscheck_z_um']
    top, bottom = settings['top_um'], settings['bottom_um']
    origin = (top+bottom)/2
    at = lambda r,z: mp.Vector3(r,0,z-origin)
    mats = {key:mp.Medium(index=value['n']) for key,value in cfg['materials'].items()}
    geometry = []
    if design:
        geometry.append(mp.Block(material=mats['substrate'],size=mp.Vector3(sr,mp.inf,-bottom),
                                 center=at(sr/2,bottom/2)))
        if design.get('kind') != 'bare_fiber':
            shape = {'annuli':cfg['annuli'],'aperture_diameter_um':cfg['cylinder_diameter_um']}
            for polygon in profile_polygons(heights+args.H, shape):
                geometry.append(mp.Prism(vertices=[mp.Vector3(r,.5,z-origin) for r,z in polygon],
                                         axis=mp.Vector3(0,-1,0),height=1,material=mats['polymer']))
    wavelength = float(cfg['wavelength_um'])
    frequency = 1/wavelength
    pulse = mp.GaussianSource(frequency,fwidth=.2*frequency,is_integrated=True)
    ns, waist, source_z = cfg['materials']['substrate']['n'], cfg['mfd_um']/2, mw['source_z_um']
    def profile(v):
        return source_envelope(v.x+sr/2,source_z,waist,wavelength,ns)
    sources = [mp.Source(pulse,component=c,center=at(sr/2,source_z),size=mp.Vector3(sr,0,0),
                         amplitude=amp,amp_func=profile)
               for c,amp in ((mp.Er,1),(mp.Ep,1j*args.m))]
    sim = mp.Simulation(cell_size=mp.Vector3(sr,0,top-bottom),dimensions=mp.CYLINDRICAL,m=args.m,
                        resolution=args.resolution,boundary_layers=[mp.PML(settings['pml_um'])],
                        sources=sources,geometry=geometry,
                        default_material=mats['ambient'] if design else mats['substrate'],
                        force_complex_fields=True,accurate_fields_near_cylorigin=True,
                        Courant=settings['Courant'])
    def zflux(z):
        return sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(radius/2,z),
                            size=mp.Vector3(radius,0,0),direction=mp.Z))
    reflection, transmitted = zflux(mw['reflection_z_um']), zflux(exit_z)
    side = sim.add_flux(frequency,0,1,mp.FluxRegion(
        center=at(radius,(exit_z+mw['reflection_z_um'])/2),
        size=mp.Vector3(0,0,exit_z-mw['reflection_z_um']),direction=mp.R))
    components = [mp.Er,mp.Ep,mp.Ez,mp.Hr,mp.Hp,mp.Hz]
    field_zs = [exit_z,cross_z] if design else [mw['reflection_z_um'],exit_z,cross_z]
    monitors = [sim.add_dft_fields(components,[frequency],center=at(radius/2,z),
                                 size=mp.Vector3(radius,0,0)) for z in field_zs]
    delta = 1/args.resolution
    pair_zs, pair_weights = [], []
    sim.init_sim()
    for z in field_zs:
        sampled_z=set()
        def stencil_probe(v):
            sampled_z.add(float(v.z+origin));return 1.
        sim.integrate_field_function([],stencil_probe,center=at(radius/2,z),size=mp.Vector3(radius,0,0))
        positions,weights=centered_stencil_from_probe(z,sampled_z,delta)
        pair_zs.extend(positions);pair_weights.append(weights)
    pair_monitors = [sim.add_dft_fields(components,[frequency],center=at(radius/2,z),
                                      size=mp.Vector3(radius,0,0),decimation_factor=1) for z in pair_zs]
    audit_fluxes = [zflux(z) for z in field_zs]
    aperture_radii = [3.,8.,15.,radius]
    aperture_fluxes = [sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(a/2,exit_z),
                                   size=mp.Vector3(a,0,0),direction=mp.Z)) for a in aperture_radii]
    if design:
        ref = json.loads(Path(args.reference).read_text(encoding='utf-8-sig'))
        if ref['settings'] != settings or ref['config_fingerprint'] != fingerprint(cfg):
            raise ValueError('Reference physical/numerical configuration differs')
        saved = np.load(ref['flux_data_file'],allow_pickle=False)
        sim.load_minus_flux_data(reflection,FluxData(E=saved['E'],H=saved['H']))
    sim.run(until_after_sources=settings['duration'])
    radial_positions, axial_positions = set(), set()
    def position_probe(v):
        radial_positions.add(float(v.x));axial_positions.add(float(v.z+origin));return 1.
    sim.integrate_field_function([],position_probe,center=at(radius/2,exit_z),size=mp.Vector3(radius,0,0))
    actual_r = np.asarray(sorted(radial_positions))
    if not np.allclose(actual_r,(np.arange(len(actual_r))+.5)/args.resolution,rtol=0,atol=1e-9):
        raise RuntimeError('Unexpected actual radial sample coordinates; no guessed metadata')
    get = lambda monitor: float(mp.get_fluxes(monitor)[0])
    arrays = [[np.asarray(sim.get_dft_array(monitor,c,0)).squeeze() for c in components]
              for monitor in monitors]
    if any(len(v) != len(actual_r) for plane in arrays for v in plane):
        raise RuntimeError('Field components and actual radial coordinates differ')
    field_path = str(Path(args.output).with_suffix('.fields.npz'))
    np.savez_compressed(field_path,r_um=actual_r,z_um=field_zs,fields=np.asarray(arrays),
                        radial_coordinate_uncertainty_um=delta)
    result = dict(solver='CPU Meep '+mp.__version__,settings=settings,resolved_maxwell=mw,
        config_fingerprint=fingerprint(cfg),requested_config=args.config,fields_file=field_path,
        result_label=cfg['result_label'],worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source='Gaussian electric-current surrogate; supplied MFD, assumed input phase index; no fiber eigenmode',
        power_convention='m=+1/reflected-m=-1 centered LP reconstruction; common azimuth factor cancels',
        input_index=ns,polymer_index=cfg['materials']['polymer']['n'],
        ambient_index=cfg['materials']['ambient']['n'],wavelength_um=wavelength)
    if design:
        incident = ref['incident_power']
        T,R,S = get(transmitted)/incident,-get(reflection)/incident,get(side)/incident
        result.update(kind=design.get('kind','height_family'),transmission=T,reflection=R,side_power=S,
                      power_balance_error=abs(T+R+S-1),design_file=args.design,reference_file=args.reference)
    else:
        incident = get(reflection)
        data = sim.get_flux_data(reflection)
        fluxfile = str(Path(args.output).with_suffix('.flux.npz'))
        np.savez_compressed(fluxfile,E=data.E,H=data.H)
        result.update(incident_power=incident,transmission=get(transmitted)/incident,
                      side_power=get(side)/incident,flux_data_file=fluxfile,
                      power_balance_error=abs((get(transmitted)+get(side))/incident-1))
    paired = [[np.asarray(sim.get_dft_array(monitor,c,0)).squeeze() for c in components]
              for monitor in pair_monitors]
    audit_path = str(Path(args.output).with_suffix('.power-audit.npz'))
    data = {'r_um':actual_r,'plane_z_um':np.asarray(field_zs),
        'pair_z_um':np.asarray(pair_zs).reshape((-1,2)),
        'paired_fields':np.asarray(paired).reshape((len(field_zs),2,6,-1)),
        'monitor_fluxes':np.asarray([get(f) for f in audit_fluxes]),
        'resolution_per_um':args.resolution,'Courant':settings['Courant'],
        'pair_weights':np.asarray(pair_weights),'aperture_radii_um':np.asarray(aperture_radii),
        'aperture_fluxes':np.asarray([get(f) for f in aperture_fluxes])}
    for i,monitor in enumerate(audit_fluxes):
        fd = sim.get_flux_data(monitor);data[f'flux_E_{i}']=fd.E;data[f'flux_H_{i}']=fd.H
    np.savez_compressed(audit_path,**data)
    result['power_audit_fields_file'] = audit_path
    result['grid_metadata'] = {
        'method':'Actual centered coordinate-only Meep integration probe',
        'r_first_um':float(actual_r[0]),'r_last_um':float(actual_r[-1]),'delta_um':delta,
        'sampled_z_um':sorted(axial_positions),'physical_z_origin_offset':origin,
        'monitor_radius_um':radius,
        'thin_plane_collapse':'Coordinate-probed neighboring centered fields and actual geometric pair_weights exported',
        'pair_coordinate_method':'Coordinate-only centered stencil probe after initialization, before timesteps; no assumed bottom/grid alignment'}
    result['wall_s'] = time.monotonic()-start
    write(args.output,result)
    print(json.dumps({'output':args.output,'wall_s':result['wall_s'],
                      'balance':result['power_balance_error']},indent=2),flush=True)
    sim.reset_meep()


if __name__ == '__main__':
    main()
