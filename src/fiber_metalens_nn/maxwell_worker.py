"""One serial CPU Meep job, isolated from the CUDA/JAX environment.

An m=+1 circular sector and its reflected m=-1 partner reconstruct x-linear
illumination. Circular and linear azimuth-integrated power differ by 2;
the same factor cancels in every reported normalized flux ratio.
"""
import argparse
import json
import time
from pathlib import Path

import meep as mp
from meep.simulation import FluxData
import numpy as np

from .geometry import load_config, edges, feasible, profile_polygons
from .metrics import peak_position, width_metrics
from .trial import write_json
from .maxwell_analysis import analyze_fields


def settings(config, resolution, pml=None, padding=None, duration=None, m=1):
    c = config["maxwell"]
    pml = c["pml_um"] if pml is None else pml
    padding = c["radial_padding_um"] if padding is None else padding
    # All scan points and the transmission plane must lie outside the PML.
    top = max(c["top_z_um"],config["exit_plane_z_um"]+config["proxy"]["focus_scan_max_um"]+pml+2.5)
    return {"resolution":resolution,"pml":pml,"padding":padding,
            "duration":c["after_sources_time"] if duration is None else duration,
            "m":m,"top":top,"bottom":c["bottom_z_um"],
            "transmission_z":top-pml-1,"requested_top":c["top_z_um"],
            "requested_transmission_z":c["transmission_z_um"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/desktop_trial.json")
    parser.add_argument("--design")
    parser.add_argument("--reference",help="Reference JSON to normalize a device job")
    parser.add_argument("--output",required=True)
    parser.add_argument("--resolution",type=float,required=True)
    parser.add_argument("--pml",type=float)
    parser.add_argument("--padding",type=float)
    parser.add_argument("--duration",type=float)
    parser.add_argument("--m",type=int,choices=[-1,1],default=1)
    args = parser.parse_args()
    start = time.monotonic()
    cfg = load_config(args.config)
    s = settings(cfg,args.resolution,args.pml,args.padding,args.duration,args.m)
    if any(value["kappa"] != 0 for value in cfg["materials"].values()):
        raise ValueError("This lossless synthetic worker requires kappa=0; dispersive loss is not implemented")
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    a = cfg["aperture_diameter_um"]/2
    radius = a+s["padding"]
    sr = radius+s["pml"]
    origin = (s["top"]+s["bottom"])/2
    sz = s["top"]-s["bottom"]
    at = lambda r,z:mp.Vector3(r,0,z-origin)
    substrate = mp.Medium(index=cfg["materials"]["substrate"]["n"])
    polymer = mp.Medium(index=cfg["materials"]["polymer"]["n"])
    ambient = mp.Medium(index=cfg["materials"]["ambient"]["n"])
    geometry = []
    design = None
    if args.design:
        design = json.loads(Path(args.design).read_text())
        heights = np.asarray(design["heights_um"])
        assert not feasible(heights,cfg)
        geometry.append(mp.Block(material=substrate,size=mp.Vector3(sr,mp.inf,-s["bottom"]),center=at(sr/2,s["bottom"]/2)))
        e = edges(cfg)
        rings = list(zip(e[:-1],e[1:],heights))
        if cfg["maxwell"].get("merge_equal_height_annuli",False):
            merged = []
            for inner,outer,h in rings:
                if merged and h == merged[-1][2]:
                    merged[-1] = (merged[-1][0],outer,h)
                else:
                    merged.append((inner,outer,h))
            rings = merged
        if cfg["maxwell"].get("profile_prism",False):
            for points in profile_polygons(heights,cfg):
                geometry.append(mp.Prism(vertices=[mp.Vector3(r,.5,z-origin) for r,z in points],
                                         axis=mp.Vector3(0,-1,0),height=1,material=polymer))
        else:
            for inner,outer,h in rings:
                if h > 0:
                    geometry.append(mp.Block(material=polymer,size=mp.Vector3(outer-inner,mp.inf,h),center=at((inner+outer)/2,h/2)))
        thickness = cfg["illumination"]["aperture_stop_thickness_um"]
        geometry.append(mp.Block(material=mp.metal,size=mp.Vector3(sr-a,mp.inf,thickness),center=at((a+sr)/2,-thickness/2)))
    frequency = 1/cfg["wavelength_um"]
    pulse = mp.GaussianSource(frequency,fwidth=frequency*cfg["maxwell"]["source_fwidth_fraction"],is_integrated=True)
    sources = [mp.Source(pulse,component=component,center=at(sr/2,cfg["maxwell"]["source_z_um"]),
                         size=mp.Vector3(sr,0,0),amplitude=amplitude)
               for component,amplitude in [(mp.Er,1),(mp.Ep,1j*args.m)]]
    sim = mp.Simulation(cell_size=mp.Vector3(sr,0,sz),dimensions=mp.CYLINDRICAL,m=args.m,
                        resolution=s["resolution"],boundary_layers=[mp.PML(s["pml"])],
                        sources=sources,geometry=geometry,default_material=ambient if design else substrate,
                        force_complex_fields=True,accurate_fields_near_cylorigin=True,Courant=.45)
    refl_z = cfg["maxwell"]["reflection_z_um"]
    def axial_flux(rad,z):
        return sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(rad/2,z),size=mp.Vector3(rad,0,0),direction=mp.Z))
    refl = axial_flux(radius,refl_z)
    incident_aperture = axial_flux(a,refl_z) if not design else None
    trans = axial_flux(radius,s["transmission_z"])
    side = sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(radius,(refl_z+s["transmission_z"])/2),
                         size=mp.Vector3(0,0,s["transmission_z"]-refl_z),direction=mp.R))
    field_monitors = []
    if design:
        if not args.reference:
            raise ValueError("A device requires an independent homogeneous reference")
        reference = json.loads(Path(args.reference).read_text())
        assert reference["settings"] == s
        sim.init_sim()
        ref_data = np.load(reference["flux_data_file"])
        sim.load_minus_flux_data(refl,FluxData(E=ref_data["E"],H=ref_data["H"]))
        target = design["target_focal_um"]
        collected = axial_flux(cfg["collection_radius_um"],cfg["exit_plane_z_um"]+target)
        focus_z = np.arange(cfg["proxy"]["focus_scan_min_um"],cfg["proxy"]["focus_scan_max_um"]+.01,cfg["proxy"]["focus_scan_step_um"])
        components = [mp.Er,mp.Ep,mp.Ez]
        for z in focus_z:
            # Point monitors use exact requested axial coordinates. Radial slices
            # are centered on Yee voxels; width coordinate uncertainty <= 1 cell.
            axis = sim.add_dft_fields(components,[frequency],center=at(0,cfg["exit_plane_z_um"]+z),size=mp.Vector3())
            line = sim.add_dft_fields(components,[frequency],center=at(1.65,cfg["exit_plane_z_um"]+z),size=mp.Vector3(3.3,0,0))
            field_monitors.append((axis,line))
    sim.run(until_after_sources=s["duration"])
    flux = lambda monitor:float(mp.get_fluxes(monitor)[0])
    result = {"result_label":cfg["result_label"],"solver":"CPU Meep "+mp.__version__,"settings":s,
              "m_reconstruction":"equal m=+1 and reflected m=-1; integrated LP power is half the circular-sector power; ratio cancels",
              "requested_config":args.config,"physical_origin_z_um":origin}
    if not design:
        incoming = flux(refl)
        aperture_power = flux(incident_aperture)
        data = sim.get_flux_data(refl)
        filename = str(output.with_suffix(".flux.npz"))
        np.savez_compressed(filename,E=data.E,H=data.H)
        result.update(incident_total_power=incoming,incident_aperture_power=aperture_power,flux_data_file=filename,
                      transmission_ratio=flux(trans)/incoming,side_ratio=flux(side)/incoming,
                      aperture_area_ratio=aperture_power/incoming,expected_aperture_area_ratio=a*a/(radius*radius))
        result["pass"] = bool(incoming > 0 and abs(result["transmission_ratio"]-1)<.02 and abs(result["side_ratio"])<.02
                              and abs(result["aperture_area_ratio"]/result["expected_aperture_area_ratio"]-1)<.02)
    else:
        incident = reference["incident_total_power"]
        reflection,transmission,side_power = -flux(refl),flux(trans),flux(side)
        balance = abs((reflection+transmission+side_power)/incident-1)
        axial = []
        profiles = []
        for axis,line in field_monitors:
            axis_values = [np.asarray(sim.get_dft_array(axis,c,0)).squeeze().item() for c in components]
            axial.append(float(abs(axis_values[0])**2+abs(axis_values[2])**2))
            profiles.append([np.asarray(sim.get_dft_array(line,c,0)).squeeze() for c in components])
        focal,boundary = peak_position(focus_z,np.asarray(axial))
        index = int(np.argmax(axial))
        er,ep,ez = profiles[index]
        # Meep's public cylindrical API does not provide array metadata. The
        # radial line covers [0,3.3], following its official zone-plate example.
        r = np.linspace(0,3.3,len(er))
        x = np.concatenate([-r[:0:-1],r])
        ix = abs(er)**2+abs(ez)**2
        iy = abs(ep)**2
        widths = width_metrics(x,np.concatenate([ix[:0:-1],ix]),np.concatenate([iy[:0:-1],iy]))
        fieldfile = output.with_suffix(".fields.npz")
        np.savez_compressed(fieldfile,r_um=r,focus_scan_um=focus_z,axis_intensity=axial,
                            Er=np.asarray(profiles)[:,0],Ep=np.asarray(profiles)[:,1],Ez=np.asarray(profiles)[:,2])
        result.update(design_file=args.design,kind=design["kind"],target_um=target,focal_um=float(focal),focus_on_scan_boundary=bool(boundary),
                      focal_reference_plane_z_um=cfg["exit_plane_z_um"],eta_R=flux(collected)/reference["incident_aperture_power"],
                      collection_radius_um=cfg["collection_radius_um"],reflection=reflection/incident,
                      transmission=transmission/reference["incident_aperture_power"],transmission_full_control_surface=transmission/incident,
                      side_power=side_power/incident,power_balance_error=balance,widths=widths,
                      width_axial_plane_um=float(focus_z[index]),width_radial_coordinate_uncertainty_um=1/s["resolution"],
                      fields_file=str(fieldfile),reference_file=args.reference,
                      sidelobe_peak_ratio_x=float(np.max(ix[r>=1.1])/max(np.max(ix),1e-30)),
                      sidelobe_peak_ratio_y=float(np.max(iy[r>=1.1])/max(np.max(iy),1e-30)))
        result["power_balance_pass"] = balance < cfg["maxwell"]["balance_tolerance"]
    result = analyze_fields(result)
    result["wall_s"] = time.monotonic()-start
    write_json(output,result)
    print(json.dumps({"output":str(output),"wall_s":result["wall_s"],"eta_R":result.get("eta_R"),"reference_pass":result.get("pass")}),flush=True)
    sim.reset_meep()


if __name__ == "__main__":
    main()
