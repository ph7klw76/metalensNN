"""Serial cylindrical Maxwell worker for a finite pedestal and Gaussian input.

Run with .venv-meep; no JAX imports. Separate homogeneous reference subtraction.
"""
import argparse
import json
import time
from pathlib import Path
import meep as mp
from meep.simulation import FluxData
import numpy as np
from .geometry import profile_polygons
from .collimation_optics import write,fingerprint


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--design")
    p.add_argument("--reference")
    p.add_argument("--output",required=True)
    p.add_argument("--resolution",type=float,required=True)
    p.add_argument("--H",type=float,default=35)
    p.add_argument("--pml",type=float)
    p.add_argument("--padding",type=float)
    p.add_argument("--duration",type=float)
    p.add_argument("--m",type=int,choices=(-1,1),default=1)
    args=p.parse_args()
    start=time.monotonic()
    cfg=json.loads(Path(args.config).read_text())
    if any(m["kappa"]!=0 for m in cfg["materials"].values()):
        raise ValueError("Loss is not implemented; no silent kappa substitution")
    mw=cfg["maxwell"]
    pml=args.pml or mw["pml_um"]
    padding=args.padding or mw["padding_um"]
    duration=args.duration or mw["after_sources_time"]
    radius=cfg["cylinder_diameter_um"]/2+padding
    sr=radius+pml
    exit_z=args.H+cfg["height_max_um"]+cfg["proxy"]["output_clearance_um"]
    cross_z=exit_z+mw["crosscheck_distance_um"]
    top=cross_z+2+pml
    bottom=mw["bottom_um"]
    origin=(top+bottom)/2
    at=lambda r,z:mp.Vector3(r,0,z-origin)
    settings=dict(resolution=args.resolution,pml_um=pml,padding_um=padding,duration=duration,
                  H_um=args.H,output_z_um=exit_z,crosscheck_z_um=cross_z,top_um=top,bottom_um=bottom,m=args.m)
    mats={k:mp.Medium(index=v["n"]) for k,v in cfg["materials"].items()}
    design=json.loads(Path(args.design).read_text()) if args.design else None
    geometry=[]
    if design:
        geometry.append(mp.Block(material=mats["substrate"],size=mp.Vector3(sr,mp.inf,-bottom),center=at(sr/2,bottom/2)))
        if design["kind"]!="bare_fiber":
            heights=np.asarray(design["heights_um"],dtype=float)
            if len(heights)!=cfg["annuli"] or np.any(heights<0) or np.any(heights>cfg["height_max_um"]+1e-7):
                raise ValueError("Invalid relief heights")
            shape={"annuli":cfg["annuli"],"aperture_diameter_um":cfg["cylinder_diameter_um"]}
            for polygon in profile_polygons(heights+args.H,shape):
                geometry.append(mp.Prism(vertices=[mp.Vector3(r,.5,z-origin) for r,z in polygon],
                                         axis=mp.Vector3(0,-1,0),height=1,material=mats["polymer"]))
    frequency=1/cfg["wavelength_um"]
    pulse=mp.GaussianSource(frequency,fwidth=.2*frequency,is_integrated=True)
    ns=cfg["materials"]["substrate"]["n"]
    waist=cfg["mfd_um"]/2
    zr=np.pi*ns*waist**2/cfg["wavelength_um"]
    source_z=mw["source_z_um"]
    wz=waist*np.sqrt(1+(source_z/zr)**2)
    inverse_R=source_z/(source_z**2+zr**2)
    def profile(v):
        r=v.x+sr/2  # amp_func coordinates are relative to the source center.
        return waist/wz*np.exp(-r*r/wz**2+1j*np.pi*ns/cfg["wavelength_um"]*inverse_R*r*r)
    sources=[mp.Source(pulse,component=c,center=at(sr/2,source_z),size=mp.Vector3(sr,0,0),
                       amplitude=amp,amp_func=profile) for c,amp in ((mp.Er,1),(mp.Ep,1j*args.m))]
    sim=mp.Simulation(cell_size=mp.Vector3(sr,0,top-bottom),dimensions=mp.CYLINDRICAL,m=args.m,
                      resolution=args.resolution,boundary_layers=[mp.PML(pml)],sources=sources,geometry=geometry,
                      default_material=mats["ambient"] if design else mats["substrate"],
                      force_complex_fields=True,accurate_fields_near_cylorigin=True,Courant=.45)
    def zflux(z):
        return sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(radius/2,z),size=mp.Vector3(radius,0,0),direction=mp.Z))
    reflection=zflux(mw["reflection_z_um"])
    transmitted=zflux(exit_z)
    side=sim.add_flux(frequency,0,1,mp.FluxRegion(center=at(radius,(exit_z+mw["reflection_z_um"])/2),
                      size=mp.Vector3(0,0,exit_z-mw["reflection_z_um"]),direction=mp.R))
    components=[mp.Er,mp.Ep,mp.Ez,mp.Hr,mp.Hp,mp.Hz]
    field_zs=[exit_z,cross_z] if design else [0.,exit_z]
    monitors=[sim.add_dft_fields(components,[frequency],center=at(radius/2,z),size=mp.Vector3(radius,0,0)) for z in field_zs]
    if design:
        ref=json.loads(Path(args.reference).read_text())
        if ref["settings"]!=settings or ref["config_fingerprint"]!=fingerprint(cfg):
            raise ValueError("Reference physics or settings differ")
        sim.init_sim()
        saved=np.load(ref["flux_data_file"])
        sim.load_minus_flux_data(reflection,FluxData(E=saved["E"],H=saved["H"]))
    sim.run(until_after_sources=duration)
    get=lambda monitor:float(mp.get_fluxes(monitor)[0])
    arrays=[]
    for monitor in monitors:
        values=[np.asarray(sim.get_dft_array(monitor,c,0)).squeeze() for c in components]
        if len({len(v) for v in values})!=1:
            raise RuntimeError("DFT components require explicit Yee-grid interpolation")
        arrays.append(values)
    field_path=str(Path(args.output).with_suffix(".fields.npz"))
    np.savez_compressed(field_path,r_um=np.linspace(0,radius,len(arrays[0][0])),z_um=field_zs,fields=np.asarray(arrays),
                        radial_coordinate_uncertainty_um=1/args.resolution)
    result=dict(solver="CPU Meep "+mp.__version__,settings=settings,config_fingerprint=fingerprint(cfg),
                requested_config=args.config,fields_file=field_path,result_label=cfg["result_label"],
                source="Gaussian equivalent electric-current surrogate, backwards paraxial phase to nominal facet waist; measured reference checked separately",
                power_convention="m=+1 circular and reflected m=-1 reconstruct LP; common azimuth factor cancels in normalized powers")
    if design:
        incident=ref["incident_power"]
        T,R,S=get(transmitted)/incident,-get(reflection)/incident,get(side)/incident
        result.update(kind=design["kind"],transmission=T,reflection=R,side_power=S,power_balance_error=abs(T+R+S-1),
                      design_file=args.design,reference_file=args.reference)
    else:
        incident=get(reflection)
        data=sim.get_flux_data(reflection)
        fluxfile=str(Path(args.output).with_suffix(".flux.npz"))
        np.savez_compressed(fluxfile,E=data.E,H=data.H)
        result.update(incident_power=incident,transmission=get(transmitted)/incident,side_power=get(side)/incident,
                      flux_data_file=fluxfile,power_balance_error=abs((get(transmitted)+get(side))/incident-1))
    result["wall_s"]=time.monotonic()-start
    write(args.output,result)
    print(json.dumps({"output":args.output,"wall_s":result["wall_s"],"balance":result["power_balance_error"]}),flush=True)
    sim.reset_meep()


if __name__=="__main__":
    main()
