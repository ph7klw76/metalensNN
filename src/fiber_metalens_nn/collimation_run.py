"""Reproducible Phase 8--10 CLI, with distinct configs, raw data, and gates."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .collimation_optics import (write,fingerprint,cylindrical_to_cartesian,forward_spectrum,
                                 propagate,describe,overlap,grid,vector_spectrum,flux)


def effective_duration(config,H,requested=None):
    """Allow a complete first round trip through even a taller comparison body."""
    requested=config["maxwell"]["after_sources_time"] if requested is None else requested
    nmax=max(m["n"] for m in config["materials"].values())
    flight=np.ceil(2*nmax*(H+config["height_max_um"])+20)
    return float(max(requested,flight))


class Experiment:
    def __init__(self,config,root):
        self.config=json.loads(Path(config).read_text())
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.key=fingerprint(self.config)
        frozen=self.root/"frozen-config.json"
        if frozen.exists() and fingerprint(json.loads(frozen.read_text()))!=self.key:
            raise ValueError("Frozen config differs: preserve run and choose a new root")
        write(frozen,self.config)
        self.config_file=str(frozen)
        self.worker_hash=hashlib.sha256(b"".join((Path(__file__).parent/name).read_bytes() for name in
                    ("collimation_worker.py","collimation_optics.py","geometry.py"))).hexdigest()
        self.started=time.monotonic()

    def design(self,name,heights,H=35,diameter=30,kind=None):
        step=self.config["fabrication"]["height_step_um"]
        heights=np.clip(np.round(np.asarray(heights)/step)*step,0,self.config["height_max_um"])
        value=dict(kind=kind or name,H_um=float(H),target_diameter_um=float(diameter),heights_um=heights.tolist(),
                   cylinder_diameter_um=self.config["cylinder_diameter_um"],architecture=self.config["architecture"],
                   config_fingerprint=self.key,units="um",output_reference_z_um=float(H+self.config["height_max_um"]+self.config["proxy"]["output_clearance_um"]))
        path=self.root/"designs"/(name+".json")
        write(path,value)
        return str(path)

    def maxwell(self,design,H,resolution,_transit_attempt=0,**options):
        duration=effective_duration(self.config,H,options.get("duration"))
        if duration!=options.get("duration",self.config["maxwell"]["after_sources_time"]):
            options={**options,"duration":duration}
        folder=self.root/"maxwell"
        folder.mkdir(exist_ok=True)
        params=dict(H=H,resolution=resolution,**options)
        contents=json.loads(Path(design).read_text()) if design else None
        cache=fingerprint(dict(config=self.config,design=contents,params=params,source=self.worker_hash))[:20]
        output=folder/(('device-' if design else 'reference-')+cache+".json")
        # JSON distinguishes 6 and 6.0 although they request identical numerics.
        # Search equivalent legacy keys before scheduling an expensive solver.
        numeric=[key for key,value in params.items() if isinstance(value,(int,float)) and float(value).is_integer()]
        if not output.exists():
            for representations in itertools.product((int,float),repeat=len(numeric)):
                equivalent=dict(params)
                for key,cast in zip(numeric,representations): equivalent[key]=cast(params[key])
                key=fingerprint(dict(config=self.config,design=contents,params=equivalent,source=self.worker_hash))[:20]
                candidate=folder/(('device-' if design else 'reference-')+key+".json")
                if candidate.exists():
                    output=candidate
                    break
        if output.exists():
            value=json.loads(output.read_text())
            required=[value["fields_file"]]+([] if design else [value["flux_data_file"]])
            if value["config_fingerprint"]!=self.key or not all(Path(p).is_file() for p in required):
                raise RuntimeError(f"Incomplete or mismatched cached measurement: {output}")
            if "artifact_file" not in value:
                value["artifact_file"]=str(output)
                value["worker_source_sha256"]=self.worker_hash
                write(output,value)
            return self.transit_checked(value,design,H,resolution,options,_transit_attempt)
        if design:
            ledger=folder/"geometry-ledger.json"
            entries=json.loads(ledger.read_text()) if ledger.exists() else []
            geometry=fingerprint(dict(heights=contents["heights_um"],H=H,kind="bare" if contents["kind"]=="bare_fiber" else "polymer"))
            if geometry not in entries:
                if len(entries)>=self.config["maxwell"]["max_unique_candidates"]:
                    raise RuntimeError("Predeclared unique-geometry budget exhausted")
                entries.append(geometry)
                write(ledger,entries)
            reference=self.maxwell(None,H,resolution,**options)
        args=[".venv-meep/bin/python","-m","fiber_metalens_nn.collimation_worker","--config",self.config_file,
              "--output",str(output),"--H",str(H),"--resolution",str(resolution)]
        for key,value in options.items():
            args.extend(["--"+key,str(value)])
        if design:
            args.extend(["--design",str(design),"--reference",reference["artifact_file"]])
        print("Maxwell:",contents["kind"] if contents else "homogeneous",params,flush=True)
        with output.with_suffix(".log").open("w") as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=self.config["maxwell"]["child_timeout_s"],check=False)
        if result.returncode!=0 or not output.exists():
            raise RuntimeError(f"Meep exit {result.returncode}; inspect {output.with_suffix('.log')}")
        value=json.loads(output.read_text())
        value["artifact_file"]=str(output)
        value["worker_source_sha256"]=self.worker_hash
        write(output,value)
        return self.transit_checked(value,design,H,resolution,options,_transit_attempt)

    def transit_checked(self,value,design,H,resolution,options,attempt):
        """Verify long-path completion instead of trusting a geometrical estimate.

        Coarse-grid numerical group delay and reflected pulses can outlast the
        nominal round trip. Keep failed records; bounded retries add no geometry.
        """
        mw=self.config["maxwell"]
        if effective_duration(self.config,H)<=mw["after_sources_time"] or value["power_balance_error"]<=mw["balance_tolerance"]:
            return value
        if attempt>=2:
            raise RuntimeError(f"Power closure still fails after two duration extensions: {value['artifact_file']}")
        duration=float(np.ceil(value["settings"]["duration"]*1.5))
        log=self.root/"transit-duration-repairs.json"
        repairs=json.loads(log.read_text()) if log.exists() else []
        entry=dict(source_record=value["artifact_file"],power_balance_error=value["power_balance_error"],
                   next_duration=duration,reason="Tall-device reflected pulse not settled; bounded duration test includes numerical group delay",
                   geometries_added=0,threshold_changed=False)
        if entry not in repairs:
            repairs.append(entry)
            write(log,repairs)
        return self.maxwell(design,H,resolution,_transit_attempt=attempt+1,**dict(options,duration=duration))

    def analyze(self,record,diameter=30,count=512,window=128,save=None):
        from .collimation_metrics import attach_divergence_fit
        saved=np.load(record["fields_file"])
        x,dx,fields=cylindrical_to_cartesian(saved["r_um"],saved["fields"][0],count,window)
        spectrum=forward_spectrum(fields,dx,self.config["wavelength_um"])
        value=describe(spectrum,dx,self.config["wavelength_um"],diameter,self.config,record["transmission"],save)
        value=attach_divergence_fit(value,spectrum,dx,self.config["wavelength_um"])
        x,_,cross=cylindrical_to_cartesian(saved["r_um"],saved["fields"][1],count,window)
        measured=propagate(forward_spectrum(cross,dx,self.config["wavelength_um"]),dx,self.config["wavelength_um"],0)
        predicted=propagate(spectrum,dx,self.config["wavelength_um"],self.config["maxwell"]["crosscheck_distance_um"])
        value.update(crosscheck_overlap=overlap(predicted,measured,dx),power_balance_error=record["power_balance_error"],
                     record=record["artifact_file"],result_label=self.config["result_label"],
                     radial_coordinate_uncertainty_um=float(saved["radial_coordinate_uncertainty_um"]))
        return value


def phase8(experiment):
    import jax
    import jax.numpy as jnp
    from .collimation_proxy import CylinderProxy
    cfg=experiment.config
    folder=experiment.root/"phase8"
    folder.mkdir(exist_ok=True)
    if not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("RTX GPU required for this run")
    zr=np.pi*(cfg["mfd_um"]/2)**2/cfg["wavelength_um"]
    report=dict(input="Gaussian surrogate; nominal NA not imposed as divergence",materials=cfg["material_provenance"],
                air_Rayleigh_um=zr,air_diameters_30_35_40_um=[cfg["mfd_um"]*np.sqrt(1+(z/zr)**2) for z in (30,35,40)],
                air_distance_for_30um=zr*np.sqrt((30/cfg["mfd_um"])**2-1),
                polymer_homogeneous_Rayleigh_um=zr*cfg["materials"]["polymer"]["n"],
                polymer_homogeneous_diameter_at_35um=cfg["mfd_um"]*np.sqrt(1+(35/(zr*cfg["materials"]["polymer"]["n"]))**2),
                target_Rayleigh_um=np.pi*15**2/cfg["wavelength_um"],
                target_diffraction_half_angle_rad=cfg["wavelength_um"]/(np.pi*15),
                Gaussian_99percent_clear_aperture_um=30*np.sqrt(np.log(100)/2),
                preferred_architecture=cfg["architecture"],H_definition="support top / relief base; output is H+hmax+clearance",
                alternatives="Hollow wall and integrated expansion stages are not implemented in this initial solid-pedestal branch")
    write(folder/"feasibility.json",report)
    proxy=CylinderProxy(cfg)
    H,D=34.,28.
    seed=proxy.seed(H)
    pedestal=proxy.pedestal(H)
    target=proxy.target(D)
    loss,grad=proxy.value_gradient(seed,pedestal,target,D)
    gradient=[]
    for i in (0,2,4,6):
        eps=.01
        direction=np.eye(cfg["annuli"],dtype=np.float32)[i]*eps
        fd=float((proxy.objective(seed+direction,pedestal,target,D)-proxy.objective(seed-direction,pedestal,target,D))/(2*eps))
        ad=float(grad[i])
        gradient.append(dict(ring=i,autodiff=ad,finite_difference=fd,relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-5)))
    write(folder/"gradients.json",gradient)
    flat=experiment.design("phase8-flat-H34",np.zeros(cfg["annuli"]),H,D,"flat_cylinder")
    record=experiment.maxwell(flat,H,cfg["maxwell"]["resolutions_per_um"][0])
    measured=experiment.analyze(record,D,save=folder/"flat-maxwell-planes.npz")
    predicted=proxy.evaluate(np.zeros(cfg["annuli"],np.float32),H,D,folder/"flat-proxy-planes.npz")
    # Independent input reference at the facet, measured before any polymer.
    ref=json.loads(Path(record["reference_file"]).read_text())
    saved=np.load(ref["fields_file"])
    x,dx,field=cylindrical_to_cartesian(saved["r_um"],saved["fields"][0],512,128)
    from .metrics import full_width
    intensity=np.sum(abs(field[:3])**2,axis=0)
    m=len(x)//2
    input_width=full_width(x,intensity[m-1:m+1].mean(axis=0),np.exp(-2))
    discrepancy=dict(diameter_relative=abs(predicted["planes"][0]["diameter_x_um"]/measured["planes"][0]["diameter_x_um"]-1),
                     transmission_relative=abs(predicted["transmission"]/measured["transmission"]-1),
                     measured_input_diameter_um=input_width,input_diameter_relative_error=abs(input_width/cfg["mfd_um"]-1))
    # This diagnostic does not calibrate to the held-out target or change its metric.
    write(folder/"flat-cylinder-proxy-vs-Maxwell.json",dict(proxy=predicted,Maxwell=measured,discrepancy=discrepancy))
    sampling=[]
    for label,kwargs in (("base",{}),("axial",dict(dz=.25,slices=48)),("spatial",dict(grid_n=256))):
        p=CylinderProxy(cfg,**kwargs)
        metric=p.evaluate(np.asarray(p.seed(H)),H,D)
        sampling.append(dict(label=label,eta_mode=metric["eta_mode"],diameter=metric["planes"][0]["diameter_x_um"]))
    errors=[abs(s["eta_mode"]/sampling[0]["eta_mode"]-1) for s in sampling[1:]]
    write(folder/"sampling.json",sampling)
    passed=max(g["relative_error"] for g in gradient)<.05 and max(errors)<.05 and discrepancy["diameter_relative"]<.05 and discrepancy["transmission_relative"]<.05 and discrepancy["input_diameter_relative_error"]<.03 and measured["crosscheck_overlap"]>.98 and record["power_balance_error"]<.05
    result=dict(phase=8,status="PASS" if passed else "FAIL_MODEL_OR_NUMERICS",gradient_max_relative_error=max(g["relative_error"] for g in gradient),
                sampling_eta_relative_changes=errors,flat_model_discrepancy=discrepancy,crosscheck_overlap=measured["crosscheck_overlap"],
                ram_gates=False,devices=[str(d) for d in jax.devices()],result_label=cfg["result_label"])
    write(folder/"gate.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def smoke(experiment):
    import jax
    import jax.numpy as jnp
    from .collimation_proxy import CylinderProxy
    proxy=CylinderProxy(experiment.config)
    h=proxy.seed(34)
    value,g=proxy.value_gradient(h,proxy.pedestal(34),proxy.target(28),28.)
    passed=bool(np.isfinite(float(value)) and np.isfinite(np.asarray(g)).all() and any(d.platform=="gpu" for d in jax.devices()))
    result=dict(status="PASS_SMOKE" if passed else "FAIL_SMOKE",loss=float(value),gradient_norm=float(jnp.linalg.norm(g)),
                devices=[str(d) for d in jax.devices()],full_phase_acceptance=False,ram_gates=False)
    write(experiment.root/"smoke.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("smoke","phase8","train","validate","transit","refine","package","all"))
    parser.add_argument("--config",default="configs/1550nm_collimation.json")
    parser.add_argument("--root",default="artifacts/1550nm_collimation")
    args=parser.parse_args()
    experiment=Experiment(args.config,args.root)
    if args.stage=="smoke":
        smoke(experiment)
    if args.stage in ("phase8","all"):
        phase8(experiment)
    if args.stage in ("train","all"):
        from .collimation_train import train
        train(experiment)
    if args.stage in ("validate","all"):
        from .collimation_validate import validate
        result=validate(experiment)
        if args.stage=="all" and result["engineering_gate"]!="PASS":
            from .collimation_refine import refine
            refine(experiment)
    if args.stage=="refine":
        from .collimation_refine import refine
        refine(experiment)
    if args.stage=="transit":
        path=experiment.root/"phase9/taller-cylinder-comparison.json"
        values=json.loads(path.read_text())
        for label,item in values.items():
            original=json.loads(Path(item["metrics"]["record"]).read_text())
            record=experiment.maxwell(original["design_file"],item["H_um"],original["settings"]["resolution"])
            item["metrics"]=experiment.analyze(record,30)
            item["timing_verified"]=record["power_balance_error"]<experiment.config["maxwell"]["balance_tolerance"]
            write(path,values)
    if args.stage in ("package","all"):
        from .collimation_package import package
        package(experiment)


if __name__=="__main__":
    main()
