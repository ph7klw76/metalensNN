"""Read canonical evidence; write separate diagnostics. No GPU/JAX/Meep jobs.

The NumPy scalar split-step below independently transcribes the inspected
one-way proxy for two fixed-index evaluations of unchanged geometry. It is
not a full-vector dielectric solver or an optimization experiment.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def scalar_overlap(field, target):
    return float(abs(np.vdot(target, field))**2 / (np.vdot(field, field).real * np.vdot(target, target).real))


def amplitude_limit(field, target):
    return float(np.sum(abs(field)*abs(target))**2 / (np.vdot(field, field).real * np.vdot(target, target).real))


def scalar_summary(field, x, wavelength, n, target_diameter=30):
    xx, yy = np.meshgrid(x, x, indexing="xy")
    intensity = abs(field)**2
    total = intensity.sum()
    w_x = float(2*np.sqrt(np.sum(intensity*xx**2)/total))
    w_y = float(2*np.sqrt(np.sum(intensity*yy**2)/total))
    # w_x, w_y are 1/e^2-equivalent radii inferred from second moments.
    target = np.exp(-(xx**2+yy**2)/(target_diameter/2)**2)
    mid = len(x)//2
    phase = np.unwrap(np.angle(field[mid]))
    bright = (abs(x)<target_diameter/2) & (abs(field[mid])>.15*abs(field).max())
    coefficient = np.polyfit(x[bright], phase[bright], 2)
    inverse_R = float(2*coefficient[0]/(2*np.pi*n/wavelength))
    fitted = np.polyval(coefficient,x[bright])
    weights = intensity[mid,bright]
    rms = float(np.sqrt(np.sum(weights*(phase[bright]-fitted)**2)/weights.sum()))
    inverse_q = complex(inverse_R,-wavelength/(np.pi*n*((w_x+w_y)/2)**2))
    q = 1/inverse_q
    return {
        "scalar_Ex_D4sigma_equivalent_diameter_x_um":2*w_x,
        "scalar_Ex_D4sigma_equivalent_diameter_y_um":2*w_y,
        "scalar_Ex_complex_overlap_with_flat_30um_Gaussian":scalar_overlap(field,target),
        "scalar_Ex_fixed_amplitude_optimal_phase_overlap_limit":amplitude_limit(field,target),
        "limit_scope":"same saved scalar Ex amplitude, local phase only, same plane; not a bound on finite-thickness devices or vector fields",
        "core_fitted_inverse_R_per_um":inverse_R,
        "core_fitted_R_um":None if inverse_R==0 else 1/inverse_R,
        "core_phase_quadratic_fit_weighted_RMS_rad":rms,
        "Gaussian_equivalent_waist_distance_ahead_um":float(-q.real),
        "Gaussian_equivalent_Rayleigh_um":float(q.imag),
        "waist_fit_scope":"paraxial estimate from scalar second moment and central phase; not a full caustic M2 fit",
        "normalized_power_outside_radius_15um":float(intensity[xx**2+yy**2>225].sum()/total)
    }


def scalar_proxy(cfg, H, heights):
    p=cfg["proxy"]
    count=p["grid_n"]*p["padding"]
    dx=p["window_um"]/p["grid_n"]
    x=(np.arange(count)-count/2+.5)*dx
    xx,yy=np.meshgrid(x,x,indexing="xy")
    r=np.hypot(xx,yy)
    radius=cfg["cylinder_diameter_um"]/2
    disk=np.clip((radius-r)/dx+.5,0,1)
    ring_index=np.minimum((r/(radius/cfg["annuli"])).astype(int),cfg["annuli"]-1)
    k0=2*np.pi/cfg["wavelength_um"]
    ns=cfg["materials"]["substrate"]["n"]
    n=cfg["materials"]["polymer"]["n"]
    na=cfg["materials"]["ambient"]["n"]
    launch=np.exp(-(r/(cfg["mfd_um"]/2))**2).astype(np.complex64)
    incident=float(.5*ns*np.sum(abs(launch)**2)*dx**2)
    f=2*np.pi*np.fft.fftfreq(count,d=dx)
    transverse=f[:,None]**2+f[None,:]**2
    def kernel(index,d):
        return np.exp(1j*np.sqrt(((k0*index)**2-transverse).astype(np.complex64))*d).astype(np.complex64)
    def step(field,k,phase):
        first=np.fft.ifft2(np.fft.fft2(field)*k)*phase
        return np.fft.ifft2(np.fft.fft2(first)*k).astype(np.complex64)
    steps=int(np.ceil(H/p["pedestal_dz_um"]))
    dz=H/steps
    k=kernel(n,dz/2)
    phase=np.exp(1j*k0*(na+(n-na)*disk-n)*dz).astype(np.complex64)
    field=launch*(2*ns/(ns+n))
    for _ in range(steps):field=step(field,k,phase)
    pedestal=field.copy()
    dz=cfg["height_max_um"]/p["relief_slices"]
    ref=(n+na)/2
    k=kernel(ref,dz/2)
    local=np.asarray(heights)[ring_index]
    for z in np.arange(p["relief_slices"])*dz:
        fill=np.clip((local-z)/dz,0,1)*disk
        phase=np.exp(1j*k0*(na+(n-na)*fill-ref)*dz).astype(np.complex64)
        field=step(field,k,phase)
    field*=2*n/(n+na)
    return x,dx,pedestal,field,incident


def objective_terms(spectrum, target, cfg, dx, incident, heights, r, kernels, propagate):
    count=spectrum.shape[-1]
    def inner(a,b):
        return .25*np.sum(a[0]*np.conj(b[4])-a[1]*np.conj(b[3])+
            np.conj(b[0])*a[4]-np.conj(b[1])*a[3])*dx**2/count**2
    useful=float(abs(inner(spectrum,target))**2/(inner(target,target).real*incident))
    widths=[]
    for z in (0,100):
        field=propagate(spectrum,dx,cfg["wavelength_um"],z)
        flux=np.maximum(.5*np.real(field[0]*np.conj(field[4])-field[1]*np.conj(field[3])),0)
        widths.append(float(2*np.sqrt(2*np.sum(flux*r**2)/flux.sum())))
    loss=cfg["loss"]
    terms={"negative_useful_mode_power":-loss["mode"]*useful,
           "diameter_penalty":loss["diameter"]*(widths[0]/30-1)**2,
           "endpoint_stability_penalty":loss["stability"]*(widths[1]/widths[0]-1)**2,
           "height_roughness_penalty":float(loss["roughness"]*np.mean(np.diff(heights)**2))}
    return {"weighted_terms":terms,"total":sum(terms.values()),"useful_mode_power":useful,
            "radial_D4sigma_equivalent_diameters_0_100_um":widths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.canonical_root.resolve()
    output=args.output.resolve()
    if output.exists():raise FileExistsError("preserve previous diagnostics; choose new output")
    if root==output or root in output.parents:raise ValueError("outputs must be outside canonical project")
    start=time.perf_counter()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root/"src"))
    from fiber_metalens_nn.collimation_optics import (cylindrical_to_cartesian, forward_spectrum,
        propagate, vector_spectrum, overlap, flux, describe, kernels)
    cfg=read_json(root/"artifacts/1550nm_collimation/frozen-config.json")
    artifact=root/"artifacts/1550nm_collimation"
    gate=read_json(artifact/"phase9/gate.json")
    record=read_json(root/gate["NN"]["record"])
    design=read_json(root/record["design_file"])
    file_paths=list((root/"src/fiber_metalens_nn").glob("collimation_*.py"))
    file_paths += [root/"src/fiber_metalens_nn/metrics.py",root/"artifacts/1550nm_collimation/frozen-config.json",root/record["design_file"],root/gate["NN"]["record"],root/record["fields_file"],root.parent/"metalens_inverse_design_gpu_trial_prompts.md"]
    manifest={str(p):sha(p) for p in file_paths}
    output.mkdir(parents=True)
    plan=(root.parent/"metalens_inverse_design_gpu_trial_prompts.md").read_text(encoding="utf-8-sig")
    phase_start=re.search(r"^## Phase 8\b",plan,re.M).start()
    phase_end=re.search(r"^## Primary references",plan,re.M).start()
    (output/"revised_phases_8_to_10_verbatim.md").write_text(plan[phase_start:phase_end],encoding="utf-8")
    write_json(output/"preserved_config_n1p52.json",cfg)
    alternative=copy.deepcopy(cfg)
    alternative["materials"]["polymer"]["n"]=1.50
    alternative["result_label"]="DIAGNOSTIC_ONLY_PROVISIONAL_N1P50_UNCHANGED_GEOMETRY"
    write_json(output/"config_n1p50_sensitivity.json",alternative)
    saved=np.load(root/record["fields_file"],allow_pickle=False)
    x,dx,fields=cylindrical_to_cartesian(saved["r_um"],saved["fields"][0],512,128)
    spectrum=forward_spectrum(fields,dx,cfg["wavelength_um"])
    field=propagate(spectrum,dx,cfg["wavelength_um"],0)
    xx,yy=np.meshgrid(x,x,indexing="xy")
    gaussian=np.exp(-(xx**2+yy**2)/225)
    target=vector_spectrum(np.fft.fft2(gaussian),dx,cfg["wavelength_um"])
    target_field=propagate(target,dx,cfg["wavelength_um"],0)
    computed=overlap(field,target_field,dx)
    assert abs(computed-gate["NN"]["overlap"])<1e-10,"raw field overlap does not reproduce existing gate"
    maxwell_summary=scalar_summary(field[0],x,cfg["wavelength_um"],1)
    checks={"existing_vector_overlap_reproduced":abs(computed-gate["NN"]["overlap"])<1e-10,
            "Gaussian_identical_overlap":abs(scalar_overlap(gaussian,gaussian)-1)<1e-12,
            "Gaussian_radius5_15_overlap_0p36":abs(scalar_overlap(np.exp(-(xx**2+yy**2)/25),gaussian)-.36)<1e-12,
            "phase_limit_exceeds_actual_scalar_overlap":maxwell_summary["scalar_Ex_fixed_amplitude_optimal_phase_overlap_limit"]>=maxwell_summary["scalar_Ex_complex_overlap_with_flat_30um_Gaussian"]}
    proxies=[]
    for config in [cfg,alternative]:
        xp,dp,pedestal,exit_field,incident=scalar_proxy(config,design["H_um"],design["heights_um"])
        spec=vector_spectrum(np.fft.fft2(exit_field),dp,config["wavelength_um"])
        *_,valid,_=kernels(len(xp),dp,config["wavelength_um"])
        _,_,kz,_,_,_=kernels(len(xp),dp,config["wavelength_um"])
        spec*=np.exp(1j*kz*config["proxy"]["output_clearance_um"])[None]*valid[None]
        power=float(flux(propagate(spec,dp,config["wavelength_um"],0)).sum()*dp**2)
        metrics=describe(spec,dp,config["wavelength_um"],30,config,power/incident)
        xxp,yyp=np.meshgrid(xp,xp,indexing="xy")
        targetp=vector_spectrum(np.fft.fft2(np.exp(-(xxp**2+yyp**2)/225)),dp,config["wavelength_um"])
        summary={"polymer_n":config["materials"]["polymer"]["n"],"evidence":"fresh NumPy transcription of one-way scalar proxy, no optimization or Maxwell run",
                 "internal_pedestal_top_z_um":design["H_um"],"pedestal_top":scalar_summary(pedestal,xp,config["wavelength_um"],config["materials"]["polymer"]["n"]),
                 "output_metrics":{k:metrics[k] for k in ["overlap","transmission","eta_mode","maximum_diameter_change_100um","inverse_wavefront_radius_per_um"]},
                 "output_diameters_um":[metrics["planes"][0]["diameter_x_um"],metrics["planes"][0]["diameter_y_um"]],
                 "loss":objective_terms(spec,targetp,config,dp,incident,design["heights_um"],np.hypot(xxp,yyp),kernels,propagate)}
        proxies.append(summary)
        np.savez_compressed(output/f"proxy_n{summary['polymer_n']:.2f}_diagnostic.npz",x_um=xp,pedestal_top_scalar=pedestal,output_Ex=propagate(spec,dp,config["wavelength_um"],0)[0].astype(np.complex64))
    baseline=read_json(artifact/"phase9/trained-sampling.json")["samples"][0]["metrics"]
    parity={"overlap_absolute_error":abs(proxies[0]["output_metrics"]["overlap"]-baseline["overlap"]),
            "diameter_x_absolute_error_um":abs(proxies[0]["output_diameters_um"][0]-baseline["planes"][0]["diameter_x_um"])}
    checks["NumPy_proxy_parity_with_saved_JAX_metrics"]=parity["overlap_absolute_error"]<1e-4 and parity["diameter_x_absolute_error_um"]<.01
    # Construct a legitimate Gaussian with waist midway through the interval.
    zR=50.
    w0=math.sqrt(cfg["wavelength_um"]*zR/math.pi)
    widths=[2*w0*math.sqrt(1+((z-50)/zR)**2) for z in [0,25,50,75,100]]
    blindspot={"scope":"analytic counterexample, not the actual NN field","diameters_0_25_50_75_100_um":widths,
               "endpoint_relative_change":abs(widths[-1]/widths[0]-1),"maximum_interval_relative_change":max(abs(w/widths[0]-1) for w in widths)}
    checks["endpoint_stability_blind_spot_demonstrated"]=blindspot["endpoint_relative_change"]<1e-12 and blindspot["maximum_interval_relative_change"]>.05
    seed_reports=[]
    for path in (artifact/"phase9").glob("seed-*-training.json"):
        data=read_json(path);v=data["validation"]
        seed_reports.append({"seed":data["seed"],"steps":data["steps"],"objective_forwards":data["objective_forwards"],"best_validation_loss":data["best_validation_loss"],"validation":v})
    phase_source=(root/"src/fiber_metalens_nn/collimation_proxy.py").read_text(encoding="utf-8")
    loss_def=next(n for n in ast.walk(ast.parse(phase_source)) if isinstance(n,ast.FunctionDef) and n.name=="_objective")
    (output/"audited_objective_verbatim.py.txt").write_text(ast.get_source_segment(phase_source,loss_def)+"\n",encoding="utf-8")
    assert manifest=={str(p):sha(p) for p in file_paths},"canonical source/evidence changed during audit"
    checks["canonical_inputs_unchanged_during_audit"]=True
    result={"utc":datetime.now(timezone.utc).isoformat(),"evidence_class":"saved Maxwell-field reanalysis plus non-optimized scalar-proxy diagnostics; synthetic materials",
            "canonical_project":str(root),"actual_saved_Maxwell_planes_um":saved["z_um"].tolist(),
            "internal_cylinder_top_Maxwell_field_available":False,
            "internal_top_limit":"Saved NN fields are at z=39 and 44, beyond the structured relief. They cannot be propagated backward through vacuum to z=35 inside that relief. Only scalar-proxy top fields were recomputed.",
            "recomputed_vector_overlap":computed,"saved_NN_output_scalar_diagnostic":maxwell_summary,
            "fixed_geometry_proxy_sensitivity":proxies,"JAX_reference_parity":parity,"objective_stability_counterexample":blindspot,
            "objective_audit":{"genuine_target":"complex forward vector Gaussian power overlap at output, diameter conditioned on D; useful power normalized to launched surrogate",
                "terms":"-1*useful_mode_power + 0.15*(D4radial(0)/D-1)^2 + 0.1*(D4radial(100)/D4radial(0)-1)^2 + 0.002*mean(diff(height)^2)",
                "missing_explicit_terms":["intermediate-plane maximum stability", "separate residual-curvature/divergence term", "separate sidelobe/clipping term", "calibrated fabrication penalties"],
                "metric_mismatch":"training uses radial D4sigma-equivalent diameter; acceptance uses x/y 1/e^2 intensity crossings; identical for ideal circular Gaussian but not arbitrary halos",
                "qualification":"overlap implicitly penalizes phase/amplitude mismatch; missing explicit penalties do not mean target is ignored"},
            "proxy_model":{"includes":["full 35 um one-way pedestal propagation", "finite 48 um radial index mask and scalar diffraction", "24 finite-thickness annular relief slices", "approximate normal Fresnel transmission", "vector E/H reconstruction in homogeneous output"],
                "omits":["internal backward waves and coherent multi-interface reflections", "full vector boundary/TE-TM coupling inside dielectric", "self-consistent real fibre eigenmode and facet loading", "loss and dispersion"],
                "qualification":"finite-cylinder propagation is present; vector Maxwell validates selected outputs but current saved monitors cannot isolate internal causes"},
            "training_records":seed_reports,"checks":checks,"all_checks_passed":all(checks.values()),
            "wall_seconds":time.perf_counter()-start,"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),"command":sys.argv,
            "GPU_used":False,"new_Maxwell_runs":0,"training_steps":0,
            "heldout_integrity":"existing held-out result diagnosed without changing geometry/weights; any future correction requires fresh prospective held-out combinations"}
    write_json(output/"audit.json",result)
    write_json(output/"source_manifest.json",{"inputs_sha256":manifest,"audit_script_sha256":sha(__file__)})
    print(json.dumps({k:result[k] for k in ["all_checks_passed","wall_seconds","GPU_used","new_Maxwell_runs","training_steps","JAX_reference_parity","saved_NN_output_scalar_diagnostic"]},indent=2))
    print(json.dumps({"output":str(output),"proxy_sensitivity":[{"n":p["polymer_n"],"diameters":p["output_diameters_um"],"metrics":p["output_metrics"],"top":p["pedestal_top"]} for p in proxies]},indent=2))
    return 0 if result["all_checks_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
