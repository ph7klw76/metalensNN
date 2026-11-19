"""Independent Maxwell acceptance, bounded refinement, and robustness audits."""
import copy
import json
from pathlib import Path
import numpy as np
from .collimation_proxy import CylinderProxy
from .collimation_optics import write


def difference(a,b,cfg):
    wa=np.array([a["planes"][0][f"diameter_{axis}_um"] for axis in ("x","y")],float)
    wb=np.array([b["planes"][0][f"diameter_{axis}_um"] for axis in ("x","y")],float)
    da,db=np.array(a["divergence_half_angle_rad"]),np.array(b["divergence_half_angle_rad"])
    widths=float(np.max(abs(wa-wb)/np.maximum(abs(wb),1e-6)))
    divergence=float(np.max(abs(da-db)/np.maximum(abs(db),1e-8)))
    divergence_abs=float(np.max(abs(da-db)))
    eta=abs(a["eta_mode"]-b["eta_mode"])/max(abs(b["eta_mode"]),1e-5)
    mw=cfg["maxwell"]
    fits_valid=a.get("fitted_divergence",{}).get("valid",True) and b.get("fitted_divergence",{}).get("valid",True)
    return dict(diameter_relative=widths,divergence_relative=divergence,divergence_absolute_rad=divergence_abs,
                eta_mode_relative=eta,fitted_divergence_valid=bool(fits_valid),passed=bool(fits_valid and widths<mw["metric_relative_tolerance"] and
                (divergence<mw["metric_relative_tolerance"] or divergence_abs<mw["divergence_absolute_tolerance_rad"]) and
                eta<mw["eta_relative_tolerance"] and b["power_balance_error"]<mw["balance_tolerance"]))


def validate(experiment):
    cfg=experiment.config
    folder=experiment.root/"phase9"
    training=json.loads((folder/"training-gate.json").read_text())
    if training["status"]!="PASS":
        raise RuntimeError("Training/resume gate must pass before Maxwell acceptance")
    heldout=json.loads((folder/"heldout.json").read_text())
    selected=training["selected_seed"]
    cases=[entry for entry in heldout if entry["seed"]==selected]
    primary=next(entry for entry in cases if entry["pair"]==[35,30])
    proxy=CylinderProxy(cfg)
    H,D=35,30
    zero=np.zeros(cfg["annuli"])
    designs=dict(bare_fiber=experiment.design("bare-fiber-H35",zero,H,D,"bare_fiber"),
                 flat_cylinder=experiment.design("flat-cylinder-H35",zero,H,D,"flat_cylinder"),
                 realizable_collimator=experiment.design("curvature-relief-H35",np.asarray(proxy.seed(H)),H,D,"realizable_collimator"),
                 direct=primary["direct_design"],NN=primary["NN_design"])
    results={}
    convergences={}
    resolutions=cfg["maxwell"]["resolutions_per_um"]
    for name,design in designs.items():
        series=[]
        # Preserve a finer mesh already executed by an earlier diagnostic run.
        history_file=folder/f"{name}-mesh.json"
        prior=json.loads(history_file.read_text()) if history_file.exists() else []
        previous_resolutions=[m["resolution_per_um"] for m in prior]
        current_resolutions=sorted(set(resolutions+previous_resolutions))
        for resolution in current_resolutions:
            record=experiment.maxwell(design,H,resolution)
            analysis=experiment.analyze(record,D,save=folder/f"{name}-r{resolution}-planes.npz")
            analysis["resolution_per_um"]=resolution
            series.append(analysis)
            write(folder/f"{name}-mesh.json",series)
        convergence=difference(series[-2],series[-1],cfg)
        if not convergence["passed"]:
            extra=cfg["maxwell"]["extra_resolution_per_um"]
            record=experiment.maxwell(design,H,extra)
            analysis=experiment.analyze(record,D,save=folder/f"{name}-r{extra}-planes.npz")
            analysis["resolution_per_um"]=extra
            series.append(analysis)
            convergence=difference(series[-2],series[-1],cfg)
            write(folder/f"{name}-mesh.json",series)
        results[name]=series[-1]
        convergences[name]=convergence
        write(folder/"Maxwell-summary.json",results)
        write(folder/"convergence.json",convergences)
        print(name,"eta",results[name]["eta_mode"],"diameter",results[name]["planes"][0]["diameter_x_um"],"converged",convergence["passed"],flush=True)
    # Boundary changes independently of mesh; no new geometry candidates.
    boundary={}
    for name in ("NN","realizable_collimator"):
        resolution=results[name]["resolution_per_um"]
        base=results[name]
        checks={}
        for label,options in (("spacing",dict(padding=15)),("PML",dict(pml=3)),("duration",dict(duration=240))):
            record=experiment.maxwell(designs[name],H,resolution,**options)
            metric=experiment.analyze(record,D)
            checks[label]=dict(comparison=difference(base,metric,cfg),metrics=metric)
        boundary[name]=checks
        write(folder/"boundary-checks.json",boundary)
    # Heldout outputs not selected/retrained by these measurements.
    other_heldouts=[]
    for entry in cases:
        if entry["pair"]==[35,30]:
            continue
        h,d=entry["pair"]
        record=experiment.maxwell(entry["NN_design"],h,resolutions[0])
        other_heldouts.append(dict(pair=entry["pair"],metrics=experiment.analyze(record,d),
                                   status="COARSE_DIAGNOSTIC_NOT_CONVERGED_ACCEPTANCE"))
    write(folder/"additional-heldout-Maxwell.json",other_heldouts)
    # Explicitly outside-preference architecture: same radius, taller polymer.
    longH=round(np.pi*(cfg["mfd_um"]/2)**2/cfg["wavelength_um"]*cfg["materials"]["polymer"]["n"]*np.sqrt((30/cfg["mfd_um"])**2-1),2)
    longer={}
    for name,heights in (("flat",zero),("curvature_relief",np.asarray(proxy.seed(longH)))):
        path=experiment.design("tall-"+name,heights,longH,30,"tall_"+name)
        record=experiment.maxwell(path,longH,resolutions[0])
        longer[name]=dict(H_um=longH,metrics=experiment.analyze(record,30),
                          status="OUTSIDE_HEIGHT_PREFERENCE_COARSE_ONLY")
    write(folder/"taller-cylinder-comparison.json",longer)
    # Final trained-design proxy sampling, after quantization.
    nn_heights=np.asarray(json.loads(Path(designs["NN"]).read_text())["heights_um"])
    sample=[]
    for label,options in (("base",{}),("axial",dict(dz=.25,slices=48)),("spatial",dict(grid_n=256)),("window",dict(padding=3))):
        p=CylinderProxy(cfg,**options)
        metric=p.evaluate(nn_heights,H,D)
        from .collimation_metrics import attach_divergence_fit
        metric=attach_divergence_fit(metric,np.asarray(p.spectrum(nn_heights,p.pedestal(H))),p.dx,cfg["wavelength_um"])
        sample.append(dict(label=label,metrics=metric))
    sample_changes=[difference(dict(sample[0]["metrics"],power_balance_error=0),dict(s["metrics"],power_balance_error=0),cfg) for s in sample[1:]]
    write(folder/"trained-sampling.json",dict(samples=sample,comparisons=sample_changes))
    # Independent reconstruction/propagation window checks on final Maxwell data.
    record=json.loads(Path(results["NN"]["record"]).read_text())
    analysis_checks=[]
    for count,window in ((768,128),(768,192)):
        metric=experiment.analyze(record,D,count=count,window=window)
        analysis_checks.append(dict(count=count,window_um=window,comparison=difference(results["NN"],metric,cfg)))
    write(folder/"Maxwell-analysis-sampling.json",analysis_checks)
    # Robustness is proxy-only, with assumptions rather than production tolerances.
    robustness=[]
    for parameter,values in cfg["robustness_assumed_variations"].items():
        for value in values:
            altered=copy.deepcopy(cfg)
            h=H
            relief=nn_heights
            if parameter=="height_um": h+=value
            elif parameter=="polymer_n": altered["materials"]["polymer"]["n"]=value
            elif parameter=="relief_scale": relief=np.minimum(nn_heights*value,cfg["height_max_um"])
            else: altered[parameter]=value
            metric=CylinderProxy(altered).evaluate(relief,h,D)
            robustness.append(dict(parameter=parameter,value=value,eta_mode=metric["eta_mode"],diameter_um=metric["planes"][0]["diameter_x_um"],
                                   stability=metric["maximum_diameter_change_100um"],evidence="proxy_only_assumed_tolerance"))
    write(folder/"robustness.json",robustness)
    engineering=all(c["passed"] for c in convergences.values()) and all(c["comparison"]["passed"] for values in boundary.values() for c in values.values()) and all(c["passed"] for c in sample_changes) and all(c["comparison"]["passed"] for c in analysis_checks) and all(m["crosscheck_overlap"]>.98 for m in results.values())
    baseline_ratio=results["NN"]["eta_mode"]/max(results["realizable_collimator"]["eta_mode"],1e-20)
    optical=engineering and results["NN"]["optical_targets_pass"] and baseline_ratio>=cfg["evaluation"]["minimum_baseline_ratio"]
    gate=dict(phase=9,engineering_gate="PASS" if engineering else "FAIL_NUMERICAL_CHECKS",optical_gate="PASS_SYNTHETIC" if optical else "FAIL_PREFERRED_OPTICAL_TARGETS",
              result_label=cfg["result_label"],selected_seed=selected,preferred_H_um=H,target_diameter_um=D,
              NN=results["NN"],eta_relative_to_realizable_baseline=baseline_ratio,convergence=convergences,
              heldout_not_used_for_selection=True,geometry_budget_used=len(json.loads((experiment.root/"maxwell/geometry-ledger.json").read_text())),
              full_wave_feedback_rounds_used=0,ram_gates=False,
              unexecuted="Hollow cylinder, integrated second expansion stage, 3D offset/tilt defects, measured-material/fiber validation, experimental fabrication")
    write(folder/"gate.json",gate)
    print(json.dumps(dict(engineering=gate["engineering_gate"],optical=gate["optical_gate"],NN_diameter=gate["NN"]["planes"][0]["diameter_x_um"],NN_eta=gate["NN"]["eta_mode"]),indent=2),flush=True)
    return gate
