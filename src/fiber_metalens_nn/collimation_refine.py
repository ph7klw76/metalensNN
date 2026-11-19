"""Bounded numerical repair after the initial schedule; no new optical designs."""
import json
from pathlib import Path
import numpy as np
from .collimation_optics import write,cylindrical_to_cartesian,forward_spectrum,kernels
from .collimation_validate import difference


def refine(experiment):
    cfg=experiment.config
    folder=experiment.root/"phase9"
    gate=json.loads((folder/"gate.json").read_text())
    original=folder/"gate-initial-numerical-schedule.json"
    if not original.exists():
        write(original,gate)
    summary=json.loads((folder/"Maxwell-summary.json").read_text())
    convergence=json.loads((folder/"convergence.json").read_text())
    boundary=json.loads((folder/"boundary-checks.json").read_text())
    failures=[name for name,c in convergence.items() if not c["passed"]]
    diagnostics={}
    for name in failures:
        values=[]
        for metric in json.loads((folder/f"{name}-mesh.json").read_text()):
            record=json.loads(Path(metric["record"]).read_text())
            data=np.load(record["fields_file"])
            x,dx,fields=cylindrical_to_cartesian(data["r_um"],data["fields"][0],512,128)
            spectrum=forward_spectrum(fields,dx,cfg["wavelength_um"])
            kx,ky,kz,safe,valid,k0=kernels(512,dx,cfg["wavelength_um"])
            power=np.maximum(.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])),0)
            angle=np.arctan2(np.hypot(kx,ky),safe)
            values.append(dict(resolution_per_um=metric["resolution_per_um"],divergence_half_angle_rad=metric["divergence_half_angle_rad"],
                                power_fraction_beyond_0p2rad=float(power[angle>.2].sum()/power.sum()),
                                cartesian_min_radius_um=float(np.hypot(x[256],x[256])),
                                first_nonzero_radial_sample_um=float(data["r_um"][1])))
        diagnostics[name]=values
    write(folder/"divergence-refinement-diagnosis.json",dict(cases=diagnostics,
          note="Second angular moments are sensitive to weak scattered fields. The anomalous raw r=0 cylindrical point is outside the interpolation neighborhood of the nearest base Cartesian sample at these meshes; no fields were altered or threshold relaxed."))
    amendment=folder/"numerical-refinement-amendment.json"
    write(amendment,dict(reason="Initial frozen mesh sequence exhausted; divergence of stepped relief still unconverged. Extend numerical fidelity on identical geometry, not candidate budget or acceptance thresholds.",
                        failed_cases=failures,additional_resolutions_per_um=[20,24],
                        geometry_candidates_added=0,thresholds_changed=False,heldout_training_or_geometry_feedback=False))
    for name in failures:
        series=json.loads((folder/f"{name}-mesh.json").read_text())
        last=series[-1]
        source=json.loads(Path(last["record"]).read_text())
        design=source["design_file"]
        H=source["settings"]["H_um"]
        for resolution in (20,24):
            if resolution<=last["resolution_per_um"]:
                continue
            record=experiment.maxwell(design,H,resolution)
            metric=experiment.analyze(record,30,save=folder/f"{name}-r{resolution}-planes.npz")
            metric["resolution_per_um"]=resolution
            comparison=difference(last,metric,cfg)
            series.append(metric)
            write(folder/f"{name}-mesh.json",series)
            convergence[name]=comparison
            summary[name]=metric
            write(folder/"convergence.json",convergence)
            write(folder/"Maxwell-summary.json",summary)
            print("Numerical refinement",name,resolution,comparison,flush=True)
            last=metric
            if comparison["passed"]:
                break
        if name in boundary:
            checks={}
            resolution=last["resolution_per_um"]
            for label,options in (("spacing",dict(padding=15)),("PML",dict(pml=3)),("duration",dict(duration=240))):
                record=experiment.maxwell(design,H,resolution,**options)
                metric=experiment.analyze(record,30)
                checks[label]=dict(comparison=difference(last,metric,cfg),metrics=metric)
            boundary[name]=checks
            write(folder/"boundary-checks.json",boundary)
    sampling=json.loads((folder/"trained-sampling.json").read_text())
    analysis=json.loads((folder/"Maxwell-analysis-sampling.json").read_text())
    engineering=all(c["passed"] for c in convergence.values()) and all(c["comparison"]["passed"] for checks in boundary.values() for c in checks.values()) and all(c["passed"] for c in sampling["comparisons"]) and all(c["comparison"]["passed"] for c in analysis) and all(m["crosscheck_overlap"]>.98 for m in summary.values())
    gate["engineering_gate"]="PASS" if engineering else "FAIL_NUMERICAL_CHECKS"
    gate["convergence"]=convergence
    gate["NN"]=summary["NN"]
    gate["eta_relative_to_realizable_baseline"]=summary["NN"]["eta_mode"]/summary["realizable_collimator"]["eta_mode"]
    gate["optical_gate"]="PASS_SYNTHETIC" if engineering and gate["NN"]["optical_targets_pass"] and gate["eta_relative_to_realizable_baseline"]>=cfg["evaluation"]["minimum_baseline_ratio"] else "FAIL_PREFERRED_OPTICAL_TARGETS"
    gate["numerical_refinement_amendment"]=str(amendment)
    write(folder/"gate.json",gate)
    print(json.dumps(dict(engineering=gate["engineering_gate"],optical=gate["optical_gate"]),indent=2),flush=True)
    return gate
