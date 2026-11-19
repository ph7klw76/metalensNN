"""Sequential independent Maxwell validation and bounded training-target feedback."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np

from .geometry import baselines, project, save_design
from .trial import gate, write_json
from .maxwell_analysis import analyze_fields


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


class MaxwellBatch:
    def __init__(self, config, root="artifacts/maxwell"):
        self.config = config
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.config_path = self.root/("config-"+digest(config)[:20]+".json")
        if not self.config_path.exists():
            write_json(self.config_path,config)
        self.ledger_path = self.root/"geometry-ledger.json"
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}

    def job(self,design=None,resolution=None,pml=None,padding=None,duration=None,m=1):
        c = self.config["maxwell"]
        options = {"resolution":resolution or c["coarse_resolution_per_um"],"pml":pml or c["pml_um"],
                   "padding":padding or c["radial_padding_um"],"duration":duration or c["after_sources_time"],"m":m}
        geometry = json.loads(Path(design).read_text()) if design else None
        heights = geometry["heights_um"] if geometry else None
        if design:
            geom_key = digest({"heights":heights,"materials":self.config["materials"],"illumination":self.config["illumination"]})
            if geom_key not in self.ledger:
                if len(self.ledger) >= c["max_initial_unique_geometries"]:
                    raise RuntimeError("Predeclared initial unique geometry budget exhausted")
                self.ledger[geom_key] = {"first_design":str(design),"target_um":geometry["target_focal_um"],"kind":geometry["kind"]}
                write_json(self.ledger_path,self.ledger)
        key = digest({"physics_version":1,"config":self.config,"heights":heights,
                      "target":geometry["target_focal_um"] if geometry else None,"settings":options})[:20]
        path = self.root/("device-"+key+".json" if design else "reference-"+key+".json")
        if path.exists():
            result = analyze_fields(json.loads(path.read_text()))
            write_json(path,result)
            return result
        reference = None
        if design:
            reference = self.job(None,**options)
            if not reference["pass"]:
                raise RuntimeError("Homogeneous reference did not pass; device evaluation is invalid")
        args = [".venv-meep/bin/python","-m","fiber_metalens_nn.maxwell_worker","--config",str(self.config_path),"--output",str(path)]
        for name,value in options.items():
            args += ["--"+name,str(value)]
        if design:
            args += ["--design",str(design),"--reference",reference["artifact_file"]]
        print(f"Meep serial: {str(design) if design else 'homogeneous reference'}, {options}",flush=True)
        with path.with_suffix(".log").open("w") as stream:
            completed = subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        if completed.returncode:
            raise RuntimeError(f"Meep job failed ({completed.returncode}); see {path.with_suffix('.log')}")
        result = analyze_fields(json.loads(path.read_text()))
        result["artifact_file"] = str(path)
        write_json(path,result)
        return result


def relative_checks(a,b,config,require_focus=True):
    focus = abs(a["focal_um"]-b["focal_um"])/max(abs(b["focal_um"]),1e-12)
    eta = abs(a["eta_R"]-b["eta_R"])/max(abs(b["eta_R"]),1e-12)
    return {"focus_relative_change":focus,"eta_relative_change":eta,
            "pass":bool(eta < config["maxwell"]["eta_tolerance_relative"]
                         and (not require_focus or (focus < config["maxwell"]["focus_tolerance_relative"] and not b["focus_on_scan_boundary"]))
                         and a["power_balance_pass"] and b["power_balance_pass"])}


def phase6(config):
    start = time.monotonic()
    batch = MaxwellBatch(config)
    folder = Path("artifacts/phase6")
    folder.mkdir(parents=True,exist_ok=True)
    if not json.loads(Path("artifacts/phase5/gate.json").read_text())["status"].startswith("PASS"):
        raise RuntimeError("Phase 5 must pass")
    reference = batch.job()
    reverse = batch.job(m=-1)
    azimuth_error = abs(reference["incident_aperture_power"]/reverse["incident_aperture_power"]-1)
    reference_check = {"m_plus":reference,"m_minus":reverse,"relative_power_difference":azimuth_error,
                       "pass":reference["pass"] and reverse["pass"] and azimuth_error < .02}
    write_json(folder/"reference-check.json",reference_check)
    if not reference_check["pass"]:
        return gate(6,"FAIL_REFERENCE",config,reference_check=reference_check)
    target = config["training"]["final_evaluation_target_um"]
    cases = {}
    for name,heights in baselines(target,config).items():
        path = folder/f"{name}-f{target:.1f}.json"
        save_design(path,heights,target,name,config)
        cases[name] = str(path)
    selected = json.loads(Path("artifacts/phase5/selection-before-heldout.json").read_text())["selected_seed"]
    cases["direct"] = f"artifacts/phase5/seed-{selected}/direct-f{target:.1f}.json"
    cases["NN"] = f"artifacts/phase5/selected-NN-f{target:.1f}.json"
    resolutions = [config["maxwell"]["coarse_resolution_per_um"]]+config["maxwell"]["refinement_resolutions_per_um"]
    results = {}
    # Mesh-first ordering allows every candidate to obtain a coarse result
    # before expensive refinement; all simulations remain sequential.
    for res in resolutions:
        for name,path in cases.items():
            report = batch.job(path,resolution=res)
            results.setdefault(name,[]).append(report)
            write_json(folder/"mesh-results.json",results)
    results = {name:[analyze_fields(report) for report in reports] for name,reports in results.items()}
    write_json(folder/"mesh-results.json",results)
    convergence = {name:relative_checks(reports[-2],reports[-1],config,require_focus=name not in ["no_lens","flat"])
                   for name,reports in results.items()}
    # Baselines without a designed focus use eta and power convergence; a
    # finite-window maximum is recorded but is not called a physical focus.
    boundary = {}
    fine = resolutions[-1]
    for name in ["NN","fresnel"]:
        standard = results[name][-1]
        c = config["maxwell"]
        tests = {"spacing":{"padding":c["radial_padding_um"]+1.5},
                 "PML":{"pml":c["pml_um"]+.5},"duration":{"duration":1.5*c["after_sources_time"]}}
        boundary[name] = {}
        for test,kwargs in tests.items():
            report = batch.job(cases[name],resolution=fine,**kwargs)
            boundary[name][test] = {"metrics":report,"comparison":relative_checks(standard,report,config)}
            write_json(folder/"boundary-tests.json",boundary)
    passed = all(value["pass"] for value in convergence.values()) and all(test["comparison"]["pass"] for tests in boundary.values() for test in tests.values())
    return gate(6,"PASS" if passed else "FAIL_NUMERICAL_CONVERGENCE",config,reference_check=reference_check,
                mesh_convergence=convergence,boundary_tests=boundary,results=results,
                wall_s=time.monotonic()-start,initial_unique_geometries=len(batch.ledger),
                independent_CPU_Meep=True,physical_IP_Dip_validation=False)


def phase7(config):
    start = time.monotonic()
    batch = MaxwellBatch(config)
    folder = Path("artifacts/phase7")
    folder.mkdir(parents=True,exist_ok=True)
    phase6 = json.loads(Path("artifacts/phase6/gate.json").read_text())
    quadrature_path = folder/"collection-quadrature-check.json"
    if not quadrature_path.exists():
        with (folder/"collection-quadrature-check.log").open("w") as stream:
            subprocess.run([".venv-meep/bin/python","-m","fiber_metalens_nn.maxwell_calibration"],
                           stdout=stream,stderr=subprocess.STDOUT,timeout=120,check=True)
    quadrature = json.loads(quadrature_path.read_text())
    # Only a training target is used for the bounded correction; 7.8 stays
    # untouched and cannot choose the correction or proxy loss weights.
    target = 7.4
    nn_path = Path(f"artifacts/phase5/selected-NN-f{target:.1f}.json")
    heights = np.asarray(json.loads(nn_path.read_text())["heights_um"])
    candidates = {"untouched_NN":str(nn_path)}
    fresnel = folder/"training-fresnel.json"
    save_design(fresnel,baselines(target,config)["fresnel"],target,"training_fresnel",config)
    candidates["fresnel"] = str(fresnel)
    for direction in [-1,1]:
        changed = heights.copy()
        changed[16:] += direction*config["fabrication"]["height_step_um"]
        path = folder/f"corrected-NN-outer-{direction:+d}.json"
        save_design(path,project(changed,config),target,"Maxwell_feedback_candidate",config,
                    corrected_with_Maxwell=True,correction_target_is_training=True,correction="outer 16 rings +/- one height step")
        candidates[f"outer_{direction:+d}"] = str(path)
    reports = {name:batch.job(path) for name,path in candidates.items()}
    # Predeclared score: fixed-radius power with a 10% focal-error penalty.
    score = lambda value:value["eta_R"]-.1*abs(value["focal_um"]/target-1)
    selected = max(["untouched_NN","outer_-1","outer_+1"],key=lambda name:score(reports[name]))
    write_json(folder/"feedback-selection.json",{"target_um":target,"heldout_used_for_correction":False,
        "score":"eta_R - 0.1*abs(f/f_target-1)","selected":selected,"reports":reports})
    refined = [reports[selected]]
    for res in config["maxwell"]["refinement_resolutions_per_um"]:
        refined.append(batch.job(candidates[selected],resolution=res))
    corrected_convergence = relative_checks(refined[-2],refined[-1],config)
    nn = phase6["results"]["NN"][-1]
    baseline = phase6["results"]["fresnel"][-1]
    error = abs(nn["focal_um"]/nn["target_um"]-1)
    ratio = nn["eta_R"]/baseline["eta_R"]
    prior_gates = {n:json.loads(Path(f"artifacts/phase{n}/gate.json").read_text())["status"] for n in range(1,7)}
    resource_checks = {n:json.loads(Path(f"artifacts/phase{n}/memory.jsonl.summary.json").read_text())["stop_reasons"] == [] for n in range(2,7)}
    maxwell_memory = json.loads(Path("artifacts/phase6/memory.jsonl.summary.json").read_text())
    diagnostic_path = Path("artifacts/phase5/trained-state-resume-memory.jsonl.summary.json")
    overlapping_rss_bound = None
    if diagnostic_path.exists():
        diagnostic_memory = json.loads(diagnostic_path.read_text())
        resource_checks["nonoptical_resume_diagnostic"] = not diagnostic_memory["stop_reasons"]
        overlapping_rss_bound = maxwell_memory["peak_project_rss_bytes"]+diagnostic_memory["peak_project_rss_bytes"]
        resource_checks["conservative_sum_of_recorded_RSS_peaks"] = bool(overlapping_rss_bound < min(
            maxwell_memory["budgets"]["project_ram_bytes"],diagnostic_memory["budgets"]["project_ram_bytes"]))
    current_samples = [json.loads(line) for line in (folder/"memory.jsonl").read_text().splitlines() if line.strip()]
    resource_checks[7] = all(not value["violations"] for value in current_samples)
    engineering = bool(all(value.startswith("PASS") for value in prior_gates.values())
                       and all(resource_checks.values()) and corrected_convergence["pass"] and quadrature["pass"])
    optical = bool(engineering and error <= config["acceptance"]["focal_error_relative_max"] and ratio >= config["acceptance"]["eta_relative_to_fresnel_min"])
    return gate(7,"PASS" if engineering and optical else "FAIL_DESKTOP_ACCEPTANCE",config,
                engineering_gate="PASS" if engineering else "FAIL",optical_gate="PASS" if optical else "FAIL",
                heldout_focal_relative_error=error,heldout_eta_relative_to_fresnel=ratio,
                heldout_untouched_NN=nn,matched_fresnel=baseline,feedback_target_um=target,
                selected_feedback_candidate=selected,feedback_refinement=refined,feedback_convergence=corrected_convergence,
                prior_phase_gates=prior_gates,resource_checks=resource_checks,
                sum_of_recorded_project_RSS_peaks_bytes=overlapping_rss_bound,
                combined_resource_bound_note="Sum of original per-job peak RSS values; conservative for possible concurrency, with no temporal overlap assertion",
                collection_quadrature_check=quadrature,
                initial_unique_geometries=len(batch.ledger),unique_geometry_budget=config["maxwell"]["max_initial_unique_geometries"],
                wall_s=time.monotonic()-start,physical_IP_Dip_validation=False)
