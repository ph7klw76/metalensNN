"""Generate the report from executed artifacts, never inferred pass statuses."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def main():
    config = read("configs/desktop_trial.json")
    gates = {n:read(f"artifacts/phase{n}/gate.json") for n in range(1,8)}
    phase6,phase7 = gates[6],gates[7]
    heldout = read("artifacts/phase5/heldout.json")
    selected = str(heldout["selected_seed"])
    target = config["training"]["final_evaluation_target_um"]
    index = config["training"]["heldout_targets_um"].index(target)
    diagnostic_path = Path("artifacts/phase5/trained-state-resume.json")
    diagnostic = read(diagnostic_path) if diagnostic_path.exists() else None
    proxy_baselines = read("artifacts/phase4/baselines.json")
    proxy = {name:proxy_baselines[name] for name in ["no_lens","flat","fresnel"]}
    proxy["direct"] = heldout["seeds"][selected]["direct"][index]["metrics"]
    proxy["NN"] = heldout["seeds"][selected]["NN"][index]
    names = ["no_lens","flat","fresnel","direct","NN"]
    maxwell = {name:phase6["results"][name][-1] for name in names}
    folder = Path("artifacts/desktop-report")
    folder.mkdir(exist_ok=True)
    x = np.arange(len(names))
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(x-.18,[100*proxy[n]["eta_R"] for n in names],width=.36,label="GPU proxy")
    axes[0].bar(x+.18,[100*maxwell[n]["eta_R"] for n in names],width=.36,label="CPU Maxwell (finest mesh)")
    axes[0].set(ylabel="Fixed-radius power / incident aperture power (%)",title="Synthetic materials; R = 0.55 µm")
    for n in ["fresnel","direct","NN"]:
        axes[1].plot([r["settings"]["resolution"] for r in phase6["results"][n]],
                     [r["focal_um"] for r in phase6["results"][n]],"o-",label=n)
    axes[1].axhline(target,color="black",ls="--",label="Target")
    axes[1].set(xlabel="Meep resolution (pixels/µm)",ylabel="Focus from exit plane (µm)",title="Maxwell mesh refinement")
    axes[0].set_xticks(x,["No lens","Flat","Fresnel","Direct","NN"])
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/"proxy-maxwell.png",dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for n in ["fresnel","direct","NN"]:
        data = np.load(maxwell[n]["fields_file"])
        axial = data["axis_intensity"]
        axes[0].plot(data["focus_scan_um"],axial/np.max(axial),label=n)
        i = int(np.argmax(axial))
        er,ep,ez = data["Er"][i],data["Ep"][i],data["Ez"][i]
        r = data["r_um"]
        radial = abs(er)**2+abs(ez)**2
        axes[1].plot(r,radial/np.max(radial),label=n)
    axes[0].axvline(target,color="black",ls="--",label="Target")
    axes[0].set(xlabel="Distance from exit plane (µm)",ylabel="Normalized on-axis |E|²",title="Independent vector Maxwell fields")
    axes[1].set(xlabel="Radius on x axis (µm)",ylabel="Normalized |E|² at axial maximum",xlim=(0,2),title="Profiles at each design's own peak")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/"maxwell-fields.png",dpi=160)
    plt.close(fig)

    lines = ["# RTX 4060 desktop trial — executed results", "",
             f"Engineering gate: **{phase7['engineering_gate']}**. Optical gate: **{phase7['optical_gate']}**.","",
             "All requested Phases 1–7 were executed. These are **synthetic generic-material results**; measured cured-resin optical constants and exact paper CAD remain unavailable. No experimental or IP-Dip performance is established.","",
             "| Phase | Executed acceptance status |", "|---|---|"]
    for n,g in gates.items():
        lines.append(f"| {n} | {g['status']} |")
    lines += ["", "Hardware: RTX 4060, 8188 MiB VRAM, SM 8.9, driver 610.88; WSL2 Ubuntu-20.04, Python 3.12.15. Driver CUDA compatibility 13.3, separately installed toolkit 12.8, pinned JAX CUDA 12 libraries. GPU JAX/Flax training and independent serial **CPU** Meep 1.35.0 were actually run.","",
              "The fixed Windows free-RAM gate is disabled for every Phase 1–7 job, including the former 0.56 GiB stop. Windows RAM is telemetry only. Process RSS, WSL available RAM, GPU budgets and timeouts remain monitored. Historical RAM stops are preserved and do not govern this run.","",
              "The synthetic assumptions are polymer n=1.52, substrate n=1.45, ambient n=1.0, all κ=0; diameter 24.8 µm, 32 supported annuli, 0–2.5 µm heights quantized to 0.05 µm, and an explicit PEC aperture stop. Assumed fabrication bounds are not printer calibration. Source provenance and separate 980/780 nm literature benchmarks are in `reference.yaml` and `assumptions.md`.","",
              "The proxy's thin-element approximation and vector propagation passed derivative, energy, known-phase, padding, and 256/512-grid checks. Its omitted internal propagation, ring coupling and scattering remain model errors even when differentiation and sampling tests pass.","",
              "Three 3×64 conditional MLP seeds (11, 29, 47), batch one, ran 120 updates each. Train targets: 7.0, 7.4, 8.1, 8.5, 9.0 µm; validation: 7.2, 8.3, 8.8 µm; held-out: 7.6, 7.8, 8.6 µm. Seed 11 was selected solely by validation loss before held-out evaluation. Every seed passed checkpoint recovery and conditional response checks.","",
              "Each NN seed used 158 objective forwards (120 training + 36 validation + 2 recovery), or 474 propagated objective planes. Its direct comparison used the same aggregate 158 optimization forwards split 53/53/52 across held-out targets. Final axial evaluation scans are excluded equally from this optimization budget. The direct candidates are independently optimized per target; this budget accounting does not imply equal amortized cost at every target.","",
              "| Seed | Training/resume wall time (s) | Best validation loss | Held-out target/focus correlation |", "|---|---:|---:|---:|"]
    for seed in gates[5]["seeds"]:
        lines.append(f"| {seed['seed']} | {seed['training_wall_s']:.2f} | {seed['best_validation_loss']:.6f} | {heldout['seeds'][str(seed['seed'])]['conditional_test']['target_focus_correlation']:.6f} |")
    lines += ["", "| Seed at 7.8 µm | NN inference (s, warm) | Direct optimization (s) | Direct objective forwards |", "|---|---:|---:|---:|"]
    for seed in gates[5]["seeds"]:
        data = heldout["seeds"][str(seed["seed"])]
        prediction,comparison = data["NN"][index],data["direct"][index]
        lines.append(f"| {seed['seed']} | {prediction['inference_wall_s_including_first_compile_if_cold']:.6f} | {comparison['optimization_cost']['wall_s']:.3f} | {comparison['optimization_cost']['objective_forwards']} |")
    recovery_note = ""
    if diagnostic is not None:
        recovery_note = " Additional trained-state diagnostic: "+("PASS" if diagnostic["pass"] else "FAIL")+". It compares the next Adam updates after serialization with a fixed diagnostic gradient and zero optical forwards, checking nonzero trained optimizer states."
    lines += ["", "A report-export failure after the first completed optimization run was repaired; checkpoints and direct designs were reused. The original failed-at-export guard report is retained, and the resumed run repeated final evaluations. The logical matched optimization budgets above exclude this extra evaluation/report recovery overhead."+recovery_note+" Inference measurements and per-target direct optimization costs are recorded in `artifacts/phase5/heldout.json`.","",
              "![Learning curves](artifacts/phase5/learning-curves.png)","",
              "| Design at held-out 7.8 µm | Proxy focus (µm) | Proxy ηR | Finest Maxwell focus (µm) | Maxwell ηR | Maxwell x/y FWHM (µm) | Power closure error |", "|---|---:|---:|---:|---:|---|---:|"]
    for n in names:
        p,m = proxy[n],maxwell[n]
        w = m["widths"]
        widths = "/".join("unresolved" if w[k] is None else f"{w[k]:.4f}" for k in ["fwhm_x_um","fwhm_y_um"])
        lines.append(f"| {n} | {p['focal_um']:.4f} | {p['eta_R']:.6f} | {m['focal_um']:.4f} | {m['eta_R']:.6f} | {widths} | {m['power_balance_error']:.5f} |")
    lines += ["", "ηR is Poynting power through a fixed 0.55 µm radius at the **target** plane divided by incident aperture power. It is not peak intensity and is not directly the paper's experimental efficiency definition. No-lens and flat values called 'focus' in raw arrays are simply the maximum within the declared axial scan, not evidence of a designed focus. Widths use the nearest 0.25 µm scan plane to each design's peak, with one-cell radial-coordinate uncertainty explicitly recorded; full exp(−2) widths and sidelobes are in the job JSON files.","",
              "Meep's homogeneous reference passed forward power, aperture-area scaling, and m=±1 power equivalence. Equal reflected azimuthal sectors reconstruct x-linear fields; the circular-to-linear factor cancels in normalized fluxes. The effective domain top is 18.5 µm and transmission plane 16 µm, enlarged from requested values to keep monitors out of PML. Focal distances retain the fixed z=2.5 µm origin. Meep transmission is measured through the finite top control surface; proxy transmission integrates the full transverse plane, so these transmission numbers should not be identified. Full reflection/top/side power closure is tested independently.","",
              "![Proxy and Maxwell comparison](artifacts/desktop-report/proxy-maxwell.png)","",
              "| Case | Last-mesh focal change | Last-mesh ηR change | Convergence gate |", "|---|---:|---:|---|"]
    for name,value in phase6["mesh_convergence"].items():
        lines.append(f"| {name} | {100*value['focus_relative_change']:.3f}% | {100*value['eta_relative_change']:.3f}% | {'PASS' if value['pass'] else 'FAIL'} |")
    lines += ["", "For no-lens and flat controls only ηR/power closure define mesh convergence; their finite-window intensity maxima are reported without a focal-convergence requirement. Focus uses the first near-axis centered Yee voxel. Cylindrical point DFTs returned alternating zeros at half-grid axial positions at resolution 30; extraction was repaired from valid saved radial slices, with raw point data retained. Geometry and power fluxes were unchanged.","",
              "| Boundary test | Focal change | ηR change | Gate |", "|---|---:|---:|---|"]
    for name,tests in phase6["boundary_tests"].items():
        for test,value in tests.items():
            c = value["comparison"]
            lines.append(f"| {name}: {test} | {100*c['focus_relative_change']:.3f}% | {100*c['eta_relative_change']:.3f}% | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", "![Maxwell field profiles](artifacts/desktop-report/maxwell-fields.png)","",
              "The independent small-domain uniform-field collection quadrature check is **"+("PASS" if phase7["collection_quadrature_check"]["pass"] else "FAIL")+"**. It compares centered 0.55/1.1 µm circular flux regions with the expected area ratio 1/4 at all three meshes; this diagnoses small-collection-region power integration separately from lens geometry convergence.","",
              f"Bounded Maxwell feedback used only the 7.4 µm training target and +/- one 0.05 µm step on the outer 16 rings. Selected candidate: `{phase7['selected_feedback_candidate']}`. Feedback results remain separate from untouched NN outputs. Total initial unique geometries: **{phase7['initial_unique_geometries']} / {phase7['unique_geometry_budget']}**; reference and numerical convergence repetitions are separate. The final held-out 7.8 µm design was not corrected or used to choose the feedback candidate.","",
              "| Selected feedback candidate at training target 7.4 µm | Focus (µm) | ηR |", "|---|---:|---:|"]
    for r in phase7["feedback_refinement"]:
        lines.append(f"| {r['settings']['resolution']:g} pixels/µm | {r['focal_um']:.5f} | {r['eta_R']:.6f} |")
    c = phase7["feedback_convergence"]
    lines += ["",f"The selected correction's last-mesh focus change is {100*c['focus_relative_change']:.3f}% and ηR change {100*c['eta_relative_change']:.3f}%: mesh check {'PASS' if c['pass'] else 'FAIL'}. This training-target refinement does not repair the untouched held-out design or the unconverged Fresnel/flat controls.","",
              f"Final held-out NN focal error: **{100*phase7['heldout_focal_relative_error']:.3f}%** (target ≤10%). Its finest-mesh ηR / matched realizable Fresnel ηR is **{phase7['heldout_eta_relative_to_fresnel']:.4f}** (target ≥0.90), with the Fresnel efficiency still unconverged. This ratio does not establish superiority. Optical PASS also requires engineering PASS; numerical thresholds are not relaxed to produce success.","",
              "| Guarded run | Wall time (s) | Peak sampled owned RSS (MiB) | Peak total-device GPU delta (MiB) | Resource stops |", "|---|---:|---:|---:|---|"]
    reports = [(f"phase {n}",f"artifacts/phase{n}/memory.jsonl.summary.json") for n in range(2,8)]
    reports += [("phase 5 original export failure","artifacts/phase5/first-export-failure-memory.jsonl.summary.json")]
    reports += [("trained-state resume diagnostic","artifacts/phase5/trained-state-resume-memory.jsonl.summary.json")]
    for label,path in reports:
        if not Path(path).exists():
            continue
        r = read(path)
        lines.append(f"| {label} | {r['wall_s']:.2f} | {r['peak_project_rss_bytes']/2**20:.1f} | {r['peak_total_device_delta_bytes']/2**20:.1f} | {', '.join(r['stop_reasons']) or 'none'} |")
    lines += ["", "Sampled RSS sums can double-count shared pages and miss transient peaks. GPU total-device deltas can include other applications. CPU Meep's own optical work does not use GPU acceleration. Job-specific hardware audits and JSONL measurements are preserved. The test suite result is saved in `artifacts/desktop-report/pytest.txt`.","",
              "Reproduction: follow `README.md`; pinned environments, frozen physical/training configuration, geometry JSONs, selected planes, deterministic checkpoints, direct optimization histories, Meep reference DFT data and logs are retained. No driver/WSL setting changes or unrelated-process termination was performed. Phases 8–10 are outside this executed scope."]
    Path("results.md").write_text("\n".join(lines)+"\n")
    write = {"gates":{str(n):g["status"] for n,g in gates.items()},"engineering":phase7["engineering_gate"],"optical":phase7["optical_gate"],
             "heldout_focus_error":phase7["heldout_focal_relative_error"],"heldout_eta_ratio":phase7["heldout_eta_relative_to_fresnel"]}
    (folder/"summary.json").write_text(json.dumps(write,indent=2)+"\n")
    print(json.dumps(write,indent=2))


if __name__ == "__main__":
    main()
