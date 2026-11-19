"""Repair only metric extraction; saved Maxwell fluxes and physics are untouched."""
import json
from pathlib import Path
from fiber_metalens_nn.maxwell_analysis import analyze_fields
from fiber_metalens_nn.trial import write_json
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.maxwell_batch import relative_checks

for path in Path("artifacts/maxwell").glob("device-*.json"):
    report = analyze_fields(json.loads(path.read_text()))
    write_json(path,report)
    print(path.name,report["focal_um"],report["eta_R"])

# A coordinator already running when extraction was repaired can hold stale
# metrics in memory. Refresh its completed gate from the valid saved fields.
gate_path = Path("artifacts/phase6/gate.json")
if gate_path.exists():
    config = load_config("configs/desktop_trial.json")
    gate = json.loads(gate_path.read_text())
    if "results" in gate:
        gate["results"] = {name:[analyze_fields(report) for report in reports] for name,reports in gate["results"].items()}
        gate["mesh_convergence"] = {name:relative_checks(reports[-2],reports[-1],config,require_focus=name not in ["no_lens","flat"])
                                    for name,reports in gate["results"].items()}
        for name,tests in gate["boundary_tests"].items():
            for test,value in tests.items():
                value["metrics"] = analyze_fields(value["metrics"])
                value["comparison"] = relative_checks(gate["results"][name][-1],value["metrics"],config)
        passed = all(v["pass"] for v in gate["mesh_convergence"].values()) and all(v["comparison"]["pass"] for tests in gate["boundary_tests"].values() for v in tests.values())
        gate["status"] = "PASS" if passed else "FAIL_NUMERICAL_CONVERGENCE"
        gate["metric_extraction_repaired_from_saved_fields"] = True
        write_json(gate_path,gate)
        write_json("artifacts/phase6/mesh-results.json",gate["results"])
        write_json("artifacts/phase6/boundary-tests.json",gate["boundary_tests"])
        print("Refreshed phase 6:",gate["status"])
