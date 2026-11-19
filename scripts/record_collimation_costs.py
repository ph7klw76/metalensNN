"""Capture completed guard summaries after the package process has exited."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--root",default="artifacts/1550nm_collimation")
args=parser.parse_args()
root=Path(args.root)
stages=[]
for path in sorted(root.glob("*-memory.jsonl.summary.json")):
    record=json.loads(path.read_text())
    command=record.get("command",[])
    stage=command[command.index("fiber_metalens_nn.collimation_run")+1] if "fiber_metalens_nn.collimation_run" in command else path.name
    stages.append(dict(log=str(path),wall_s=record.get("wall_s"),peak_RSS_bytes=record.get("peak_project_rss_bytes"),
                       sampled_device_delta_bytes=record.get("peak_total_device_delta_bytes"),returncode=record.get("returncode"),
                       stop_reasons=record.get("stop_reasons"),budgets=record.get("budgets"),stage=stage))
solver_records=[]
for path in (root/"maxwell").glob("*.json"):
    record=json.loads(path.read_text())
    if isinstance(record,dict) and "solver" in record:
        solver_records.append(record)
result=dict(guard_stages=stages,completed_solver_record_wall_s=sum(r["wall_s"] for r in solver_records),
            note="Actual completed solver records counted once each; cached reanalysis is not a new solver run. Controlled interruptions and any duplicated computation remain included in guard costs. RSS sums can double-count; GPU total delta includes other apps.")
output=root/"phase10/measured-costs.json"
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+"\n")
report=root/"report.md"
if report.exists():
    marker="\n## Measured resource use\n"
    base=report.read_text().split(marker)[0].rstrip()
    rows=["", "| Stage | Wall (s) | Peak RSS (GiB) | Sampled device delta (GiB) | Exit |",
          "|---|---:|---:|---:|---:|"]
    for stage in stages:
        rss=stage["peak_RSS_bytes"]/(1024**3) if stage["peak_RSS_bytes"] is not None else 0
        gpu=stage["sampled_device_delta_bytes"]/(1024**3) if stage["sampled_device_delta_bytes"] is not None else 0
        rows.append(f"| {stage['stage']} | {stage['wall_s']:.1f} | {rss:.3f} | {gpu:.3f} | {stage['returncode']} |")
    rows.extend(["", "Exit −2 rows are deliberate owned-process restarts to load the documented metric/cache fixes; original elapsed cost is retained. Successful final runs exit 0. RSS can double-count shared pages and device deltas can include other applications. RAM is telemetry only. See `phase10/measured-costs.json` for individual log paths and budgets.", ""])
    report.write_text(base+"\n"+marker+"\n".join(rows))
# This postprocessor runs after package exit; retain its final source digest.
manifest=root/"source-manifest.json"
if manifest.exists():
    sources=json.loads(manifest.read_text())
    sources["scripts/record_collimation_costs.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest.write_text(json.dumps(sources,indent=2)+"\n")
print(json.dumps(dict(output=str(output),guard_stage_count=len(stages),completed_solver_records=len(solver_records)),indent=2))
