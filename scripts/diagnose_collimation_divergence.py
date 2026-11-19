"""Read-only analysis of saved mesh fields; no optical geometry changes."""
import json
from pathlib import Path
import numpy as np
from fiber_metalens_nn.collimation_metrics import fitted_divergence
from fiber_metalens_nn.collimation_optics import cylindrical_to_cartesian,forward_spectrum,write

root=Path("artifacts/1550nm_collimation/phase9")
report={}
for path in root.glob("*-mesh.json"):
    values=[]
    for metric in json.loads(path.read_text()):
        record=json.loads(Path(metric["record"]).read_text())
        saved=np.load(record["fields_file"])
        x,dx,fields=cylindrical_to_cartesian(saved["r_um"],saved["fields"][0],512,128)
        spectrum=forward_spectrum(fields,dx,1.55)
        fitted=fitted_divergence(spectrum,dx,1.55)
        values.append(dict(resolution=metric["resolution_per_um"],second_moments=metric["divergence_half_angle_rad"],fit=fitted))
    report[path.stem]=values
write(root/"divergence-definition-diagnosis.json",report)
for name,values in report.items():
    print(name,[(v["resolution"],v["fit"]) for v in values[-2:]])
