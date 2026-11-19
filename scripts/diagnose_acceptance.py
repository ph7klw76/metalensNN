"""Controlled reruns; never overwrite the original trial or its gates."""
import json
from pathlib import Path
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.maxwell_batch import MaxwellBatch, relative_checks
from fiber_metalens_nn.trial import write_json

c = load_config("configs/desktop_trial.json")
c["maxwell"]["merge_equal_height_annuli"] = True
root = Path("artifacts/acceptance-repair")
root.mkdir(exist_ok=True)
batch = MaxwellBatch(c,root/"merged-maxwell")
results = {}
for resolution in [20,30,40]:
    results[str(resolution)] = batch.job("artifacts/phase6/flat-f7.8.json",resolution=resolution)
    write_json(root/"flat-merged-mesh.json",results)
print(json.dumps({"flat_merged":relative_checks(results['30'],results['40'],c,False)}),flush=True)
