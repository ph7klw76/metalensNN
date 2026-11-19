"""Check resumed Adam updates on actual trained, nonzero optimizer states.

Uses a fixed diagnostic gradient, no optical simulation and no model selection.
It leaves all trained checkpoints and geometries unchanged.
"""
import json
import argparse
import pickle
from pathlib import Path
from fiber_metalens_nn.runtime import configure
configure()
import jax
import jax.numpy as jnp
import numpy as np
import optax
from fiber_metalens_nn.trial import write_json

parser = argparse.ArgumentParser()
parser.add_argument('--config',default='configs/desktop_trial.json')
parser.add_argument('--folder',default='artifacts/phase5')
args = parser.parse_args()
folder = Path(args.folder)
cfg = json.loads(Path(args.config).read_text())
optimizer = optax.chain(optax.clip_by_global_norm(1.),optax.adam(cfg["training"]["learning_rate"]))
checks = []
for seed in cfg["training"]["seeds"]:
    checkpoint = pickle.loads((folder/f"seed-{seed}/latest.pkl").read_bytes())
    before = (checkpoint["params"],checkpoint["state"])
    restored = pickle.loads(pickle.dumps(before,protocol=5))
    values = []
    for params,state in [before,restored]:
        params,state = jax.tree.map(jnp.asarray,(params,state))
        diagnostic = jax.tree.map(lambda x:jnp.full_like(x,1e-4),params)
        changes,next_state = optimizer.update(diagnostic,state,params)
        values.append((optax.apply_updates(params,changes),next_state))
    error = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree.leaves(values[0]),jax.tree.leaves(values[1])))
    nonzero = sum(np.count_nonzero(np.asarray(leaf)) for leaf in jax.tree.leaves(checkpoint["state"]))
    checks.append({"seed":seed,"checkpoint_step":checkpoint["step"],"nonzero_optimizer_entries":int(nonzero),
                   "next_update_max_abs_difference":error,"pass":error==0 and nonzero>0})
report = {"checks":checks,"pass":all(value["pass"] for value in checks),"optical_forwards":0,
          "diagnostic_gradient":"constant 1e-4; verifies trained optimizer-state serialization and continuation only"}
write_json(folder/"trained-state-resume.json",report)
gate_path = folder/"gate.json"
gate = json.loads(gate_path.read_text())
gate["trained_nonzero_optimizer_recovery"] = report
if not report["pass"]:
    gate["status"] = "FAIL_TRAINED_STATE_RESUME"
write_json(gate_path,gate)
print(json.dumps(report,indent=2))
