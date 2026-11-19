"""Small conditional Flax network trained through the verified optical proxy."""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np

from .runtime import configure
configure()
import os
os.environ["JAX_PLATFORMS"] = "cuda"
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from .geometry import baselines, feasible, fresnel_heights, project, save_design
from .proxy import OpticalProxy
from .trial import bounded_correction, direct_optimize, gate, write_json


class ConditionalMLP(nn.Module):
    annuli: int

    @nn.compact
    def __call__(self, target):
        value = target
        for _ in range(3):
            value = nn.tanh(nn.Dense(64)(value))
        return nn.Dense(self.annuli,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(value)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value,stream,protocol=5)
    temporary.replace(path)


def numpy_tree(value):
    return jax.tree.map(lambda x:np.asarray(x),value)


def phase5(config, folder="artifacts/phase5", phase4_gate="artifacts/phase4/gate.json"):
    if not json.loads(Path(phase4_gate).read_text())["status"].startswith("PASS"):
        raise RuntimeError("Phase 4 must pass before neural training")
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    cfg = config["training"]
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write_json(folder/"frozen-config.json",config)
    if config["proxy"].get("model") == "split_step":
        from .thick_proxy import ThickProxy
        proxy = ThickProxy(config)
    else:
        proxy = OpticalProxy(config)
    model = ConditionalMLP(config["annuli"])
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),optax.adam(cfg["learning_rate"]))

    def predict(params,target):
        raw = model.apply(params,jnp.reshape(target-8.0,(1,1)))[0]
        return bounded_correction(raw,target,config,jnp)

    predict_jit = jax.jit(predict)
    value_gradient = jax.value_and_grad(lambda params,target:proxy._objective(predict(params,target),target))

    @jax.jit
    def update(params,state,target):
        value,gradient = value_gradient(params,target)
        changes,state = optimizer.update(gradient,state,params)
        return optax.apply_updates(params,changes),state,value

    validation_loss = jax.jit(lambda params,target:proxy._objective(predict(params,target),target))
    seeds = []
    trained = {}
    for seed in cfg["seeds"]:
        start = time.monotonic()
        seed_folder = folder/f"seed-{seed}"
        seed_folder.mkdir(exist_ok=True)
        checkpoint_path = seed_folder/"latest.pkl"
        rng = np.random.default_rng(seed)
        params = model.init(jax.random.PRNGKey(seed),jnp.zeros((1,1),jnp.float32))
        state = optimizer.init(params)
        history, validation = [], []
        best_score, best_params, patience, forward_count, step = float("inf"), params, 0, 0, 0
        recovery = None
        previous_wall = 0.0
        if checkpoint_path.exists():
            checkpoint = pickle.loads(checkpoint_path.read_bytes())
            if checkpoint["fingerprint"] != fingerprint:
                raise RuntimeError("Checkpoint configuration differs; preserve it and choose a new run directory")
            params = jax.tree.map(jnp.asarray,checkpoint["params"])
            state = jax.tree.map(jnp.asarray,checkpoint["state"])
            best_params = jax.tree.map(jnp.asarray,checkpoint["best_params"])
            history,validation = checkpoint["history"],checkpoint["validation"]
            best_score,patience = checkpoint["best_score"],checkpoint["patience"]
            forward_count,step = checkpoint["forward_count"],checkpoint["step"]
            rng.bit_generator.state = checkpoint["rng"]
            recovery,previous_wall = checkpoint["recovery"],checkpoint["wall_s"]
            print(f"Resume NN seed {seed} at step {step}",flush=True)
        if recovery is None:
            # Compare an uninterrupted optimizer update with the same update
            # after serialization. Both use a declared training target.
            save_checkpoint(seed_folder/"recovery.pkl",{"params":numpy_tree(params),"state":numpy_tree(state)})
            first = update(params,state,jnp.asarray(cfg["train_targets_um"][0],jnp.float32))
            restored = pickle.loads((seed_folder/"recovery.pkl").read_bytes())
            second = update(jax.tree.map(jnp.asarray,restored["params"]),jax.tree.map(jnp.asarray,restored["state"]),jnp.asarray(cfg["train_targets_um"][0],jnp.float32))
            differences = [float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second))]
            recovery = {"pass":max(differences) < 1e-7,"max_abs_error":max(differences),"objective_forwards":2}
            forward_count += 2
        while step < cfg["steps"] and patience < cfg["patience"]:
            target = float(rng.choice(cfg["train_targets_um"]))
            params,state,loss = update(params,state,jnp.asarray(target,jnp.float32))
            step += 1
            forward_count += 1
            history.append({"step":step,"target_um":target,"loss":float(loss)})
            if step % cfg["validation_interval"] == 0:
                scores = [float(validation_loss(params,jnp.asarray(target,jnp.float32))) for target in cfg["validation_targets_um"]]
                forward_count += len(scores)
                score = float(np.mean(scores))
                validation.append({"step":step,"mean_loss":score,"individual_losses":scores})
                if score < best_score-cfg["min_improvement"]:
                    best_score,best_params,patience = score,params,0
                else:
                    patience += 1
                save_checkpoint(checkpoint_path,{"fingerprint":fingerprint,"params":numpy_tree(params),"state":numpy_tree(state),
                                                "best_params":numpy_tree(best_params),"step":step,"rng":rng.bit_generator.state,
                                                "history":history,"validation":validation,"best_score":best_score,"patience":patience,
                                                "forward_count":forward_count,"recovery":recovery,"wall_s":previous_wall+time.monotonic()-start})
                print(f"NN seed {seed}: step {step}, validation loss {score:.6f}",flush=True)
        report = {"seed":seed,"steps":step,"best_validation_loss":best_score,"objective_forwards":forward_count,
                  "optical_plane_forwards":3*forward_count,"training_wall_s":previous_wall+time.monotonic()-start,
                  "checkpoint_recovery":recovery,"history":history,"validation":validation}
        write_json(seed_folder/"training.json",report)
        seeds.append(report)
        trained[seed] = best_params
    selected = min(seeds,key=lambda report:report["best_validation_loss"])["seed"]
    write_json(folder/"selection-before-heldout.json",{"selected_seed":selected,"rule":"minimum mean validation loss","heldout_used_for_selection":False})
    outputs = {}
    for report in seeds:
        seed = report["seed"]
        params = trained[seed]
        seed_folder = folder/f"seed-{seed}"
        # Equal aggregate optimization-forward budgets across the three heldouts.
        count,remainder = divmod(report["objective_forwards"],len(cfg["heldout_targets_um"]))
        predictions,comparisons = [], []
        for i,target in enumerate(cfg["heldout_targets_um"]):
            before = time.monotonic()
            heights = project(np.asarray(predict_jit(params,jnp.asarray(target,jnp.float32))),config)
            inference_wall = time.monotonic()-before
            filename = seed_folder/f"nn-f{target:.1f}.json"
            save_design(filename,heights,target,"untouched_NN",config,seed=seed,corrected_with_Maxwell=False)
            metric = proxy.evaluate(heights,target,seed_folder/f"nn-f{target:.1f}-planes.npz")
            metric["inference_wall_s_including_first_compile_if_cold"] = inference_wall
            predictions.append(metric)
            steps = count+(i < remainder)
            direct_path = seed_folder/f"direct-f{target:.1f}.json"
            if direct_path.exists():
                cached = json.loads(direct_path.read_text())
                cost = cached["optimization_cost"]
                assert cost["objective_forwards"] == steps and cached["seed"] == seed
                direct = np.asarray(cached["heights_um"])
            else:
                direct,cost = direct_optimize(proxy,target,steps,seed,config)
                save_design(direct_path,direct,target,"matched_direct",config,seed=seed,optimization_cost=cost)
            direct_metric = proxy.evaluate(direct,target,seed_folder/f"direct-f{target:.1f}-planes.npz")
            comparisons.append({"target_um":target,"optimization_cost":cost,"metrics":direct_metric})
        raw_a = np.asarray(model.apply(params,jnp.asarray([[-1.0]],jnp.float32)))
        raw_b = np.asarray(model.apply(params,jnp.asarray([[1.0]],jnp.float32)))
        focal_values = [p["focal_um"] for p in predictions]
        correlation = float(np.corrcoef(cfg["heldout_targets_um"],focal_values)[0,1])
        conditional = {"target_focus_correlation":correlation,"raw_correction_rms_7_to_9":float(np.sqrt(np.mean((raw_b-raw_a)**2))),
                       "pass":bool(correlation > 0.5 and np.sqrt(np.mean((raw_b-raw_a)**2)) > 1e-3)}
        outputs[str(seed)] = {"NN":predictions,"direct":comparisons,"conditional_test":conditional,
                              "matched_total_objective_forwards":sum(c["optimization_cost"]["objective_forwards"] for c in comparisons)}
        assert outputs[str(seed)]["matched_total_objective_forwards"] == report["objective_forwards"]
    for target in [7.4,7.8]:
        heights = project(np.asarray(predict_jit(trained[selected],jnp.asarray(target,jnp.float32))),config)
        save_design(folder/f"selected-NN-f{target:.1f}.json",heights,target,"untouched_NN",config,seed=selected,corrected_with_Maxwell=False)
    baseline_metrics = {str(target):proxy.evaluate(baselines(target,config)["fresnel"],target) for target in cfg["heldout_targets_um"]}
    write_json(folder/"heldout.json",{"selected_seed":selected,"seeds":outputs,"fresnel":baseline_metrics,
                                      "budget_definition":"NN training+validation+recovery objective forwards equals direct aggregate over heldout targets; final scans excluded equally", "planes_per_objective":3})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for report in seeds:
        axes[0].plot([p["step"] for p in report["history"]],[p["loss"] for p in report["history"]],alpha=.55,label=f'seed {report["seed"]}')
        axes[1].plot([p["step"] for p in report["validation"]],[p["mean_loss"] for p in report["validation"]],label=f'seed {report["seed"]}')
    for axis,title in zip(axes,["Synthetic proxy training loss","Validation loss (selection only)"]):
        axis.set(xlabel="Optimizer update",ylabel="Fixed proxy loss",title=title)
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/"learning-curves.png",dpi=160)
    plt.close(fig)
    passed = all(report["checkpoint_recovery"]["pass"] for report in seeds) and all(value["conditional_test"]["pass"] for value in outputs.values())
    # Keep separate repair runs from overwriting the original scientific gate.
    result = dict(phase=5,status="PASS" if passed else "FAIL_CONDITIONAL_OR_RESUME",
                  result_label=config["result_label"],selected_seed=selected,seeds=seeds,
                  heldout_targets_um=cfg["heldout_targets_um"],matched_budgets=True,full_wave_validated=False)
    write_json(folder/"gate.json",result)
    return result
