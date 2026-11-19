"""Fresh two-condition NN training; held-out pairs never select seeds/weights."""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from .learning import ConditionalMLP,save_checkpoint,numpy_tree
from .collimation_proxy import CylinderProxy
from .collimation_optics import write,fingerprint


def train(experiment):
    cfg=experiment.config
    phase8=json.loads((experiment.root/"phase8/gate.json").read_text())
    if phase8["status"]!="PASS":
        raise RuntimeError("Diagnose and fix Phase 8 before training")
    folder=experiment.root/"phase9"
    folder.mkdir(exist_ok=True)
    tc=cfg["training"]
    sets=[set(map(tuple,tc[key])) for key in ("train_pairs","validation_pairs","heldout_pairs")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Training/validation/test pairs overlap")
    write(folder/"predeclared-split-and-budget.json",dict(training=tc,Maxwell=cfg["maxwell"],
          selection="minimum mean validation loss only; heldouts evaluated after selection",
          correction_rounds_used=0,phase8_diagnostic_pair=[34,28]))
    proxy=CylinderProxy(cfg)
    # Pedestal fields depend on the input H, never the desired output.
    pairs=tc["train_pairs"]+tc["validation_pairs"]+tc["heldout_pairs"]
    for H,D in pairs:
        proxy.pedestal(H)
        proxy.target(D)
    model=ConditionalMLP(cfg["annuli"])
    optimizer=optax.chain(optax.clip_by_global_norm(1),optax.adam(tc["learning_rate"]))

    def prediction(params,condition,seed):
        raw=model.apply(params,condition[None])[0]
        fraction=jnp.clip(seed/cfg["height_max_um"],.001,.999)
        logit=jnp.log(fraction/(1-fraction))
        return cfg["height_max_um"]*jax.nn.sigmoid(logit+raw)

    def objective(params,condition,seed,pedestal,target,D):
        return proxy._objective(prediction(params,condition,seed),pedestal,target,D)

    vg=jax.value_and_grad(objective)
    evaluate=jax.jit(objective)
    predict=jax.jit(prediction)

    @jax.jit
    def update(params,state,condition,seed,pedestal,target,D):
        loss,gradient=vg(params,condition,seed,pedestal,target,D)
        changes,state=optimizer.update(gradient,state,params)
        return optax.apply_updates(params,changes),state,loss

    def inputs(pair):
        H,D=pair
        return (jnp.asarray([(H-35)/5,(D-30)/3],jnp.float32),proxy.seed(H),proxy.pedestal(H),proxy.target(D),jnp.asarray(D,jnp.float32))

    reports=[]
    trained={}
    physics=(Path(__file__).parent/"collimation_proxy.py").read_text()
    if cfg["loss"].get("version")=="five_plane_exp2_v2":
        from .collimation_loss_v2 import source_text
        physics += "\n"+source_text()
    signature=fingerprint(dict(config=cfg,physics=physics))
    for seed in tc["seeds"]:
        start=time.monotonic()
        checkpoint=folder/f"seed-{seed}.pkl"
        rng=np.random.default_rng(seed)
        params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,2),jnp.float32))
        state=optimizer.init(params)
        history=[]
        validation=[]
        best=params
        best_loss=1e30
        step=forwards=patience=0
        recovery=None
        previous=0.
        if checkpoint.exists():
            data=pickle.loads(checkpoint.read_bytes())
            if data["signature"]!=signature:
                raise RuntimeError("Checkpoint physics/config differs; use a new run root")
            params,state,best=(jax.tree.map(jnp.asarray,data[k]) for k in ("params","state","best"))
            history,validation=data["history"],data["validation"]
            best_loss,step,forwards,patience=data["best_loss"],data["step"],data["forwards"],data["patience"]
            rng.bit_generator.state=data["rng"]
            recovery,previous=data["recovery"],data["wall_s"]
        while step<tc["steps"] and patience<tc["patience"]:
            pair=tc["train_pairs"][int(rng.integers(len(tc["train_pairs"])))]
            params,state,loss=update(params,state,*inputs(pair))
            step+=1
            forwards+=1
            history.append(dict(step=step,pair=pair,loss=float(loss)))
            if step%tc["validation_interval"]==0:
                scores=[float(evaluate(params,*inputs(pair))) for pair in tc["validation_pairs"]]
                forwards+=len(scores)
                score=float(np.mean(scores))
                validation.append(dict(step=step,mean_loss=score))
                if score<best_loss-1e-5:
                    best,best_loss,patience=params,score,0
                else:
                    patience+=1
                if recovery is None:
                    # Nonzero Adam moments; same next update after serialization.
                    restored=pickle.loads(pickle.dumps(dict(params=numpy_tree(params),state=numpy_tree(state))))
                    a=update(params,state,*inputs(tc["train_pairs"][0]))
                    b=update(jax.tree.map(jnp.asarray,restored["params"]),jax.tree.map(jnp.asarray,restored["state"]),*inputs(tc["train_pairs"][0]))
                    error=max(float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
                    recovery=dict(trained_step=step,max_next_update_difference=error,passed=error<1e-7)
                    forwards+=2
                data=dict(signature=signature,params=numpy_tree(params),state=numpy_tree(state),best=numpy_tree(best),
                          history=history,validation=validation,best_loss=best_loss,step=step,forwards=forwards,patience=patience,
                          rng=rng.bit_generator.state,recovery=recovery,wall_s=previous+time.monotonic()-start)
                save_checkpoint(checkpoint,data)
                print(f"Collimation seed {seed}: step {step}, validation {score:.6f}",flush=True)
        report=dict(seed=seed,steps=step,objective_forwards=forwards,best_validation_loss=best_loss,
                    trained_state_recovery=recovery,training_wall_s=previous+time.monotonic()-start,
                    history=history,validation=validation)
        write(folder/f"seed-{seed}-training.json",report)
        reports.append(report)
        trained[seed]=best
    selected=min(reports,key=lambda s:s["best_validation_loss"])["seed"]
    write(folder/"selection-before-heldout.json",dict(seed=selected,rule="minimum validation loss",heldout_used=False))
    # Budget-matched direct optimization across heldout pairs for every seed.
    heldouts=[]
    for report in reports:
        seed=report["seed"]
        total=report["objective_forwards"]
        q,remainder=divmod(total,len(tc["heldout_pairs"]))
        for i,pair in enumerate(tc["heldout_pairs"]):
            H,D=pair
            inputs_pair=inputs(pair)
            before=time.monotonic()
            heights=np.asarray(predict(trained[seed],*inputs_pair[:2]))
            elapsed=time.monotonic()-before
            path=experiment.design(f"NN-seed{seed}-H{H}-D{D}",heights,H,D,"untouched_NN")
            rounded=json.loads(Path(path).read_text())["heights_um"]
            nn_metric=proxy.evaluate(rounded,H,D)
            direct_steps=q+(i<remainder)
            direct_path=experiment.root/"designs"/f"direct-seed{seed}-H{H}-D{D}.json"
            if direct_path.exists():
                direct=np.asarray(json.loads(direct_path.read_text())["heights_um"])
            else:
                fraction=jnp.clip(proxy.seed(H)/cfg["height_max_um"],.001,.999)
                raw=jnp.log(fraction/(1-fraction))+jnp.asarray(np.random.default_rng(seed).normal(0,.03,cfg["annuli"]),jnp.float32)
                opt=optax.adam(tc["direct_learning_rate"])
                state=opt.init(raw)
                def loss_direct(raw):
                    return proxy._objective(cfg["height_max_um"]*jax.nn.sigmoid(raw),*inputs_pair[2:])
                gradient=jax.jit(jax.value_and_grad(loss_direct))
                direct_start=time.monotonic()
                for _ in range(direct_steps):
                    loss,g=gradient(raw)
                    changes,state=opt.update(g,state,raw)
                    raw=optax.apply_updates(raw,changes)
                direct=np.asarray(cfg["height_max_um"]*jax.nn.sigmoid(raw))
                experiment.design(direct_path.stem,direct,H,D,"direct_optimized")
                write(folder/f"{direct_path.stem}-cost.json",dict(objective_forwards=direct_steps,wall_s=time.monotonic()-direct_start))
                direct=np.asarray(json.loads(direct_path.read_text())["heights_um"])
            direct_metric=proxy.evaluate(direct,H,D)
            entry=dict(seed=seed,pair=pair,NN_design=path,direct_design=str(direct_path),NN_proxy=nn_metric,
                       direct_proxy=direct_metric,direct_objective_forwards=direct_steps,inference_wall_s=elapsed)
            heldouts.append(entry)
            write(folder/"heldout.json",heldouts)
    # Verify conditional behavior on a predeclared train-domain pair, not test scores.
    a=np.asarray(predict(trained[selected],*inputs([30,27])[:2]))
    b=np.asarray(predict(trained[selected],*inputs([40,33])[:2]))
    conditional_rms=float(np.sqrt(np.mean((a-b)**2)))
    result=dict(phase=9,stage="proxy_training",status="PASS" if all(r["trained_state_recovery"]["passed"] for r in reports) and conditional_rms>.001 else "FAIL_RESUME_OR_CONDITIONAL",
                selected_seed=selected,conditional_geometry_rms_um=conditional_rms,three_seeds=True,matched_total_forward_budgets=True,
                direct_budget_definition="Per seed, sum of direct objective calls across heldouts equals all NN train+validation+resume calls. Final metric scans excluded on both sides.",
                heldout_used_for_selection=False,full_wave_validated=False)
    write(folder/"training-gate.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result
