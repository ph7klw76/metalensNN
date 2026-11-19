"""Reproducible desktop phase entrypoints; each artifact records its model label."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from .geometry import (baselines, edges, epsilon, epsilon_rz, feasible, fresnel_heights,
                       load_config, project, save_design)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    def scientific_scalar(item):
        if isinstance(item,np.generic):
            return item.item()
        if isinstance(item,np.ndarray):
            return item.tolist()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=scientific_scalar)+"\n")


def gate(phase, status, config, **data):
    result = {"phase":phase,"status":status,"result_label":config["result_label"],**data}
    write_json(f"artifacts/phase{phase}/gate.json",result)
    print(json.dumps({"phase":phase,"status":status,"result_label":config["result_label"]}),flush=True)
    return result


def phase2(config):
    import yaml
    reference = yaml.safe_load(Path("reference.yaml").read_text())
    for key in ["hadibrata","ou"]:
        source = reference["sources"][key]
        path = Path(source["supplied_file"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    source = reference["sources"]["hadibrata"]
    assert hashlib.sha256(Path(source["supporting_information"]).read_bytes()).hexdigest() == source["supporting_information_sha256"]
    assert set(config["training"]["train_targets_um"]).isdisjoint(config["training"]["validation_targets_um"])
    assert set(config["training"]["train_targets_um"]).isdisjoint(config["training"]["heldout_targets_um"])
    assert set(config["training"]["validation_targets_um"]).isdisjoint(config["training"]["heldout_targets_um"])
    assert 7.8 in config["training"]["heldout_targets_um"]
    indices = {key:{"n":value["n"],"kappa":value["kappa"],"epsilon_real":epsilon(value).real,"epsilon_imag":epsilon(value).imag,"provenance":value["provenance"]} for key,value in config["materials"].items()}
    write_json("artifacts/phase2/materials.json",indices)
    write_json("artifacts/predeclared-config.json",config)
    write_json("artifacts/predeclared-config-sha256.json",{"sha256":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()})
    return gate(2,"PASS_SYNTHETIC_ONLY",config,source_audit="reference.yaml",assumptions="assumptions.md",materials=indices,
                missing_physical_data=reference["unresolved"],physical_IP_Dip_validation=False)


def phase3(config):
    folder = Path("artifacts/phase3")
    folder.mkdir(parents=True,exist_ok=True)
    cases = baselines(7.8,config)
    cases["maximum_height"] = np.full(config["annuli"],config["height_max_um"])
    cases["alternating_supported"] = project(np.tile([0,1.5],config["annuli"]//2),config)
    results = {}
    for name,heights in cases.items():
        assert not feasible(heights,config), (name,feasible(heights,config))
        save_design(folder/f"{name}.json",heights,7.8,name,config)
        r = np.linspace(0,14,701)
        z = np.linspace(-0.2,2.6,141)
        er, polymer, stop = epsilon_rz(heights,r,z,config)
        np.savez_compressed(folder/f"{name}-epsilon.npz",r_um=r,z_um=z,epsilon_real=er.real,epsilon_imag=er.imag,polymer=polymer,PEC_stop=stop)
        results[name] = {"feasible":True,"nonzero_annuli":int(np.count_nonzero(heights)),"max_height_um":float(np.max(heights))}
    np.savetxt(folder/"fresnel-rings-um.csv",np.column_stack([edges(config)[:-1],edges(config)[1:],cases["fresnel"]]),delimiter=",",header="inner_radius_um,outer_radius_um,height_um",comments="")
    return gate(3,"PASS",config,cases=results,ring_width_um=float(np.diff(edges(config))[0]),collection_radius_um=config["collection_radius_um"],binary_epsilon_and_supports=True)


def bounded_correction(raw, target, config, xp):
    base = fresnel_heights(target,config,xp=xp)
    fraction = xp.clip(base/config["height_max_um"],1e-4,1-1e-4)
    logits = xp.log(fraction/(1-fraction))+config["training"]["correction_logit_scale"]*xp.tanh(raw)
    return config["height_max_um"]/(1+xp.exp(-logits))


def direct_optimize(proxy, target, steps, seed, config):
    import jax
    import jax.numpy as jnp
    import optax
    rng = np.random.default_rng(seed)
    raw = jnp.asarray(rng.normal(0,0.002,config["annuli"]),jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),optax.adam(config["training"]["direct_learning_rate"]))
    state = optimizer.init(raw)
    objective = lambda value:proxy._objective(bounded_correction(value,jnp.asarray(target,jnp.float32),config,jnp),jnp.asarray(target,jnp.float32))
    value_gradient = jax.jit(jax.value_and_grad(objective))
    history = []
    start = time.monotonic()
    for _ in range(steps):
        loss, grads = value_gradient(raw)
        updates,state = optimizer.update(grads,state,raw)
        raw = optax.apply_updates(raw,updates)
        history.append(float(loss))
    heights = np.asarray(bounded_correction(raw,target,config,jnp))
    return project(heights,config), {"objective_forwards":steps,"optical_plane_forwards":3*steps,"wall_s":time.monotonic()-start,"loss":history}


def phase4(config):
    from .runtime import configure
    configure()
    import os
    os.environ["JAX_PLATFORMS"] = "cuda"
    import jax
    import jax.numpy as jnp
    from .proxy import OpticalProxy
    folder = Path("artifacts/phase4")
    folder.mkdir(parents=True,exist_ok=True)
    device = jax.devices("gpu")[0]
    # Small complex128 checks only; production fields remain complex64.
    jax.config.update("jax_enable_x64",True)
    small = OpticalProxy(config,grid_n=64,high_precision=True)
    heights = jnp.asarray(np.linspace(0.71,1.63,config["annuli"]),jnp.float64)
    value, gradient = small.value_gradient(heights,jnp.asarray(7.8,jnp.float64))
    derivatives = []
    for index in [0,7,16,31]:
        delta = 1e-4
        fd = float((small.objective(heights.at[index].add(delta),7.8)-small.objective(heights.at[index].add(-delta),7.8))/(2*delta))
        analytic = float(gradient[index])
        error = abs(fd-analytic)
        derivatives.append({"annulus":index,"analytic":analytic,"finite_difference":fd,"absolute_error":error,"pass":error < 5e-5+0.01*abs(fd)})
    # Homogeneous plane-wave: phase, longitudinal transversality, flux.
    seed = jnp.ones((small.size,small.size),jnp.complex128)
    spectra = jnp.fft.fft2(seed)[None,:,:]*small.vector_kernel
    wave = small.fields(spectra,1.7)
    expected = np.exp(1j*small.k0*small.na.real*1.7)
    plane_error = float(jnp.max(jnp.abs(wave[0]-expected)))
    h = project(fresnel_heights(7.8,config),config)
    spectra = small.spectrum(h)
    powers = [float(jnp.sum(small.flux(small.fields(spectra,z)))*small.dx**2) for z in [0.0,4.0,9.0]]
    energy_error = float(np.ptp(powers)/np.mean(powers))
    thin_zero = complex(np.asarray(small.transmission(jnp.zeros(config["annuli"])))[0])
    zero_expected = 2*small.ns/(small.ns+small.na)*np.exp(1j*small.k0*small.na*config["exit_plane_z_um"])
    interface_error = abs(thin_zero-zero_expected)
    del small
    jax.config.update("jax_enable_x64",False)
    jax.clear_caches()
    proxy = OpticalProxy(config)
    # A known ideal phase is used ONLY in this propagator test.
    xx,yy = np.meshgrid(proxy.x,proxy.x,indexing="xy")
    aperture = np.hypot(xx,yy) < config["aperture_diameter_um"]/2
    phase = -proxy.k0*proxy.na.real*(np.sqrt(xx*xx+yy*yy+7.8**2)-7.8)
    ideal = jnp.asarray(aperture*np.exp(1j*phase),jnp.complex64)
    spec = jnp.fft.fft2(ideal)[None,:,:]*proxy.vector_kernel
    zs = np.arange(6,9.51,0.25)
    axial = []
    mid = proxy.size//2
    for z in zs:
        field = proxy.fields(spec,z)
        axial.append(float(jnp.mean(jnp.sum(jnp.abs(field[:3,mid-1:mid+1,mid-1:mid+1])**2,axis=0))))
    from .metrics import peak_position
    known_focus,_ = peak_position(zs,axial)
    checks = {"derivatives":derivatives,"plane_wave_max_error":plane_error,"energy_relative_error":energy_error,
              "zero_height_interface_error":interface_error,"known_phase_focus_um":known_focus,
              "known_phase_is_test_only":True,"gradient_device":str(device)}
    checks["pass"] = all(d["pass"] for d in derivatives) and plane_error < 1e-8 and energy_error < 1e-8 and interface_error < 1e-8 and abs(known_focus-7.8)/7.8 < 0.1
    write_json(folder/"physics-checks.json",checks)
    if not checks["pass"]:
        gate(4,"FAIL_PHYSICS_TEST",config,checks=checks)
        raise RuntimeError("Proxy verification failed; see physics-checks.json")
    baseline_reports = {}
    cases = baselines(7.8,config)
    for name,heights in cases.items():
        baseline_reports[name] = proxy.evaluate(heights,7.8,folder/f"{name}-planes.npz")
    direct, cost = direct_optimize(proxy,7.8,config["training"]["phase4_direct_steps"],11,config)
    save_design(folder/"direct.json",direct,7.8,"phase4_direct",config,optimization_cost=cost)
    baseline_reports["direct"] = proxy.evaluate(direct,7.8,folder/"direct-planes.npz")
    write_json(folder/"baselines.json",baseline_reports)
    # Baseline sampling and padding checked before training or test selection.
    higher = OpticalProxy(config,grid_n=512)
    fine = higher.evaluate(cases["fresnel"],7.8,folder/"sampling-512-planes.npz")
    del higher
    jax.clear_caches()
    padded = OpticalProxy(config,padding=3)
    padding = padded.evaluate(cases["fresnel"],7.8)
    coarse = baseline_reports["fresnel"]
    sampling = {"n256":coarse,"n512":fine,"padding3_n256":padding}
    for name,report in [("grid",fine),("padding",padding)]:
        sampling[name+"_eta_relative_change"] = abs(report["eta_R"]-coarse["eta_R"])/max(abs(report["eta_R"]),1e-12)
        sampling[name+"_focus_relative_change"] = abs(report["focal_um"]-coarse["focal_um"])/abs(report["focal_um"])
    sampling["pass"] = all(sampling[n+"_eta_relative_change"] < config["proxy"]["sampling_eta_relative_tolerance"] and sampling[n+"_focus_relative_change"] < config["proxy"]["sampling_focus_relative_tolerance"] for n in ["grid","padding"])
    write_json(folder/"sampling.json",sampling)
    return gate(4,"PASS" if sampling["pass"] else "FAIL_SAMPLING",config,physics_checks=checks,sampling=sampling,baseline_metrics=baseline_reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",type=int,choices=[2,3,4,5,6,7])
    parser.add_argument("--config",default="configs/desktop_trial.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.phase in [6,7]:
        from .maxwell_batch import phase6, phase7
        result = {6:phase6,7:phase7}[args.phase](config)
    elif args.phase == 5:
        from .learning import phase5
        result = phase5(config)
    else:
        result = {2:phase2,3:phase3,4:phase4}[args.phase](config)
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
