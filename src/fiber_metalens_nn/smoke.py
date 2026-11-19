"""CPU/GPU analytic forward, autodiff, finite difference and FFT smoke."""
import importlib.metadata
import json
import os
import time
from pathlib import Path

from .runtime import configure


def smoke(backend, output):
    configure()
    os.environ["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import linen as nn
    import optax

    start = time.monotonic()
    devices = jax.devices(backend)  # raises rather than silently falling back
    with jax.default_device(devices[0]):
        x = jnp.linspace(-0.5, 0.5, 32, dtype=jnp.float32)
        loss = jax.jit(lambda v: jnp.sum(jnp.sin(v)**2))
        value, gradient = jax.jit(jax.value_and_grad(loss))(x)
        gradient.block_until_ready()
        expected = np.sin(2*np.asarray(x))
        error = float(np.max(np.abs(np.asarray(gradient)-expected)))
        i, delta = 7, 1e-3
        finite = float((loss(x.at[i].add(delta))-loss(x.at[i].add(-delta)))/(2*delta))
        fd_error = abs(finite-float(gradient[i]))
        field = jnp.exp(1j*jnp.arange(256*256, dtype=jnp.float32).reshape(256,256)/100)
        recovered = jnp.fft.ifft2(jnp.fft.fft2(field))
        fft_error = float(jnp.max(jnp.abs(recovered-field)))

        class SmallMLP(nn.Module):
            @nn.compact
            def __call__(self, v):
                for _ in range(3):
                    v = nn.tanh(nn.Dense(64)(v))
                return nn.sigmoid(nn.Dense(32)(v))*2.5

        model = SmallMLP()
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1,1)))
        optimizer = optax.adam(1e-3)
        state = optimizer.init(params)
        fn = lambda p: jnp.mean(model.apply(p, jnp.ones((1,1)))**2)
        mlp_value, grads = jax.value_and_grad(fn)(params)
        updates, state = optimizer.update(grads, state, params)
        updated = optax.apply_updates(params, updates)
        mlp_after = float(fn(updated))
        report = {"backend": backend, "devices": [str(d) for d in devices],
                  "value": float(value), "analytic_gradient_max_abs_error": error,
                  "centered_difference_abs_error": fd_error, "fft_round_trip_max_abs_error": fft_error,
                  "mlp_loss_before": float(mlp_value), "mlp_loss_after": mlp_after,
                  "dtype": str(field.dtype), "preallocate": False,
                  "wall_s": time.monotonic()-start,
                  "versions": {p: importlib.metadata.version(p) for p in ["jax","jaxlib","flax","optax","numpy"]},
                  "cuda_package_versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                                            if d.metadata["Name"].lower().startswith("nvidia-")},
                  "jax_memory_stats": devices[0].memory_stats(),
                  "pass": error < 1e-5 and fd_error < 1e-3 and fft_error < 2e-6 and mlp_after < float(mlp_value)}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1
