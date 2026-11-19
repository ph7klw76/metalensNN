"""Configure allocators before importing JAX anywhere in the project."""
import os
from pathlib import Path


def configure():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["JAX_ENABLE_X64"] = "false"
    # No persistent JAX compilation cache: disk use is bounded to zero.
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    cuda_cache = Path.cwd() / ".cache/cuda"
    cuda_cache.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_CACHE_PATH"] = str(cuda_cache)
    os.environ["CUDA_CACHE_MAXSIZE"] = str(256*1024**2)
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache/matplotlib"))
