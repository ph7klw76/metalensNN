import json
from pathlib import Path
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.learning import phase5

c = load_config('configs/finite_thickness_repair.json')
phase5(c,folder='artifacts/acceptance-repair/phase5',phase4_gate='artifacts/acceptance-repair/proxy-validation.json')
