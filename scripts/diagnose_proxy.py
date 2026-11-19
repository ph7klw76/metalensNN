import os
os.environ['JAX_PLATFORMS'] = 'cuda'
import json
from pathlib import Path
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.thick_proxy import ThickProxy
from fiber_metalens_nn.trial import write_json

c = load_config()
proxy = ThickProxy(c)
out = {}
for name,path in [('fresnel','artifacts/phase6/fresnel-f7.8.json'),('NN','artifacts/phase5/selected-NN-f7.8.json')]:
    d = json.loads(Path(path).read_text())
    out[name] = proxy.evaluate(d['heights_um'],7.8)
    print(json.dumps(out[name]),flush=True)
write_json('artifacts/acceptance-repair/thick-proxy-diagnostic.json',out)
