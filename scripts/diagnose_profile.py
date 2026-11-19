from pathlib import Path
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.maxwell_batch import MaxwellBatch
from fiber_metalens_nn.trial import write_json

c = load_config()
c['maxwell']['profile_prism'] = True
batch = MaxwellBatch(c,'artifacts/acceptance-repair/profile-maxwell')
out = {}
for name,path in [('flat','artifacts/phase6/flat-f7.8.json'),('fresnel','artifacts/phase6/fresnel-f7.8.json'),('NN','artifacts/phase5/selected-NN-f7.8.json'),('training_NN','artifacts/phase5/selected-NN-f7.4.json')]:
    out[name] = batch.job(path,resolution=20)
    write_json('artifacts/acceptance-repair/profile-diagnostic.json',out)
