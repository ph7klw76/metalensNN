"""Reuse measured homogeneous references only after explicit configuration equivalence.

Training and lens-representation changes cannot affect a homogeneous source run.
No measured value, settings, data file, or source provenance is changed.
"""
import copy
import hashlib
import json
from pathlib import Path
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.maxwell_batch import digest
from fiber_metalens_nn.trial import write_json

old = load_config()
new = load_config('configs/finite_thickness_repair.json')
# These are the complete permitted config differences. Every other input must
# agree, including scan limits that control the simulation cell's upper edge.
reduced = copy.deepcopy(new)
reduced.pop('repair_provenance')
reduced['training'] = copy.deepcopy(old['training'])
for key in ['model','thickness_slices','split_step_reference_index']:
    reduced['proxy'].pop(key)
reduced['maxwell'].pop('profile_prism')
assert reduced == old
root = Path('artifacts/acceptance-repair/final-maxwell')
records = []
for source in Path('artifacts/maxwell').glob('reference-*.json'):
    data = json.loads(source.read_text())
    if not data.get('pass') or not Path(data['flux_data_file']).exists():
        continue
    options = {k:data['settings'][k] for k in ['resolution','pml','padding','duration','m']}
    # Preserve JSON numeric types used by batch.job so cache keys agree.
    options['m'] = int(options['m'])
    for resolution in [int(options['resolution']),float(options['resolution'])]:
        options['resolution'] = resolution
        key = digest(dict(physics_version=1,config=new,heights=None,target=None,settings=options))[:20]
        dest = root/('reference-'+key+'.json')
        if dest.exists():
            continue
        data['reference_reuse_provenance'] = dict(source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            justification='Only training, proxy internals, and lens-only geometry representation differ. All homogeneous-reference inputs agree.',
            measured_values_unchanged=True)
        write_json(dest,data)
        records.append(dict(source=str(source),cache=str(dest)))
manifest = Path('artifacts/acceptance-repair/reused-homogeneous-references.json')
previous = json.loads(manifest.read_text()) if manifest.exists() else []
write_json(manifest,previous+records)
print(json.dumps(records),flush=True)
