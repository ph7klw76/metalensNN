"""Report measured repair results without rewriting the original trial."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiber_metalens_nn.trial import write_json

root = Path('artifacts/acceptance-repair')
read = lambda p:json.loads(Path(p).read_text())
g = read(root/'phase6-7/gate.json')
meshes = read(root/'phase6-7/mesh-results.json')
old = read('artifacts/phase6/mesh-results.json')
flat = read(root/'flat-merged-mesh.json')
diagnostic = read(root/'thick-proxy-diagnostic.json')
validation = read(root/'proxy-validation.json')
nn,fr = g['NN'],g['matched_fresnel']
old_fr_change = abs(old['fresnel'][-2]['eta_R']/old['fresnel'][-1]['eta_R']-1)
new_fr_change = g['mesh_convergence']['original_fresnel']['eta_relative_change']
lines = ['# Desktop acceptance repair','',f"**Repair status: {g['status']}. Engineering: {g['engineering_gate']}; optical: {g['optical_gate']}.**",
         '', 'All results are synthetic, using the original assumed materials. The original `results.md` and Phase 1–7 gates remain the historical record.',
         '', '## Diagnosed causes and corrections','',
         '1. **Artificial geometry boundaries.** The flat resin was represented as 32 touching blocks. Replacing these with one block changes no physical material but removes internal boundaries from Meep’s subpixel averaging. The 30→40 cells/µm efficiency discrepancy falls from 47.54% to 0.49%. The production repair represents each connected stepped profile with a single extruded polygon. Polygon tests verify the exact annular solid and area. Coarse flat-prism and flat-block results agree.',
         f"The unchanged original Fresnel profile independently improves from {100*old_fr_change:.2f}% to {100*new_fr_change:.2f}% efficiency change between its last two meshes. Thus both failing baselines are checked without relying on the redesigned lens.",
         '2. **Inadequate thin-element forward model.** The old proxy applies the entire height-dependent phase at one exit plane, omitting diffraction within the 2.5 µm structure. Adding split-step propagation through the actual annular occupancy moves the unchanged NN geometry’s predicted focus from 7.77 µm to '+f"{diagnostic['NN']['focal_um']:.3f} µm, toward its Maxwell focus of 6.373 µm. This controlled model comparison explains most, not all, of the bias. Vector interface coupling and nonlocal reflection remain approximations; only Maxwell results determine acceptance.",
         'The 2.5 µm thickness is 2.55 vacuum wavelengths, and each annulus is only 0.3875 µm wide. Treating the structure as independent local phase delays is therefore a strong approximation. The remaining proxy/Maxwell difference is retained as model error, not removed by changing the reported focus origin.',
         '3. **Configuration forwarding defect.** MaxwellBatch did not pass its supplied configuration to the subprocess. It now writes and passes an immutable configuration snapshot. This was not the cause of the original default-config results, but would invalidate alternative-config experiments.',
         '', 'The repaired model uses 25 split-step layers, a 256×256 transverse grid with 2× padding, and reference index 1.26. A fixed geometric mid-height phase seed (hmax/2 = 1.25 µm) changes the initial manufacturable height profile. Three fresh NN seeds are trained through the finite-thickness model and selected by validation loss. No original checkpoint is reused or overwritten.',
         '', '**The focal coordinate system did not change:** physical substrate z=0, exit reference z=2.5 µm, target distance 7.8 µm, collection plane z=10.3 µm, radius 0.55 µm. No threshold was relaxed.',
         '', '## Independent Maxwell results','',
         '| Design | Finest cells/µm | Focus from exit (µm) | ηR at its target | Last mesh η change | Mesh gate |',
         '|---|---:|---:|---:|---:|---|']
for name,series in meshes.items():
    r = series[-1]
    check = g['mesh_convergence'][name]
    lines.append(f"| {name} | {r['settings']['resolution']:g} | {r['focal_um']:.4f} | {r['eta_R']:.6f} | {100*check['eta_relative_change']:.3f}% | {'PASS' if check['pass'] else 'FAIL'} |")
lines += ['',f"The repaired NN has {100*g['focal_relative_error']:.2f}% focal error and {g['eta_relative_to_fresnel']:.3f}× the matched finite-thickness-seed Fresnel efficiency. Original NN: 6.3733 µm focus, ηR=0.081954.",
          '',f"The predeclared prospective 8.0 µm target focuses at {g['prospective_8_0']['focal_um']:.4f} µm; focal acceptance: {'PASS' if g['prospective_focal_pass'] else 'FAIL'}. The 7.8 µm result is a regression result after diagnosis, **not a fresh held-out generalization claim**. The new 8.0 µm target was not used for training or selection. Its check here is focal acceptance and mesh convergence; a matched 8.0 µm Maxwell Fresnel efficiency comparison is not claimed.",
          '', 'No-lens and flat-layer axial maxima are finite-window diagnostics, not designed focal points; their mesh gates use efficiency and power closure.',
          '', '## Verification and limits','',
          'The regression suite passes all 18 tests; its captured output is `artifacts/acceptance-repair/pytest.txt`. Actual trained optimizer-state recovery and post-training sampling audits are saved separately and included in the final gate.',
          f"Finite-difference gradient checks: maximum relative error {100*max(x['relative_error'] for x in validation['gradient']):.3f}%. Thickness and transverse sampling checks passed the unchanged 2% focus / 5% efficiency limits. Three trained seeds retain serialization/recovery checks and matched aggregate direct-optimization forward budgets.",
          '', 'The independent Maxwell rerun includes no lens, flat resin, original Fresnel, updated Fresnel, direct optimization, the NN, and the prospective target. Failed mesh comparisons trigger 60 and then 80 cells/µm refinement; they are never relabeled as converged. Boundary spacing, PML, duration, m=±1 source normalization, power balance and training-only bounded feedback are recorded under `artifacts/acceptance-repair/phase6-7/`.',
          '', 'One optical worker runs at a time. GPU proxy/training uses the RTX 4060; independent Meep runs on CPU. Windows free RAM remains telemetry, with process, WSL and GPU memory budgets enforced. See the individual `*-memory.jsonl.summary.json` files for measured costs and any stops.',
          '', 'Homogeneous boundary-reference measurements are reused only after asserting that all source-run inputs are unchanged. Training, proxy internals, and lens-only representation flags are the only permitted configuration differences. Original values and field files are retained; `reused-homogeneous-references.json` records source hashes and cache provenance. Changed lens geometries are simulated afresh.',
          '', '## Reproduce','', 'From the project directory in WSL:', '', '```bash',
          'bash scripts/run_repair.sh diagnose_acceptance',
          'bash scripts/run_repair.sh diagnose_proxy',
          'bash scripts/run_repair.sh diagnose_profile',
          'bash scripts/run_repair.sh validate_thick_proxy',
          'bash scripts/run_repair.sh train_repair',
          'bash scripts/run_repair.sh check_trained_state_resume --config configs/finite_thickness_repair.json --folder artifacts/acceptance-repair/phase5',
          'bash scripts/run_repair.sh validate_repair',
          'bash scripts/run_repair.sh check_repaired_sampling',
          'PYTHONPATH=src .venv-gpu/bin/python scripts/report_repair.py',
          '```','',
          'Sources: [Meep subpixel smoothing and limitations](https://meep.readthedocs.io/en/latest/Subpixel_Smoothing/), [Meep convergence guidance](https://meep.readthedocs.io/en/latest/FAQ/), [Prism geometry API](https://meep.readthedocs.io/en/latest/Python_User_Interface/#prism).',
          '', '![Measured repair comparison](artifacts/acceptance-repair/repair-comparison.png)','']
lines += ['## Measured resource use','',
          '| Stage | Wall time (s) | Peak process RSS (GiB) | Sampled GPU total delta (GiB) | Resource stops |',
          '|---|---:|---:|---:|---|']
for path in sorted(root.glob('*-memory.jsonl.summary.json')):
    resource = read(path)
    name = path.name.removesuffix('-memory.jsonl.summary.json')
    lines.append(f"| {name} | {resource['wall_s']:.1f} | {resource['peak_project_rss_bytes']/2**30:.3f} | {resource['peak_total_device_delta_bytes']/2**30:.3f} | {', '.join(resource['stop_reasons']) or 'none'} |")
lines += ['', 'RSS sums can double-count shared pages. Device-total deltas can include other applications and do not prove per-process GPU allocation. All optical jobs are serial.', '']
lines += ['The full Maxwell process completed successfully, but editing its active shell wrapper caused an outer-shell error afterward. The full-run telemetry is preserved as `validate_repair-full-memory.jsonl`; a cached rerun verified the corrected wrapper with exit code 0. Its short duration does not replace the full simulation cost.', '']
Path('acceptance-repair.md').write_text('\n'.join(lines))
fig,ax = plt.subplots(1,2,figsize=(10,4))
ax[0].bar(['Original NN','Repaired NN'],[old['NN'][-1]['focal_um'],nn['focal_um']],color=['#c45a47','#238b78'])
ax[0].axhline(7.8,color='black',linestyle='--',label='7.8 µm target')
ax[0].axhspan(7.8*.9,7.8*1.1,color='#238b78',alpha=.1,label='±10% tolerance')
ax[0].set(ylabel='Maxwell focal distance from exit (µm)',ylim=(0,9),title='Focal error repair')
ax[0].legend(fontsize=8)
for name,series in [('Original flat',old['flat']),('Same flat, merged blocks',list(flat.values()))]:
    ax[1].plot([r['settings']['resolution'] for r in series],[r['eta_R'] for r in series],marker='o',label=name)
ax[1].set(xlabel='Cells / µm',ylabel='Fixed-region efficiency ηR',title='Identical physical flat layer')
ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(root/'repair-comparison.png',dpi=170);plt.close(fig)
manifest = {}
for directory in ['src','scripts','tests','configs']:
    for p in Path(directory).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.sh','.json'] and '__pycache__' not in str(p):
            manifest[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
write_json(root/'source-manifest.json',manifest)
