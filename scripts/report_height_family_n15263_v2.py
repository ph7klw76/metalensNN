"""Package provisional height-family results without changing physics/designs."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2'
def read(p):return json.loads(Path(p).read_text())
def main():
    cfg=read(FOLDER/'config.json');result=read(FOLDER/'training_result.json')
    analytic=read(FOLDER/'analytic_feasibility.json')
    rows=[r for r in result['records'] if r['method']=='NN' and r['height_domain']=='training']
    H=np.array([r['H_um'] for r in rows]);diameter=np.array([r['metrics']['widths_um'][0] for r in rows])
    theta=np.array([[f['half_angle_rad']*180/math.pi for f in r['fitted_core_divergence']['axes']] for r in rows])
    target_theta=np.array([[f['half_angle_rad']*180/math.pi for f in r['target_fitted_core_divergence']['axes']] for r in rows])
    eta=np.array([r['metrics']['declared_useful_efficiency'] for r in rows]);C=np.array([r['metrics']['normalized_curvature'] for r in rows])
    radii=(np.arange(cfg['annuli'])+.5)*cfg['cylinder_diameter_um']/2/cfg['annuli']
    profiles=np.array([read(FOLDER/f'designs/H{h:g}/NN.json')['heights_um'] for h in H])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for h,profile in zip(H,profiles):axes[0,0].step(radii,profile,where='mid',label=f'{h:g} µm')
    axes[0,0].set(xlabel='Radius (µm)',ylabel='Top relief height (µm)',title='A  Different quantized profile for each cylinder height')
    axes[0,0].legend(ncol=2,fontsize=8,title='Cylinder height H',frameon=False)
    axes[0,1].plot(H,diameter[:,0],'o-',label='x width');axes[0,1].plot(H,diameter[:,1],'s--',label='y width')
    axes[0,1].plot(H,[a['Gaussian_top_diameter_um'] for a in analytic['rows']],':',color='black',label='Homogeneous Gaussian top reference')
    axes[0,1].set(xlabel='Cylinder height H (µm)',ylabel='Output 1/e² intensity diameter (µm)',title='B  Output beam size');axes[0,1].legend(fontsize=8,frameon=False)
    axes[1,0].plot(H,theta[:,0],'o-',label='Fitted core x');axes[1,0].plot(H,theta[:,1],'s--',label='Fitted core y')
    axes[1,0].plot(H,target_theta.mean(axis=1),':',color='black',label='Declared Gaussian target core')
    axes[1,0].set(xlabel='Cylinder height H (µm)',ylabel='Fitted core half-angle (degrees)',title='C  Core divergence and declared target');axes[1,0].legend(fontsize=8,frameon=False)
    axes[1,1].plot(H,100*eta,'o-',color='#206d53',label='Useful modal power')
    axes[1,1].set(xlabel='Cylinder height H (µm)',ylabel='Incident power in target mode (%)',title='D  Useful efficiency in the proxy')
    axes[1,1].set_ylim(90,97);axes[1,1].grid(alpha=.2)
    for ax in axes.ravel():ax.grid(alpha=.18)
    fig.suptitle('Height-specific inverse design  |  1550 nm  |  polymer n = 1.5263\nPROVISIONAL SCALAR-PROXY RESULTS  —  independent Maxwell validation pending',fontsize=13)
    fig.savefig(FOLDER/'height_family_proxy_results.png',dpi=190);plt.close(fig)
    with (FOLDER/'NN_height_profiles_um.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['ring_index_zero_based','inner_radius_um','outer_radius_um','center_radius_um']+[f'H{h:g}_relief_um' for h in H])
        pitch=cfg['cylinder_diameter_um']/2/cfg['annuli']
        for j,r in enumerate(radii):writer.writerow([j,j*pitch,(j+1)*pitch,r]+profiles[:,j].tolist())
    table=[]
    for row,d,t,e,c in zip(rows,diameter,theta,eta,C):
        table.append({'H_um':row['H_um'],'output_Dx_um':float(d[0]),'output_Dy_um':float(d[1]),'fitted_core_x_half_angle_deg':float(t[0]),'fitted_core_y_half_angle_deg':float(t[1]),'useful_efficiency':float(e),'normalized_curvature':float(c),'declared_overlap':row['metrics']['declared_overlap'],'retained_angular_ratio':row['metrics']['angular_second_moment_ratio'],'profile_sha256':row['profile_sha256'],'status':'PROXY_ONLY_NOT_MAXWELL_VALIDATED'})
    with (FOLDER/'NN_height_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    sampling=read(FOLDER/'proxy_sampling.json') if (FOLDER/'proxy_sampling.json').exists() else None
    protected=read(FOLDER/'provenance.json')['protected_before']
    unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in protected.items())
    tolerance_failures=[]
    if sampling:
        for diagnostic in sampling['MFD_tolerance_diagnostics']:
            if abs(diagnostic['metrics']['normalized_curvature'])>.1:
                tolerance_failures.append({'MFD_um':diagnostic['MFD_um'],'H_um':diagnostic['H_um'],'normalized_curvature':diagnostic['metrics']['normalized_curvature'],'failed':'abs(C)<=0.1 nominal investigation criterion'})
    gate={'status':'PROXY_FAMILY_COMPLETE_INDEPENDENT_VALIDATION_PENDING','source_MFD_um':10.4,'polymer_n':1.5263,'input_phase_n_assumed':1.45,'different_profile_per_H':True,'requested_H_um':H.tolist(),'unique_quantized_requested_profiles':len(set(r['profile_sha256'] for r in rows)),'heldout_selection_order_preserved':result['heldout_created_after_selection'],'matched_calls':result['matched_training_validation_recovery_calls'],'all_trained_recovery_checks_pass':all(r['recovery']['passed'] for r in result['reports']),'proxy_sampling_status':sampling['status'] if sampling else 'PENDING','MFD_tolerance_failures':tolerance_failures,'full_MFD_tolerance_robustness_demonstrated':False,'protected_evidence_unchanged':unchanged,'independent_Maxwell_runs':0,'Maxwell_allocation_status':'AWAITING_EXPLICIT_USER_AMENDMENT','manufacturing_file_exported':False,'global_diffraction_optimum_claimed':False,'full_M2_claimed':False,'rows':table}
    (FOLDER/'proxy_family_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    lines=['HEIGHT SPECIFIC METALENS INVERSE DESIGN','',gate['status'],'',
        'Goal: a different optimized top relief for each H = 40, 50, 60, 70, 80, 90 and 100 um.',
        'Input: Corning SMF-28 Ultra specification PI-1424, July 2025, gives MFD = 10.4 +/- 0.5 um at 1550 nm.',
        'The user core label of 8 um is not the mode-field diameter. NA = 0.14 is a 1%-power far-field descriptor at 1310 nm.',
        'A Gaussian source and phase index 1.45 are assumed; no measured/eigenmode field or radial fiber index profile is available.',
        'Polymer real index 1.5263 is supplied by the user; air index is 1; material losses are assumed zero.',
        'Cylinder diameter: 50 um. Top relief: 50 radial annuli, each 0.5 um wide; heights 0-3 um in 0.05 um steps.',
        'Facet/base: z = 0. Relief base: z = H. Maximum assembly height: H + 3 um. Output reference: H + 4 um.',
        '', 'PROVISIONAL SCALAR-PROXY OUTPUTS',
        'H(um)  Dx(um)  Dy(um)  core half-angle x/y(deg)  useful power(%)  C']
    for r in table:lines.append(f"{r['H_um']:5g}  {r['output_Dx_um']:7.3f} {r['output_Dy_um']:7.3f}   {r['fitted_core_x_half_angle_deg']:.4f}/{r['fitted_core_y_half_angle_deg']:.4f}             {100*r['useful_efficiency']:.4f}      {r['normalized_curvature']:+.6f}")
    lines+=['','TRAINING AND CHECKS',f"Three NN and three matched direct-table seeds, 100 updates each; 162 objective calls per arm; 972 matched calls total.",f"Selected NN seed {result['selection']['NN']['seed']}; direct seed {result['selection']['direct']['seed']}; validation-only selection before withheld-target construction.",'Training H: 40/50/60/70/80/90/100 um; validation: 45/55/65/75/85/95 um; withheld: 42.5/57.5/72.5/87.5/97.5 um.',
        'The NN is conditioned on H and learns an H-specific residual to a field-derived phase seed; it does not learn one shared physical mask.',
        'No NN advantage over the matched direct optimizer or global optimality is claimed.',
        'Nine focused loss tests pass; three sampled finite-difference checks pass; all six trained-state recovery checks pass.',
        f"Protected historical sources/exports/ledger unchanged: {unchanged}.",
        'Final-profile sampling: '+(sampling['status'] if sampling else 'PENDING'),
        'Independent Maxwell adapter: 14 focused setup/analytic-field tests pass; no real Meep run yet.',
        'Nominal-family sampling covers grid 256, 48 relief slices and 0.25 um pedestal steps at all 7 heights (21 checks).',
        'The sampling report initially hit a JAX-to-JSON formatting error, corrected and verified on identical frozen designs; original failure preserved.',
        '', 'INTERPRETATION',
        'These are inverse-designed candidates evaluated by a finite-thickness one-way scalar proxy with vector output.',
        'Independent full-Maxwell physics/numerical acceptance remains pending. Core agreement is not full all-angle M2.',
        'The Gaussian diffraction reference is matched to the declared footprint; it is not a global 50 um aperture bound.',
        'Small core angles below the declared-target curve reflect changed beam shape/size; they are not evidence of beating diffraction.',
        'Finite beams still diffract downstream; the old 5% absolute-growth criterion is not reused.',
        'The drawing beam diameters 17.5/19.4/28.7 um were not used as measured input evidence.',
        'Corning group index 1.4682 is not used as a propagation phase index.',
        'Full MFD-tolerance robustness is not established. The unchanged H = 100 um design at MFD = 9.9 um gives C = 0.12436, exceeding the nominal |C| <= 0.1 criterion.',
        '', 'NEXT INDEPENDENT STAGE',
        'The historical Maxwell allocation is exhausted (13/13). See MAXWELL_VALIDATION_PROPOSAL.txt and its JSON.',
        'Proposed cap: 11 new device cases (cumulative 24), 60 local CPU solves including references/retries, 4 hours, one worker, no RAM gates.',
        'No independent solve, SLC fabrication export or printing has been launched under this new campaign.',
        '', 'FILES','designs/H40/NN.json ... designs/H100/NN.json: quantized per-height candidates.',
        'NN_height_profiles_um.csv: printable-height parameter table, not an exposure job.',
        'NN_height_summary.csv; height_family_proxy_results.png; training_result.json; selection_before_heldout.json.',
        'config.json; source_snapshots/; loss_tests.xml; pilot_gradient.json; trained checkpoints and GPU logs.',
        '', 'PRIMARY REFERENCES',cfg['input_mode_provenance']['source_url'],
        'https://www.edmundoptics.com/knowledge-center/application-notes/lasers/gaussian-beam-propagation/']
    (FOLDER/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':gate['status'],'sampling':gate['proxy_sampling_status'],'requested_profiles':len(table),'protected_unchanged':unchanged},indent=2))

if __name__=='__main__':main()
