"""Independent CPU accounting/transcription of the saved direct comparison.

This re-evaluates the approximate proxy, not Maxwell or physical acceptance.
No GPU imports or optimization are performed here.
"""
import csv,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

BASE=Path(__file__).parent
ROOT=Path(__file__).resolve().parents[3]
RUN=BASE/'direct_loss_comparison_01'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.perf_counter();sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(BASE))
    from audit_saved_fields import scalar_proxy,scalar_summary
    from fiber_metalens_nn.collimation_optics import vector_spectrum,kernels,propagate,describe
    from fiber_metalens_nn.collimation_metrics import attach_divergence_fit
    result=read(RUN/'result.json');execution=read(RUN/'execution_completed.json');declaration=read(RUN/'predeclared.json');cfg=read(RUN/'config_v2.json')
    output=RUN/'assessment'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    histories={label:[json.loads(line) for line in (RUN/label/'history.jsonl').read_text().splitlines()] for label in ['legacy','v2']}
    checks={
      'owned_child_exit_zero':execution['owned_child_exit_code']==0,
      'same_50_update_budgets':all(v['updates']==50 and v['gradient_calls_for_updates']==50 for v in result['matched_50'].values()),
      'all_51_history_states_saved':all([v['step'] for v in h]==list(range(51)) for h in histories.values()),
      'identical_initial_heights':histories['legacy'][0]['heights_um']==histories['v2'][0]['heights_um'],
      'finite_bounded_heights_all_steps':all(np.isfinite(v['heights_um']).all() and np.all((np.array(v['heights_um'])>=0)&(np.array(v['heights_um'])<=3)) for h in histories.values() for v in h),
      'all_five_plane_metrics_saved':all(np.asarray(v['metrics']['exp2_widths_um']).shape==(5,2) and np.asarray(v['metrics']['D4sigma_widths_um']).shape==(5,2) and np.asarray(v['metrics']['valid_crossings']).all() for h in histories.values() for v in h),
      'all_metrics_finite':all(np.isfinite(v['metrics']['exp2_widths_um']).all() and all(np.isfinite(v['metrics'][k]) for k in ['eta_mode','overlap','transmission','maximum_diameter_change_100um','legacy_objective','v2_objective']) for h in histories.values() for v in h),
      'finite_difference_checks_passed':all(v['passed'] for label in ['legacy','v2'] for v in read(RUN/(label+'_preflight.json'))['finite_differences']),
      'trained_checkpoint_recovery_passed':all(read(RUN/label/'checkpoint_recovery.json')['passed'] for label in ['legacy','v2']),
      'optional_extension_skipped_as_declared':not result['optional_v2_extension']['executed'] and not result['optional_v2_extension']['useful_progress'],
      'resource_bounds_met':execution['wall_s']<600 and execution['peak_total_device_delta_mib']<=6144 and result['allocator']['peak_bytes_in_use']<=6*1024**3 and not execution['violations'],
      'historical_protected_files_unchanged':execution['protected_unchanged'] and all(sha(Path(p.replace('/mnt/c/','C:/')))==h for p,h in declaration['protected_sha256'].items()),
      'heldout_not_used':not result['heldout_H35_used'] and declaration['training_pair']==[32,30],
      'no_NN_or_Maxwell_launched':not result['NN_training_launched'] and not result['Maxwell_launched']}
    metrics={'initial':result['initial_quantized_metrics'],**{label+'_50':v['quantized_metrics'] for label,v in result['matched_50'].items()}}
    rows=[];cpu={};loss_terms={}
    for label,metric in metrics.items():
        h=np.asarray(metric['quantized_heights_um'],np.float32)
        checks[label+'_quantized_constraints']=all(metric['constraints'][k] for k in ['finite','bounded','annuli','within_declared_aspect_ratio']) and np.max(abs(h/.05-np.round(h/.05)))<1e-5
        for plane in metric['planes'][:5]:
            rows.append(dict(case=label,z_from_output_um=plane['z_um'],diameter_x_um=plane['diameter_x_um'],diameter_y_um=plane['diameter_y_um'],D4sigma_x_um=plane['D4sigma_x_um'],D4sigma_y_um=plane['D4sigma_y_um'],power_relative_to_output=plane['power_relative_to_output'],edge_power_fraction=plane['edge_power_fraction']))
        # Independent NumPy transcription of this same one-way proxy.
        x,dx,pedestal,field,incident=scalar_proxy(cfg,32,h)
        spectrum=vector_spectrum(np.fft.fft2(field),dx,1.55)
        _,_,kz,_,valid,_=kernels(len(x),dx,1.55)
        spectrum*=np.exp(1j*kz*cfg['proxy']['output_clearance_um'])[None]
        power=float(.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])).sum()*dx**2/len(x)**2)
        recalculated=describe(spectrum,dx,1.55,30,cfg,power/incident)
        attach_divergence_fit(recalculated,spectrum,dx,1.55)
        diffs=dict(overlap=abs(recalculated['overlap']-metric['overlap']),eta_mode=abs(recalculated['eta_mode']-metric['eta_mode']),maximum_output_diameter_um=max(abs(recalculated['planes'][0][key]-metric['planes'][0][key]) for key in ['diameter_x_um','diameter_y_um']))
        checks[label+'_numpy_proxy_agreement']=diffs['overlap']<1e-4 and diffs['eta_mode']<1e-4 and diffs['maximum_output_diameter_um']<.002
        widths=np.array([[p['diameter_x_um'],p['diameter_y_um']] for p in metric['planes'][:5]])
        terms=dict(negative_useful_mode_power=-metric['eta_mode'],output_diameter_penalty=.15*float(np.mean((widths[0]/30-1)**2)),five_plane_stability_penalty=.1*metric['maximum_diameter_change_100um']**2,roughness_penalty=.002*float(np.mean(np.diff(h)**2)),invalid_profile_penalty=0.)
        loss_terms[label]=dict(weighted_terms=terms,total=sum(terms.values()))
        cpu[label]=dict(metrics=recalculated,absolute_differences_from_GPU_proxy=diffs,output_Ex_shape=scalar_summary(propagate(spectrum,dx,1.55,0)[0],x,1.55,1.,30))
    with (output/'all_five_plane_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    legacy=metrics['legacy_50'];revised=metrics['v2_50'];initial=metrics['initial']
    outcome=dict(status='EXECUTION_AND_DIAGNOSTIC_CHECKS_PASS_PROXY_OPTICAL_TARGETS_FAIL',utc=datetime.now(timezone.utc).isoformat(),checks=checks,all_checks_pass=all(checks.values()),initial_quantized=initial,legacy_50_quantized=legacy,v2_50_quantized=revised,old_vs_v2=dict(eta_absolute_change=revised['eta_mode']-legacy['eta_mode'],eta_relative_change=revised['eta_mode']/legacy['eta_mode']-1,stability_error_absolute_reduction=legacy['maximum_diameter_change_100um']-revised['maximum_diameter_change_100um'],stability_error_relative_reduction=1-revised['maximum_diameter_change_100um']/legacy['maximum_diameter_change_100um']),weighted_v2_objective_terms_for_same_quantized_geometries=loss_terms,numpy_proxy_accounting=cpu,execution=execution,child_profile=result['profiles'],sampling_checks=result['sampling_checks'],sampling_scope='Selected legacy design only: single pedestal dz refinement, relief slice refinement, transverse-grid refinement; not full window/padding convergence, not convergence of both optimized arms, not independent Maxwell.',optional_extension=result['optional_v2_extension'],n1p50_sensitivity=result['n1p50_sensitivity'],best_overall_by_useful_efficiency='initial' if initial['eta_mode']>=max(legacy['eta_mode'],revised['eta_mode']) else 'optimized',Maxwell_recommendation=dict(justified_as_next_optical_acceptance_run=False,reason='Neither optimized proxy design approaches diameter, stability, overlap or useful-efficiency targets; both lose useful power versus their shared baseline. No new successful candidate is indicated. A future Maxwell run could diagnose proxy bias, but would need a specific hypothesis and explicit cumulative candidate-budget amendment, not an optical PASS expectation.',existing_unique_candidate_ledger_count=10,new_unique_Maxwell_geometries=0,required_future_analysis='versioned actual centered radial metadata, native weighted/paired quadrature, validated LP reconstruction and explicit continuation scope; legacy defaults remain unchanged'),interpretation='The corrected loss improves the trained stability metric in this finite 50-update experiment by trading useful power. This supports an objective/optimization tradeoff; it does not establish impossibility of the architecture, NN superiority, global optimality, or physical validation.',remaining=['No new independently validated Maxwell candidate.','No measured polymer dispersion/loss or actual complex fibre mode.','Full sampling/continuation convergence for a future chosen candidate remains required.','No three-seed NN training launched.'],source_sha256={str(p):sha(p) for p in [Path(__file__),BASE/'audit_saved_fields.py',ROOT/'src/fiber_metalens_nn/collimation_metrics.py']},assessment_wall_s=time.perf_counter()-started)
    (output/'assessment.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:outcome[k] for k in ['status','all_checks_pass','checks','old_vs_v2','best_overall_by_useful_efficiency','assessment_wall_s']},indent=2))
    if not outcome['all_checks_pass']:raise RuntimeError('Post-run evidence check failed; retain report and investigate')

if __name__=='__main__':main()
