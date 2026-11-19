"""Fresh, bounded direct-optimization comparison; no NN or Maxwell launch.

Supervisor owns one child and only that child can be stopped. No CIM,
host-memory acceptance gates, process inventory, or remote access.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,pickle,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu():
    line=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()[0]
    v=[x.strip() for x in line.split(',')]
    return dict(name=v[0],total_mib=float(v[1]),used_mib=float(v[2]),free_mib=float(v[3]),utilization_percent=float(v[4]))

def child(out):
    # The supervisor sets allocation controls before this process imports JAX.
    import copy
    import numpy as np
    sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
    from fiber_metalens_nn.runtime import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from fiber_metalens_nn.collimation_proxy import CylinderProxy
    from fiber_metalens_nn.collimation_loss_v2 import plane_widths,stability_change
    from fiber_metalens_nn.collimation_optics import kernels,describe
    began=time.monotonic();declaration=read(out/'predeclared.json');cfg1=read(out/'config_legacy.json');cfg2=read(out/'config_v2.json')
    H=32.;D=30.;versions=[('legacy',cfg1),('v2',cfg2)]
    def event(name,**kwargs):
        row=dict(utc=utc(),elapsed_s=time.monotonic()-began,event=name,**kwargs)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def ready(value):
        return jax.tree.map(lambda x:x.block_until_ready() if hasattr(x,'block_until_ready') else x,value)
    def host(value):return jax.tree.map(lambda x:np.asarray(x) if hasattr(x,'dtype') else x,value)
    def plain(value):
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        return value
    def mem():
        data=jax.devices()[0].memory_stats() or {}
        useful={k:int(v) for k,v in data.items() if isinstance(v,(int,float))}
        peak=max(useful.get('peak_bytes_in_use',0),useful.get('bytes_in_use',0))
        if peak>6*1024**3:raise RuntimeError('JAX allocator in-use peak exceeded 6 GiB')
        return useful
    devices=jax.devices();assert len(devices)>=1 and devices[0].platform=='gpu',str(devices)
    a=jnp.arange(1024,dtype=jnp.float32);smoke=float(jnp.sum(a*a).block_until_ready())
    event('GPU_SMOKE_COMPLETE',devices=[str(d) for d in devices],sum_squares=smoke,JAX_version=jax.__version__,allocator=mem())
    measurement= CylinderProxy(cfg2)
    start=time.monotonic();pedestal=ready(measurement.pedestal(H));target=ready(measurement.target(D))
    pedestal_s=time.monotonic()-start
    initial_heights=np.asarray(measurement.seed(H),np.float32)
    fraction=np.clip(initial_heights/cfg2['height_max_um'],.001,.999)
    raw0=jnp.asarray(np.log(fraction/(1-fraction)),jnp.float32)
    initial_heights=np.asarray(cfg2['height_max_um']*jax.nn.sigmoid(raw0))
    np.savez_compressed(out/'shared_initialization.npz',raw_logits=np.asarray(raw0),heights_um=initial_heights,pedestal=np.asarray(pedestal))
    _,_,kz,_,valid,_=kernels(measurement.count,measurement.dx,1.55,xp=jnp)
    x=jnp.asarray(measurement.x,jnp.float32)
    def measure_heights(heights):
        spectrum=measurement.spectrum(heights,pedestal)
        def inner(a,b):
            return .25*jnp.sum(a[0]*jnp.conj(b[4])-a[1]*jnp.conj(b[3])+jnp.conj(b[0])*a[4]-jnp.conj(b[1])*a[3])*measurement.dx**2/measurement.count**2
        p=jnp.real(inner(spectrum,spectrum));pt=jnp.real(inner(target,target))
        useful=jnp.abs(inner(spectrum,target))**2/(pt*measurement.incident)
        widths=[];validity=[];moments=[];radial=[]
        for z in cfg2['evaluation']['processing_planes_um']:
            field=jnp.fft.ifft2(spectrum*(jnp.exp(1j*kz*z)*valid)[None],axes=(-2,-1))
            w,v,m=plane_widths(field,x);widths.append(w);validity.append(v);moments.append(m)
            sz=jnp.maximum(.5*jnp.real(field[0]*jnp.conj(field[4])-field[1]*jnp.conj(field[3])),0)
            radial.append(2*jnp.sqrt(2*jnp.sum(sz*measurement.r**2)/jnp.maximum(sz.sum(),1e-20)))
        widths=jnp.stack(widths);validity=jnp.stack(validity);moments=jnp.stack(moments);radial=jnp.stack(radial)
        change=stability_change(widths);rough=jnp.mean(jnp.diff(heights)**2);l=cfg2['loss']
        legacy=-l['mode']*useful+l['diameter']*(radial[0]/D-1)**2+l['stability']*(radial[-1]/radial[0]-1)**2+l['roughness']*rough
        revised=-l['mode']*useful+l['diameter']*jnp.mean((widths[0]/D-1)**2)+l['stability']*change**2+l['roughness']*rough+l['invalid_profile']*jnp.mean((~validity).astype(jnp.float32))
        return dict(transmission=p/measurement.incident,overlap=useful/(p/measurement.incident),eta_mode=useful,exp2_widths_um=widths,D4sigma_widths_um=moments,valid_crossings=validity,maximum_diameter_change_100um=change,legacy_objective=legacy,v2_objective=revised,roughness_um2=rough)
    measure=jax.jit(measure_heights)
    start=time.monotonic();initial=plain(host(ready(measure(jnp.asarray(initial_heights)))))
    measure_compile_s=time.monotonic()-start
    def quantize(heights):return np.clip(np.round(np.asarray(heights)/.05)*.05,0,3).astype(np.float32)
    def constraints(heights):
        h=np.asarray(heights);return dict(finite=bool(np.isfinite(h).all()),bounded=bool(np.all((h>=0)&(h<=3))),annuli=bool(len(h)==24),min_width_um=1.,max_relief_aspect_ratio=float(h.max()),within_declared_aspect_ratio=bool(h.max()<=8))
    def evaluate_full(heights,config,path):
        p=measurement if config['materials']['polymer']['n']==1.52 else CylinderProxy(config)
        start=time.monotonic();spectrum=np.asarray(ready(p.spectrum(jnp.asarray(heights),p.pedestal(H))))
        power=.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])).sum()*p.dx**2/p.count**2
        report=describe(spectrum,p.dx,1.55,D,config,float(power/p.incident),save=path.with_suffix('.fields.npz'))
        report.update(result_label='SYNTHETIC_PROXY_ONLY_NOT_MAXWELL',H_um=H,target_diameter_um=D,cylinder_diameter_um=48.,top_relief_max_bound_um=3.,output_reference_z_um=36.,polymer_n=config['materials']['polymer']['n'],source='Gaussian surrogate, 9 um MFD; nominal fibre NA 0.14 not independent angular cutoff',model_limitations='one-way scalar internal propagation, approximate normal interface transmission; no full-vector backward/interface coupling',quantized_heights_um=np.asarray(heights).tolist(),constraints=constraints(heights),evaluation_wall_s=time.monotonic()-start)
        write(path,report);return report
    initial_full=evaluate_full(quantize(initial_heights),cfg2,out/'initial_quantized_metrics.json')
    event('SHARED_INITIALIZATION_READY',pedestal_compile_and_forward_s=pedestal_s,measurement_compile_and_forward_s=measure_compile_s,initial_quantized_eta_mode=initial_full['eta_mode'],initial_quantized_stability=initial_full['maximum_diameter_change_100um'],allocator=mem())
    reports={};states={};profiles={};function_cache={}
    for label,config in versions:
        proxy=CylinderProxy(config)
        def objective(raw,proxy=proxy):return proxy._objective(3*jax.nn.sigmoid(raw),pedestal,target,jnp.asarray(D,jnp.float32))
        vg=jax.jit(jax.value_and_grad(objective));value=jax.jit(objective)
        start=time.monotonic();loss,g=ready(vg(raw0));compile_gradient_s=time.monotonic()-start
        start=time.monotonic();ready(vg(raw0));steady_gradient_s=time.monotonic()-start
        start=time.monotonic();base_value=float(value(raw0).block_until_ready());compile_forward_s=time.monotonic()-start
        start=time.monotonic();value(raw0).block_until_ready();steady_forward_s=time.monotonic()-start
        fd=[]
        for index in [2,5,8]:
            for epsilon in [.01,.005]:
                step=np.zeros(24,np.float32);step[index]=epsilon
                plus=float(value(raw0+jnp.asarray(step)).block_until_ready());minus=float(value(raw0-jnp.asarray(step)).block_until_ready())
                finite=(plus-minus)/(2*epsilon);ad=float(g[index]);error=abs(finite-ad);tol=.0003+.1*max(abs(finite),abs(ad))
                fd.append(dict(raw_index=index,epsilon=epsilon,finite_difference=finite,autodiff=ad,absolute_error=error,tolerance=tol,passed=error<=tol))
        consistent=abs(base_value-float(initial[label+'_objective']))<2e-5
        report=dict(compile_and_gradient_s=compile_gradient_s,steady_gradient_s=steady_gradient_s,compile_and_forward_s=compile_forward_s,steady_forward_s=steady_forward_s,finite_differences=fd,common_metrics_objective_agreement=consistent,initial_loss=base_value)
        write(out/(label+'_preflight.json'),report)
        if not all(row['passed'] for row in fd) or not consistent or not np.isfinite(np.asarray(g)).all():raise RuntimeError(label+' gradient/objective preflight failed')
        event('GRADIENT_PREFLIGHT_PASS',label=label,**report,allocator=mem())
        opt=optax.adam(.025)
        @jax.jit
        def apply(raw,state,gradient):
            changes,state=opt.update(gradient,state,raw)
            return optax.apply_updates(raw,changes),state
        raw=raw0.copy();state=opt.init(raw);function_cache[label]=(vg,value,apply,config)
        arm=out/label;arm.mkdir();start_arm=time.monotonic()
        with (arm/'history.jsonl').open('w') as history:
            row=dict(step=0,elapsed_s=0.,heights_um=initial_heights.tolist(),metrics=initial)
            history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
            for step in range(1,51):
                before=time.monotonic();loss,gradient=ready(vg(raw));grad_s=time.monotonic()-before
                if not np.isfinite(float(loss)) or not np.isfinite(np.asarray(gradient)).all():raise RuntimeError(label+' nonfinite loss or gradient')
                raw,state=ready(apply(raw,state,gradient));heights=ready(3*jax.nn.sigmoid(raw))
                before=time.monotonic();metrics=plain(host(ready(measure(heights))));measure_s=time.monotonic()-before
                row=dict(step=step,elapsed_s=time.monotonic()-start_arm,objective_before_update=float(loss),gradient_l2=float(jnp.linalg.norm(gradient)),gradient_wall_s=grad_s,metric_wall_s=measure_s,heights_um=np.asarray(heights).tolist(),metrics=metrics,allocator=mem())
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush()
                if step%10==0:
                    data=dict(source_manifest=declaration['source_sha256'],label=label,step=step,raw=host(raw),state=host(state))
                    with (arm/f'checkpoint_step{step:03d}.pkl').open('wb') as f:pickle.dump(data,f)
                    if step==10:
                        restored=pickle.loads(pickle.dumps(data));rg=ready(vg(raw))[1];restored_g=ready(vg(jnp.asarray(restored['raw'])))[1]
                        expected=ready(apply(raw,state,rg));actual=ready(apply(jnp.asarray(restored['raw']),jax.tree.map(jnp.asarray,restored['state']),restored_g))
                        error=max(float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)))
                        write(arm/'checkpoint_recovery.json',dict(trained_step=10,next_update_max_absolute_error=error,passed=error<1e-7,extra_gradient_calls=2))
                        if error>=1e-7:raise RuntimeError(label+' trained checkpoint recovery failed')
                    event('OPTIMIZATION_PROGRESS',label=label,step=step,loss=float(loss),eta_mode=metrics['eta_mode'],widths=metrics['exp2_widths_um'],stability=metrics['maximum_diameter_change_100um'])
        final=quantize(np.asarray(3*jax.nn.sigmoid(raw)))
        report=evaluate_full(final,config,arm/'matched_50_quantized_metrics.json')
        write(arm/'matched_50_design.json',dict(kind='direct_optimized_'+label,H_um=H,target_diameter_um=D,heights_um=final.tolist(),polymer_n=1.52,architecture=config['architecture'],cylinder_diameter_um=48,matched_updates=50))
        report_summary=dict(updates=50,wall_s=time.monotonic()-start_arm,gradient_calls_for_updates=50,checkpoint_recovery_extra_gradient_calls=2,common_metric_calls_during_updates=50,quantized_metrics=report)
        write(arm/'matched_50_result.json',report_summary);reports[label]=report_summary;states[label]=(raw,state);profiles[label]=read(out/(label+'_preflight.json'))
    # Common initialization and equal 50-update budgets above are the controlled comparison.
    initial_eta=initial_full['eta_mode'];initial_stability=initial_full['maximum_diameter_change_100um'];v2=reports['v2']['quantized_metrics']
    remaining=600-(time.monotonic()-began)
    progress=v2['eta_mode']>=initial_eta*1.01 and v2['maximum_diameter_change_100um']<=initial_stability*.95
    optional=dict(executed=False,useful_progress=progress,remaining_seconds=remaining,criteria='quantized eta improves >=1% relative AND five-plane stability error reduces >=5% relative; all geometry constraints pass; >=120 s left in same 600 s run; allocator peak <=5 GiB',reason='progress or resource criterion not met')
    if (progress and remaining>=120 and all(v2['constraints'][k] for k in ['finite','bounded','annuli','within_declared_aspect_ratio']) and mem().get('peak_bytes_in_use',0)<=5*1024**3):
        raw,state=states['v2'];vg,value,apply,config=function_cache['v2'];arm=out/'v2';start_extra=time.monotonic()
        with (arm/'history_extension.jsonl').open('w') as history:
            for step in range(51,101):
                loss,gradient=ready(vg(raw));raw,state=ready(apply(raw,state,gradient));heights=ready(3*jax.nn.sigmoid(raw));metrics=plain(host(ready(measure(heights))))
                if not np.isfinite(float(loss)) or not np.isfinite(np.asarray(gradient)).all():raise RuntimeError('v2 extension nonfinite loss/gradient')
                history.write(json.dumps(dict(step=step,elapsed_total_s=time.monotonic()-began,objective_before_update=float(loss),heights_um=np.asarray(heights).tolist(),metrics=metrics,allocator=mem()),allow_nan=False)+'\n');history.flush()
                if step%10==0:
                    with (arm/f'checkpoint_step{step:03d}.pkl').open('wb') as f:pickle.dump(dict(source_manifest=declaration['source_sha256'],label='v2_extended',step=step,raw=host(raw),state=host(state)),f)
                    event('OPTIONAL_EXTENSION_PROGRESS',step=step,eta_mode=metrics['eta_mode'],stability=metrics['maximum_diameter_change_100um'])
        final=quantize(np.asarray(3*jax.nn.sigmoid(raw)));extra=evaluate_full(final,config,arm/'extended_100_quantized_metrics.json')
        write(arm/'extended_100_design.json',dict(kind='direct_optimized_v2_extended',H_um=H,target_diameter_um=D,heights_um=final.tolist(),polymer_n=1.52,architecture=config['architecture'],cylinder_diameter_um=48,updates=100))
        optional.update(executed=True,reason='predeclared progress and resource criteria passed',extra_updates=50,extra_wall_s=time.monotonic()-start_extra,quantized_metrics=extra)
    write(out/'optional_extension_decision.json',optional)
    # Separately labeled n=1.50 forward-only check; no optimizer reuse or training.
    sensitivity=dict(executed=False,reason='insufficient remaining time')
    candidates=[('legacy_matched_50',reports['legacy']['quantized_metrics']),('v2_matched_50',reports['v2']['quantized_metrics'])]
    if optional['executed']:candidates.append(('v2_extended_100',optional['quantized_metrics']))
    best_name,best=max(candidates,key=lambda item:item[1]['eta_mode'])
    if 600-(time.monotonic()-began)>60:
        folder=out/'sensitivity_n1p50';folder.mkdir();c=copy.deepcopy(cfg2);c['materials']['polymer']['n']=1.50;c['result_label']='SYNTHETIC_PROVISIONAL_N1P50_FORWARD_ONLY_NOT_OPTIMIZED';write(folder/'config.json',c)
        sensitivity=dict(executed=True,initial=evaluate_full(quantize(initial_heights),c,folder/'initial_quantized_metrics.json'),selected_geometry=evaluate_full(np.array(best['quantized_heights_um'],np.float32),c,folder/'selected_geometry_metrics.json'),selected_from_n1p52=best_name,optimization_updates=0)
    # Sample/grid checks are fixed-shape forward tests, not new design optimization.
    sampling=[]
    if 600-(time.monotonic()-began)>90:
        for name,kwargs in [('pedestal_dz_0p25',dict(dz=.25)),('relief_slices_48',dict(slices=48)),('transverse_grid_256',dict(grid_n=256))]:
            if 600-(time.monotonic()-began)<45:break
            p=CylinderProxy(cfg2,**kwargs);s=p.evaluate(best['quantized_heights_um'],H,D)
            sampling.append(dict(setting=name,eta_mode=s['eta_mode'],diameters=[s['planes'][0]['diameter_x_um'],s['planes'][0]['diameter_y_um']],stability=s['maximum_diameter_change_100um'],eta_relative_change=s['eta_mode']/best['eta_mode']-1,maximum_output_diameter_relative_change=max(abs(s['planes'][0]['diameter_x_um']/best['planes'][0]['diameter_x_um']-1),abs(s['planes'][0]['diameter_y_um']/best['planes'][0]['diameter_y_um']-1))))
            p.pedestal_cache.clear();p.target_cache.clear();event('SAMPLING_CHECK',**sampling[-1])
    result=dict(utc=utc(),status='COMPLETED_SYNTHETIC_PROXY_COMPARISON',GPU_devices=[str(d) for d in devices],H_um=H,target_diameter_um=D,polymer_n=1.52,initial_quantized_metrics=initial_full,matched_50=reports,optional_v2_extension=optional,n1p50_sensitivity=sensitivity,sampling_checks=sampling,best_proxy_geometry=best_name,best_proxy_metrics=best,profiles=profiles,wall_s=time.monotonic()-began,allocator=mem(),heldout_H35_used=False,NN_training_launched=False,Maxwell_launched=False,optical_acceptance='Not granted: proxy estimates only, no new Maxwell validation',source_manifest_unchanged=all(sha(Path(p))==h for p,h in declaration['source_sha256'].items()))
    write(out/'result.json',result);event('COMPARISON_COMPLETE',wall_s=result['wall_s'],best_proxy_geometry=best_name,best_eta_mode=best['eta_mode'],allocator=result['allocator'])

def supervise(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);before=gpu()
    if 'RTX 4060' not in before['name']:raise RuntimeError('Intended RTX4060 hardware mismatch')
    if before['free_mib']<3072:raise RuntimeError('Less than 3072 MiB free GPU memory for this declared workload; no run started')
    import copy
    config=read(ROOT/'configs/1550nm_collimation_loss_v2.json');cfg2=copy.deepcopy(config);cfg1=copy.deepcopy(config);cfg1['loss']['version']='legacy_endpoints_v1'
    for c in [cfg1,cfg2]:c['result_label']='SYNTHETIC_DIRECT_LOSS_COMPARISON_H32_D30_N1P52'
    assert [32,30] in config['training']['train_pairs'] and [35,30] in config['training']['heldout_pairs']
    assert {k:v for k,v in cfg1.items() if k!='loss'}=={k:v for k,v in cfg2.items() if k!='loss'}
    sources=[Path(__file__),ROOT/'configs/1550nm_collimation_loss_v2.json']+[ROOT/'src/fiber_metalens_nn'/n for n in ['collimation_proxy.py','collimation_loss_v2.py','collimation_optics.py','runtime.py']]
    protected=[ROOT/'artifacts/1550nm_collimation/phase9/gate.json',ROOT/'artifacts/1550nm_collimation/phase9/seed-11.pkl',ROOT/'artifacts/1550nm_collimation/maxwell/geometry-ledger.json',ROOT/'configs/1550nm_collimation.json',ROOT/'configs/1550nm_collimation_loss_v2.json']
    protected_hashes={str(p):sha(p) for p in protected}
    write(out/'config_legacy.json',cfg1);write(out/'config_v2.json',cfg2)
    declaration=dict(utc=utc(),H_um=32,target_diameter_um=30,updates_each=50,optional_v2_extra_updates=50,timeout_s=600,workers=1,batch=1,VRAM_cap_GiB=6,allocator_fraction=.60,precision='float32/complex64',RAM_acceptance_gates=False,GPU_before=before,source_sha256={str(p):sha(p) for p in sources},protected_sha256=protected_hashes,optional_criteria='quantized eta >=1.01*initial AND stability <=.95*initial AND >=120 s remaining AND peak allocator <=5 GiB; geometry valid',Maxwell_unique_geometry_budget_used=len(read(ROOT/'artifacts/1550nm_collimation/maxwell/geometry-ledger.json')),training_pair=[32,30],heldout_pair_not_used=[35,30],checkpoint_interval=10,metric_planes_um=[0,25,50,75,100],normalization='proxy forward launched-Gaussian E/H Poynting; no Meep field extraction is performed',same_initialization='same bounded curvature-cancelling relief logits; no perturbation',gradient_tolerance='abs error <=0.0003+0.1*max(abs(AD),abs(FD)), raw logits epsilon .01/.005, rings2/5/8',constraint_policy='fixed cylinder48um;24 rings1um;heights0..3um;final .05um quantization; max relief aspect<=8',native_flux_reconstruction='Any future Maxwell analysis requires verified half-cell radial coordinates, paired centered quadrature and tested output-plane continuation; legacy defaults are not promoted by this experiment')
    write(out/'predeclared.json',declaration)
    environment=os.environ.copy();environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'),XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.60',JAX_PLATFORMS='cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');environment.pop('LD_LIBRARY_PATH',None)
    started=time.monotonic();violations=[];peak_delta=0.;min_free=before['free_mib'];samples=0
    with (out/'child.log').open('w') as log,(out/'GPU_telemetry.jsonl').open('w') as telemetry:
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child','--output',str(out)],cwd=out,env=environment,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        write(out/'execution_started.json',dict(utc=utc(),owned_child_pid=p.pid,command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--output',str(out)],GPU_before=before))
        print(json.dumps(dict(event='ACTUAL_COMPARISON_STARTED',utc=utc(),owned_child_pid=p.pid,output=str(out),GPU_free_mib=before['free_mib'])),flush=True)
        while True:
            elapsed=time.monotonic()-started
            try:
                g=gpu();delta=max(g['used_mib']-before['used_mib'],0);peak_delta=max(peak_delta,delta);min_free=min(min_free,g['free_mib']);samples+=1
                row=dict(utc=utc(),elapsed_s=elapsed,total_device_delta_mib=delta,**g)
                telemetry.write(json.dumps(row)+'\n');telemetry.flush()
                if delta>6144:violations.append('total-device GPU allocation delta exceeded 6 GiB')
                if g['free_mib']<384:violations.append('GPU free memory below declared 384 MiB safety reserve')
            except Exception as error:violations.append('GPU telemetry unavailable: '+str(error))
            if elapsed>=600:violations.append('600-second owned-run timeout')
            if p.poll() is not None:break
            if violations:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(.5)
    result=dict(utc=utc(),owned_child_exit_code=p.returncode,wall_s=time.monotonic()-started,peak_total_device_delta_mib=peak_delta,minimum_sampled_free_GPU_mib=min_free,GPU_before=before,GPU_after=gpu(),GPU_samples=samples,violations=violations,protected_unchanged=all(sha(Path(p))==h for p,h in protected_hashes.items()),RAM_acceptance_gates=False,telemetry_limitations='Sampled total-device deltas can include other applications and can miss transients; JAX allocator peaks are recorded in child. No CIM or unrelated-process control.')
    write(out/'execution_completed.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if p.returncode==0 and not violations and result['protected_unchanged'] else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:
        try:child(a.output)
        except BaseException as error:
            write(a.output/'child_failure.json',dict(utc=utc(),type=type(error).__name__,reason=str(error)))
            raise
        return 0
    return supervise(a.output)

if __name__=='__main__':sys.exit(main())
