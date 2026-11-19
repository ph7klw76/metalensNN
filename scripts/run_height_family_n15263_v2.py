"""Bounded RTX4060 per-height NN/direct-table inverse design, proxy only.

Run pilot, then train with the frozen new config. No independent Maxwell
allocation, previous checkpoint or historical geometry registry is modified.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/1550nm_height_family_n15263_v2'

def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkpoint(p,value):
    temp=p.with_suffix('.tmp');temp.write_bytes(pickle.dumps(value,protocol=5));temp.replace(p)
def gpu():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    values=rows[0].split(',')
    return dict(name=values[0].strip(),total_mib=float(values[1]),used_mib=float(values[2]),free_mib=float(values[3]))

def child(folder,stage):
    import numpy as np
    sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'src'))
    from fiber_metalens_nn.runtime import configure
    configure()
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from fiber_metalens_nn.height_conditioned_loss_v2 import HeightConditionedLoss,selection_key
    from fiber_metalens_nn.collimation_metrics import fitted_divergence
    from fiber_metalens_nn.collimation_optics import propagate,flux
    cfg=read(folder/'config.json');tc=cfg['natural_training'];criteria=tc['investigation_gates'];begin=time.monotonic()
    dependencies=[Path(__file__),ROOT/'src/fiber_metalens_nn/height_conditioned_loss_v2.py',ROOT/'src/fiber_metalens_nn/collimation_proxy.py',ROOT/'src/fiber_metalens_nn/collimation_optics.py',ROOT/'src/fiber_metalens_nn/collimation_loss_v2.py',ROOT/'src/fiber_metalens_nn/natural_collimation.py',ROOT/'src/fiber_metalens_nn/natural_nn_loss_v1.py']
    signature=hashlib.sha256((folder/'config.json').read_bytes()+b''.join(p.read_bytes() for p in dependencies)).hexdigest()
    if jax.devices()[0].platform!='gpu':raise RuntimeError('Actual GPU backend required')
    def ready(x):return jax.tree.map(lambda a:a.block_until_ready() if hasattr(a,'block_until_ready') else a,x)
    def host(x):return jax.tree.map(lambda a:np.asarray(a) if hasattr(a,'dtype') else a,x)
    def plain(x):
        if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return x
    def event(name,**kw):
        row=dict(utc=utc(),elapsed_s=time.monotonic()-begin,event=name,**kw)
        print(json.dumps(row),flush=True)
        with (folder/(stage+'_events.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
    loss=HeightConditionedLoss(cfg);p=loss.p
    train_heights=tc['train_H_um'];validation=tc['validation_H_um'];heldout=tc['heldout_H_um']
    if set(train_heights)&set(validation) or (set(train_heights)|set(validation))&set(heldout):raise ValueError('Height split overlap')
    for H in train_heights+validation:loss.bundle(H)
    write(folder/(stage+'_frozen_train_validation_targets.json'),loss.metadata)
    write(folder/(stage+'_source_manifest.json'),{'config_sha256':sha(folder/'config.json'),'signature':signature,'sources':{str(q.relative_to(ROOT)):sha(q) for q in dependencies},'GPU_devices':[str(d) for d in jax.devices()]})
    class Model(nn.Module):
        @nn.compact
        def __call__(self,H):
            for _ in range(3):H=nn.tanh(nn.Dense(64)(H))
            return nn.Dense(cfg['annuli'],kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros)(H)
    model=Model();hgrid=jnp.asarray(train_heights,jnp.float32);hmax=cfg['height_max_um'];height_step=cfg['fabrication']['height_step_um']
    def phenotype(parameters,H,bundle,method):
        fraction=jnp.clip(bundle[3]/hmax,.001,.999)
        base=jnp.log(fraction/(1-fraction))
        if method=='NN':raw=model.apply(parameters,jnp.reshape((H-tc['normalization_center_um'])/tc['normalization_half_range_um'],(1,1)))[0]
        else:raw=jax.vmap(lambda row:jnp.interp(H,hgrid,row),in_axes=1)(parameters)
        return hmax*jax.nn.sigmoid(base+raw)
    functions={}
    for method in ['NN','direct']:
        def objective(parameters,H,bundle,method=method):
            metrics=loss.terms(phenotype(parameters,H,bundle,method),bundle)
            return metrics['loss'],metrics
        functions[method]={'vg':jax.jit(jax.value_and_grad(objective,has_aux=True)),
                           'predict':jax.jit(lambda params,H,b,method=method:phenotype(params,H,b,method))}
    terms=jax.jit(loss.terms)
    def metric(parameters,H,method):
        bundle=loss.bundle(H);h=ready(functions[method]['predict'](parameters,jnp.asarray(H,jnp.float32),bundle))
        q=jnp.clip(jnp.round(h/height_step)*height_step,0,hmax)
        m=plain(host(ready(terms(q,bundle))));m['H_um']=float(H);return m
    if stage=='pilot':
        H=60.;bundle=loss.bundle(H);h=np.asarray(bundle[3]);objective=jax.jit(lambda a:loss.terms(a,bundle)['loss']);gradient=np.asarray(ready(jax.grad(objective)(jnp.asarray(h))))
        checks=[];delta=.01
        for ring in [2,8,16]:
            a=h.copy();b=h.copy();a[ring]+=delta;b[ring]-=delta
            fd=(float(ready(objective(jnp.asarray(a))))-float(ready(objective(jnp.asarray(b)))))/(2*delta)
            allowed=.0003+.1*max(abs(fd),abs(float(gradient[ring])))
            checks.append(dict(ring_zero_based=ring,autodiff=float(gradient[ring]),finite_difference=fd,absolute_error=abs(fd-gradient[ring]),allowed=allowed,passed=bool(abs(fd-gradient[ring])<=allowed)))
        write(folder/'pilot_gradient.json',plain(checks))
        if not all(c['passed'] for c in checks):raise RuntimeError('Finite difference preflight failed')
    reports=[]
    for method in (['NN'] if stage=='pilot' else ['NN','direct']):
        for seed in ([tc['seeds'][0]] if stage=='pilot' else tc['seeds']):
            arm=folder/f'{method}_seed{seed}';arm.mkdir(exist_ok=True);path=arm/'checkpoint.pkl'
            rng=np.random.default_rng(seed)
            optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(tc['learning_rate'] if method=='NN' else tc['direct_learning_rate']))
            @jax.jit
            def update(parameters,state,gradient):
                changes,state=optimizer.update(gradient,state,parameters)
                return optax.apply_updates(parameters,changes),state
            params=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,1),jnp.float32)) if method=='NN' else jnp.zeros((len(train_heights),cfg['annuli']),jnp.float32)
            state=optimizer.init(params);step=0;calls=0;history=[];validations=[];best=None;best_key=[1e30]*3;recovery=None
            if path.exists():
                old=pickle.loads(path.read_bytes())
                if old['signature']!=signature:raise RuntimeError('Checkpoint config/source signature mismatch')
                params,state=(jax.tree.map(jnp.asarray,old[k]) for k in ['parameters','state']);step=old['step'];calls=old['calls'];history=old['history'];validations=old['validations'];best=old['best'];best_key=old['best_key'];recovery=old['recovery'];rng.bit_generator.state=old['rng']
                if best is not None:best=jax.tree.map(jnp.asarray,best)
            vg=functions[method]['vg'];stop=tc['pilot_steps'] if stage=='pilot' else tc['steps'];arm_start=time.monotonic()
            while step<stop:
                H=float(rng.choice(train_heights));bundle=loss.bundle(H)
                ((value,m),grad)=ready(vg(params,jnp.asarray(H,jnp.float32),bundle));calls+=1
                if not np.isfinite(float(value)) or not all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad)):raise RuntimeError('Nonfinite training value/gradient')
                params,state=ready(update(params,state,grad));step+=1
                history.append(dict(evaluated_before_update=step,H_um=H,metrics=plain(host(m))))
                if step%tc['validation_interval']==0:
                    metrics=[metric(params,H,method) for H in validation];calls+=len(metrics);key=selection_key(metrics,criteria)
                    validations.append(dict(step=step,key=key,metrics=metrics))
                    if key<best_key:best_key=key;best=jax.tree.map(lambda x:x.copy(),params)
                    if recovery is None:
                        restored=pickle.loads(pickle.dumps({'parameters':host(params),'state':host(state)}))
                        args=(jnp.asarray(60.,jnp.float32),loss.bundle(60.))
                        g1=ready(vg(params,*args))[1];rp=jax.tree.map(jnp.asarray,restored['parameters']);rs=jax.tree.map(jnp.asarray,restored['state']);g2=ready(vg(rp,*args))[1];calls+=2
                        a=ready(update(params,state,g1));b=ready(update(rp,rs,g2))
                        error=max(float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
                        recovery=dict(step=step,next_update_max_error=error,passed=error<1e-7)
                        if not recovery['passed']:raise RuntimeError('Trained-state recovery failure')
                    checkpoint(path,dict(signature=signature,parameters=host(params),state=host(state),step=step,calls=calls,history=history,validations=validations,best=host(best),best_key=best_key,recovery=recovery,rng=rng.bit_generator.state))
                    write(arm/'history.json',history);write(arm/'validation.json',validations)
                    peak=(jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use',0)
                    if peak>6*1024**3:raise RuntimeError('Allocator VRAM bound exceeded')
                    event('LEARNING_PROGRESS',method=method,seed=seed,step=step,validation_key=key,worst_useful_efficiency=min(m['declared_useful_efficiency'] for m in metrics),allocator_peak_bytes=peak)
            report=dict(method=method,seed=seed,steps=step,objective_calls=calls,validation_key=best_key,recovery=recovery,arm_wall_s=time.monotonic()-arm_start)
            write(arm/('pilot_report.json' if stage=='pilot' else 'training_report.json'),report);reports.append(report)
    if stage=='pilot':
        write(folder/'pilot_gate.json',dict(status='PASS',gradient_checks=read(folder/'pilot_gradient.json'),reports=reports,heldout_used=False,independent_Maxwell_launched=False,wall_s=time.monotonic()-begin))
        event('PILOT_COMPLETE');return
    expected=tc['steps']+tc['steps']//tc['validation_interval']*len(validation)+2
    if not all(r['objective_calls']==expected for r in reports):raise RuntimeError('Objective call-budget accounting mismatch')
    selection={method:min([r for r in reports if r['method']==method],key=lambda x:x['validation_key']) for method in ['NN','direct']}
    write(folder/'selection_before_heldout.json',dict(utc=utc(),selection=selection,heldout_used=False,checkpoint_sha256={m:sha(folder/f'{m}_seed{r["seed"]}/checkpoint.pkl') for m,r in selection.items()},signature=signature))
    parameters={}
    for method in ['NN','direct']:
        d=pickle.loads((folder/f'{method}_seed{selection[method]["seed"]}/checkpoint.pkl').read_bytes())
        parameters[method]=jax.tree.map(jnp.asarray,d['best'])
    records=[]
    for H in train_heights+validation+heldout:
        bundle=loss.bundle(H)
        domain='training' if H in train_heights else 'validation' if H in validation else 'initially_withheld'
        for method in ['baseline','NN','direct']:
            h=bundle[3] if method=='baseline' else ready(functions[method]['predict'](parameters[method],jnp.asarray(H,jnp.float32),bundle))
            q=np.asarray(jnp.clip(jnp.round(h/height_step)*height_step,0,hmax))
            m=plain(host(ready(terms(jnp.asarray(q),bundle))))
            spectrum=np.asarray(p.spectrum(jnp.asarray(q),bundle[0]))
            fit=fitted_divergence(spectrum,p.dx,cfg['wavelength_um'])
            targetfit=fitted_divergence(np.asarray(bundle[1]),p.dx,cfg['wavelength_um'])
            row=dict(method=method,H_um=H,height_domain=domain,metrics=m,fitted_core_divergence=fit,target_fitted_core_divergence=targetfit,profile_sha256=hashlib.sha256(q.tobytes()).hexdigest(),declared_target_diameter_um=float(bundle[4]))
            records.append(row)
            design=dict(kind='height_specific_'+method,H_um=H,cylinder_diameter_um=cfg['cylinder_diameter_um'],heights_um=q.tolist(),units='um',architecture=cfg['architecture'],materials=cfg['materials'],mfd_um=cfg['mfd_um'],target_diameter_um=float(bundle[4]),output_reference_z_um=H+hmax+cfg['proxy']['output_clearance_um'],source_signature=signature,validation_status='PROXY_ONLY_NOT_MAXWELL_VALIDATED',height_domain=domain)
            write(folder/f'designs/H{H:g}/{method}.json',design)
            if method=='NN' and H in train_heights:
                f=propagate(spectrum,p.dx,cfg['wavelength_um'],0)
                np.savez_compressed(folder/f'designs/H{H:g}/NN_proxy_output.npz',fields=f,x_um=p.x)
    write(folder/'all_target_metadata.json',loss.metadata)
    result=dict(status='COMPLETED_INVERSE_DESIGN_PROXY_ONLY',selection=selection,reports=reports,objective_calls_per_arm=expected,
                matched_training_validation_recovery_calls=expected*len(reports),records=records,one_profile_per_height=True,
                heldout_created_after_selection=True,heldout_corrections=0,independent_Maxwell_launched=False,
                diffraction_limit_claim='No global aperture-optimum or full M2 claim; compare declared Gaussian core and retained band',
                GPU_allocator=plain(jax.devices()[0].memory_stats()),wall_s=time.monotonic()-begin)
    write(folder/'training_result.json',result);event('TRAINING_COMPLETE',selected_seeds={m:r['seed'] for m,r in selection.items()},records=len(records))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',type=Path,default=DEFAULT);parser.add_argument('--stage',choices=['pilot','train'],required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();folder=a.folder.resolve()
    if a.child:
        try:child(folder,a.stage)
        except BaseException as e:write(folder/(a.stage+'_failure.json'),dict(utc=utc(),type=type(e).__name__,reason=str(e)));raise
        return 0
    if a.stage=='train' and read(folder/'pilot_gate.json')['status']!='PASS':raise RuntimeError('Pilot must pass first')
    if (folder/(a.stage+'_execution_completed.json')).exists():raise FileExistsError('Completed stage exists; retain its cost/status')
    cfg=read(folder/'config.json');before=gpu()
    if 'RTX 4060' not in before['name']:raise RuntimeError('Requested RTX4060 not detected')
    if before['free_mib']<3072:raise RuntimeError('Insufficient free GPU VRAM for declared workload')
    limits=cfg['campaign_limits'];timeout=limits['pilot_timeout_s' if a.stage=='pilot' else 'training_timeout_s']
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src'),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.60',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('LD_LIBRARY_PATH',None)
    start=time.monotonic();peak=0.;minfree=before['free_mib'];errors=[]
    with (folder/(a.stage+'_child.log')).open('w') as log,(folder/(a.stage+'_GPU.jsonl')).open('w') as telemetry:
        process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--folder',str(folder),'--stage',a.stage,'--child'],cwd=folder,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        write(folder/(a.stage+'_execution_started.json'),dict(utc=utc(),owned_child_pid=process.pid,GPU_before=before));print(json.dumps(dict(event='GPU_STAGE_STARTED',stage=a.stage,pid=process.pid)),flush=True)
        while True:
            try:
                now=gpu();delta=max(0.,now['used_mib']-before['used_mib']);peak=max(peak,delta);minfree=min(minfree,now['free_mib'])
                telemetry.write(json.dumps(dict(elapsed_s=time.monotonic()-start,delta_mib=delta,**now))+'\n');telemetry.flush()
                if delta>limits['GPU_delta_max_mib'] or now['free_mib']<limits['GPU_free_min_mib']:errors.append('GPU resource guard')
            except Exception as e:errors.append(str(e))
            if time.monotonic()-start>timeout:errors.append('Stage timeout')
            if process.poll() is not None:break
            if errors:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(.5)
    report=dict(utc=utc(),stage=a.stage,returncode=process.returncode,wall_s=time.monotonic()-start,peak_GPU_delta_mib=peak,minimum_free_GPU_mib=minfree,violations=errors,RAM_gates=False)
    write(folder/(a.stage+'_execution_completed.json'),report);print(json.dumps(report,indent=2));return 0 if process.returncode==0 and not errors else 2

if __name__=='__main__':sys.exit(main())
