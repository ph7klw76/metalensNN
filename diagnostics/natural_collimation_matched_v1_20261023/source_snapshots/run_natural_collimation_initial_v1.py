"""One fixed-geometry natural-size forward study; no optimizer/NN/Meep."""
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).parent/'natural_collimation_staging'
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu():
    v=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=8).strip().splitlines()[0].split(',')
    return dict(name=v[0].strip(),total_mib=float(v[1]),used_mib=float(v[2]),free_mib=float(v[3]))

def child(out):
    import numpy as np
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(STAGE/'src'),str(ROOT/'src')]
    from fiber_metalens_nn.runtime import configure
    configure()
    import jax
    import jax.numpy as jnp
    from fiber_metalens_nn.collimation_proxy import CylinderProxy
    from fiber_metalens_nn.natural_collimation import (require_index_matched,derive_annular_relief,ideal_top_phase_spectrum,measure_natural_output)
    from fiber_metalens_nn.collimation_optics import propagate
    cfg=read(out/'config.json');require_index_matched(cfg);start=time.monotonic();H=32.
    assert jax.devices()[0].platform=='gpu'
    def event(name,**kw):
        v=dict(utc=utc(),elapsed_s=time.monotonic()-start,event=name,**kw)
        print(json.dumps(v),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    p=CylinderProxy(cfg);top=p.pedestal(H);top.block_until_ready();array=np.asarray(top)
    continuous,quantized,encoding=derive_annular_relief(array,p.x,cfg)
    write(out/'field_derived_relief.json',encoding)
    np.savez_compressed(out/'cylinder_top_field.npz',x_um=p.x,scalar_complex_field=array)
    event('FIELD_DERIVED_RELIEF_READY',core_n=p.ns,cylinder_n=p.np,bright_phase_RMS_rad=encoding['encoded_bright_phase_RMS_rad'],fitted_polymer_R_um=encoding['polymer_wavefront_radius_um'])
    cases={};spectra={}
    for name,heights in [('flat',np.zeros(24,np.float32)),('annular_continuous',continuous),('annular_quantized',quantized)]:
        before=time.monotonic();s=np.asarray(p.spectrum(jnp.asarray(heights),top));report=measure_natural_output(s,p.dx,p.incident,cfg,H)
        report['height_profile_um']=heights.tolist();report['profile_quantized']=name!='annular_continuous';report['forward_and_analysis_wall_s']=time.monotonic()-before
        write(out/(name+'_metrics.json'),report);cases[name]=report;spectra[name]=s
        np.savez_compressed(out/(name+'_output.npz'),x_um=p.x,spectrum=s.astype(np.complex64),E_H_fields=propagate(s,p.dx,1.55,0).astype(np.complex64))
        event('FORWARD_CASE_COMPLETE',case=name,diameters_um=report['diameters_output_um'],curvature=report['wavefront'],matched_overlap=report['overlap_with_actual_size_Gaussian'],transmission=report['transmission'],half_angles=[a.get('half_angle_rad') for a in report['fitted_core_divergence']['axes']])
    s=ideal_top_phase_spectrum(array,p.dx,cfg);ideal=measure_natural_output(s,p.dx,p.incident,cfg,H)
    ideal['ideal_operator_scope']='Exact scalar phase flattening at H, zero-thickness local phase and normal interface amplitude, then air propagation4um to the common output. No finite relief or fabrication constraints. Reference, not universal optical upper bound.'
    write(out/'ideal_phase_only_metrics.json',ideal);cases['ideal_phase_only']=ideal
    np.savez_compressed(out/'ideal_phase_only_output.npz',x_um=p.x,spectrum=s.astype(np.complex64),E_H_fields=propagate(s,p.dx,1.55,0).astype(np.complex64))
    checks=[]
    for name,kw in [('pedestal_dz_0p25',dict(dz=.25)),('relief_slices_48',dict(slices=48)),('grid_256',dict(grid_n=256))]:
        p2=CylinderProxy(cfg,**kw);s2=np.asarray(p2.spectrum(jnp.asarray(quantized),p2.pedestal(H)))
        m=measure_natural_output(s2,p2.dx,p2.incident,cfg,H);write(out/(name+'_same_geometry_metrics.json'),m)
        base=cases['annular_quantized'];checks.append(dict(setting=name,diameter_relative_changes=[m['diameters_output_um'][i]/base['diameters_output_um'][i]-1 for i in range(2)],matched_overlap_absolute_change=m['overlap_with_actual_size_Gaussian']-base['overlap_with_actual_size_Gaussian'],transmission_relative_change=m['transmission']/base['transmission']-1,curvature=m['wavefront'],fitted_core_divergence=m['fitted_core_divergence']))
        p2.pedestal_cache.clear();event('FIXED_GEOMETRY_SAMPLING_COMPLETE',setting=name)
    stats={k:int(v) for k,v in (jax.devices()[0].memory_stats() or {}).items() if isinstance(v,(int,float))}
    if stats.get('peak_bytes_in_use',0)>6*1024**3:raise RuntimeError('allocator peak exceeds6GiB')
    flat=cases['flat'];ann=cases['annular_quantized'];angles=lambda m:[a['half_angle_rad'] for a in m['fitted_core_divergence']['axes']]
    comparison=dict(abs_curvature_reduction_fraction=1-abs(ann['wavefront']['inverse_R_per_um'])/abs(flat['wavefront']['inverse_R_per_um']),core_divergence_reduction_fraction=[1-a/b for a,b in zip(angles(ann),angles(flat))],transmission_absolute_change=ann['transmission']-flat['transmission'],actual_size_overlap_change=ann['overlap_with_actual_size_Gaussian']-flat['overlap_with_actual_size_Gaussian'])
    result=dict(utc=utc(),status='COMPLETED_SYNTHETIC_INITIAL_FIELD_DERIVED_COMPARISON',objective='natural-size wavefront flattening/reduced excess divergence; 30um remains separate aspiration',H_um=H,cylinder_diameter_um=48,indices=dict(core_surrogate=1.5,cylinder=1.5,relief=1.5,ambient=1.),encoding=encoding,cases=cases,annular_vs_flat=comparison,same_geometry_sampling=checks,wall_s=time.monotonic()-start,GPU_allocator=stats,Maxwell_unique_candidates_added=0,Maxwell_launched=False,optimizer_updates=0,NN_training_launched=False,old_gate_files_changed=False,physical_validation=False,next_Maxwell_budget='Historical ledger10/10; at least3 new device identities (bare H32, flat H32, annular H32) for this new source/material cohort, cumulative13, plus references/repeats explicitly budgeted. Ideal phase-only operator is not a Maxwell-realizable geometry. No solve is launched here.')
    write(out/'result.json',result);event('INITIAL_STUDY_COMPLETE',wall_s=result['wall_s'],comparison=comparison,allocator=stats)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();out=a.output
    if a.child:
        try:child(out)
        except BaseException as e:write(out/'failure.json',dict(utc=utc(),type=type(e).__name__,reason=str(e)));raise
        return 0
    if out.exists():raise FileExistsError(out)
    before=gpu();assert 'RTX 4060' in before['name']
    if before['free_mib']<3072:raise RuntimeError('insufficient free VRAM for declared unchanged grid; no start')
    cfg=read(ROOT/'configs/1550nm_collimation.json');cfg['materials']['substrate']['n']=1.50;cfg['materials']['polymer']['n']=1.50
    cfg['material_provenance']='SYNTHETIC ASSUMPTION: Gaussian fibre-surrogate medium and polymer bothn1.50, air1, allkappa0. No verified real core/resin index or cladding/eigenmode supplied.'
    cfg['result_label']='SYNTHETIC_INDEX_MATCHED_N1P50_NATURAL_SIZE_INITIAL_NO_TRAINING'
    cfg['historical_30um_evaluation']=cfg.pop('evaluation')
    cfg['evaluation']=dict(processing_planes_um=[0,10,25,50,75,100],diagnostic_planes_um=[200])
    cfg['loss']=dict(version='natural_output_wavefront_initial_v1',optimization_enabled=False,note='Forward-only initial phase encoding. Legacy training rejects this unknown loss version rather than silently forcing30um.')
    cfg['scope']='Same solid cylinder/top relief. Primary objective is actual-size wavefront flattening and reduced excess divergence. 30um and old5percent gate remain separately reported aspirations; no NN/generalization claim.'
    out.mkdir(parents=True);write(out/'config.json',cfg)
    source=[Path(__file__),STAGE/'src/fiber_metalens_nn/natural_collimation.py']+[ROOT/'src/fiber_metalens_nn'/n for n in ['collimation_proxy.py','collimation_optics.py','collimation_metrics.py']]
    protected=[ROOT/'configs/1550nm_collimation.json',ROOT/'artifacts/1550nm_collimation/phase9/gate.json',ROOT/'artifacts/1550nm_collimation/maxwell/geometry-ledger.json',ROOT/'artifacts/1550nm_collimation/phase9/seed-11.pkl']
    protected_hash={str(p):sha(p) for p in protected}
    write(out/'predeclared.json',dict(utc=utc(),H_um=32,cylinder_diameter_um=48,annuli=24,hmax_um=3,height_step_um=.05,phase_encoding='field-derived unwrapped bounded center-high relief; no optical optimizer',cases=['flat','annular_continuous','annular_quantized','ideal_phase_only_reference'],same_geometry_checks=['dz.25','slices48','grid256'],timeout_s=600,workers=1,VRAM_limit_GiB=6,RAM_gates=False,source_sha256={str(p):sha(p) for p in source},protected_sha256=protected_hash,GPU_before=before,old_objective_unchanged_in_historical_records=True,training_or_Maxwell_authorized=False))
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(STAGE/'src')+':'+str(ROOT/'src'),XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.60',JAX_PLATFORMS='cuda',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('LD_LIBRARY_PATH',None)
    start=time.monotonic();peak=0.;minimum=before['free_mib'];violations=[]
    with (out/'child.log').open('w') as log,(out/'GPU_telemetry.jsonl').open('w') as telemetry:
        p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child','--output',str(out)],cwd=out,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        write(out/'execution_started.json',dict(utc=utc(),owned_child_pid=p.pid));print(json.dumps(dict(event='ACTUAL_FORWARD_STUDY_STARTED',utc=utc(),owned_child_pid=p.pid,output=str(out))),flush=True)
        while True:
            elapsed=time.monotonic()-start
            try:
                g=gpu();delta=max(0,g['used_mib']-before['used_mib']);peak=max(peak,delta);minimum=min(minimum,g['free_mib']);telemetry.write(json.dumps(dict(elapsed_s=elapsed,delta_mib=delta,**g))+'\n');telemetry.flush()
                if delta>6144 or g['free_mib']<384:violations.append('GPU allocation/free-memory reserve bound')
            except Exception as e:violations.append('GPU telemetry unavailable:'+str(e))
            if elapsed>600:violations.append('600second timeout')
            if p.poll() is not None:break
            if violations:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(.5)
    result=dict(utc=utc(),owned_child_exit_code=p.returncode,wall_s=time.monotonic()-start,peak_total_device_delta_mib=peak,minimum_free_GPU_mib=minimum,violations=violations,protected_unchanged=all(sha(Path(p))==h for p,h in protected_hash.items()),RAM_gates=False,Maxwell_launched=False,optimization_launched=False)
    write(out/'execution_completed.json',result);print(json.dumps(result,indent=2),flush=True)
    return 0 if p.returncode==0 and not violations and result['protected_unchanged'] else 2

if __name__=='__main__':sys.exit(main())
