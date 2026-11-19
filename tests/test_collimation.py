import copy
import json
from pathlib import Path
import numpy as np
from fiber_metalens_nn.collimation_optics import (grid,vector_spectrum,propagate,forward_spectrum,overlap,describe,cylindrical_to_cartesian)
from fiber_metalens_nn.audit import effective_hardware,preflight
from fiber_metalens_nn.guard import violations
from fiber_metalens_nn.collimation_run import effective_duration,Experiment
from fiber_metalens_nn.collimation_optics import fingerprint,write


def test_gaussian_propagation_power_overlap_and_diameter():
    cfg=json.loads(Path("configs/1550nm_collimation.json").read_text())
    x,dx=grid(512,128)
    xx,yy=np.meshgrid(x,x,indexing="xy")
    spectrum=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/225)),dx,1.55)
    result=describe(spectrum,dx,1.55,30,cfg)
    assert result["optical_targets_pass"]
    assert result["overlap"]>1-1e-10
    assert abs(result["planes"][0]["diameter_x_um"]/30-1)<.005
    expected=np.sqrt(1+(100/(np.pi*225/1.55))**2)
    assert abs(result["planes"][4]["diameter_x_um"]/result["planes"][0]["diameter_x_um"]-expected)<.002
    assert max(abs(p["power_relative_to_output"]-1) for p in result["planes"])<1e-10


def test_forward_projection_rejects_backward_plane_wave():
    f=np.zeros((6,32,32),complex)
    f[0]=1
    f[4]=1
    assert np.max(abs(np.fft.ifft2(forward_spectrum(f,.5,1.55),axes=(-2,-1))-f))<1e-12
    f[4]=-1
    assert np.max(abs(forward_spectrum(f,.5,1.55)))<1e-10


def test_cylindrical_linear_reconstruction():
    # Cover the Cartesian grid corners; otherwise intentional zero padding
    # beyond the measured radial extent is not the analytic Gaussian.
    r=np.linspace(0,64,1281)
    g=np.exp(-(r/15)**2)
    cyl=np.asarray([g,1j*g,0*g,-1j*g,g,0*g])
    x,dx,f=cylindrical_to_cartesian(r,cyl,256,64)
    xx,yy=np.meshgrid(x,x,indexing="xy")
    expected=np.exp(-(xx*xx+yy*yy)/225)
    assert np.max(abs(f[0]-expected))<1e-5
    assert np.max(abs(f[1]))<1e-12
    assert np.max(abs(f[4]-expected))<1e-5


def test_phase8_ram_override_preserves_vram_controls():
    data=dict(phase=8,budgets=dict(project_ram_bytes=1,host_free_floor_bytes=100,wsl_free_floor_bytes=100,
              project_gpu_bytes=1024**3,gpu_free_floor_bytes=10),gpu=dict(devices=[{}]))
    effective=effective_hardware(data)
    assert preflight(effective)==[]
    sample=dict(project_rss_bytes=10**20,host_available_bytes=None,wsl_available_bytes=0,gpu_free_bytes=20,gpu_delta_bytes=0)
    assert violations(sample,effective["budgets"])==[]
    sample["gpu_free_bytes"]=0
    assert violations(sample,effective["budgets"])==["memory pressure: gpu_free_bytes"]
    assert data["budgets"]["project_ram_bytes"]==1


def test_taller_comparison_waits_for_transit_and_returned_reflection():
    cfg=json.loads(Path("configs/1550nm_collimation.json").read_text())
    assert effective_duration(cfg,35)==160
    assert effective_duration(cfg,198.37)>2*1.52*(198.37+3)
    assert effective_duration(cfg,35,240)==240


def test_identical_integer_float_numerics_reuse_completed_solver_record(tmp_path,monkeypatch):
    import subprocess
    cfg=json.loads(Path("configs/1550nm_collimation.json").read_text())
    config=tmp_path/"config.json"
    write(config,cfg)
    exp=Experiment(config,tmp_path/"run")
    field=tmp_path/"fields.npz";field.touch()
    fluxfile=tmp_path/"flux.npz";fluxfile.touch()
    key=fingerprint(dict(config=cfg,design=None,params=dict(H=35,resolution=6),source=exp.worker_hash))[:20]
    cached=exp.root/"maxwell"/("reference-"+key+".json")
    write(cached,dict(config_fingerprint=exp.key,fields_file=str(field),flux_data_file=str(fluxfile),
                      artifact_file=str(cached),power_balance_error=0,settings=dict(duration=160)))
    def must_not_submit(*args,**kwargs):
        raise AssertionError("Equivalent numerical settings must not launch Meep")
    monkeypatch.setattr(subprocess,"run",must_not_submit)
    assert exp.maxwell(None,35.,6.)["artifact_file"]==str(cached)


def test_failed_tall_power_closure_extends_time_with_a_bounded_retry(tmp_path,monkeypatch):
    cfg=json.loads(Path("configs/1550nm_collimation.json").read_text())
    config=tmp_path/"config.json";write(config,cfg)
    exp=Experiment(config,tmp_path/"run")
    bad=dict(artifact_file="original.json",power_balance_error=.062,settings=dict(duration=633))
    good=dict(artifact_file="longer.json",power_balance_error=.002,settings=dict(duration=950))
    requests=[]
    def fake_solver(*args,**kwargs):
        requests.append(kwargs)
        return good
    monkeypatch.setattr(exp,"maxwell",fake_solver)
    assert exp.transit_checked(bad,"design.json",198.37,6,{},0)==good
    assert requests==[dict(_transit_attempt=1,duration=950.)]
    import pytest
    with pytest.raises(RuntimeError,match="after two duration extensions"):
        exp.transit_checked(bad,"design.json",198.37,6,{},2)
