"""Repeat only the existing H34 pair, exporting data lost by thin-plane collapse."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import psutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--canonical-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    root=args.canonical_root.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if root==out or root in out.parents:raise ValueError("separate run required")
    out.mkdir(parents=True)
    source=root/"src/fiber_metalens_nn/collimation_worker.py"
    config=root/"artifacts/1550nm_collimation/frozen-config.json"
    design=root/"artifacts/1550nm_collimation/designs/phase8-flat-H34.json"
    hashes={str(f):sha(f) for f in [source,config,design]}
    cfg=json.loads(config.read_text());shape=json.loads(design.read_text())
    assert cfg["wavelength_um"]==1.55 and cfg["mfd_um"]==9 and cfg["cylinder_diameter_um"]==48
    assert cfg["materials"]["polymer"]["n"]==1.52 and shape["H_um"]==34 and all(h==0 for h in shape["heights_um"])
    save(out/"config.json",cfg);save(out/"design.json",shape)
    text=source.read_text().replace("from .geometry import", "from fiber_metalens_nn.geometry import").replace("from .collimation_optics import", "from fiber_metalens_nn.collimation_optics import")
    old='    field_zs=[exit_z,cross_z] if design else [0.,exit_z]\n'
    replacement='    field_zs=[exit_z,cross_z,args.H-.5,args.H+.5] if design else [0.,exit_z,args.H-.5,args.H+.5]\n'
    assert text.count(old)==1;text=text.replace(old,replacement)
    anchor='    if design:\n        ref=json.loads(Path(args.reference).read_text())'
    insert='''    # Additional planes preserve the two centered-grid samples whose
    # correlation is lost by the thin-plane export's collapse operation.
    offset=.5/args.resolution
    pair_zs=[z+d for z in field_zs for d in (-offset,offset)]
    pair_monitors=[sim.add_dft_fields(components,[frequency],center=at(radius/2,z),size=mp.Vector3(radius,0,0),decimation_factor=1) for z in pair_zs]
    audit_fluxes=[zflux(z) for z in field_zs]
    if design:
        ref=json.loads(Path(args.reference).read_text())'''
    assert text.count(anchor)==1;text=text.replace(anchor,insert)
    anchor='    result["wall_s"]=time.monotonic()-start\n'
    insert='''    paired=[]
    for monitor in pair_monitors:
        paired.append([np.asarray(sim.get_dft_array(monitor,c,0)).squeeze() for c in components])
    audit_path=str(Path(args.output).with_suffix(".power-audit.npz"))
    data={"r_um":np.linspace(0,radius,len(paired[0][0])),"plane_z_um":np.asarray(field_zs),"pair_z_um":np.asarray(pair_zs).reshape((-1,2)),"paired_fields":np.asarray(paired).reshape((len(field_zs),2,6,-1)),"monitor_fluxes":np.asarray([get(f) for f in audit_fluxes]),"resolution_per_um":args.resolution,"Courant":.45}
    # FluxData E includes the signed quadrature weights once; H does not.
    # Their contraction independently reproduces the monitor API convention.
    for index,monitor in enumerate(audit_fluxes):
        fd=sim.get_flux_data(monitor)
        data[f"flux_E_{index}"]=fd.E
        data[f"flux_H_{index}"]=fd.H
    np.savez_compressed(audit_path,**data)
    result["power_audit_fields_file"]=audit_path
    result["wall_s"]=time.monotonic()-start
'''
    assert text.count(anchor)==1;text=text.replace(anchor,insert)
    worker=out/"paired_export_worker.py";worker.write_text(text)
    save(out/"predeclared.json",{"H_um":34,"n":1.52,"resolution":6,"geometry_unchanged":True,"reason":"collapsed saved fields lack correlations between the two adjacent centered planes; export both plus flux quadrature data","simulations":2,"new_unique_geometries":0,"training_steps":0,"RAM_gates":False,"canonical_hashes":hashes,"diagnostic_worker_sha256":sha(worker)})
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(root/"src"),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1");env.pop("LD_LIBRARY_PATH",None)
    telemetry=[];runs=[]
    for kind in ["reference","device"]:
        command=[str(root/".venv-meep/bin/python"),"-B",str(worker),"--config",str(out/"config.json"),"--output",str(out/(kind+".json")),"--H","34","--resolution","6"]
        if kind=="device":command += ["--design",str(out/"design.json"),"--reference",str(out/"reference.json")]
        start=time.monotonic()
        with (out/(kind+".log")).open("w") as log:
            process=subprocess.Popen(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT);owned=psutil.Process(process.pid)
            while process.poll() is None:
                now=time.monotonic()
                try:rss=sum(t.memory_info().rss for t in [owned]+owned.children(recursive=True) if t.is_running())
                except psutil.Error:rss=None
                telemetry.append({"kind":kind,"elapsed_s":now-start,"rss_bytes":rss,"available_host_ram_bytes":psutil.virtual_memory().available})
                if now-start>90:process.terminate();process.wait(timeout=10);raise TimeoutError("owned pair worker timeout")
                time.sleep(.1)
        runs.append({"kind":kind,"command":command,"wall_s":time.monotonic()-start,"exit_code":process.returncode})
        save(out/"execution.json",{"runs":runs});(out/"telemetry.jsonl").write_text("".join(json.dumps(t)+"\n" for t in telemetry))
        if process.returncode:raise RuntimeError(f"{kind} failed, retain log")
        print(json.dumps(runs[-1]),flush=True)
    save(out/"execution.json",{"utc":datetime.now(timezone.utc).isoformat(),"runs":runs,"peak_sampled_rss_mib":max(t["rss_bytes"] or 0 for t in telemetry)/2**20,"canonical_inputs_unchanged":hashes=={str(f):sha(f) for f in [source,config,design]},"RAM_gates":False,"GPU_used":False})


if __name__=="__main__":main()
