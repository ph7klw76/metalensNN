"""Evidence-based report, figures and verified connected geometry exports."""
import hashlib
import json
import struct
from pathlib import Path
from collections import Counter
import numpy as np
from .geometry import profile_polygons
from .collimation_optics import write,cylindrical_to_cartesian,forward_spectrum,grid,vector_spectrum,propagate,overlap,flux


def cylinder_mesh(heights,H,radius,segments=128):
    heights=np.asarray(heights,float)
    if H<=0 or np.any(heights<0):
        raise ValueError("Supported nonnegative radial graph required")
    cfg=dict(annuli=len(heights),aperture_diameter_um=2*radius)
    polygon=profile_polygons(heights+H,cfg)[0]
    vertices=[]
    rings=[]
    for r,z in polygon:
        if r==0:
            rings.append([len(vertices)])
            vertices.append([0.,0.,z])
        else:
            rings.append(list(range(len(vertices),len(vertices)+segments)))
            vertices.extend([[r*np.cos(2*np.pi*j/segments),r*np.sin(2*np.pi*j/segments),z] for j in range(segments)])
    faces=[]
    for i in range(len(rings)):
        a,b=rings[i],rings[(i+1)%len(rings)]
        if len(a)==len(b)==1:
            continue
        for j in range(segments):
            k=(j+1)%segments
            if len(a)==1:
                faces.append([a[0],b[j],b[k]])
            elif len(b)==1:
                faces.append([a[j],b[0],a[k]])
            else:
                faces.extend([[a[j],b[j],b[k]],[a[j],b[k],a[k]]])
    vertices,faces=np.asarray(vertices),np.asarray(faces,dtype=np.int32)
    volume=np.sum(np.einsum("ij,ij->i",vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])))/6
    if volume<0:
        faces=faces[:,::-1]
    return vertices,faces


def mesh_audit(vertices,faces,heights,H,radius):
    edges=Counter(tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in faces for i in range(3))
    triangles=vertices[faces]
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    volume=float(np.sum(np.einsum("ij,ij->i",triangles[:,0],np.cross(triangles[:,1],triangles[:,2])))/6)
    e=np.linspace(0,radius,len(heights)+1)
    exact=float(np.sum(np.pi*(e[1:]**2-e[:-1]**2)*(H+np.asarray(heights))))
    # One connected component, not merely pairwise watertight separate shells.
    adjacent={i:set() for i in range(len(vertices))}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            adjacent[int(a)].add(int(b));adjacent[int(b)].add(int(a))
    visited={0};pending=[0]
    while pending:
        for node in adjacent[pending.pop()]-visited:
            visited.add(node);pending.append(node)
    checks=dict(watertight=all(n==2 for n in edges.values()),single_connected_component=len(visited)==len(vertices),
                no_degenerate_faces=bool(np.all(area>1e-12)),positive_volume=volume>0,
                volume_relative_error=abs(volume/exact-1),mesh_volume_um3=volume,analytic_volume_um3=exact,
                dimensions_um=np.ptp(vertices,axis=0).tolist(),vertices=len(vertices),triangles=len(faces))
    checks["passed"]=checks["watertight"] and checks["single_connected_component"] and checks["no_degenerate_faces"] and checks["positive_volume"] and checks["volume_relative_error"]<.001
    return checks


def export_stl(path,vertices,faces):
    with Path(path).open("wb") as stream:
        stream.write(b"units: micrometres; see manifest; geometry only".ljust(80,b" "))
        stream.write(struct.pack("<I",len(faces)))
        for triangle in vertices[faces]:
            normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
            normal/=np.linalg.norm(normal)
            stream.write(struct.pack("<12fH",*normal,*triangle.ravel(),0))


def caustic_fit(metric,wavelength):
    """Second-moment caustic fit, accepted only if its window and fit are valid."""
    planes=metric["planes"][:5]
    if max(p["edge_power_fraction"] for p in planes)>1e-4:
        return dict(valid=False,reason="processing-plane edge power exceeds 1e-4; extend measured/analysis window")
    z=np.asarray([p["z_um"] for p in planes])
    axes={}
    for axis in ("x","y"):
        squared=np.asarray([p[f"D4sigma_{axis}_um"] for p in planes])**2
        a,b,c=np.polyfit(z,squared,2)
        residual=float(np.max(abs(np.polyval([a,b,c],z)/squared-1)))
        if a<=0 or c-b*b/(4*a)<=0 or residual>.01:
            return dict(valid=False,reason="nonphysical or >1% residual second-moment quadratic fit")
        waist_diameter=float(np.sqrt(c-b*b/(4*a)))
        divergence=float(np.sqrt(a)/2)
        axes[axis]=dict(waist_z_relative_to_output_um=float(-b/(2*a)),waist_D4sigma_um=waist_diameter,
                        divergence_half_angle_rad=divergence,M2=float(np.pi*(waist_diameter/2)*divergence/wavelength),
                        maximum_fit_relative_residual=residual)
    return dict(valid=True,axes=axes,note="Diagnostic paraxial second-moment fit over 0--100 um. Algebra/window validity is not mesh acceptance of M2 or all-angle moments; inspect retained moment-convergence data separately.")


def profile_proxy(cfg):
    import time
    import jax
    import jax.numpy as jnp
    from .collimation_proxy import CylinderProxy
    start=time.monotonic()
    proxy=CylinderProxy(cfg)
    pedestal=proxy.pedestal(34)
    pedestal.block_until_ready()
    pedestal_wall=time.monotonic()-start
    inputs=(proxy.seed(34),pedestal,proxy.target(28),jnp.asarray(28,jnp.float32))
    report=dict(pedestal_setup_and_first_propagation_s=pedestal_wall)
    for name,fn in (("forward",proxy.objective),("forward_and_backward",proxy.value_gradient)):
        start=time.monotonic()
        compiled=fn.lower(*inputs).compile()
        report[name+"_explicit_compile_s"]=time.monotonic()-start
        jax.block_until_ready(compiled(*inputs))
        start=time.monotonic()
        for _ in range(5):
            jax.block_until_ready(compiled(*inputs))
        report[name+"_warm_mean_s"]=(time.monotonic()-start)/5
    start=time.monotonic()
    proxy.evaluate(np.asarray(inputs[0]),34,28)
    report["all_metric_planes_s_including_transfer_and_analysis"]=time.monotonic()-start
    report["devices"]=[str(d) for d in jax.devices()]
    return report


def audit_conditioning(cfg,folder,seed):
    import pickle
    import jax
    import jax.numpy as jnp
    from .learning import ConditionalMLP
    # Trusted local checkpoint from this run, selected before heldout metrics.
    checkpoint=pickle.loads((folder/f"seed-{seed}.pkl").read_bytes())
    params=jax.tree.map(jnp.asarray,checkpoint["best"])
    model=ConditionalMLP(cfg["annuli"])
    raw=np.asarray(model.apply(params,jnp.asarray([[0.,-1.],[0.,1.],[-1.,0.],[1.,0.]],jnp.float32)))
    diameter_response=float(np.sqrt(np.mean((raw[0]-raw[1])**2)))
    H_response=float(np.sqrt(np.mean((raw[2]-raw[3])**2)))
    return dict(raw_network_diameter_response_rms=diameter_response,raw_network_height_response_rms=H_response,
                passed=diameter_response>1e-3 and H_response>1e-3,
                note="Raw network outputs tested separately from the already-conditional analytic phase seed; no heldout scores select parameters")


def expansion_decision(cfg,summary):
    wavelength=cfg["wavelength_um"]
    w0=cfg["mfd_um"]/2
    target=15.
    H=35.
    proposals=[]
    for medium,n in (("air",1.),("homogeneous_polymer_limit",cfg["materials"]["polymer"]["n"])):
        zr=np.pi*n*w0*w0/wavelength
        magnification=np.sqrt((target/w0)**2-(H/zr)**2)
        first_f=-H/(magnification-1)
        q_after=1/(1/(1j*zr)-1/first_f)
        q_top=q_after+H
        second_f=1/(1/q_top).real
        proposals.append(dict(medium=medium,inter_surface_distance_um=H,first_diverging_focal_um=float(first_f),
                              second_curvature_cancel_focal_um=float(second_f),
                              predicted_diameter_um=float(2*np.sqrt(wavelength/(np.pi*n)*abs(q_top)**2/q_top.imag)),
                              status="PARAXIAL_TWO_THIN_SURFACE_FEASIBILITY_ONLY_NOT_FABRICATED_OR_MAXWELL_VALIDATED"))
    w=summary["flat_cylinder"]["planes"][0]["diameter_x_um"]/2
    return dict(primary_result="Executed top-relief candidates do not meet preferred beam-expansion/collimation requirements",
                local_fixed_amplitude_Gaussian_overlap_limit=float((2*w*target/(w*w+target*target))**2),
                limit_scope="Illustrative local-phase-only same-plane Gaussian approximation. Not a bound on arbitrary finite-thickness or multi-surface structures.",
                next_architecture_to_test="Explicitly modeled near-facet diverging stage plus top collimator, potentially inside a hollow cylindrical support with top membrane; requires fresh geometry/material/Maxwell validation.",
                paraxial_proposals=proposals)


def audit_input_mode(cfg,root):
    diagnostic=json.loads((root/"phase8/flat-cylinder-proxy-vs-Maxwell.json").read_text())
    record=json.loads(Path(diagnostic["Maxwell"]["record"]).read_text())
    reference=json.loads(Path(record["reference_file"]).read_text())
    raw=np.load(reference["fields_file"])
    x,dx,field=cylindrical_to_cartesian(raw["r_um"],raw["fields"][0],512,128)
    n=cfg["materials"]["substrate"]["n"]
    measured=propagate(forward_spectrum(field,dx,cfg["wavelength_um"],n),dx,cfg["wavelength_um"],0,n)
    xx,yy=np.meshgrid(x,x,indexing="xy")
    scalar=np.exp(-(xx*xx+yy*yy)/(cfg["mfd_um"]/2)**2)
    target=propagate(vector_spectrum(np.fft.fft2(scalar),dx,cfg["wavelength_um"],n),dx,cfg["wavelength_um"],0,n)
    return dict(declared_surrogate_MFD_um=cfg["mfd_um"],measured_diameter_um=diagnostic["discrepancy"]["measured_input_diameter_um"],
                forward_vector_overlap_with_declared_Gaussian=overlap(measured,target,dx),
                reference_file=record["reference_file"],note="Input consistency audit, not a measured eigenmode of the unspecified real fiber")


def audit_power_normalization(cfg,summary):
    audits={}
    for name,metric in summary.items():
        record=json.loads(Path(metric["record"]).read_text())
        reference=json.loads(Path(record["reference_file"]).read_text())
        n=cfg["materials"]["substrate"]["n"]
        raw=np.load(reference["fields_file"])
        _,dx,input_field=cylindrical_to_cartesian(raw["r_um"],raw["fields"][0],512,128)
        input_field=propagate(forward_spectrum(input_field,dx,cfg["wavelength_um"],n),dx,cfg["wavelength_um"],0,n)
        raw=np.load(record["fields_file"])
        _,_,output_field=cylindrical_to_cartesian(raw["r_um"],raw["fields"][0],512,128)
        output_field=propagate(forward_spectrum(output_field,dx,cfg["wavelength_um"]),dx,cfg["wavelength_um"],0)
        transmission=float(np.sum(flux(output_field))/np.sum(flux(input_field)))
        audits[name]=dict(independent_field_transmission=transmission,Meep_flux_transmission=record["transmission"],
                          relative_discrepancy=abs(transmission/record["transmission"]-1),
                          independent_field_eta_mode=transmission*metric["overlap"])
    return dict(cases=audits,maximum_relative_discrepancy=max(a["relative_discrepancy"] for a in audits.values()),
                note="Independent E/H field power ratios versus Meep flux monitors. Differences include radial interpolation and propagating-band truncation; no renormalization conceals them.")


def package(experiment):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg=experiment.config
    root=experiment.root
    phase9=root/"phase9"
    gate=json.loads((phase9/"gate.json").read_text())
    summary=json.loads((phase9/"Maxwell-summary.json").read_text())
    selected=gate["selected_seed"]
    folder=root/"phase10"
    folder.mkdir(exist_ok=True)
    write(folder/"proxy-profile.json",profile_proxy(cfg))
    write(folder/"design-decision.json",expansion_decision(cfg,summary))
    write(folder/"input-mode-audit.json",audit_input_mode(cfg,root))
    normalization=audit_power_normalization(cfg,summary)
    write(folder/"power-normalization-audit.json",normalization)
    if normalization["maximum_relative_discrepancy"]>.05:
        raise RuntimeError("Field/flux power normalization differs by >5%; diagnose before packaging")
    conditional=audit_conditioning(cfg,phase9,selected)
    write(phase9/"raw-NN-conditioning-audit.json",conditional)
    if not conditional["passed"]:
        raise RuntimeError("NN raw outputs do not respond to both conditions; diagnose before packaging")
    write(folder/"caustic-fits.json",{name:caustic_fit(metric,cfg["wavelength_um"]) for name,metric in summary.items()})
    design=json.loads((root/"designs"/f"NN-seed{selected}-H35-D30.json").read_text())
    exports={}
    for label,path in (("preferred_NN",root/"designs"/f"NN-seed{selected}-H35-D30.json"),
                       ("taller_curvature_comparator",root/"designs/tall-curvature_relief.json")):
        item=json.loads(path.read_text())
        h=np.asarray(item["heights_um"])
        vertices,faces=cylinder_mesh(h,item["H_um"],cfg["cylinder_diameter_um"]/2)
        audit=mesh_audit(vertices,faces,h,item["H_um"],cfg["cylinder_diameter_um"]/2)
        if not audit["passed"]:
            raise RuntimeError(f"Invalid export mesh: {audit}")
        stl=folder/(label+".stl")
        export_stl(stl,vertices,faces)
        write(folder/(label+"-parameters.json"),dict(item,ring_edges_um=np.linspace(0,cfg["cylinder_diameter_um"]/2,cfg["annuli"]+1).tolist()))
        exports[label]=dict(stl_file=str(stl),stl_coordinate_unit="um",to_mm_scale=.001,facet_z_um=0,
                            relief_base_z_um=item["H_um"],total_polymer_height_um=float(item["H_um"]+h.max()),
                            output_reference_z_um=item["output_reference_z_um"],mesh=audit,
                            optical_status=gate["optical_gate"] if label=="preferred_NN" else "OUTSIDE_PREFERENCE_COARSE_COMPARATOR")
    write(folder/"geometry-manifest.json",dict(exports=exports,material=cfg["material_provenance"],
          warning="STL has no inherent units. Files use micrometre coordinates. Not a calibrated fabrication/exposure recipe."))
    # Dimensioned cross section and exterior view of exactly the exported design.
    h=np.asarray(design["heights_um"])
    radius=cfg["cylinder_diameter_um"]/2
    H=design["H_um"]
    edges=np.linspace(0,radius,len(h)+1)
    xr=np.r_[-edges[::-1],edges[1:]]
    hr=np.r_[h[::-1],h]
    fig=plt.figure(figsize=(10,5))
    ax=fig.add_subplot(121)
    ax.stairs(H+hr,xr,baseline=0,fill=True,color="#64a7bf",alpha=.85)
    ax.axhline(0,color="#656565",lw=3,label="fiber facet z=0")
    ax.axhline(H,color="k",ls="--",lw=.7)
    ax.axhline(design["output_reference_z_um"],color="#aa5533",ls=":",label="output reference")
    ax.annotate(f"H={H:g} µm",xy=(-radius-2,H/2),rotation=90,va="center")
    ax.set(xlabel="transverse position (µm)",ylabel="z above facet (µm)",title="Solid polymer pedestal + NN relief",xlim=(-radius-5,radius+5),ylim=(-3,H+7))
    ax.text(0,H/2,"polymer",ha="center");ax.text(radius+1,H+2,"air",ha="left",fontsize=8)
    ax.legend(fontsize=7,loc="lower center")
    ax=fig.add_subplot(122,projection="3d")
    vertices,faces=cylinder_mesh(h,H,radius,segments=64)
    ax.plot_trisurf(vertices[:,0],vertices[:,1],vertices[:,2],triangles=faces,color="#64a7bf",linewidth=0,shade=True)
    ax.set(xlabel="x (µm)",ylabel="y (µm)",zlabel="z (µm)",title="Connected geometry; optical target may fail")
    ax.set_box_aspect((48,48,H+3))
    fig.tight_layout();fig.savefig(folder/"assembly.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,metric in summary.items():
        p=metric["planes"][:5]
        axes[0].plot([v["z_um"] for v in p],[v["diameter_x_um"] for v in p],label=name)
    axes[0].axhspan(27,33,alpha=.1,color="green")
    axes[0].set(xlabel="distance after output reference (µm)",ylabel="x 1/e² beam diameter (µm)")
    axes[0].legend(fontsize=6)
    axes[1].bar(["bare fiber","flat cylinder","curvature relief","direct","NN"],[m["eta_mode"] for m in summary.values()])
    axes[1].axhline(.7,color="red",ls="--");axes[1].tick_params(axis="x",labelrotation=45)
    axes[1].set(ylabel="useful-mode efficiency",ylim=(0,1))
    for seed in cfg["training"]["seeds"]:
        r=json.loads((phase9/f"seed-{seed}-training.json").read_text())
        axes[2].plot([v["step"] for v in r["validation"]],[v["mean_loss"] for v in r["validation"]],label=str(seed))
    axes[2].set(xlabel="NN update",ylabel="validation loss");axes[2].legend(title="seed")
    fig.tight_layout();fig.savefig(folder/"comparison.png",dpi=170);plt.close(fig)
    planes=np.load(phase9/f"NN-r{summary['NN']['resolution_per_um']}-planes.npz")
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    extent=[planes["x_um"][0],planes["x_um"][-1]]*2
    intensity_scale=float(planes["intensity_output"].max())
    for axis,key,title in zip(axes,["intensity_output","intensity_100um","output_Ex"],["output intensity","100 µm downstream intensity","output Ex phase"]):
        data=planes[key]
        data=np.angle(data) if key=="output_Ex" else data/intensity_scale
        limits=(-np.pi,np.pi) if key=="output_Ex" else (0,1)
        picture=axis.imshow(data,extent=extent,origin="lower",cmap="twilight" if key=="output_Ex" else "magma",vmin=limits[0],vmax=limits[1])
        fig.colorbar(picture,ax=axis,shrink=.7,label="phase (rad)" if key=="output_Ex" else "I / output peak")
        axis.set(xlim=(-30,30),ylim=(-30,30),xlabel="x (µm)",ylabel="y (µm)",title=title)
    fig.tight_layout();fig.savefig(folder/"NN-fields.png",dpi=170);plt.close(fig)
    raw=json.loads(Path(summary["NN"]["record"]).read_text())
    field=np.load(raw["fields_file"])
    _,dx,cartesian=cylindrical_to_cartesian(field["r_um"],field["fields"][0],512,128)
    spectrum=forward_spectrum(cartesian,dx,cfg["wavelength_um"])
    angular=np.fft.fftshift(np.maximum(.5*np.real(spectrum[0]*np.conj(spectrum[4])-spectrum[1]*np.conj(spectrum[3])),0))
    spatial=np.fft.fftshift(np.fft.fftfreq(512,d=dx))*cfg["wavelength_um"]
    fig,ax=plt.subplots(figsize=(5,4))
    image=ax.imshow(np.log10(np.maximum(angular/angular.max(),1e-6)),extent=[spatial[0],spatial[-1]]*2,origin="lower",cmap="magma",vmin=-6,vmax=0)
    ax.set(xlim=(-.3,.3),ylim=(-.3,.3),xlabel="kx/k0",ylabel="ky/k0",title="NN angular power spectrum (log10 relative)")
    fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(folder/"NN-angular-spectrum.png",dpi=170);plt.close(fig)
    robustness=json.loads((phase9/"robustness.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[f"{r['parameter']}={r['value']}" for r in robustness]
    axes[0].bar(labels,[r["eta_mode"] for r in robustness])
    axes[0].axhline(.7,color="red",ls="--");axes[0].set(ylabel="proxy useful-mode efficiency",title="Assumed tolerance sweeps; not Maxwell validation")
    axes[1].bar(labels,[r["diameter_um"] for r in robustness])
    axes[1].axhspan(27,33,color="green",alpha=.1);axes[1].set(ylabel="proxy output x diameter (µm)")
    for axis in axes: axis.tick_params(axis="x",labelrotation=80,labelsize=6)
    fig.tight_layout();fig.savefig(folder/"robustness.png",dpi=170);plt.close(fig)
    nn=summary["NN"]
    lines=["# 1550 nm fiber cylinder metalens investigation","",
           f"**Engineering: {gate['engineering_gate']}. Preferred optical target: {gate['optical_gate']}.**","",
           "This is a synthetic investigation with a Gaussian equivalent-current input and assumed lossless materials. No experimental fiber or resin data were supplied.","",
           "The primary assembly is a 48 µm diameter solid polymer cylinder, H=35 µm, with a trained annular top relief. The 30–40 µm requirement refers to cylinder height; total polymer height and the output plane are recorded separately in the geometry manifest.","",
           f"The validation-selected NN seed is {selected}. Its converged reported output diameters are {nn['planes'][0]['diameter_x_um']:.3f} × {nn['planes'][0]['diameter_y_um']:.3f} µm; useful-mode efficiency is {nn['eta_mode']:.4f}, vector target overlap {nn['overlap']:.4f}, and maximum 100 µm diameter change {100*nn['maximum_diameter_change_100um']:.2f}%.","",
           "## Independent Maxwell comparison","","| Design | Output x/y diameter (µm) | Overlap | Useful efficiency | 100 µm change | Last mesh |","|---|---:|---:|---:|---:|---|"]
    for name,m in summary.items():
        p=m["planes"][0]
        lines.append(f"| {name} | {p['diameter_x_um']:.3f} / {p['diameter_y_um']:.3f} | {m['overlap']:.4f} | {m['eta_mode']:.4f} | {100*m['maximum_diameter_change_100um']:.1f}% | {'PASS' if gate['convergence'][name]['passed'] else 'FAIL'} |")
    lines.extend(["","The preferred optical target requires 27–33 µm x/y diameter, ≤5% diameter change over a provisional 100 µm processing interval, overlap ≥0.90, useful efficiency ≥0.70, and ≥90% of matched realizable baseline efficiency. Each failed criterion remains visible in `phase9/gate.json`.","",
                  "## Taller-cylinder comparator — outside the requested height, coarse only","",
                  "| Geometry | H (µm) | Output x/y diameter (µm) | Useful efficiency | 100 µm change |","|---|---:|---:|---:|---:|"])
    for name,value in json.loads((phase9/"taller-cylinder-comparison.json").read_text()).items():
        m=value["metrics"];p=m["planes"][0]
        lines.append(f"| {name} | {value['H_um']:.2f} | {p['diameter_x_um']:.3f} / {p['diameter_y_um']:.3f} | {m['eta_mode']:.4f} | {100*m['maximum_diameter_change_100um']:.1f}% |")
    lines.extend(["",
                  "## What the result supports","",
                  "A mechanical cylinder establishes the separation, but does not by itself expand the optical mode to 30 µm. Top-relief optimization can correct phase while retaining a much narrower illuminated footprint. The comparison tests this specific solid-cylinder/top-relief family; it does not prove that all distributed, hollow, or two-surface architectures are impossible.","",
                  "An explicitly modeled near-facet diverging stage followed by a top collimator is the next architecture to test within 35 µm. `phase10/design-decision.json` gives a paraxial two-surface calculation, including about −15.75 µm first-surface focal length and +49.30 µm second-surface focal length for an ideal air-filled 35 µm separation. These are conceptual thin-lens powers, not validated polymer shapes or a claim that the solid-cylinder design meets the target.","",
                  "The taller-cylinder cases are explicitly outside the 30–40 µm preference and were tested only at coarse Maxwell resolution. See `phase9/taller-cylinder-comparison.json`; do not call these converged accepted alternatives. Additional heldout pairs are likewise coarse diagnostics, not extra acceptance claims.","",
                  "## Correctness and limits","",
                  "Phase 8 records finite-difference gradients, axial/spatial sampling, measured input width, and flat-pedestal proxy/Maxwell discrepancy. Phase 9 records final-design sampling, mesh/PML/spacing/duration checks, and direct-monitor checks of propagated complex E/H fields. NN checkpoints contain nonzero Adam state and a verified next-update recovery check. Direct baselines have matched aggregate objective-forward counts, recorded separately from inference cost.","",
                  "The initial divergence implementation used an all-angle second moment, although the phase plan requests fitted divergence. The corrected implementation fits log angular intensity over the contiguous 1/e² core and its crossing neighbors, with a fixed fit-validity check tested against analytical Gaussian and non-Gaussian examples. Original moment values remain in `second_moment_divergence_half_angle_rad`; original failed comparisons are archived under `phase9/initial-second-moment-analysis/`. Weak scattered fields make the all-angle moments more mesh-sensitive. A core-fit convergence PASS does not assert convergence of those moments, M², or every electromagnetic field component. No beam-diameter, overlap, efficiency, or target threshold was relaxed.","",
                  f"An independent E/H field-power audit differs from the authoritative Meep flux-monitor transmission by at most {100*normalization['maximum_relative_discrepancy']:.2f}%. It exposes interpolation/finite-band uncertainty rather than silently renormalizing fields; reported transmission uses the flux monitors. See `phase10/power-normalization-audit.json`. The input-mode overlap and measured reference-plane diameter are in `phase10/input-mode-audit.json`.","",
                  "MFD and nominal NA do not uniquely specify a real fiber eigenmode. The injected Gaussian is checked in a homogeneous reference; its measured width differs slightly from the nominal MFD. The facet must accommodate the 48 µm support footprint; the actual cladding diameter and attachment process were not supplied. The one-way proxy omits internal backward coupling. Maxwell is the independent authority. Conditional second-moment waist/M² fits are saved in `phase10/caustic-fits.json`; invalid fits carry reasons, not invented values. Long-range 500/1000 µm planes are diagnostics: inspect edge power for finite-window contamination. Robustness sweeps are proxy-only and use assumed tolerances. Offset, tilt, asymmetric supports, hollow cylinders, and an integrated second expansion stage were not validated.","",
                  "## Reproduce / resume","","From the project directory in WSL:","","```bash",
                  "bash scripts/run_collimation.sh phase8","bash scripts/run_collimation.sh train",
                  "bash scripts/run_collimation.sh validate","bash scripts/run_collimation.sh package",
                  "PYTHONPATH=src .venv-gpu/bin/python scripts/record_collimation_costs.py",
                  "PYTHONPATH=src .venv-gpu/bin/python -m pytest -q","```","",
                  "The train command resumes compatible checkpoints. Maxwell reuses only hashes matching configuration, design, numerical settings, and solver source. To reproduce from scratch, use `--root artifacts/1550nm_collimation-reproduction`; preserve the original run. No host/RSS/WSL RAM gates apply. GPU VRAM guards and timeouts remain active. The original 980 nm repair commands/configuration are separate.","",
                  "## Exports and provenance","",
                  "`phase10/geometry-manifest.json` records dimensions, micrometre STL coordinates, conversion to mm, connectedness, watertightness, and analytic-versus-mesh volume checks. Exported preferred geometry may fail optical targets; an STL is neither optical acceptance nor a calibrated fabrication recipe. See `source-manifest.json`, environment locks, frozen configuration, raw field NPZs, training checkpoints, and guard logs for provenance.","",
                  "![Assembly](phase10/assembly.png)","","![Comparison](phase10/comparison.png)","","![Fields](phase10/NN-fields.png)","",
                  "![Angular spectrum](phase10/NN-angular-spectrum.png)","","![Assumed robustness](phase10/robustness.png)","",
                  "Primary numerical references: [Meep cylindrical symmetry and azimuthal conventions](https://meep.readthedocs.io/en/latest/Python_Tutorials/Cylindrical_Coordinates/), [Meep near-to-far field principles](https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/), [Meep numerical-dispersion guidance](https://meep.readthedocs.io/en/latest/FAQ/), and the Gaussian propagation reference in the root phase plan. These describe numerical methods, not measured performance of this design."])
    # Sum executed solver costs once, distinct from cached validation invocation.
    resource=[]
    for path in root.glob("*-memory.jsonl.summary.json"):
        s=json.loads(path.read_text())
        resource.append(dict(stage=path.name,wall_s=s.get("wall_s"),peak_RSS_bytes=s.get("peak_project_rss_bytes"),
                             sampled_device_delta_bytes=s.get("peak_total_device_delta_bytes"),returncode=s.get("returncode"),stop_reasons=s.get("stop_reasons")))
    costs=[json.loads(p.read_text()) for p in (root/"maxwell").glob("*.json") if p.name!="geometry-ledger.json"]
    write(folder/"measured-costs.json",dict(guard_stages=resource,unique_solver_wall_s=sum(p.get("wall_s",0) for p in costs),
          attribution_note="RSS may double-count shared pages; device delta includes other applications; cached runs are not new solver cost"))
    taller=json.loads((phase9/"taller-cylinder-comparison.json").read_text())
    transit=[]
    for value in taller.values():
        record=json.loads(Path(value["metrics"]["record"]).read_text())
        transit.append(dict(H_um=value["H_um"],effective_after_sources_time=record["settings"]["duration"],
                            power_balance_error=record["power_balance_error"],
                            passed=record["power_balance_error"]<cfg["maxwell"]["balance_tolerance"]))
    if not all(item["passed"] for item in transit):
        raise RuntimeError("Taller comparison has unsettled power: run the bounded transit repair before packaging")
    write(folder/"fixes-and-verification.json",dict(
        divergence_definition=dict(problem="Acceptance used an all-angle moment instead of the requested Gaussian-core fit",
                                   fix="Validated fixed-interval log-intensity fit; original moments and failures preserved",
                                   verification="Analytic Gaussian/halo/non-Gaussian/undersampling regression tests plus unchanged raw Maxwell-field mesh comparisons",
                                   all_final_mesh_checks_pass=all(v["passed"] for v in gate["convergence"].values())),
        taller_time_of_flight=dict(problem="Fixed desktop tail and nominal round-trip estimate can miss the tall-device reflected pulse, including coarse-grid delay",
                                  fix="Minimum optical round trip followed by at most two 1.5x duration extensions if measured power closure fails; primary duration retained",verification=transit),
        cache_numeric_types=dict(problem="JSON cache keys distinguished equivalent integer/float numerical requests",
                                 fix="Search equivalent numerical representations under identical config/design/source hashes before launching a solver",
                                 verification="test_identical_integer_float_numerics_reuse_completed_solver_record"),
        radial_fixture=dict(problem="Gaussian reconstruction test expected nonzero values outside its supplied radial support",
                            fix="Extend fixture beyond Cartesian corners; physical zero-padding behavior retained",verification="test_cylindrical_linear_reconstruction"),
        resource_policy=dict(fix="Phase-scoped null host/RSS/WSL RAM gates while retaining VRAM limits",verification="test_phase8_ram_override_preserves_vram_controls and actual phase hardware/guard logs"),
        process_lifecycle=dict(controlled_restart="If present, initial-second-moment-analysis/controlled-restart.json and the original validate-memory log describe the deliberate validator restart; original cost is retained",unique_future_logs=True),
        optical_failure="Unmet diameter/stability/mode targets are retained as physical results of the tested geometry family, not relabeled by a code fix"))
    (root/"report.md").write_text("\n".join(lines)+"\n")
    paths=list(Path("src/fiber_metalens_nn").glob("*.py"))+list(Path("tests").glob("*.py"))+list(Path("locks").glob("*"))+[Path("configs/1550nm_collimation.json"),Path("configs/1550nm_collimation_smoke.json"),Path("scripts/run_collimation.sh"),Path("scripts/record_collimation_costs.py"),Path("README.md"),Path("../metalens_inverse_design_gpu_trial_prompts.md"),Path("../AGENTS.md"),Path("skills/metalens-phase-workflow/SKILL.md")]
    write(root/"source-manifest.json",{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()})
    result=dict(phase=10,status="PASS_PACKAGE_VERIFIED",scientific_engineering=gate["engineering_gate"],
                scientific_optical=gate["optical_gate"],geometry_checks_passed=all(v["mesh"]["passed"] for v in exports.values()),
                report=str(root/"report.md"),ram_gates=False)
    write(folder/"gate.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result
