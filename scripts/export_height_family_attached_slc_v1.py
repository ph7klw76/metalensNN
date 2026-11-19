"""Export verified H40/75/100 frozen NN profiles with their solid cylinders.

Uses the current user builder's native minimum-Z convention unchanged. No
legacy midpoint correction, optical calculation, GUI, printer job or exposure
operation is invoked. Actual-size connected H+h(r) solids use 128-gon sections.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
STAGE=CAMPAIGN/'maxwell_validation_v3'
BUILDER=ROOT.parent.parent/'fiber_builder_gui_v10_fast.py'
BUILDER_SHA='d03f067d62b3592dd47cfae670a99ce7fbe4a9d48f66d1ddabe924c1532f1800'
CONFIG=STAGE/'config.json';CONFIG_SHA='ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734'
COMPARISON=STAGE/'forward_comparison/recovery_final_v1/comparison.json'
COMPARISON_SHA='8d54aad5ea03ddf3de21f35d0062a22464008ed5cd7919c02922bf93de0f778d'
DELIVERY_SHA='f602c5e0ec7157c81fc7b7f4ebf7d4226e1f784d67088ae21ef5391dbb9122f1'
ASSESSMENT=STAGE/'assessment/recovery_final_verification_v2/assessment.json'
ASSESSMENT_SHA='e120ec628e1add2eb61dffb23dc15f7ec7ac0d2c1b86bb19ea657bb246dd4e8b'
SOURCE_AUDIT=CAMPAIGN/'slc_source_independent_review_v1/review.json'
SOURCE_AUDIT_SHA='039412b97b8515c3905ece09f0f9a2b972e8cd0f8324c70e92e9d8e14a8e494f'
BUILDER_REVIEW=CAMPAIGN/'slc_builder_independent_review_v1/review.json'
BUILDER_REVIEW_SHA='91ad7945a353ea85fc38a11ee6b8302b0f04f40a20bb690c1c3b5ae888845ed3'
HELPERS={
 'scripts/export_metalens_slc.py':'2015a18d380c0d9344b1fcd4c97566d75cca37dbe898a4b6af598a9324c5420a',
 'scripts/export_cylinder_metalens_slc.py':'3a2e2205178929e3c09f4ebdf9ffa909fbf40ef3022ffff81c3d73ba3e607346',
 'scripts/audit_metalens_slc.py':'28e2bb508c6e2b89d057aeee23276f6186c519bb7ddfdfef25c6dd7c678c2164',
 'src/fiber_metalens_nn/geometry.py':'6da888fa1dbad634efe067ed7a26435696f47b6512f30f5bd3876e4a557baf4d'}
SOURCES={
 40:dict(path=CAMPAIGN/'maxwell_validation_v1/frozen_designs/NN_H40.json',sha256='16a765787ce3dd50ad34bc6e8f3dac89e0fd0e6c3723ec6460cfa8bbe1b0b362',identity='maxwell_validation_v1:43',filename='C01.slc'),
 75:dict(path=STAGE/'frozen_designs/NN_H75.json',sha256='7861a7d4a8409026b1d57ed20bb5fcb9e30d2dbe7b4a68b8214ae1ee9a458c69',identity='maxwell_validation_v3:69',filename='C02.slc'),
 100:dict(path=STAGE/'frozen_designs/NN_H100.json',sha256='0e83c24f7978f764c937b74f15b2d6e6f71da96d3ce320677e941b8cc94b3772',identity='maxwell_validation_v3:83',filename='C03.slc')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def local(value):
    value=str(value)
    if sys.platform=='win32' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif sys.platform!='win32' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load_helpers():
    for name,digest in HELPERS.items():
        if sha(ROOT/name)!=digest:raise ValueError('Reviewed SLC helper changed: '+name)
    sys.path.insert(0,str(ROOT/'scripts'))
    import export_metalens_slc as core
    import export_cylinder_metalens_slc as assembly
    return core,assembly
def validate_profile(design,layer_um):
    if design.get('units')!='um':raise ValueError('Source must explicitly use micrometres')
    heights=np.asarray(design['heights_um'],float);H=float(design['H_um']);D=float(design['cylinder_diameter_um'])
    if (heights.ndim!=1 or not len(heights) or not np.isfinite(heights).all() or np.any(heights<0)
            or np.any(np.diff(heights)>1e-6) or heights.max()<=0 or not math.isfinite(H) or H<=0 or not math.isfinite(D) or D<=0):
        raise ValueError('Finite positive cylinder and connected nonincreasing relief required')
    if (not math.isfinite(layer_um) or layer_um<=0 or abs(H/layer_um-round(H/layer_um))>1e-4
            or np.max(abs((H+heights)/layer_um-np.round((H+heights)/layer_um)))>1e-4):
        raise ValueError('Layer spacing must preserve cylinder and frozen relief steps')
    return heights,H,D
def authenticate_sources():
    protected={}
    def pin(path,digest):
        if sha(path)!=digest:raise ValueError('Frozen export provenance differs: '+str(path))
        protected[str(Path(path).resolve())]=digest
    for path,digest in [(CONFIG,CONFIG_SHA),(COMPARISON,COMPARISON_SHA),(COMPARISON.parent/'delivery_manifest.json',DELIVERY_SHA),
                        (ASSESSMENT,ASSESSMENT_SHA),(SOURCE_AUDIT,SOURCE_AUDIT_SHA),(BUILDER_REVIEW,BUILDER_REVIEW_SHA),(BUILDER,BUILDER_SHA)]:pin(path,digest)
    config=read(CONFIG);comparison=read(COMPARISON);assessment=read(ASSESSMENT);records=[]
    if comparison['config']!=config or config['cylinder_diameter_um']!=50. or config['materials']['polymer']['n']!=1.5263:
        raise ValueError('Verified configuration differs')
    for H,source in SOURCES.items():
        pin(source['path'],source['sha256']);design=read(source['path']);validate_profile(design,.05)
        row=next(r for r in comparison['rows'] if r['H_um']==H);point=next(r for r in assessment['NN_heights'] if r['H_um']==H)
        measured=row['Meep']
        if (design['H_um']!=H or design['cylinder_diameter_um']!=50. or len(design['heights_um'])!=50
                or measured['identity']!=source['identity'] or measured['design_sha256']!=source['sha256']
                or point['Meep']['identity']!=source['identity']
                or any(r[key]['status']!='PASS' for r in (row,point) for key in ('native','nominal_optical','selected_mesh'))):
            raise ValueError('Source does not match verified selected43/69/83 geometry/gates')
        for pathkey,digestkey in [('record_file','record_sha256'),('analysis_file','analysis_sha256')]:pin(local(measured[pathkey]),measured[digestkey])
        records.append(dict(H_um=H,design=design,source=source,selected_saved_record=measured,
            scoped_optical_status={key:row[key]['status'] for key in ('native','nominal_optical','selected_mesh')}))
    for name,digest in HELPERS.items():pin(ROOT/name,digest)
    return records,protected,dict(comparison_sha256=COMPARISON_SHA,assessment_sha256=ASSESSMENT_SHA,
        original_execution_and_current_clock_are_not_accepted=True,
        strict_all_eight_workflow_status=assessment['all_eight_nominal_heights_acceptance']['status'],
        scoped_selected_saved_field_checks='PASS for unchanged selected H40/H75/H100 designs; not printer or experimental acceptance')
def native_min_z_check(parsed,design,layer_um):
    heights,H,D=validate_profile(design,layer_um);top=H+float(heights.max());zs=np.asarray([z*1000 for z,_ in parsed['layers']])
    expected=round(top/layer_um)
    if (len(zs)!=expected or abs(zs[0])>1e-10 or abs(parsed['sampling'][0]*1000)>1e-10
            or abs(parsed['sampling'][1]*1000-layer_um)>1e-5
            or np.max(abs(zs-np.arange(expected)*layer_um))>1e-5 or abs(parsed['top_mm']*1000-top)>1e-5):
        raise ValueError('Current builder output must already use native minimum-Z; no legacy relabeling allowed')
    intervals=np.diff(np.r_[zs,parsed['top_mm']*1000])
    if np.any(intervals<=0) or max(abs(intervals-layer_um))>1e-5:raise ValueError('Serialized slab spacing differs from requested interval')
    return dict(layer_count=expected,first_minimum_z_um=float(zs[0]),last_minimum_z_um=float(zs[-1]),terminal_top_um=parsed['top_mm']*1000,
        binary_sampling_spacing_um=float(parsed['sampling'][1]*1000),actual_slab_interval_min_um=float(intervals.min()),
        actual_slab_interval_max_um=float(intervals.max()),terminal_slab_interval_um=float(intervals[-1]),maximum_slab_interval_error_um=float(max(abs(intervals-layer_um))))
def preview(parsed,design,vertices,faces,destination,layer_um):
    from PIL import Image,ImageDraw,ImageFont
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    heights,H,D=validate_profile(design,layer_um);top=H+float(heights.max());radius=D/2
    image=Image.new('RGB',(1800,1120),'white');draw=ImageDraw.Draw(image)
    draw.text((55,28),f'Frozen NN H{H:g}: cylinder + attached metalens',fill='#172a3a',font=font(34))
    draw.text((55,83),f'Cylinder D={D:g} um, H={H:g} um | Relief max={heights.max():g} um | Total={top:g} um',fill='black',font=font(25))
    draw.text((55,125),f'SLC: MM, actual size; {len(parsed["layers"])} minimum-Z slabs at {layer_um:g} um; facet base z=0',fill='black',font=font(22))
    baseline=910;cx=400;scale=min(650/top,510/D);project=lambda x,z:(cx+x*scale,baseline-z*scale)
    draw.text((135,190),'Cross-section from serialized SLC',fill='#172a3a',font=font(24))
    for i,(z,loops) in enumerate(parsed['layers']):
        lower=z*1000;upper=(parsed['layers'][i+1][0] if i+1<len(parsed['layers']) else parsed['top_mm'])*1000
        r=max(float(np.linalg.norm(q,axis=1).max()) for q in loops)*1000
        draw.rectangle((*project(-r,upper),*project(r,lower)),fill='#73b7c8' if lower<H-1e-5 else '#e3aa5a')
    draw.line((*project(-radius,H),*project(radius,H)),fill='#35647a',width=2)
    draw.line((cx-radius*scale,baseline+28,cx+radius*scale,baseline+28),fill='black',width=2)
    draw.text((cx-90,baseline+43),f'{D:g} um diameter',fill='black',font=font(21))
    for z in (0,H,top):
        x,y=project(-radius,z);draw.line((x-15,y,x-3,y),fill='black',width=2)
        draw.text((x-95,y-12 if z!=H else y+3),f'{z:g} um',fill='black',font=font(18))
    draw.text((110,1010),'Fiber facet z=0; fiber/substrate are not included',fill='black',font=font(20))
    draw.text((1010,190),'Source-derived connected exterior',fill='#172a3a',font=font(24))
    iso_scale=min(650/(top+.7*radius),610/(1.6*D));center=1250
    points=np.column_stack([center+.8*(vertices[:,0]-vertices[:,1])*iso_scale,baseline+(.34*(vertices[:,0]+vertices[:,1])-vertices[:,2])*iso_scale])
    triangles=vertices[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);view=np.asarray([1.,1.,.68])
    order=np.argsort(triangles.mean(axis=1)@view);light=np.asarray([-.3,.2,1.]);light/=np.linalg.norm(light)
    for index in order:
        normal=normals[index];normal/=np.linalg.norm(normal)
        if normal@view<=0:continue
        base=np.asarray([93,168,187]) if triangles[index,:,2].mean()<=H+1e-6 else np.asarray([218,158,76])
        rgb=tuple(np.clip(base*(.72+.28*max(0.,normal@light)),0,255).astype(int))
        draw.polygon([tuple(p) for p in points[faces[index]]],fill=rgb)
    draw.text((1020,1010),'One material body; no internal cap or extra support',fill='black',font=font(20))
    draw.text((55,1065),'Geometry only. Printer import, exposure, scan paths and resin shrinkage remain unverified.',fill='black',font=font(21))
    image.save(destination)
def export_component(design,source_path,output,builder,core,assembly,filename,layer_um=.05,make_preview=True):
    source_bytes=Path(source_path).read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest()
    if json.loads(source_bytes.decode('utf-8-sig'))!=design:raise ValueError('Parsed source bytes differ before export')
    heights,H,D=validate_profile(design,layer_um);vertices,faces=core.relief_mesh(H+heights,D/2,segments=128)
    mesh_checks=assembly.check_mesh(vertices,faces);output.mkdir(parents=True,exist_ok=False)
    audit=output/'audit';audit.mkdir();staging=audit/'native_builder';staging.mkdir()
    np.savez_compressed(audit/'connected_mesh_um.npz',vertices_um=vertices,faces=faces)
    (audit/'source_design_exact.json').write_bytes(source_bytes)
    options=builder.slc_options(layer_um=layer_um,z_origin='preserve',xy_origin='preserve',xy_mm_per_um=.001,z_mm_per_um=.001)
    result=builder._convert_stl_to_slc_only(None,staging/filename,options,source_mesh=builder.slc_mesh_from_arrays(vertices*.001,faces))
    actual_builder_path=Path(result['slc']);parsed=core.independent_read(actual_builder_path);z_checks=native_min_z_check(parsed,design,layer_um)
    final=output/filename
    with final.open('xb') as stream:stream.write(actual_builder_path.read_bytes())
    if sha(final)!=sha(actual_builder_path):raise ValueError('Native minimum-Z bytes changed during final copy')
    internal,parsed=core.verify(final,dict(heights_um=(H+heights).tolist(),cylinder_diameter_um=D),vertices,faces,128,layer_um)
    builder.slc_readback(final)
    internal.update(component='cylinder_with_attached_metalens',cylinder_included=True,
        analytic_circular_assembly_volume_um3=internal.pop('analytic_circular_relief_volume_um3'))
    if make_preview:preview(parsed,design,vertices,faces,output/'assembly_preview.png',layer_um)
    if sha(source_path)!=source_sha:raise RuntimeError('Source design bytes changed during export')
    manifest=dict(component='cylinder_with_attached_metalens',architecture='one_connected_solid_H_plus_h',
        source_design=str(Path(source_path).resolve()),source_design_sha256=source_sha,source_design_bytes_unchanged=True,
        cylinder_height_um=H,cylinder_diameter_um=D,annuli=len(heights),annular_width_um=D/2/len(heights),
        nonzero_top_relief_diameter_um=(int(np.flatnonzero(heights>0)[-1])+1)*D/len(heights),maximum_relief_um=float(heights.max()),total_height_um=H+float(heights.max()),
        dimensions_um=np.ptp(vertices,axis=0).tolist(),azimuthal_segments=128,layer_interval_um=layer_um,**z_checks,
        native_minimum_z_builder_output=True,legacy_min_z_relabeling_applied=False,actual_builder_output=str(actual_builder_path.resolve()),
        final_slc=str(final.resolve()),final_slc_sha256=sha(final),final_slc_bytes=final.stat().st_size,
        source_mesh_units='um',slc_units='MM',mesh_um_to_mm_conversions=1,scale_overrides_mm_per_source_um=dict(xy=.001,z=.001),
        assembly_base_um=0,internal_material_cap_or_duplicate_cylinder=False,fiber_substrate_extra_raft_included=False,
        builder_source_sha256=builder.source_sha256,extracted_builder_functions=builder.extracted_functions,
        mesh_checks=mesh_checks,exporter_internal_checks=internal,
        independent_audit_required='Run unchanged audit_metalens_slc.py --component assembly against this exact source and final binary',
        printer_import_tested=False,exposure_calibration_verified=False,optical_run_performed_by_export=False)
    write_new(output/'assembly_manifest.json',manifest);write_new(audit/'exporter_verification.json',internal)
    (output/'README.txt').write_text(f'H{H:g} attached NN assembly: {filename}\nD={D:g} um, cylinder H={H:g} um, total={H+heights.max():g} um; source and geometry unchanged.\nNative minimum-Z SLC2.0 PART/MM; {len(parsed["layers"])} slabs at {layer_um:g} um; base z=0 at fiber facet.\nNo fiber/substrate/raft/internal cap. Independent assembly audit is required.\nContour geometry only; printer import, laser settings, shrinkage and exposure are unverified.\n',encoding='utf-8')
    return manifest
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--layer-um',type=float,default=.05)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh batch output required; preserve all earlier exports')
    if args.layer_um!=.05:raise ValueError('This frozen batch uses the existing0.05um layer contract')
    core,assembly=load_helpers();records,protected,optics=authenticate_sources();protected[str(Path(__file__).resolve())]=sha(__file__);builder=core.load_builder(BUILDER)
    if builder.source_sha256!=BUILDER_SHA:raise ValueError('Current reviewed native-minimum-Z builder changed')
    args.output.mkdir(parents=True,exist_ok=False);outputs=[]
    software=args.output/'audit/software';software.mkdir(parents=True)
    (software/'builder_function_subset.py.txt').write_text(builder.subset_source,encoding='utf-8')
    write_new(software/'builder_constants_and_closure.json',dict(builder_source_sha256=builder.source_sha256,
        extracted_functions=builder.extracted_functions,original_literal_constants=builder.original_constants,
        native_minimum_z_output_expected=True,legacy_midpoint_relabeler_called=False))
    for record in records:
        H=record['H_um'];folder=args.output/f'H{H:03d}'
        manifest=export_component(record['design'],record['source']['path'],folder,builder,core,assembly,record['source']['filename'],args.layer_um)
        write_new(folder/'selected_source_provenance.json',dict(source=record['source']['identity'],source_design_sha256=record['source']['sha256'],
            selected_saved_field_checks=record['scoped_optical_status'],verification_context=optics))
        outputs.append(dict(H_um=H,file=str(Path(manifest['final_slc']).relative_to(args.output.resolve())),sha256=manifest['final_slc_sha256'],
            cylinder_diameter_um=manifest['cylinder_diameter_um'],cylinder_height_um=H,total_height_um=manifest['total_height_um'],layer_count=manifest['layer_count']))
    if any(sha(path)!=digest for path,digest in protected.items()):raise RuntimeError('Source/builder/verified evidence changed during export')
    (args.output/'README.txt').write_text('Frozen verified NN attached assemblies\n'+''.join(f"H{x['H_um']:g}: {x['file']} | D50um x total{x['total_height_um']:g}um | {x['layer_count']} minimum-Z layers\n" for x in outputs)+
        'Unchanged source geometry; native-minimum-Z builder; true-size MM. Independent assembly audits must accompany delivery.\nSynthetic scoped optical evidence is separate from printer import/exposure calibration and the preserved strict workflow failure.\n',encoding='utf-8')
    files={p.relative_to(args.output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_new(args.output/'batch_manifest.json',dict(status='EXPORTED_GEOMETRY_REQUIRES_INDEPENDENT_ASSEMBLY_AUDITS',created_utc=datetime.now(timezone.utc).isoformat(),
        exporter_source_sha256=sha(__file__),protected_input_sha256=protected,verification_context=optics,outputs=outputs,files=files,
        user_builder_GUI_opened=False,legacy_midpoint_fix_used=False,optical_or_training_runs=0,printing_or_calibrated_laser_job_performed=False))
    print(json.dumps(dict(status='BATCH_EXPORTED_GEOMETRY_ONLY_INDEPENDENT_AUDIT_REQUIRED',outputs=outputs),indent=2))
if __name__=='__main__':main()
