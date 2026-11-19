"""Export the frozen NN cylinder and attached top relief as one true-size SLC.

Uses the original connected assembly mesh and the user builder's contour
functions, with interior sampling and corrected SLC minimum-Z serialization.
No optical optimization, printer execution or laser exposure job is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

import export_metalens_slc as core

ROOT = core.ROOT
CAMPAIGN = core.CAMPAIGN


def load_frozen_assembly(design_path):
    design = json.loads(Path(design_path).read_text(encoding='utf-8-sig'))
    source_sha = core.digest(design_path)
    geometry_manifest = CAMPAIGN/'geometry_exports/geometry_manifest.json'
    expected = json.loads(geometry_manifest.read_text(encoding='utf-8'))['NN']
    if source_sha != expected['source_sha256']:
        raise ValueError('This exporter requires the frozen NN source recorded by the assembly manifest')
    for key in ('heights_um', 'H_um', 'cylinder_diameter_um'):
        if design[key] != expected[key]:
            raise ValueError(f'Frozen design differs from the assembly manifest: {key}')
    h = np.asarray(design['heights_um'], dtype=float)
    H, diameter = float(design['H_um']), float(design['cylinder_diameter_um'])
    if (not np.isfinite(h).all() or np.any(h < 0) or np.any(np.diff(h) > 1e-6)
            or not np.isfinite(H) or H <= 0 or not np.isfinite(diameter) or diameter <= 0):
        raise ValueError('Positive cylinder and finite nonincreasing annular relief required')
    # This is one radial height function, not separately capped/overlapping bodies.
    expected_v, expected_f = core.relief_mesh(H+h, diameter/2, segments=128)
    mesh_path = CAMPAIGN/'geometry_exports/NN_mesh.npz'
    with np.load(mesh_path, allow_pickle=False) as mesh:
        v, f = mesh['vertices_um'].copy(), mesh['faces'].astype(np.int64)
    if (v.shape != expected_v.shape or f.shape != expected_f.shape
            or not np.allclose(v, expected_v, rtol=0, atol=1e-12)
            or not np.array_equal(f, expected_f)):
        # Compare actual oriented triangles if the frozen mesh only has a
        # different vertex numbering. No geometric change is accepted.
        def triangles(vertices, faces):
            result = []
            for tri in vertices[faces]:
                pts = [tuple(float(x) for x in p) for p in tri]
                result.append(min(tuple(pts[i:]+pts[:i]) for i in range(3)))
            return sorted(result)
        if triangles(v, f) != triangles(expected_v, expected_f):
            raise ValueError('Frozen assembly mesh differs from the exact connected H+relief profile')
    return design, v, f, {
        'source_design': str(Path(design_path).resolve()), 'source_design_sha256': source_sha,
        'source_assembly_mesh': str(mesh_path.resolve()), 'source_assembly_mesh_sha256': core.digest(mesh_path),
        'source_geometry_manifest': str(geometry_manifest.resolve()),
        'source_geometry_manifest_sha256': core.digest(geometry_manifest),
        'frozen_assembly_mesh_matches_connected_height_profile': True,
    }


def check_mesh(v, f):
    undirected = Counter(tuple(sorted((int(a), int(b)))) for tri in f for a,b in zip(tri,np.roll(tri,-1)))
    directed = Counter((int(a),int(b)) for tri in f for a,b in zip(tri,np.roll(tri,-1)))
    if not undirected or any(n != 2 for n in undirected.values()):
        raise ValueError('Assembly mesh must have exactly two incident faces per edge')
    if any(n != 1 or directed[(b,a)] != 1 for (a,b),n in directed.items()):
        raise ValueError('Assembly mesh has inconsistent directed edge winding')
    adjacency = {i:set() for i in range(len(v))}
    for a,b in undirected:
        adjacency[a].add(b); adjacency[b].add(a)
    seen={0}; pending=[0]
    while pending:
        for j in adjacency[pending.pop()]-seen:
            seen.add(j);pending.append(j)
    tri=v[f]
    doubled_areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    signed_volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    if len(seen)!=len(v) or np.min(doubled_areas)<=1e-12 or signed_volume<=0:
        raise ValueError('Assembly must be connected with nondegenerate positive-volume surface')
    if len(v)-len(undirected)+len(f)!=2:
        raise ValueError('Expected one closed genus-zero connected assembly')
    return {'watertight':True,'single_connected_component':True,'consistent_surface_winding':True,
            'no_degenerate_triangles':True,'genus_zero':True,'positive_volume':True,
            'vertices':len(v),'triangles':len(f),'mesh_volume_um3':signed_volume,
            'dimensions_um':np.ptp(v,axis=0).tolist()}


def preview(parsed, design, destination):
    """Dimensioned assembly view reconstructed from the serialized contours."""
    from PIL import Image, ImageDraw, ImageFont
    font_path='C:/Windows/Fonts/arial.ttf'
    font=lambda size:ImageFont.truetype(font_path,size)
    im=Image.new('RGB',(1600,1040),'white');dr=ImageDraw.Draw(im)
    H=float(design['H_um']);radius=float(design['cylinder_diameter_um'])/2
    top=parsed['top_mm']*1000
    dr.text((55,30),'Cylinder + attached NN metalens | connected solid',fill='#172a3a',font=font(34))
    dr.text((55,85),f'Cylinder: {2*radius:g} um diameter x {H:g} um high    Relief: up to {top-H:g} um    Total: {top:g} um',fill='black',font=font(24))
    dr.text((55,123),'Actual scale in SLC: mm | Layer interval: 0.05 um | Assembly base z = 0 at the intended fiber facet',fill='black',font=font(22))
    x_center,base,scale=415,865,15
    to_xy=lambda x,z:(x_center+x*scale,base-z*scale)
    for i,(z,loops) in enumerate(parsed['layers']):
        lower=z*1000;upper=(parsed['layers'][i+1][0] if i+1<len(parsed['layers']) else parsed['top_mm'])*1000
        r=max(float(np.linalg.norm(q,axis=1).max()) for q in loops)*1000
        color='#75b5c6' if lower < H-1e-5 else '#e3a259'
        dr.rectangle((*to_xy(-r,upper),*to_xy(r,lower)),fill=color)
    dr.line((*to_xy(-radius,H),*to_xy(radius,H)),fill='#687c85',width=2)
    for z in (0,H,top):
        x,y=to_xy(-radius,z)
        dr.line((x-12,y,x-2,y),fill='black',width=2)
        dr.text((x-55,y-14),f'{z:g}',fill='black',font=font(20))
    dr.text((45,235),'z (um)',fill='black',font=font(21))
    dr.line((x_center-radius*scale,base+30,x_center+radius*scale,base+30),fill='black',width=2)
    dr.text((x_center-95,base+45),f'{2*radius:g} um diameter',fill='black',font=font(22))
    dr.text((135,base+90),'Facet plane (fiber itself is excluded)',fill='black',font=font(21))
    dr.text((880,220),'XY sections from the final SLC',fill='black',font=font(25))
    for slot,z_wanted in enumerate((0,H,H+1.5,H+2.95)):
        i=min(range(len(parsed['layers'])),key=lambda j:abs(parsed['layers'][j][0]*1000-z_wanted))
        z,loops=parsed['layers'][i];cx=1000+(slot%2)*350;cy=435+(slot//2)*360
        dr.text((cx-145,cy-162),f'Lower z = {z*1000:.2f} um',fill='black',font=font(21))
        for q in loops:
            pts=[(cx+x*1000/radius*125,cy-y*1000/radius*125) for x,y in q]
            dr.polygon(pts,fill='#d9e9ed',outline='#35647a',width=2)
    dr.text((55,995),'Geometry only: printer import, laser paths, resin exposure and shrinkage require receiving-system validation.',fill='black',font=font(21))
    im.save(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=core.DEFAULT_DESIGN)
    parser.add_argument('--builder',type=Path,default=core.DEFAULT_BUILDER)
    parser.add_argument('--output',type=Path,default=ROOT/'exports/cylinder_attached_slc_v1')
    parser.add_argument('--layer-um',type=float,default=.05)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output folder; preserve all earlier exports')
    design,v,f,provenance=load_frozen_assembly(args.design)
    h=np.asarray(design['heights_um'],float);H=float(design['H_um']);diameter=float(design['cylinder_diameter_um'])
    total=H+h
    if (not np.isfinite(args.layer_um) or args.layer_um<=0
            or np.max(np.abs(total/args.layer_um-np.round(total/args.layer_um)))>1e-4
            or abs(H/args.layer_um-round(H/args.layer_um))>1e-4):
        raise ValueError('Layer interval must preserve cylinder and all frozen relief step heights')
    mesh_audit=check_mesh(v,f)
    builder=core.load_builder(args.builder)
    source_mesh=builder.slc_mesh_from_arrays(v*.001,f)
    options=builder.slc_options(layer_um=args.layer_um,z_origin='preserve',xy_origin='preserve',
                                xy_mm_per_um=.001,z_mm_per_um=.001)
    args.output.mkdir(parents=True)
    audit_dir=args.output/'audit';audit_dir.mkdir()
    staging=audit_dir/'builder_midpoint_samples';staging.mkdir()
    (audit_dir/'builder_slc_function_subset.py.txt').write_text(builder.subset_source,encoding='utf-8')
    np.savez_compressed(audit_dir/'cylinder_metalens_mesh.npz',vertices_um=v,faces=f)
    result=builder._convert_stl_to_slc_only(None,staging/'C01.slc',options,source_mesh=source_mesh)
    slc=args.output/'C01.slc'
    parsed=core.write_min_z_slc(Path(result['slc']),slc,args.layer_um,package_name='FiberCylinderMetalensMinZ1')
    # Check the full solid H+h, not a relief translated to the pedestal top.
    slicing_geometry={'heights_um':total.tolist(),'cylinder_diameter_um':diameter}
    internal,_=core.verify(slc,slicing_geometry,v,f,128,args.layer_um)
    builder.slc_readback(slc)
    internal.update(component='cylinder_with_attached_metalens',cylinder_included=True,
                    analytic_circular_assembly_volume_um3=internal.pop('analytic_circular_relief_volume_um3'))
    if (core.digest(args.design)!=provenance['source_design_sha256']
            or core.digest(args.builder)!=builder.source_sha256
            or core.digest(Path(provenance['source_assembly_mesh']))!=provenance['source_assembly_mesh_sha256']
            or core.digest(Path(provenance['source_geometry_manifest']))!=provenance['source_geometry_manifest_sha256']):
        raise RuntimeError('Frozen input changed during export')
    source_builder_manifest=json.loads(Path(result['slc_manifest']).read_text(encoding='utf-8'))
    for i,layer in enumerate(source_builder_manifest['layers']):
        layer.update(sampled_at_z_um=float(layer['z_original_um']),
                     z_mm=float(np.float32(i*args.layer_um*.001)),
                     z_original_mm=float(np.float32(i*args.layer_um*.001)),
                     z_original_um=float(np.float32(i*args.layer_um*.001))*1000)
    manifest={**provenance,'created_utc':datetime.now(timezone.utc).isoformat(),
        'component':'cylinder_with_attached_metalens','cylinder_included':True,
        'architecture':'one_connected_solid_cylinder_and_top_annular_relief',
        'source_design_unchanged':True,'builder_source':str(args.builder.resolve()),
        'builder_source_sha256':builder.source_sha256,'extracted_builder_functions':builder.extracted_functions,
        'source_mesh_units':'um','slc_units':'MM','mesh_um_to_mm_conversion_count':1,
        'builder_scale_overrides_mm_per_um':{'xy':.001,'z':.001},
        'builder_default_enlargement_avoided':{'xy':240,'z':100},
        'coordinate_frame':'assembly: optical axis x=y=0; flat bottom z=0 at intended fiber facet',
        'frame_transform':'identity in micrometres; convert all coordinates to mm once',
        'cylinder_diameter_um':diameter,'cylinder_height_um':H,
        'nonzero_top_relief_diameter_um':float(np.flatnonzero(h>0)[-1]+1)*diameter/len(h),
        'maximum_relief_um':float(h.max()),'total_height_um':float(total.max()),
        'dimensions_um':np.ptp(v,axis=0).tolist(),'layer_interval_um':args.layer_um,
        'layer_count':len(parsed['layers']),'first_minimum_z_um':parsed['layers'][0][0]*1000,
        'last_minimum_z_um':parsed['layers'][-1][0]*1000,'terminal_z_um':parsed['top_mm']*1000,
        'slice_convention':'interior midpoint XY samples serialized at slab minimum/lower Z; forward-held solid',
        'azimuthal_segments':128,'single_material':True,'internal_cylinder_relief_cap_or_seam':False,
        'fiber_substrate_raft_extra_support_included':False,
        'builder_midpoint_file_non_deliverable':str(Path(result['slc']).resolve()),
        'builder_midpoint_file_sha256':core.digest(Path(result['slc'])),
        'output_slc':str(slc.resolve()),'slc_sha256':core.digest(slc),'bytes':slc.stat().st_size,
        'mesh_audit':mesh_audit,'exporter_geometry_checks':internal,
        'independent_audit_required':'audit/independent_audit.json with --component assembly; not satisfied by writer readback',
        'layers':source_builder_manifest['layers'],
        'optical_status':'Unchanged frozen synthetic natural-size assembly; curvature convergence unresolved; original30um target unmet.',
        'printer_import_tested':False,'exposure_calibration_verified':False,
        'limitations':['SLC contour geometry only, not calibrated two-photon laser instructions.',
                       'Printer/resin/objective, power/speed/hatching, voxel compensation and shrinkage remain unverified.',
                       '0.05um is source quantization/slice interval, not a claim of receiving-printer resolution.',
                       'This exporter is specific to the frozen NN source and assembly mesh.']}
    (args.output/'C01_slc_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (audit_dir/'exporter_verification.json').write_text(json.dumps(internal,indent=2)+'\n',encoding='utf-8')
    preview(parsed,design,args.output/'assembly_preview.png')
    (args.output/'README.txt').write_text(
        'CYLINDER + ATTACHED NN METALENS\n\n'
        'Deliverable: C01.slc. Single connected solid, cylinder diameter48um, height35um,\n'
        'top relief0..3um, total height38um. Nonzero top relief footprint34um.\n'
        'Flat assembly base z0; optical axis x=y0; fiber/substrate/raft excluded.\n'
        'Frozen original assembly mesh and design unchanged. No internal cap atz35.\n\n'
        'SLC2.0 PART, MM, true physical size, little-endian float32/uint32.\n'
        f'Layer interval{args.layer_um:g}um; {len(parsed["layers"])} lower-bound records; terminal top38um.\n'
        'Interior samples are serialized at lower bounds, not midpoint sample Z.\n'
        'audit/builder_midpoint_samples contains non-deliverable intermediate output.\n'
        'Require the separate --component assembly independent audit for acceptance.\n\n'
        'This is contour geometry for a compatible two-photon workflow. Printer import,\n'
        'laser power, speed, hatch, objective/resin, compensation and shrinkage are untested.\n'
        'Latest assembly optical evidence is synthetic, with unresolved curvature convergence;\n'
        'the earlier30um beam aspiration remains unmet. No new optics or printing performed.\n',encoding='utf-8')
    print(json.dumps({'slc':str(slc),'sha256':manifest['slc_sha256'],
                      'dimensions_um':manifest['dimensions_um'],'layers':len(parsed['layers']),
                      'mesh_audit':mesh_audit,'independent_audit':'required'},indent=2))


if __name__=='__main__':
    main()
