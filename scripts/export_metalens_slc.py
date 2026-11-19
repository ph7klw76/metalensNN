"""Export the frozen NN top relief only through the user's SLC builder.

No optics, optimization, cylinder, supports, printer execution or exposure job.
The SLC-only functions are loaded by AST so the builder's GUI/OpenCV pipeline
does not start and no dependency or original builder file needs modification.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import shutil
import struct
import sys
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fiber_metalens_nn.geometry import profile_polygons

# Project is inverse design/fiber_metalens_nn; the GUI lives beside inverse design.
DEFAULT_BUILDER = ROOT.parent.parent/'fiber_builder_gui_v10_fast.py'
CAMPAIGN = ROOT/'artifacts/1550nm_natural_nn_v1'
DEFAULT_DESIGN = CAMPAIGN/'designs/NN.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_builder(path):
    """Compile unchanged SLC function bodies with only their dependencies."""
    path = Path(path)
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    definitions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    needed = {'_convert_stl_to_slc_only', 'slc_mesh_from_arrays', 'slc_readback'}
    while True:
        extra = {x.id for name in needed for x in ast.walk(definitions[name])
                 if isinstance(x, ast.Name) and x.id in definitions}
        if extra <= needed:
            break
        needed |= extra
    # Only literal configuration constants; never execute module-level GUI code.
    constants = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name.startswith('SLC_') or name in {'UM_TO_MM','EXPORT_ORIGIN_MODE'}:
                constants[name]=ast.literal_eval(n.value)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed]
    namespace=dict(np=np, math=math, Path=Path, PureWindowsPath=PureWindowsPath,
                   hashlib=hashlib, json=json, struct=struct, shutil=shutil,
                   os=__import__('os'), sys=sys, **constants)
    # Resolve annotations lazily just as in the original builder.
    import __future__
    exec(compile(ast.Module(body=selected,type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    module=types.SimpleNamespace(**namespace)
    module.source_sha256=digest(path)
    module.extracted_functions=sorted(needed)
    module.subset_source='\n\n'.join(ast.get_source_segment(source,n) for n in selected)
    module.original_constants=constants
    return module


def relief_mesh(heights, aperture_radius_um, segments=128):
    """Revolve the exact nonzero relief outline, including a flat base cap."""
    h=np.asarray(heights,dtype=float)
    if h.ndim != 1 or not len(h) or not np.all(np.isfinite(h)) or np.any(h<0):
        raise ValueError('Finite nonnegative annular heights required')
    if segments<16 or segments%4:
        raise ValueError('Azimuthal segment count must be a multiple of four >=16')
    cfg={'annuli':len(h),'aperture_diameter_um':2*aperture_radius_um}
    outlines=profile_polygons(h,cfg)
    if len(outlines)!=1 or outlines[0][0][0]!=0:
        raise ValueError('This standalone export requires one connected central relief')
    vertices=[]
    rings=[]
    for r,z in outlines[0]:
        if r==0:
            rings.append([len(vertices)])
            vertices.append([0.,0.,z])
        else:
            rings.append(list(range(len(vertices),len(vertices)+segments)))
            vertices.extend([[r*math.cos(2*math.pi*j/segments),
                              r*math.sin(2*math.pi*j/segments),z] for j in range(segments)])
    faces=[]
    for i,a in enumerate(rings):
        b=rings[(i+1)%len(rings)]
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
    v,f=np.array(vertices,float),np.array(faces,np.int64)
    t=v[f]
    volume=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
    if volume<0:
        f=f[:,::-1].copy()
    return v,f


def independent_read(path):
    """Separate binary parser for size, endianness, closure and record checks."""
    raw=Path(path).read_bytes()
    end=raw.find(b'\r\n\x1a',0,2048)
    if end<0:
        raise ValueError('Missing header terminator')
    header=raw[:end].decode('ascii')
    if not all(k in header for k in ['-SLCVER 2.0','-UNIT MM','-TYPE PART']):
        raise ValueError('Unsupported version or units')
    pos=end+3+256
    if pos>=len(raw) or raw[pos]!=1:
        raise ValueError('Expected one sample-table entry')
    pos+=1
    def take(fmt):
        nonlocal pos
        size=struct.calcsize(fmt)
        if pos+size>len(raw):
            raise ValueError('Truncated record')
        v=struct.unpack_from(fmt,raw,pos)
        pos+=size
        return v
    table=take('<4f')
    if not all(math.isfinite(x) for x in table) or table[1]<=0 or table[2:]!=(0.,0.):
        raise ValueError('Invalid sampling entry')
    layers=[]
    while True:
        z,count=take('<fI')
        if not math.isfinite(z):
            raise ValueError('Nonfinite plane')
        if count==0xffffffff:
            if not layers or z<=layers[-1][0] or pos!=len(raw):
                raise ValueError('Invalid terminal record')
            return {'header':header,'sampling':table,'layers':layers,'top_mm':z}
        if count>10000 or (layers and z<=layers[-1][0]):
            raise ValueError('Invalid boundary count or plane order')
        contours=[]
        for _ in range(count):
            n,gaps=take('<II')
            if n<4 or n>1000000 or gaps!=0:
                raise ValueError('Invalid contour size or gap count')
            xy=np.array(take('<'+'f'*(2*n)),float).reshape(n,2)
            if not np.isfinite(xy).all() or not np.array_equal(xy[0],xy[-1]):
                raise ValueError('Contour not finite and closed')
            if np.any(np.all(np.diff(xy,axis=0)==0,axis=1)):
                raise ValueError('Zero-length edge')
            contours.append(xy)
        layers.append((z,contours))


def area(xy):
    return .5*np.sum(xy[:-1,0]*xy[1:,1]-xy[1:,0]*xy[:-1,1])


def write_min_z_slc(sampled_path, destination, layer_um, package_name='MetalensReliefMinZ2'):
    """Serialize midpoint-sampled contours at their SLC slab LOWER bounds.

    The user builder samples safely inside each slab, but stores those sample
    heights as record Z. SLC record Z is a minimum/lower bound, not a sample
    midpoint. Keep every XY vertex and use explicit lower bounds 0, dz, ... .
    """
    sampled=independent_read(sampled_path)
    if abs(sampled['sampling'][0]*1000-layer_um/2)>1e-5:
        raise ValueError('Expected the builder half-layer-first sample sequence')
    for i,(z,_) in enumerate(sampled['layers']):
        if abs(z*1000-(i+.5)*layer_um)>1e-5:
            raise ValueError('Builder sample sequence differs from declared slabs')
    if not package_name or not package_name.isascii() or any(c.isspace() for c in package_name):
        raise ValueError('SLC package name must be a nonempty ASCII token')
    header=sampled['header'].replace('SquareFiberScaled2',package_name)
    with Path(destination).open('xb') as stream:
        stream.write(header.encode('ascii')+b'\r\n\x1a')
        stream.write(bytes(256))
        stream.write(struct.pack('<B4f',1,0.,layer_um*.001,0.,0.))
        for i,(_,loops) in enumerate(sampled['layers']):
            stream.write(struct.pack('<fI',i*layer_um*.001,len(loops)))
            for xy in loops:
                stream.write(struct.pack('<II',len(xy),0))
                stream.write(xy.astype('<f4').tobytes())
        stream.write(struct.pack('<fI',sampled['top_mm'],0xffffffff))
    corrected=independent_read(destination)
    if len(corrected['layers'])!=len(sampled['layers']):
        raise ValueError('Record count changed during minimum-Z serialization')
    for before,after in zip(sampled['layers'],corrected['layers']):
        if len(before[1])!=len(after[1]) or any(not np.array_equal(a,b) for a,b in zip(before[1],after[1])):
            raise ValueError('XY geometry changed during minimum-Z serialization')
    return corrected


def verify(path, design, vertices_um, faces, segments, layer_um):
    p=independent_read(path)
    h=np.asarray(design['heights_um'],float)
    radius=design['cylinder_diameter_um']/2
    edges=np.linspace(0,radius,len(h)+1)
    polygon_factor=segments*math.sin(2*math.pi/segments)/2
    errors=[]
    radii=[]
    areas=[]
    zs=[]
    for index,(z,loops) in enumerate(p['layers']):
        if len(loops)!=1:
            raise ValueError('Expected exactly one contour for every metalens plane')
        xy=loops[0]*1000
        if area(xy)<=0:
            raise ValueError('Exterior winding is not counterclockwise')
        upper_z=(p['layers'][index+1][0] if index+1<len(p['layers']) else p['top_mm'])*1000
        # The record contains a slab lower bound; evaluate its cross section
        # strictly inside that slab, away from float32 step boundaries.
        active=h>((z*1000+upper_z)/2)
        expected_area=polygon_factor*np.sum((edges[1:]**2-edges[:-1]**2)*active)
        measured=area(xy)
        errors.append(abs(measured/expected_area-1))
        r=float(np.sqrt((xy*xy).sum(axis=1)).max())
        expected_radius=edges[1:][active][-1]
        if abs(r-expected_radius)>1e-5:
            raise ValueError('Slice radius disagrees with the frozen annular profile')
        radii.append(r)
        areas.append(float(measured))
        zs.append(z*1000)
    expected_count=round(float(h.max())/layer_um)
    assert len(zs)==expected_count
    assert abs(zs[0])<1e-12
    assert abs(p['sampling'][0])<1e-12
    assert np.max(abs(np.array(zs)-np.arange(len(zs))*layer_um))<1e-5
    assert abs(p['top_mm']*1000-h.max())<1e-5
    assert max(errors)<2e-6
    # SLC solid is held from each record's minimum Z to the next record/top.
    sliced_volume=float(np.dot(areas,np.diff(np.r_[zs,p['top_mm']*1000])))
    t=vertices_um[faces]
    mesh_volume=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
    exact=float(np.sum(math.pi*(edges[1:]**2-edges[:-1]**2)*h))
    assert abs(sliced_volume/mesh_volume-1)<2e-6
    # Directed edge and graph checks are independent of the SLC parser.
    uses=Counter(tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in faces for i in range(3))
    assert all(x==2 for x in uses.values())
    adjacent={i:set() for i in range(len(vertices_um))}
    for f in faces:
        for a,b in zip(f,np.roll(f,-1)):
            adjacent[int(a)].add(int(b));adjacent[int(b)].add(int(a))
    visited={0};pending=[0]
    while pending:
        for j in adjacent[pending.pop()]-visited:
            visited.add(j);pending.append(j)
    assert len(visited)==len(vertices_um)
    return {'status':'PASS','layer_count':len(zs),'boundaries_per_layer':1,
            'dimensions_um':np.ptp(vertices_um,axis=0).tolist(),
            'first_plane_um':zs[0],'last_plane_um':zs[-1],'terminal_z_um':p['top_mm']*1000,
            'maximum_cross_section_relative_area_error':max(errors),
            'standard_min_z_volume_um3':sliced_volume,'mesh_volume_um3':mesh_volume,
            'standard_min_z_volume_relative_error':abs(sliced_volume/mesh_volume-1),
            'minimum_z_semantics_pass':True,
            'analytic_circular_relief_volume_um3':exact,
            'azimuthal_mesh_volume_error_relative_to_circles':abs(mesh_volume/exact-1),
            'closed_manifold':True,'single_connected_component':True,
            'builder_readback_pass':True,'independent_binary_and_geometry_check_pass':True,
            'cylinder_included':False,'extra_base_or_support_added':False,
            'printer_import_tested':False,'exposure_calibration_verified':False,
            'slc_sha256':digest(path)},p


def preview(parsed, path):
    from PIL import Image,ImageDraw,ImageFont
    def font(size):
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    im=Image.new('RGB',(1500,830),'white')
    dr=ImageDraw.Draw(im)
    dr.text((50,30),'Frozen NN metalens relief only',fill='black',font=font(34))
    dr.text((50,82),'Actual size 34 µm diameter × 3 µm height   |   No cylinder   |   0.05 µm slice spacing',fill='black',font=font(23))
    layers=parsed['layers']
    # Cross section reconstructed from SLC boundary radii, not an optical simulation.
    x0,y0,x1,y1=80,235,740,650
    dr.text((80,170),'Slice envelope reconstructed from the SLC',fill='black',font=font(23))
    dr.line((x0,y1,x1,y1),fill='black',width=2)
    dr.line((410,y0,410,y1),fill='#cccccc',width=1)
    for index,(z,loops) in enumerate(layers):
        r=max(float(np.sqrt((q*q).sum(axis=1)).max()) for q in loops)*1000
        upper=(layers[index+1][0] if index+1<len(layers) else parsed['top_mm'])*1000
        lower_y=y1-z*1000/3*(y1-y0)
        upper_y=y1-upper/3*(y1-y0)
        dr.rectangle((410-r/17*310,upper_y,410+r/17*310,lower_y),fill='#6196ac')
    for z in [0,1,2,3]:
        y=y1-z/3*(y1-y0)
        dr.text((35,y-12),str(z),fill='black',font=font(20))
    for r in [-17,0,17]:
        x=410+r/17*310
        dr.text((x-17,y1+15),str(r),fill='black',font=font(20))
    dr.text((260,y1+55),'Radial position (µm)',fill='black',font=font(22))
    dr.text((80,210),'z (µm)',fill='black',font=font(20))
    for slot,index in enumerate([0,29,59]):
        cx,cy=900+(slot%2)*350,310+(slot//2)*315
        z,loops=layers[index]
        dr.text((cx-125,cy-142),f'Lower z = {z*1000:.3f} µm',fill='black',font=font(22))
        for xy in loops:
            pts=[(cx+x*1000/17*110,cy-y*1000/17*110) for x,y in xy]
            dr.polygon(pts,fill='#abcbd9',outline='#35647a',width=2)
    dr.text((50,770),'Geometry preview only. Printer import, hatching and resin exposure require machine-specific settings.',fill='black',font=font(22))
    im.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder',type=Path,default=DEFAULT_BUILDER)
    parser.add_argument('--design',type=Path,default=DEFAULT_DESIGN)
    parser.add_argument('--output',type=Path,default=ROOT/'exports/metalens_only_slc_v2')
    parser.add_argument('--layer-um',type=float,default=.05)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output folder; previous exports are preserved')
    source_hash=digest(args.design)
    design=json.loads(args.design.read_text(encoding='utf-8-sig'))
    h=np.array(design['heights_um'])
    if not np.isfinite(args.layer_um) or args.layer_um<=0 or np.max(abs(h/args.layer_um-np.round(h/args.layer_um)))>1e-4:
        raise ValueError('Layer spacing must preserve all quantized relief steps')
    geometry_manifest=CAMPAIGN/'geometry_exports/geometry_manifest.json'
    original=json.loads(geometry_manifest.read_text())['NN']
    if source_hash!=original['source_sha256'] or design['heights_um']!=original['heights_um']:
        raise ValueError('Frozen design does not match the existing geometry manifest')
    builder=load_builder(args.builder)
    v,f=relief_mesh(h,design['cylinder_diameter_um']/2)
    mesh=builder.slc_mesh_from_arrays(v*.001,f)
    options=builder.slc_options(layer_um=args.layer_um,z_origin='preserve',
        xy_mm_per_um=.001,z_mm_per_um=.001,xy_origin='preserve')
    args.output.mkdir(parents=True)
    audit=args.output/'audit';audit.mkdir()
    (audit/'builder_slc_function_subset.py.txt').write_text(builder.subset_source,encoding='utf-8')
    np.savez_compressed(audit/'metalens_only_mesh.npz',vertices_um=v,faces=f)
    staging=audit/'builder_midpoint_samples';staging.mkdir()
    result=builder._convert_stl_to_slc_only(None,staging/'M02.slc',options,source_mesh=mesh)
    sampled_slc=Path(result['slc'])
    slc=args.output/'M02.slc'
    write_min_z_slc(sampled_slc,slc,args.layer_um)
    original_manifest=Path(result['slc_manifest'])
    manifest_path=args.output/'M02_slc_manifest.json'
    verification,parsed=verify(slc,design,v,f,128,args.layer_um)
    builder.slc_readback(slc)
    if digest(args.design)!=source_hash or digest(args.builder)!=builder.source_sha256:
        raise RuntimeError('Source design or builder changed during export')
    manifest=json.loads(original_manifest.read_text())
    for index,layer in enumerate(manifest['layers']):
        lower_z=float(np.float32(index*args.layer_um*.001))
        layer.update(z_mm=lower_z,z_original_mm=lower_z,z_original_um=lower_z*1000,
                     sampled_at_z_um=layer['z_original_um'])
    manifest.update(
        output_slc=str(slc.resolve()),
        first_plane_mm=0.,last_plane_mm=float(np.float32((len(manifest['layers'])-1)*args.layer_um*.001)),
        first_plane_above_mesh_minimum_mm=0.,first_plane_above_mesh_minimum_source_um=0.,
        last_plane_below_mesh_maximum_mm=float(h.max()*.001-np.float32((len(manifest['layers'])-1)*args.layer_um*.001)),
        last_plane_below_mesh_maximum_source_um=float(h.max()-np.float32((len(manifest['layers'])-1)*args.layer_um*.001)*1000),
        slc_sha256=digest(slc),bytes=slc.stat().st_size,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_mesh='Frozen NN top relief only; cylinder excluded; mesh converted µm to mm once',
        source_design=str(args.design.resolve()),source_design_sha256=source_hash,
        builder_source=str(args.builder.resolve()),builder_source_sha256=builder.source_sha256,
        extracted_builder_functions=builder.extracted_functions,
        builder_default_scales_mm_per_um={'xy':.24,'z':.10},
        builder_defaults_overridden_to_physical_scale=True,
        component='metalens_top_relief_only',cylinder_included=False,
        original_assembly_cylinder_height_um=design['H_um'],
        export_origin_mode='metalens_relief_base_center',
        export_origin_definition='x=y=0 on optical axis; relief base z=0 corresponds to assembly z=35 µm',
        frame_transform='export_z_um = assembly_z_um - 35; no other geometry shift or scaling',
        declared_design_aperture_um=design['cylinder_diameter_um'],
        nonzero_relief_footprint_diameter_um=float(np.ptp(v[:,0])),
        extra_base_support_or_raft_added=False,
        export_revision=2,
        slice_convention='SLC minimum-Z convention: sample contours inside slabs, serialize their lower bounds 0..2.95 µm; top=3 µm',
        builder_midpoint_serialization_corrected=True,
        builder_midpoint_file=str(sampled_slc.resolve()),builder_midpoint_file_sha256=digest(sampled_slc),
        correction_scope='Only serialized Z labels and package metadata change; every XY contour vertex is byte-equivalent after float32 encoding',
        geometry_verification=verification,
        optical_status='The reported optical investigation includes a 35 µm support. Standalone relief optics were not simulated. Latest assembly refinement retains unresolved curvature convergence.',
        limitations=[
            'SLC 2.0 PART, MM, +Z, little-endian float32 contours; no 240x/100x enlargement.',
            '128 azimuthal segments match the verified geometry export tessellation.',
            'Builder midpoint sampling is retained; serialized minimum-Z labels are corrected locally without changing the user builder.',
            'No cylinder, fiber, substrate, raft, support copy, 3DP or laser exposure job is included.',
            'Laser power, scan speed, hatch spacing, voxel/linewidth compensation and shrinkage compensation are not calibrated.',
            'The 0.05 µm slice spacing preserves the design height grid; it is not an assertion of printer resolution.',
        ])
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    (audit/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    preview(parsed,args.output/'slice_preview.png')
    (args.output/'README.txt').write_text(
        'METALENS ONLY SLC\n\n'
        'M02.slc contains only the frozen 1550 nm NN top relief. No cylinder, fiber, substrate, raft or supports.\n'
        'Nonzero footprint: 34 µm diameter. Maximum height: 3 µm.\n'
        'The original design aperture/support diameter is 48 µm, but the outer seven 1 µm radial annuli have zero relief.\n'
        'Frame: relief base at z=0; optical axis x=y=0. Original assembly base plane was z=35 µm.\n\n'
        'FORMAT AND SCALE\n'
        'SLC 2.0 PART with MM units. Actual extents: x/y=-0.017..+0.017 mm; z=0..0.003 mm.\n'
        'Slice spacing 0.05 µm = 0.00005 mm. 60 minimum-Z records at z=0..2.95 µm; terminal z=3 µm.\n'
        'Uses contour samples from fiber_builder_gui_v10_fast.py, with a local minimum-Z serialization correction.\n'
        'Each XY contour is sampled inside its slab, then stored at the slab lower boundary, as SLC specifies.\n'
        'The user builder file is unchanged. Its raw midpoint samples and manifest are retained under audit/.\n'
        'The builder defaults enlarge XY by 240x and Z by 100x; those enlargements are not applied.\n'
        'Polylines are explicitly closed; outer contours are counterclockwise viewed from +Z; no gaps.\n'
        '128 azimuthal segments retain the verified mesh tessellation. Exact original annular heights are preserved.\n\n'
        'VERIFICATION\n'
        'Builder readback and a separate binary parser pass. All 60 cross sections match the frozen annular profile.\n'
        'Mesh topology, winding, connectedness, dimensions, slice areas, standard minimum-Z volume and source hashes pass.\n'
        'Independent forward-hold solid verification is required; a midpoint-volume check alone is insufficient.\n'
        'audit/verification.json and M02_slc_manifest.json contain measured errors and provenance.\n\n'
        'CORRECTION FROM M01\n'
        'The original v1/M01.slc used midpoint coordinates as SLC minimum-Z labels. Its base and step transitions\n'
        'were therefore shifted by 0.025 µm under standard SLC interpretation. M02 corrects that metadata error.\n'
        'The original v1 file and its original audit remain preserved; use this v2/M02.slc for standard SLC import.\n\n'
        'TWO PHOTON PRINTING\n'
        'This is contour geometry for a compatible slicer, not a calibrated laser job. Configure resin/objective,\n'
        'laser power, scan speed, hatching, voxel compensation, interface alignment and shrinkage in your software.\n'
        'The original optical results are for the supported assembly; removing the support does not preserve\n'
        'those optical claims. The latest assembly curvature convergence remains unresolved.\n'
        'No printer or receiving application was started and no fabrication was performed.\n\n'
        'References\n'
        'SLC layout: user-supplied fiber_builder_gui_v10_fast.py, SLC conversion section.\n'
        'Nanoscribe CAD/import guidance: https://www.nanoscribe.com/en/contact-support/support/cad-model-creation/\n',encoding='utf-8')
    print(json.dumps({'slc':str(slc),'verification':verification},indent=2))


if __name__=='__main__':
    main()
