"""Additive 500 nm SLC export. Original optical designs and 50 nm exports are immutable.
Quantizes nominal source heights to nearest 0.5 um (ties upward); no optical solve.
"""
import argparse, copy, json, math
from pathlib import Path
import numpy as np
import export_height_family_attached_slc_v1 as e
import audit_metalens_slc as independent


def write_new(path,value):
    def convert(v):
        if isinstance(v,np.generic):return convert(v.item())
        if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [convert(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    e.write_new(path,convert(value))

def quantize(source):
    # Recover the explicitly quantized 50 nm design grid before tie handling.
    original=np.asarray(source['heights_um'],float)
    nominal=np.rint(original/.05)*.05
    if np.max(abs(nominal-original))>2e-6:raise ValueError('Not the frozen 50 nm source grid')
    indices=np.rint(original/.05).astype(int)
    coarse=((indices+5)//10)*.5
    effective=copy.deepcopy(source)
    effective['heights_um']=coarse.tolist()
    effective['kind']='fabrication_500nm_quantized_from_frozen_NN'
    effective['optical_status']='NOT_REVALIDATED_AFTER_500NM_SLICING'
    effective['fabrication_quantization']='nearest 0.5 um, nominal ties upward'
    n=len(original);r=float(source['cylinder_diameter_um'])/2
    edges=np.arange(n+1)*r/n
    areas=math.pi*np.diff(edges**2)
    base=float(source['H_um'])*math.pi*r*r
    difference=float(np.sum(areas*abs(coarse-original)))
    source_volume=float(base+np.sum(areas*original))
    details=dict(rule=effective['fabrication_quantization'],requested_spacing_um=.5,
        maximum_height_change_um=float(max(abs(coarse-original))),
        changed_annuli=int(np.count_nonzero(abs(coarse-original)>2e-6)),
        circular_solid_symmetric_difference_um3=difference,
        circular_solid_symmetric_difference_fraction=difference/source_volume,
        signed_circular_volume_change_um3=float(np.sum(areas*(coarse-original))),
        original_heights_um=original.tolist(),exported_heights_um=coarse.tolist(),
        optical_performance_of_coarse_geometry_verified=False)
    return effective,details


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Preserve all existing exports; fresh output required')
    records,protected,optics=e.authenticate_sources();core,assembly=e.load_helpers();builder=core.load_builder(e.BUILDER)
    protected[str(Path(e.__file__).resolve())]=e.sha(e.__file__)
    protected[str(Path(__file__).resolve())]=e.sha(__file__)
    out.mkdir(parents=True);sources=out/'fabrication_sources';sources.mkdir();rows=[]
    for record in records:
        H=record['H_um'];source=record['design'];effective,delta=quantize(source)
        path=sources/f'NN_H{H}_500nm_fabrication.json';write_new(path,effective)
        folder=out/f'H{H:03d}'
        manifest=e.export_component(effective,path,folder,builder,core,assembly,record['source']['filename'],.5,False)
        final=Path(manifest['final_slc'])
        result=independent.audit(final.read_bytes(),path.read_bytes(),.5,128,'assembly')
        result['provenance']=dict(slc_sha256=e.sha(final),effective_design_sha256=e.sha(path),original_design_sha256=record['source']['sha256'])
        write_new(folder/'audit/independent_500nm_audit.json',result)
        # Keep the unchanged independent auditor's mismatch result visible too.
        mismatch=independent.audit(final.read_bytes(),record['source']['path'].read_bytes(),.5,128,'assembly')
        write_new(folder/'audit/comparison_to_original_50nm_profile.json',mismatch)
        write_new(folder/'slicing_change_manifest.json',dict(original_source=str(record['source']['path']),
            original_source_sha256=record['source']['sha256'],selected_optical_record=record['source']['identity'],
            original_scoped_optical_evidence=optics,**delta))
        # Replace new-folder generic helper text only; no previous evidence is touched.
        (folder/'README.txt').write_text(f'H{H}: {final.name}, 500 nm (0.5 um) minimum-Z slabs.\nCylinder diameter 50 um; cylinder H={H} um; total top {manifest["total_height_um"]} um.\nFine relief heights are quantized to nearest 0.5 um (nominal ties upward). See slicing_change_manifest.json.\nIndependent geometry audit is against the declared 500 nm fabrication geometry, NOT unchanged optical source.\nOriginal 50 nm profiles and exports remain unchanged. Coarse optical performance, printer import and exposure remain unverified.\n',encoding='utf-8')
        rows.append(dict(H_um=H,slc=str(final),slc_sha256=e.sha(final),slc_bytes=final.stat().st_size,
            layers=manifest['layer_count'],total_height_um=manifest['total_height_um'],
            spacing_min_um=manifest['actual_slab_interval_min_um'],spacing_max_um=manifest['actual_slab_interval_max_um'],
            independent_geometry_status=result['status'],independent_checks=len(result['checks']),
            original_geometry_comparison_status=mismatch['status'],slicing_change=delta))
        if result['status']!='PASS':raise RuntimeError(result['failed_checks'])
    if any(e.sha(p)!=digest for p,digest in protected.items()):raise RuntimeError('Protected input changed')
    write_new(out/'delivery_gate.json',dict(status='PASS_DECLARED_500NM_FABRICATION_GEOMETRY',
        original_files_unchanged=True,protected_input_sha256=protected,outputs=rows,
        optical_or_training_runs=0,coarse_optical_performance_verified=False,printer_import_tested=False))
    (out/'VERIFICATION.txt').write_text('500 nm SLC export: independent declared fabrication-geometry checks PASS for all three files.\n'+
        '\n'.join(f'H{r["H_um"]}: {r["layers"]} slabs; top {r["total_height_um"]} um; {r["slc"]}' for r in rows)+
        '\nImportant: 500 nm slicing quantizes fine relief heights by up to 250 nm. Optical PASS of the original 50 nm geometry is not transferred. No optical solves or printing performed.\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()


