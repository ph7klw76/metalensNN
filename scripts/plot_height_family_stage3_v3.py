"""Stage3 training-alias adapter to unchanged v5 layout/v3 optical selection.

No new physics, metric, CSV mapping or solver. Actual reports require all
primary jobs complete unless root supplies an explicit partial-report reason.
Multiple authenticated copies of one training result are projected to its
canonical path only for CSV provenance. Full evidence inventories stay intact.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
LAYOUT_PATH=ROOT/'scripts/plot_height_family_nn_forward_publication_v5.py'
LAYOUT_SHA='8698d5cd6c70e60baa2361f67385cd0132df79816ad2edde257f9884135542ad'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_layout():
    if sha(LAYOUT_PATH)!=LAYOUT_SHA:raise ValueError('Unchanged pinned v5 publication layout required')
    spec=importlib.util.spec_from_file_location('stage3_pinned_publication_v5',LAYOUT_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def require_ready(collected,partial_reason=None):
    if collected['authorization']['status']!='PASS' or collected['budget']['status']!='PASS':
        raise RuntimeError('Verified stage3 authorization and original deadline/accounting required')
    if collected['primary_execution']['status']!='PASS' and not (partial_reason or '').strip():
        raise RuntimeError('Primary is incomplete; explicit root-requested --partial-reason required')


def csv_training_projection(data,folder,science):
    """Verify original plus byte-identical copies; never modify scientific data."""
    canonical=(science.local_path(folder).resolve().parent/'training_result.json').resolve()
    aliases=[];normalized={}
    for path,digest in data['input_sha256'].items():
        resolved=science.local_path(path).resolve()
        if resolved.name!='training_result.json':continue
        if resolved in normalized and normalized[resolved]!=digest:
            raise ValueError('Conflicting normalized training-result alias')
        normalized[resolved]=digest
        if science.sha(resolved)!=digest:raise ValueError('Training-result alias bytes changed')
        aliases.append({'input_inventory_key':path,'resolved_file':str(resolved),'sha256':digest})
    if canonical not in normalized:raise ValueError('Canonical campaign training result required for CSV')
    digest=normalized[canonical]
    if any(value!=digest for value in normalized.values()):
        raise ValueError('Training-result aliases do not share the canonical source SHA')
    projected={path:value for path,value in data['input_sha256'].items()
               if science.local_path(path).resolve().name!='training_result.json'}
    projected[str(canonical)]=digest
    proof={'status':'VERIFIED_IDENTICAL_TRAINING_RESULT_ALIASES_FOR_CSV_ONLY',
        'canonical_file':str(canonical),'canonical_sha256':digest,'authenticated_aliases':aliases,
        'full_comparison_and_delivery_input_inventory_preserved':True,
        'scientific_records_and_numeric_CSV_mapping_unchanged':True}
    return dict(data,input_sha256=projected),proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--partial-reason')
    args=parser.parse_args();layout=load_layout();science=layout.load_science()
    folder=science.local_path(args.folder).resolve();output=science.local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve prior outputs; choose a fresh stage3 version')
    if not output.is_relative_to(folder/'forward_comparison') or output==folder/'forward_comparison':
        raise ValueError('Output must be below stage3/forward_comparison/version')
    from collect_height_family_stage3_v2 import collect_stage3
    collected=collect_stage3(folder);require_ready(collected,args.partial_reason)
    data=science.build_comparison(collected['data'],collected['numerical_limits'])
    data.update(utility_sha256=layout.EXPECTED_SCIENTIFIC_SHA,publication_layout_sha256=LAYOUT_SHA,
        stage3_publication_adapter_sha256=sha(__file__),stage3_collector_sha256=sha(ROOT/'scripts/collect_height_family_stage3_v2.py'),
        authorization=collected['authorization'],primary_execution=collected['primary_execution'],adaptive_execution=collected['adaptive_execution'],
        selected_adaptive_diagnostics=collected['selected_adaptive_diagnostics'],partial_report_reason=args.partial_reason,
        publication_layout={'figure_width_in':layout.FIGURE_WIDTH_IN,'figure_height_in':layout.FIGURE_HEIGHT_IN,
            'raster_dpi':layout.RASTER_DPI,'formats':['png','svg','pdf'],'scientific_logic_changed':False,
            'minimum_font_pt':8.5,'patterns_differ_between_heights':True,'same_frozen_geometry_within_each_proxy_Meep_pair':True})
    csv_data,csv_proof=csv_training_projection(data,folder,science)
    csv_values=layout.csv_rows(csv_data,science)
    data['CSV_training_result_lineage']=csv_proof
    output.mkdir(parents=True,exist_ok=False)
    with (output/'comparison.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:
        stream.write('THREE-STAGE SAVED-EVIDENCE COMPARISON\n'
            'Unchanged v3 scientific selection, v5 layout and CSV mapping. Different patterns across heights; same frozen pattern within each height.\n'
            'Diameters are at distance0 after the H+4um monitor; Gaussian-core HALF-angles are existing fitted values in DEGREES.\n'
            f"Additional stage3 attempts={collected['budget']['additional_attempts']}; cumulative={collected['budget']['cumulative_attempts']}; original deadline unchanged.\n\n"+
            science.human_report(data))
    with (output/'proxy_vs_selected_Meep.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=layout.CSV_COLUMNS,lineterminator='\n');writer.writeheader();writer.writerows(csv_values)
    layout.render_publication(data,output,science)
    if sha(LAYOUT_PATH)!=LAYOUT_SHA or not all(science.sha(path)==digest for path,digest in data['input_sha256'].items()):
        raise RuntimeError('Pinned source or immutable evidence changed during export')
    manifest={'status':'STAGE3_DESCRIPTIVE_PUBLICATION_NO_OPTICAL_RUN','scientific_source_sha256':layout.EXPECTED_SCIENTIFIC_SHA,
        'publication_layout_sha256':LAYOUT_SHA,'stage3_adapter_sha256':sha(__file__),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()},
        'input_sha256':data['input_sha256'],'optical_solves_launched':0,'training_runs_launched':0,'scientific_logic_changed':False}
    manifest['CSV_training_result_lineage']=csv_proof
    with (output/'delivery_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'status':manifest['status'],'optical_solves_launched':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
