"""Recovery-aware publication wrapper; same frozen scientific rules/layout.

Original ledger/time acceptance stays non-PASS. Newly authenticated83 fields
replace older67 even if native/nominal/mesh gates fail. No optical computation.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import types

ROOT=Path(__file__).resolve().parents[1]
ALIAS_SHA='79c421c77ea49dfbd1d468e635e01c17b317664359e32bd3c9785be411bf6a08'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_alias():
    path=ROOT/'scripts/plot_height_family_stage3_v3.py'
    if sha(path)!=ALIAS_SHA:raise ValueError('Reviewed canonical training-alias mapper changed')
    spec=importlib.util.spec_from_file_location('recovery_pinned_alias_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def recovery_footer(context):
    date=datetime.fromisoformat(context['postprocessing']['started_utc'].replace('Z','+00:00'))
    day=date.astimezone(timezone(timedelta(hours=8))).strftime('%d %b')
    return f'{day} postprocessing; controller exit unrecorded. Scoped optics only; no workflow PASS.'


def decorate_recovery_footer(fig,context):
    """Use the existing lower text slots; curves/axes/fonts stay unchanged."""
    for text in list(fig.texts):
        x,y=text.get_position()
        if abs(y-.02)<1e-8:text.set_position((x,.038))
        elif abs(y-.06)<1e-8:text.set_position((x,.078))
    fig.text(.5,.006,recovery_footer(context),ha='center',fontsize=8.5)


def render_recovery(data,output,science,layout,context):
    original_save=layout.save_formats
    def save(fig,destination,stem):
        decorate_recovery_footer(fig,context);return original_save(fig,destination,stem)
    # Clone only this function's namespace; the pinned module is never mutated.
    scope=dict(layout.render_publication.__globals__);scope['save_formats']=save
    render=types.FunctionType(layout.render_publication.__code__,scope,
        layout.render_publication.__name__,layout.render_publication.__defaults__)
    render(data,output,science)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True)
    parser.add_argument('--overlay',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--assessment-output',required=True);args=parser.parse_args()
    from collect_height_family_stage3_recovery_v1 import collect_recovered,write_assessment_outputs,modules
    alias=load_alias();layout=alias.load_layout();science=layout.load_science()
    folder=science.local_path(args.folder).resolve();output=science.local_path(args.output).resolve()
    assessment_output=science.local_path(args.assessment_output).resolve()
    if output.exists() or not output.is_relative_to(folder/'forward_comparison'):raise ValueError('Choose a fresh additive recovery comparison version')
    if assessment_output.exists() or not assessment_output.is_relative_to(folder/'assessment'):raise ValueError('Choose a fresh additive assessment version')
    own_source_sha=sha(__file__)
    collected=collect_recovered(folder,args.overlay)
    collected['data']['input_sha256'][str(Path(__file__).resolve())]=own_source_sha
    data=science.build_comparison(collected['data'],collected['numerical_limits'])
    selected={r['H_um']:r.get('Meep') for r in data['rows']}
    if selected.get(100.,{}).get('identity')!=folder.name+':83':raise RuntimeError('Newest authenticated83 must replace olderH100; no fallback')
    context=collected['recovery_context'];csv_data,proof=alias.csv_training_projection(data,folder,science)
    values=layout.csv_rows(csv_data,science)
    data.update(utility_sha256=layout.EXPECTED_SCIENTIFIC_SHA,publication_layout_sha256=alias.LAYOUT_SHA,
        recovery_plot_adapter_sha256=sha(__file__),recovery_collector_sha256=sha(ROOT/'scripts/collect_height_family_stage3_recovery_v1.py'),
        recovery_context=context,CSV_training_result_lineage=proof,
        scoped_saved_field_optical_checks=collected['assessment']['scoped_saved_field_optical_checks'],
        original_execution_and_current_clock_are_not_accepted=True,
        publication_scope='Verified saved fields; original execution83 exit/finish unknown; late postprocessing is separately authorized; no full workflow PASS')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'comparison.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (output/'proxy_vs_selected_Meep.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=layout.CSV_COLUMNS,lineterminator='\n');writer.writeheader();writer.writerows(values)
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(
        'RECOVERED SAVED-FIELD COMPARISON; NO NEW OPTICAL SOLVES\n'+recovery_footer(context)+'\n'
        'Original ledger83 stays started, exact exit/finishUTC unknown. Current original-clock budget is not PASS.\n'
        'Same frozen geometry within each proxy/Meep pair; distinct patterns across heights. All original failures remain.\n\n'+science.human_report(data))
    render_recovery(data,output,science,layout,context)
    if not all(science.sha(p)==digest for p,digest in data['input_sha256'].items()):raise RuntimeError('Recovery comparison input changed')
    manifest={'status':'RECOVERED_SAVED_FIELDS_LATE_POSTPROCESSING_NO_WORKFLOW_PASS','recovery_plot_source_sha256':sha(__file__),
        'frozen_scientific_source_sha256':layout.EXPECTED_SCIENTIFIC_SHA,'frozen_layout_sha256':alias.LAYOUT_SHA,
        'frozen_alias_mapper_sha256':ALIAS_SHA,'recovery_context':context,'input_sha256':data['input_sha256'],
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()},
        'new_optical_solves':0,'training_runs':0,'original_ledger_rewritten':False}
    with (output/'delivery_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    c,core=modules();collected['assessment']['input_sha256']=dict(data['input_sha256'])
    write_assessment_outputs(collected,assessment_output,c,core)
    print(json.dumps({'output':str(output),'assessment_output':str(assessment_output),'status':manifest['status'],'new_optical_solves':0}))


if __name__=='__main__':main()
