"""Adaptive-aware collector adapter; unchanged reviewed stage3 scientific core."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
CORE_PATH=ROOT/'scripts/assess_height_family_stage3_v1.py'
CORE_SHA='77dae64d7a81f78ec9e338e2d3e1c576756a0aa7c91977ecbba89cc87db2d08f'
SCHEMA='height_family_stage3_v2'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_core():
    if sha(CORE_PATH)!=CORE_SHA:raise ValueError('Unchanged reviewed stage3 scientific assessment core required')
    spec=importlib.util.spec_from_file_location('stage3_reviewed_assessment_core_v1',CORE_PATH)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def add_execution_diagnostics(result,collected,core):
    result['adaptive_execution']=collected['adaptive_execution']
    result['continuation_execution']=core.aggregate({'primary_computation':collected['primary_execution'],
                                                   'selected_adaptive_computation':collected['adaptive_execution']})
    result['selected_adaptive_diagnostics']=collected['selected_adaptive_diagnostics']
    result['execution_completion_is_not_optical_acceptance']=True
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--partial-reason')
    args=parser.parse_args();core=load_core();science=core.load_science()
    folder=science.local_path(args.folder).resolve();output=science.local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve previous assessment; choose a new version')
    if not output.is_relative_to(folder/'assessment') or output==folder/'assessment':raise ValueError('Output must be below stage3/assessment/version')
    from collect_height_family_stage3_v2 import collect_stage3
    collected=collect_stage3(folder)
    if collected['primary_execution']['status']!='PASS' and not (args.partial_reason or '').strip():
        raise RuntimeError('Primary is incomplete; explicit root-requested --partial-reason required')
    result=core.assess_data(collected['data'],collected['numerical_limits'],collected['authorization'],
                            collected['budget'],collected['primary_execution'],science)
    result.update(assessment_schema=SCHEMA,assessment_utility_sha256=sha(__file__),assessment_core_sha256=CORE_SHA,
        partial_assessment_reason=args.partial_reason,
        analysis_source_sha256={str(p):sha(p) for p in (Path(__file__),CORE_PATH,ROOT/'scripts/collect_height_family_stage3_v2.py')})
    add_execution_diagnostics(result,collected,core)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'assessment.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:
        stream.write('Computation/analysis completion only: '+result['continuation_execution']['status']+'\n'
            'Execution completion does not establish optical acceptance. Selected diagnostics retain separate native, numerical and nominal outcomes.\n')
        for row in result['selected_adaptive_diagnostics']:
            stream.write(f"Adaptive {row['pair_id']}: native={row['native_status']}; numerical={row['numerical_status']}; target nominal={row['target_nominal_status']}; Q crossing={row.get('Q_gate_crossing','unresolved')}\n")
        stream.write(core.human_report(result))
    print(json.dumps({'output':str(output),'original_family':result['original_seven_height_family_acceptance']['status'],
        'all_eight_heights':result['all_eight_nominal_heights_acceptance']['status'],'optical_solves_launched':0}))
    return 0

if __name__=='__main__':raise SystemExit(main())
