"""Offline comparison of fixed SLC500 proxy and saved Meep r24/r32 results.

This tool reads saved evidence and plots it; it imports neither Meep nor JAX,
launches no subprocess and never edits the controller, source designs or ledger.
Use a fresh --output folder. --allow-partial keeps pending values missing.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import struct

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
FOLDER=CAMPAIGN/'slc500_meep_validation_v2'
PROXY=CAMPAIGN/'frozen_NN_500nm_proxy_v1/results.json'
HEIGHTS=(40,75,100)
PINS={
 'scripts/run_frozen_NN_500nm_meep_v2.py':'c6872c3f3c796cacb05cbc4077d692015fe2bf276d8d8a7102179817e46c9aad',
 'scripts/report_height_family_maxwell_v2.py':'d8773663c8595fc35c6c942c7fda6d95f20ab69e1db9940b8c66ce0bcf447d12',
 'scripts/height_family_maxwell_worker_v2.py':'59fdb4dee1e7074ca17463e1495b87886563bc0ea2f393f8c539b8727b019008',
 'scripts/height_family_maxwell_analysis_v2.py':'4d1c2cd04c539660928f29d66bce3e701ef41a115512931070dd0f11a6dfc55b',
 'artifacts/1550nm_height_family_n15263_v2/frozen_NN_500nm_proxy_v1/results.json':'50d04099e4da0ece42c3a4e551b68aa52efb62393b850f440d32be5381da6041',
 'artifacts/1550nm_height_family_n15263_v2/slc500_meep_validation_v2/preparation_manifest.json':'13543c4db5d3a6ac9e4dd1e5270971779f1439e48c6cf08824fe9612fe087fe5',
 'artifacts/1550nm_height_family_n15263_v2/slc500_meep_validation_v2/approval_receipt.json':'972b2c2a6cde0f7a2e599eaa181d9b77b60a603babc7a50708ba027496e293fd',
}
CONFIG_SHA='ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734'
DRIVER_PIN=PINS['scripts/run_frozen_NN_500nm_meep_v2.py']
PROXY_ARM='existing_SLC500_half_up'
LIMITS={'diameter_relative_mesh_change_max':.02,'eta_relative_mesh_change_max':.05,
        'curvature_absolute_max':.1,'curvature_last_pair_delta_max':.005}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def local(path):
    text=str(path).replace('\\','/')
    if text.startswith('/mnt/') and os.name=='nt':text=text[5].upper()+':'+text[6:]
    elif len(text)>2 and text[1]==':' and os.name!='nt':text='/mnt/'+text[0].lower()+text[2:]
    return Path(text).resolve()
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def require_sha(path,expected,inventory):
    path=local(path)
    if sha(path)!=expected:raise ValueError('Saved evidence hash differs: '+str(path))
    inventory[str(path)]=expected
def contained(folder,name):
    path=local(Path(folder)/name)
    if not path.is_relative_to(local(folder)):raise ValueError('Saved evidence escapes its folder')
    return path
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def pass_status(value):return 'PASS' if value is True else 'FAIL' if value is False else 'UNRESOLVED'
def number(value,digits=3):return f'{value:.{digits}f}' if finite(value) else 'NA'
def canonical_command(args):
    result=list(args)
    for i,value in enumerate(args):
        if i in (0,2) or (i>0 and args[i-1] in ('--config','--design','--reference','--record','--output')):
            result[i]=str(local(value))
    return result
def worker_command(job,case,ref,record_path):
    return [str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
        '--config',str(FOLDER/'config.json'),'--design',str(FOLDER/case['design_file']),
        '--reference',str(ROOT/ref['record_file']),'--output',str(record_path),'--H',str(job['H_um']),
        '--resolution',str(job['resolution']),'--pml','2.0','--padding','10.0','--duration',str(job['duration'])]
def analysis_command(case,record_path,analysis_path):
    return [str(ROOT/'.venv-meep/bin/python'),'-B',str(ROOT/'scripts/height_family_maxwell_analysis_v2.py'),
        '--record',str(record_path),'--config',str(FOLDER/'config.json'),'--target-diameter',str(case['target_diameter_um']),
        '--output',str(analysis_path),'--count','512','--window','128']
def require_command(actual,expected):
    if canonical_command(actual)!=canonical_command(expected):
        raise ValueError('Saved command differs from frozen worker/analyzer inputs or analysis sampling')
def require_analysis_attempt(entry,case,record_path,analysis_path):
    attempts=entry.get('analysis_attempts',[])
    if not attempts or attempts[-1].get('status')!='completed' or attempts[-1].get('exit_code')!=0:
        raise ValueError('Completed saved analysis requires its actual successful frozen analyzer attempt')
    attempt=attempts[-1]
    if local(ROOT/attempt['output_file'])!=analysis_path:raise ValueError('Analyzer attempt output path differs')
    require_command(attempt['command'],analysis_command(case,record_path,analysis_path))


def frozen_comparator():
    path=ROOT/'scripts/report_height_family_maxwell_v2.py'
    if sha(path)!=PINS[path.relative_to(ROOT).as_posix()]:raise ValueError('Frozen scientific comparator changed')
    spec=importlib.util.spec_from_file_location('slc500_saved_mesh_comparison',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.compare_analyses


def build_rows(proxy_rows, analyses, state, config, compare):
    """Pure reporting transform; missing Meep data is never converted to zero."""
    rows=[];meshes={}
    for H in HEIGHTS:
        matches=[r for r in proxy_rows if r['H_um']==H and r['arm']==PROXY_ARM]
        if len(matches)!=1:raise ValueError('Exactly one existing-SLC500 proxy row per height is required')
        proxy=matches[0];eta=proxy['metrics']['declared_useful_efficiency']
        if not finite(eta):raise ValueError('Finite fixed-profile proxy efficiency required')
        pair=[analyses.get((H,res)) for res in (24,32)]
        if all(pair):mesh=compare(pair[0],pair[1],config,LIMITS)
        else:mesh={'status':'UNRESOLVED','checks':{},'missing_evidence':['Both verified r24/r32 saved analyses are required']}
        meshes[str(H)]=mesh
        for res,analysis in zip((24,32),pair):
            if analysis is not None:
                for headline,details in [('reconstruction_pass','numerical_checks'),('synthetic_optical_pass','optical_gates')]:
                    if not analysis.get(details) or analysis.get(headline)!=all(analysis[details].values()):
                        raise ValueError('Saved audit/nominal headline contradicts its component checks')
            value=None if analysis is None else analysis.get('conservative_useful_efficiency')
            if value is not None and not finite(value):raise ValueError('Nonfinite saved Meep useful efficiency')
            rows.append(dict(H_um=H,resolution_cells_per_um=res,proxy_arm=PROXY_ARM,
                proxy_useful_efficiency_percent=100*eta,proxy_nominal_status=proxy['proxy_gates']['status'],
                Meep_useful_efficiency_percent=None if value is None else 100*value,
                Meep_minus_proxy_percentage_points=None if value is None else 100*(value-eta),
                record_state=state.get((H,res),'NOT_STARTED'),
                native_status='UNRESOLVED' if analysis is None else pass_status(analysis.get('reconstruction_pass')),
                nominal_optical_status='UNRESOLVED' if analysis is None else pass_status(analysis.get('synthetic_optical_pass')),
                mesh_r24_to_r32_status=mesh['status'],
                useful_efficiency_relative_mesh_change_percent=(100*mesh['measurements']['useful_efficiency_relative_change']
                    if finite(mesh.get('measurements',{}).get('useful_efficiency_relative_change')) else None),
                efficiency_mesh_check_status=pass_status(mesh.get('checks',{}).get('useful_efficiency')),
                native_checks={} if analysis is None else analysis['numerical_checks'],
                nominal_checks={} if analysis is None else analysis['optical_gates']))
    return rows,meshes


def collect():
    inventory={}
    for name,value in PINS.items():require_sha(ROOT/name,value,inventory)
    # Capture one exact atomic ledger version; later controller updates are not
    # silently blended into this report. The snapshot is copied to output.
    ledger_bytes=(FOLDER/'ledger.json').read_bytes();ledger=json.loads(ledger_bytes)
    ledger_sha=hashlib.sha256(ledger_bytes).hexdigest()
    preparation=read(FOLDER/'preparation_manifest.json')
    for name,value in preparation.items():require_sha(contained(FOLDER,name),value,inventory)
    for name,value in read(FOLDER/'source_manifest.json').items():require_sha(ROOT/name,value,inventory)
    cfg=read(FOLDER/'config.json');require_sha(FOLDER/'config.json',CONFIG_SHA,inventory)
    plan=read(FOLDER/'plan.json');cases=read(FOLDER/'frozen_cases.json');refs=read(FOLDER/'cache_catalog.json')
    if len(plan)!=6 or [j['id'] for j in plan]!=list(range(84,90)):raise ValueError('Exact six-device plan required')
    approval=read(FOLDER/'approval_receipt.json')
    if (ledger.get('approved') is not True or ledger.get('prior_attempt_count')!=83
            or ledger.get('approval_receipt_sha256')!=PINS['artifacts/1550nm_height_family_n15263_v2/slc500_meep_validation_v2/approval_receipt.json']):
        raise ValueError('Approved saved allocation lineage differs')
    entries=ledger['entries']
    if len(entries)>6 or [e['id'] for e in entries]!=list(range(84,84+len(entries))):raise ValueError('All attempts84..89, including failures, must remain charged')
    analyses={};state={};bindings=[]
    for entry in entries:
        job=next(j for j in plan if j['id']==entry['id'])
        if entry['job']!=job:raise ValueError('Saved attempt and exact planned job differ')
        key=(int(job['H_um']),int(job['resolution']));state[key]=entry['status']+'/'+entry.get('analysis_status','not_analyzed')
        receipt_path=contained(FOLDER,entry['prelaunch_receipt_file'])
        require_sha(receipt_path,entry['prelaunch_receipt_sha256'],inventory);receipt=read(receipt_path)
        if (receipt['job']!=job or receipt['command']!=entry['command'] or receipt['driver_sha256']!=DRIVER_PIN
                or receipt['approval_receipt_sha256']!=sha(FOLDER/'approval_receipt.json')
                or receipt['effective_settings']!=entry['effective_settings'] or receipt['config_sha256']!=CONFIG_SHA
                or receipt['source_manifest_sha256']!=sha(FOLDER/'source_manifest.json')):
            raise ValueError('Prospective attempt receipt/job/source/approval differs')
        case=next(c for c in cases if c['case_id']==job['case_id'])
        ref=next(r for r in refs if r['H_um']==key[0] and r['resolution']==key[1])
        record_path=contained(FOLDER,entry['record_file'])
        require_command(entry['command'],worker_command(job,case,ref,record_path))
        if entry['status']!='completed' or entry.get('analysis_status')!='completed':continue
        if entry.get('solver_launched') is not True or entry.get('solver_exit_code')!=0:
            raise ValueError('A completed new Meep measurement requires an actual launched worker with saved zero exit code')
        require_sha(record_path,entry['record_sha256'],inventory);record=read(record_path)
        design_path=contained(FOLDER,case['design_file']);require_sha(design_path,case['design_sha256'],inventory)
        if (record.get('solver')!='CPU Meep 1.35.0' or record['settings']!=entry['effective_settings'] or record['settings']!=ref['settings']
                or record['config_fingerprint']!=fingerprint(cfg) or record['worker_sha256']!=PINS['scripts/height_family_maxwell_worker_v2.py']
                or local(record['requested_config'])!=local(FOLDER/'config.json')
                or local(record['design_file'])!=design_path or local(record['reference_file'])!=local(ROOT/ref['record_file'])
                or receipt['case']!=case or receipt['reference']!=ref):raise ValueError('Saved raw device/config/design/reference identity differs')
        expected_fields={}
        for field,suffix in [('fields_file','.fields.npz'),('power_audit_fields_file','.power-audit.npz')]:
            field_path=local(record[field])
            if field_path!=record_path.with_suffix(suffix):raise ValueError('Device NPZ path does not belong to its raw record')
            rel=field_path.relative_to(ROOT.resolve()).as_posix();expected_fields[rel]=sha(field_path)
            require_sha(field_path,entry['required_file_sha256'].get(rel),inventory)
        if expected_fields!=entry['required_file_sha256']:raise ValueError('Required raw-field inventory differs')
        require_sha(ROOT/ref['record_file'],ref['record_sha256'],inventory)
        for name,value in ref['required_file_sha256'].items():require_sha(ROOT/name,value,inventory)
        reference=read(ROOT/ref['record_file'])
        if (reference['settings']!=record['settings'] or reference['config_fingerprint']!=fingerprint(cfg)
                or not finite(reference['incident_power']) or reference['incident_power']<=0
                or not 0<=reference['power_balance_error']<.05):raise ValueError('Paired homogeneous reference differs')
        analysis_path=contained(FOLDER,entry['analysis_file']);require_sha(analysis_path,entry['analysis_sha256'],inventory);analysis=read(analysis_path)
        require_analysis_attempt(entry,case,record_path,analysis_path)
        if (local(analysis['record'])!=record_path or analysis['settings']!=record['settings']
                or analysis['declared_target_diameter_um']!=case['target_diameter_um']
                or analysis['normalization']!='Actual monitor-medium indices; no amplitude fit; min(native net T,reconstructed forward T)*declared overlap'):
            raise ValueError('Saved analysis record/settings/target/efficiency definition differs')
        if (entry['native_pass']!=analysis['reconstruction_pass'] or entry['nominal_optical_pass']!=analysis['synthetic_optical_pass']
                or entry['useful_modal_efficiency']!=analysis['conservative_useful_efficiency']):raise ValueError('Ledger/analysis scalar or audit status differs')
        analyses[key]=analysis;state[key]='COMPLETED_VERIFIED'
        bindings.append(dict(id=entry['id'],H_um=key[0],resolution=key[1],record_file=str(record_path),record_sha256=entry['record_sha256'],
            analysis_file=str(analysis_path),analysis_sha256=entry['analysis_sha256'],design_sha256=case['design_sha256'],
            reference_record_sha256=ref['record_sha256']))
    proxy=read(PROXY)
    for case in cases:
        fixed=read(FOLDER/case['design_file'])
        proxy_row=next(r for r in proxy['rows'] if r['H_um']==case['H_um'] and r['arm']==PROXY_ARM)
        if hashlib.sha256(struct.pack('<50f',*fixed['heights_um'])).hexdigest()!=proxy_row['profile_sha256']:
            raise ValueError('Proxy/Meep fixed SLC500 relief arrays differ')
    rows,meshes=build_rows(proxy['rows'],analyses,state,cfg,frozen_comparator())
    complete=len(analyses)==6 and ledger['status']=='COMPLETED_REQUIRES_REVIEW'
    summary_path=None
    summaries=sorted(FOLDER.glob('results_v*.json'),key=lambda p:int(p.stem.split('_v')[-1]))
    if complete:
        if not summaries:raise ValueError('Final report requires the controller final saved summary')
        summary_path=summaries[-1];summary=read(summary_path);inventory[str(summary_path.resolve())]=sha(summary_path)
        if summary['additional_attempts']!=6 or summary['cumulative_attempts']!=89 or summary['global_acceptance_claimed'] is not False:raise ValueError('Final controller accounting/claim differs')
        for H in HEIGHTS:
            s=next(h for h in summary['heights'] if h['H_um']==H)
            if s['mesh_last_pair']!=meshes[str(H)]:raise ValueError('Final controller mesh comparison differs')
            for res in (24,32):
                item=next(r for r in s['results'] if r['resolution']==res);a=analyses[H,res]
                if (item['useful_modal_efficiency']!=a['conservative_useful_efficiency'] or item['native_pass']!=a['reconstruction_pass']
                        or item['nominal_optical_pass']!=a['synthetic_optical_pass']):raise ValueError('Final controller summary and analysis differ')
    result=dict(status='FINAL_SAVED_COMPARISON' if complete else 'PARTIAL_SAVED_COMPARISON',created_utc=utc(),
        rows=rows,mesh_last_pairs=meshes,source_folder=str(FOLDER),ledger_snapshot_sha256=ledger_sha,
        ledger_snapshot_status=ledger['status'],verified_analysis_count=len(analyses),required_analysis_count=6,
        attempted_new_solves=len(entries),cumulative_attempts=83+len(entries),new_reference_solves=0,
        approved_utc=approval['approved_utc'],absolute_deadline_utc=approval['absolute_deadline_utc'],
        postprocessing_utc=utc(),new_physical_cases=3,actual_case_total_if_all_executed=28,
        actual_new_physical_cases_attempted=len({e['job']['case_id'] for e in entries if e.get('solver_launched')}),
        record_bindings=bindings,input_sha256=inventory,utility_sha256=sha(__file__),config=cfg,numerical_limits=LIMITS,
        global_optical_acceptance_claimed=False,experimental_validation_claimed=False,printing_validation_claimed=False,
        proxy_source=str(PROXY),proxy_arm=PROXY_ARM,
        proxy_definition='HeightConditionedLoss useful EH-inner-product power / the same incident power; one-way finite-cylinder model.',
        Meep_definition='min(native net transmission, reconstructed forward transmission) times declared Gaussian overlap; same frozen target.',
        geometry_scope='Axisymmetric circular-annulus counterparts of existing 128-segment SLC solids; fixed 500 nm bands/heights; no redesign.',
        selected_output_rule='r32 is the finest executed grid; last-pair status is reported separately and can remain FAIL/UNRESOLVED.')
    return result,ledger_bytes


CSV_FIELDS=['H_um','resolution_cells_per_um','proxy_arm','proxy_useful_efficiency_percent',
    'Meep_useful_efficiency_percent','Meep_minus_proxy_percentage_points','record_state',
    'native_status','nominal_optical_status','useful_efficiency_relative_mesh_change_percent',
    'efficiency_mesh_check_status','mesh_r24_to_r32_status']
def write_csv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
def report_text(data):
    lines=['Existing 500 nm SLC designs: fixed-profile proxy versus direct saved Meep',
        data['status'],f"Source-authenticated saved Meep analyses: {data['verified_analysis_count']}/6. Attempts: {data['attempted_new_solves']} new; {data['cumulative_attempts']} cumulative.",
        'Useful modal efficiency is expressed as percent of incident power. NA means no verified completed saved result; it is not zero.',
        'H(um)  grid(cells/um)  Proxy(%)  Meep(%)  Meep-proxy(pp)  Native  Nominal  Pair24->32']
    for row in data['rows']:
        lines.append(f"{row['H_um']:3d}    {row['resolution_cells_per_um']:2d}              {number(row['proxy_useful_efficiency_percent'])}    {number(row['Meep_useful_efficiency_percent'])}    {number(row['Meep_minus_proxy_percentage_points'])}        {row['native_status']}    {row['nominal_optical_status']}    {row['mesh_r24_to_r32_status']}")
    for H in HEIGHTS:
        row=next(r for r in data['rows'] if r['H_um']==H)
        lines.append(f"H={H} um: useful-efficiency relative change 24->32={number(row['useful_efficiency_relative_mesh_change_percent'])}%; own fixed<5% check={row['efficiency_mesh_check_status']}; combined pair={row['mesh_r24_to_r32_status']}.")
    lines+=['','Native audits, nominal optical design gates, and the declared r24-to-r32 pair checks are separate findings.',
        'The unchanged pair rule includes processing-plane width and useful-efficiency changes, core divergence, curvature change, absolute curvature and its remaining margin.',
        'A low efficiency or nominal FAIL is a reported outcome. A failed/unresolved declared last pair prevents a convergence claim under this rule for that height.',
        'r32 is the finest executed grid, even when that last pair fails; neither efficiency nor convergence is replaced by the original 50 nm design result.',
        'Proxy and Meep have different forward models/efficiency estimators; their difference is not a measured quantization penalty relative to an original 50 nm Meep design.',
        data['geometry_scope'],data['proxy_definition'],data['Meep_definition'],
        'Synthetic 1550 nm Gaussian surrogate (MFD 10.4 um); supplied polymer n=1.5263, assumed input n=1.45 and air n=1. No measured fiber eigenmode, printer or experiment is claimed.',
        f"This report only postprocesses saved artifacts at {data['postprocessing_utc']}. Approval clock: {data['approved_utc']} to {data['absolute_deadline_utc']}.",
        'No training, Meep solve, field reconstruction, target fitting or change of acceptance criteria is performed by this report utility.',
        'All numeric values, source/settings/target bindings and audit statuses are included in comparison.json and the source inventory.']
    return '\n'.join(lines)+'\n'


def render(path,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.4,6.0));fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.39)
    colors=['#6F7780','#D97828','#206A9B'];offsets=[-.24,0,.24];width=.22
    series=[('Fixed SLC500 proxy',None),('Meep: 24 cells/µm',24),('Meep: 32 cells/µm',32)]
    numeric=[]
    for s,((label,res),color,offset) in enumerate(zip(series,colors,offsets)):
        labelled=False
        for x,H in enumerate(HEIGHTS):
            row=next(r for r in data['rows'] if r['H_um']==H and (res is None or r['resolution_cells_per_um']==res))
            value=row['proxy_useful_efficiency_percent'] if res is None else row['Meep_useful_efficiency_percent']
            if not finite(value):continue
            numeric.append(value);ax.bar(x+offset,value,width,color=color,label=label if not labelled else None,
                                       edgecolor='white',linewidth=.6)
            ax.annotate(f'{value:.2f}',(x+offset,value),xytext=(0,3),textcoords='offset points',ha='center',va='bottom',fontsize=8)
            labelled=True
        if not labelled:ax.plot([],[],color=color,lw=8,label=label+' (pending)')
    ax.set_xticks(range(3),['40','75','100']);ax.set_xlabel('Solid-cylinder height H (µm)')
    ax.set_xlim(-.6,2.6)
    ax.set_ylabel('Useful modal efficiency η (%)')
    ax.set_ylim(min(0,(min(numeric)-5) if numeric else 0),max(100,(max(numeric)+7) if numeric else 100))
    ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=3,frameon=False,fontsize=9)
    fig.suptitle('SYNTHETIC LAYOUT TEST (no measured values)' if data.get('synthetic_fixture') else
                 'Existing 500 nm SLC profiles: proxy and direct Meep',y=.98,fontsize=12)
    cell=[]
    for title,key in [('Native 24/32','native_status'),('Nominal 24/32','nominal_optical_status'),
                      ('η rel. change 24→32 (%)','efficiency_mesh_check_status'),('Pair 24→32','mesh_r24_to_r32_status')]:
        values=[]
        for H in HEIGHTS:
            pair=[r for r in data['rows'] if r['H_um']==H]
            if key=='efficiency_mesh_check_status':
                values.append(number(pair[0]['useful_efficiency_relative_mesh_change_percent'],2)+' · '+pair[0][key])
            else:values.append(pair[0][key] if key=='mesh_r24_to_r32_status' else '/'.join(r[key] for r in pair))
        cell.append([title]+values)
    table=ax.table(cellText=cell,colLabels=['Audit/status','H40','H75','H100'],cellLoc='center',bbox=[-.06,-.57,1.08,.31])
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (row,col),item in table.get_celld().items():
        item.set_edgecolor('#D0D5DA');item.set_linewidth(.5)
        if row==0:item.set_facecolor('#EAF0F5')
    fig.text(.12,.095,f"{data['status']} · {data['verified_analysis_count']}/6 source-authenticated saved Meep results",fontsize=8)
    fig.text(.12,.065,'η-grid change and combined pair checks are separate. Missing bars are pending, not zero.',fontsize=8)
    fig.text(.12,.035,'Axisymmetric circular counterparts of fixed SLC geometry; synthetic optical data, no experimental claim.',fontsize=8)
    for suffix in ('.png','.svg','.pdf'):fig.savefig(str(path)+suffix,dpi=350,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Fresh report folder required; preserve earlier reports')
    if output.is_relative_to(FOLDER.resolve()):raise ValueError('Report output must remain outside prepared/active controller inputs')
    data,ledger_bytes=collect()
    if data['status']!='FINAL_SAVED_COMPARISON' and not args.allow_partial:raise RuntimeError('Six completed verified results required; use --allow-partial for explicitly incomplete reports')
    output.mkdir(parents=True)
    with (output/'report_utility_source.py').open('xb') as stream:stream.write(Path(__file__).read_bytes())
    with (output/'ledger_snapshot.json').open('xb') as stream:stream.write(ledger_bytes)
    write_new(output/'comparison.json',data);write_csv(output/'comparison.csv',data['rows'])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(report_text(data))
    render(output/'efficiency_comparison',data)
    write_new(output/'delivery_manifest.json',dict(status=data['status'],created_utc=utc(),utility_sha256=sha(__file__),
        ledger_snapshot_sha256=data['ledger_snapshot_sha256'],input_sha256=data['input_sha256'],
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()},new_solves=0,training_updates=0,global_optical_acceptance_claimed=False))
    print(json.dumps({'status':data['status'],'output':str(output),'verified':data['verified_analysis_count'],'new_solves':0}))


if __name__=='__main__':main()
