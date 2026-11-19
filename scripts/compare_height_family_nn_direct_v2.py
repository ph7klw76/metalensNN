"""Report frozen NN/direct designs and saved Meep evidence at H40/75/100.

No JAX, Meep, training or optical solve is invoked. --folder must be an
already prepared maxwell_validation_v2 folder. --output is a NEW versioned
directory below its comparison/ directory. Missing evidence remains blank;
the field-derived baseline is never substituted for direct optimization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_height_family_maxwell_v2 import compare_analyses

HEIGHTS=(40.,75.,100.)
METHODS=('NN','direct')
NUMERICS=('H_um','resolution','pml_um','padding_um','duration')
NATIVE_KEYS=('native_quadrature','paired_native_ratio','monitor_alignment',
    'aperture_fraction_disagreement','filter_net_loss','crosscheck','power_closure')
OPTICAL_KEYS=('transmission','declared_overlap','useful_efficiency','diameter',
    'sidelobe','curvature','angular_second_moment','valid_width_crossings')
EXCLUDED_VARIANTS={'pml','padding','duration','reserve_duration'}


def local_path(value):
    value=str(value)
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):return Path(value[5].upper()+':/'+value[7:])
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):return Path('/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/'))
    return Path(value)


def read(path):return json.loads(local_path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(local_path(path).read_bytes()).hexdigest()
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8)
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def safe_json(value):
    if isinstance(value,dict):return {str(k):safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def contained(folder,name):
    if not isinstance(name,str) or not name:raise ValueError('Missing evidence path')
    path=(folder/local_path(name)).resolve()
    if not path.is_relative_to(folder.resolve()):raise ValueError('Evidence path escapes its stage folder')
    return path


def core_angles(value):
    if value.get('valid') is not True:return [None,None]
    axes=value.get('axes',[])
    return [math.degrees(axes[i]['half_angle_rad']) if len(axes)>i and finite(axes[i].get('half_angle_rad'))
            and axes[i]['half_angle_rad']>0 else None for i in range(2)]


def collect(folder,proxy_folder=None,*,heights=HEIGHTS,methods=METHODS,include_variants=False):
    """Read verified evidence; optional scope supports the full-family assessor.

    The public comparison defaults remain H40/75/100, NN/direct and nominal
    numerical branches. Inclusion is not authorization or optical acceptance.
    """
    heights=tuple(float(H) for H in heights);methods=tuple(methods)
    folder=local_path(folder).resolve();proxy_folder=local_path(proxy_folder).resolve() if proxy_folder else folder.parent
    required=['config.json','approval_receipt.json','prior_stage.json','frozen_cases.json','ledger.json']
    if not folder.is_dir() or any(not (folder/name).is_file() for name in required):
        raise RuntimeError('Stage2 is not prepared; no comparison output may be created yet')
    immutable={};snapshots={}
    def protected(path):
        path=local_path(path).resolve();immutable[str(path)]=sha(path);return path
    def document(path):return read(protected(path))
    protected(Path(__file__));protected(ROOT/'scripts/report_height_family_maxwell_v2.py')
    config=document(folder/'config.json');receipt=document(folder/'approval_receipt.json')
    if receipt.get('status')!='APPROVED_BY_USER':raise ValueError('Prepared stage2 approval receipt is missing')
    prior=document(folder/'prior_stage.json');cases_doc=document(folder/'frozen_cases.json')
    cases=cases_doc if isinstance(cases_doc,list) else cases_doc.get('cases',[])
    prior_folder=local_path(prior['folder']).resolve()
    for name,key in [('ledger.json','ledger_sha256'),('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256')]:
        path=protected(prior_folder/name)
        if sha(path)!=prior.get(key):raise ValueError('Prior-stage immutable hash mismatch: '+name)
    if prior.get('delivery_manifest_sha256'):
        delivery_path=protected(prior_folder/'delivery_manifest.json')
        if sha(delivery_path)!=prior['delivery_manifest_sha256']:raise ValueError('Prior delivery-manifest hash mismatch')
    if sha(prior_folder/'config.json')!=sha(folder/'config.json'):
        raise ValueError('Prior/stage2 physical configurations differ')
    delivery=document(proxy_folder/'delivery_manifest.json')
    for name in ('config.json','training_result.json','selection_before_heldout.json'):
        path=protected(proxy_folder/name)
        if sha(path)!=delivery.get('files',{}).get(name,{}).get('sha256'):
            raise ValueError('Frozen proxy delivery hash mismatch: '+name)
    if sha(proxy_folder/'config.json')!=sha(folder/'config.json'):raise ValueError('Proxy/stage2 configurations differ')
    training=read(proxy_folder/'training_result.json');selection=read(proxy_folder/'selection_before_heldout.json')
    if selection.get('selection')!=training.get('selection'):raise ValueError('Frozen selection records differ')
    source_signature=selection['signature'];proxy=[];designs={};missing=[]
    for H in heights:
        for method in ('NN','direct','baseline'):
            matches=[r for r in training['records'] if r.get('method')==method and close(r.get('H_um'),H)]
            if len(matches)!=1:
                missing.append({'scope':'proxy','H_um':H,'method':method,'reason':'Exactly one frozen record required'});continue
            row=matches[0];path=proxy_folder/f'designs/H{H:g}/{method}.json'
            try:
                design=document(path);profile_heights=np.asarray(design['heights_um'],dtype='<f4')
                if profile_heights.shape!=(config['annuli'],) or not np.isfinite(profile_heights).all():raise ValueError('Invalid annular height array')
                if hashlib.sha256(profile_heights.tobytes()).hexdigest()!=row['profile_sha256']:raise ValueError('Height-array profile hash mismatch')
                if design.get('source_signature')!=source_signature:raise ValueError('Selected source signature mismatch')
                if not close(design.get('H_um'),H) or not close(design.get('target_diameter_um'),row.get('declared_target_diameter_um')):
                    raise ValueError('Design height or frozen target differs')
                if design.get('materials')!=config['materials'] or not close(design.get('mfd_um'),config['mfd_um']):
                    raise ValueError('Design materials or source MFD differs')
                if not close(design.get('cylinder_diameter_um'),config['cylinder_diameter_um']):raise ValueError('Cylinder diameter differs')
                if design.get('units')!='um':raise ValueError('Explicit micrometre geometry units required')
                edges=np.linspace(0,config['cylinder_diameter_um']/2,config['annuli']+1).tolist()
                record={'method':method,'H_um':H,'height_domain':row.get('height_domain'),
                    'design_file':str(path),'design_sha256':sha(path),'profile_sha256':row['profile_sha256'],
                    'source_signature':source_signature,'heights_um':profile_heights.tolist(),'radial_edges_um':edges,
                    'declared_target_diameter_um':row['declared_target_diameter_um'],
                    'eta':row['metrics'].get('declared_useful_efficiency'),
                    'C':row['metrics'].get('normalized_curvature'),'Q':row['metrics'].get('angular_second_moment_ratio'),
                    'core_half_angles_deg':core_angles(row.get('fitted_core_divergence',{})),'planes':[]}
                widths=row['metrics'].get('widths_um',[]);zs=config['evaluation']['processing_planes_um']
                for i,z in enumerate(zs):
                    valid=row['metrics'].get('valid_crossings',[])
                    xy=widths[i] if len(widths)>i and len(valid)>i and valid[i]==[True,True] else [None,None]
                    record['planes'].append({'z_um':float(z),'diameter_x_um':xy[0] if len(xy)>0 and finite(xy[0]) else None,
                                             'diameter_y_um':xy[1] if len(xy)>1 and finite(xy[1]) else None})
                designs[(method,H)]=record;proxy.append(record)
            except (OSError,ValueError,KeyError,TypeError) as error:
                missing.append({'scope':'proxy','H_um':H,'method':method,'reason':str(error)})
    frozen_case_checks=[];frozen_case_valid={}
    for case in cases:
        H=case.get('H_um');method=case.get('method')
        if method not in methods or not any(close(H,h) for h in heights):continue
        original=designs.get((method,float(H)))
        try:
            path=contained(folder,case.get('design_file'))
            check=original is not None and sha(protected(path))==case.get('design_sha256')==original['design_sha256']
        except (OSError,ValueError,TypeError):check=False
        frozen_case_valid[(method,float(H))]=check
        frozen_case_checks.append({'method':method,'H_um':H,'matches_selected_proxy_design':check})
        if not check:missing.append({'scope':'stage2_frozen_case','method':method,'H_um':H,
                                     'reason':'Frozen case unavailable or differs from selected original design'})
    stages=[];all_entries=[]
    for stage in (prior_folder,folder):
        data=(stage/'ledger.json').read_bytes();ledger=json.loads(data.decode('utf-8-sig'))
        snapshots[str(stage/'ledger.json')]=hashlib.sha256(data).hexdigest()
        entries=ledger if isinstance(ledger,list) else ledger.get('entries',[])
        duplicates=[identity for identity,count in Counter(str(e.get('id')) for e in entries).items() if count>1]
        stages.append({'folder':str(stage),'ledger':ledger,'duplicates':duplicates,'entries':entries})
        all_entries.extend((stage,e,duplicates) for e in entries)
    candidate_paths={}
    for stage,entry,duplicates in all_entries:
        try:candidate_paths[str(contained(stage,entry.get('record_file')))]=(stage,entry,duplicates)
        except (OSError,ValueError,TypeError):pass
    records=[];attempts=[]
    for stage,entry,duplicates in all_entries:
        identity=f'{stage.name}:{entry.get("id")}'
        attempts.append({'identity':identity,'id':entry.get('id'),'method':entry.get('method'),'H_um':entry.get('H_um'),
            'resolution':entry.get('resolution'),'job_type':entry.get('job_type','nominal'),'status':entry.get('status'),
            'record_file':entry.get('record_file'),'analysis_status':entry.get('analysis_status'),
            'stage_folder':str(stage),**{key:entry.get(key) for key in ('pml_um','padding_um','duration','phase',
                'case_id','design_sha256','reference_id','reference_stage','selection_receipt_file','selection_receipt_sha256',
                'selection_mode','selection_scope','started_utc')}})
        method=entry.get('method');H=entry.get('H_um');kind=entry.get('job_type','nominal')
        if method not in methods or not finite(H) or float(H) not in heights or (not include_variants and kind in EXCLUDED_VARIANTS):continue
        context={'scope':'Meep','identity':identity,'method':method,'H_um':H,'resolution':entry.get('resolution'),'job_type':kind,
                 **{key:entry.get(key) for key in ('phase','selection_receipt_file','selection_receipt_sha256')}}
        try:
            if str(entry.get('id')) in duplicates:raise ValueError('Duplicate ID within a stage')
            if entry.get('status')!='completed' or entry.get('analysis_status') not in (None,'completed'):
                raise ValueError('Completed solve and analysis required')
            design=designs.get((method,float(H)))
            if design is None or entry.get('design_sha256')!=design['design_sha256']:raise ValueError('Device design differs from selected original')
            if stage==folder and not frozen_case_valid.get((method,float(H)),False):raise ValueError('Stage2 frozen case provenance unresolved')
            rp=protected(contained(stage,entry.get('record_file')));ap=protected(contained(stage,entry.get('analysis_file')))
            if sha(rp)!=entry.get('record_sha256') or sha(ap)!=entry.get('analysis_sha256'):raise ValueError('Record/analysis ledger hash mismatch')
            raw=read(rp);analysis=read(ap);settings=raw.get('settings',{})
            if raw.get('config_fingerprint')!=fingerprint(config):raise ValueError('Actual config fingerprint mismatch')
            if not all(close(entry.get(key),settings.get(key)) for key in NUMERICS) or analysis.get('settings')!=settings:
                raise ValueError('Settings mismatch between ledger/raw/analysis')
            if local_path(analysis.get('record','')).name!=rp.name:raise ValueError('Analysis names a different raw record')
            if not close(analysis.get('declared_target_diameter_um'),design['declared_target_diameter_um']):raise ValueError('Frozen target diameter mismatch')
            refpath=local_path(raw.get('reference_file','')).resolve();refinfo=candidate_paths.get(str(refpath))
            if refinfo is None:raise ValueError('Reference record is absent from both saved ledgers')
            refstage,refentry,refduplicates=refinfo
            declared_stage=entry.get('reference_stage')
            if declared_stage is not None:
                expected_stage={'prior_stage1':prior_folder,'stage2':folder}.get(declared_stage)
                if expected_stage is None or refstage!=expected_stage:raise ValueError('Declared reference_stage/path mismatch')
            if str(refentry.get('id')) in refduplicates or refentry.get('method')!='reference' or refentry.get('status')!='completed':
                raise ValueError('Reference ledger provenance unresolved')
            if entry.get('reference_id')!=refentry.get('id'):raise ValueError('Reference ID/path mismatch')
            refpath=protected(refpath)
            if sha(refpath)!=refentry.get('record_sha256'):raise ValueError('Reference raw hash mismatch')
            reference=read(refpath)
            if reference.get('settings')!=settings or reference.get('config_fingerprint')!=fingerprint(config):raise ValueError('Reference settings/config mismatch')
            native=analysis.get('numerical_checks',{})
            native_checks={key:native.get(key) is True for key in NATIVE_KEYS}
            native_missing=[f'Native analysis check {key}' for key in NATIVE_KEYS if key not in native]
            if not finite(reference.get('power_balance_error')):native_missing.append('Finite native reference closure')
            if not finite(raw.get('power_balance_error')):native_missing.append('Finite native device closure')
            if not finite(reference.get('incident_power')):native_missing.append('Finite incident reference power')
            native_checks['reference_power_closure']=finite(reference.get('power_balance_error')) and 0<=reference['power_balance_error']<.05
            native_checks['device_power_closure']=finite(raw.get('power_balance_error')) and 0<=raw['power_balance_error']<.05
            native_checks['positive_incident_reference']=finite(reference.get('incident_power')) and reference['incident_power']>0
            planes=[];saved_planes=analysis.get('metrics',{}).get('planes',[])
            for z in config['evaluation']['processing_planes_um']:
                matches=[p for p in saved_planes if close(p.get('z_um'),z)]
                p=matches[0] if len(matches)==1 else {}
                valid=p.get('valid_crossings')==[True,True]
                planes.append({'z_um':z,'diameter_x_um':p.get('diameter_x_um') if valid and finite(p.get('diameter_x_um')) else None,
                               'diameter_y_um':p.get('diameter_y_um') if valid and finite(p.get('diameter_y_um')) else None})
            records.append({**context,'stage_folder':str(stage),'record_file':str(rp),'record_sha256':sha(rp),
                'analysis_file':str(ap),'analysis_sha256':sha(ap),'design_sha256':entry['design_sha256'],
                'reference_record_file':str(refpath),'reference_record_sha256':sha(refpath),'settings':settings,
                'native_checks':native_checks,'native_all_pass':all(native_checks.values()),
                'native_missing_evidence':native_missing,
                'optical_gates':{key:analysis.get('optical_gates',{}).get(key) for key in OPTICAL_KEYS},
                'eta':analysis.get('conservative_useful_efficiency'),'C':analysis.get('normalized_curvature'),
                'Q':analysis.get('angular_second_moment_ratio'),'planes':planes,
                'core_half_angles_deg':core_angles(analysis.get('metrics',{}).get('fitted_core_divergence',{})),
                '_analysis':analysis})
        except (OSError,ValueError,KeyError,TypeError) as error:missing.append({**context,'reason':str(error)})
    # Pair only identical physical height, mesh, PML, padding and duration.
    groups=defaultdict(dict)
    for row in records:
        key=tuple(float(row['settings'][name]) for name in NUMERICS)
        groups[key][row['method']]=row  # Latest ledger-order record; all histories remain in records/attempts.
    pairs=[];comparison_missing=[]
    for H in HEIGHTS:
        for resolution in ([24.,32.] if H==40 else [24.]):
            options=[(key,group) for key,group in groups.items() if close(key[0],H) and close(key[1],resolution)
                     and all(method in group for method in METHODS)]
            if not options:
                comparison_missing.append({'H_um':H,'resolution':resolution,'reason':'No NN/direct pair with identical numerical settings'});continue
            for key,group in options:
                pairs.append({'H_um':H,'resolution':resolution,'settings':group['NN']['settings'],
                    'NN_identity':group['NN']['identity'],'direct_identity':group['direct']['identity'],
                    'native_both_pass':all(group[m]['native_all_pass'] for m in METHODS),
                    'eta_direct_minus_NN':group['direct']['eta']-group['NN']['eta'] if all(finite(group[m]['eta']) for m in METHODS) else None,
                    'NN_superiority_established':False})
    proposal_path=prior_folder/'maxwell_validation_proposal.json'
    proposal=document(proposal_path) if proposal_path.is_file() else {}
    mesh_evidence=[]
    for H in heights:
        for method in METHODS:
            for coarse,fine in ((16.,24.),(24.,32.)):
                for key,group in groups.items():
                    if not close(key[0],H) or not close(key[1],fine) or method not in group:continue
                    old_key=(key[0],coarse,*key[2:]);older=groups.get(old_key,{}).get(method)
                    if older is None:continue
                    comparison=compare_analyses(older['_analysis'],group[method]['_analysis'],config,
                                               proposal.get('numerical_acceptance',{}),require_collimation=True)
                    mesh_evidence.append({'H_um':H,'method':method,'resolutions':[coarse,fine],
                        'prior_identity':older['identity'],'fine_identity':group[method]['identity'],
                        'native_both_pass':older['native_all_pass'] and group[method]['native_all_pass'],
                        'comparison':comparison})
    for row in records:row.pop('_analysis')
    new_ledger=stages[-1]['ledger'];limits=receipt.get('limits',{})
    prior_count=len(stages[0]['entries']);new_count=len(stages[-1]['entries'])
    elapsed=new_ledger.get('elapsed_s') if isinstance(new_ledger,dict) else None
    recorded_elapsed=elapsed;elapsed_basis='Saved ledger elapsed time at its recorded execution state'
    # The original wall-clock amendment keeps running between workers and
    # while a priority choice is pending. A stopped/pending ledger does not
    # freeze elapsed time at its last write or reset the approved deadline.
    started=isinstance(new_ledger,dict) and (new_count>0 or bool(new_ledger.get('first_started_utc'))
             or str(new_ledger.get('status','')) not in ('','PREPARED_APPROVED_PHASE_A_ONLY'))
    if started:
        try:
            start=datetime.fromisoformat(new_ledger['first_started_utc'].replace('Z','+00:00'))
            if start.tzinfo is None:raise ValueError('Timezone-aware execution start required')
            live_elapsed=(datetime.now(timezone.utc)-start).total_seconds()
            if live_elapsed<0:raise ValueError('Execution start is in the future')
            elapsed=max(elapsed,live_elapsed) if finite(elapsed) else live_elapsed
            elapsed_basis='Started-stage wall clock since first_started_utc, including stopped/pending gaps; never below recorded elapsed'
        except (KeyError,TypeError,ValueError):
            elapsed=None;elapsed_basis='UNRESOLVED: started stage lacks a valid absolute execution start'
    caps={key:limits.get(key) for key in ('additional_CPU_solves','additional_wall_s','cumulative_CPU_solves')}
    budget_missing=[key for key,value in caps.items() if not finite(value)]
    budget_checks={}
    if finite(limits.get('prior_attempts')):budget_checks['prior_attempts_match']=prior_count==limits['prior_attempts']
    else:budget_missing.append('approved prior_attempts')
    if finite(caps['additional_CPU_solves']):budget_checks['additional_solve_cap']=new_count<=caps['additional_CPU_solves']
    if finite(caps['cumulative_CPU_solves']):budget_checks['cumulative_solve_cap']=prior_count+new_count<=caps['cumulative_CPU_solves']
    if finite(elapsed) and finite(caps['additional_wall_s']):budget_checks['additional_wall_cap']=elapsed<=caps['additional_wall_s']
    else:budget_missing.append('finite additional elapsed time/wall cap')
    budget_status='FAIL' if not all(budget_checks.values()) else ('UNRESOLVED' if budget_missing else 'PASS')
    unchanged=all(sha(path)==digest for path,digest in immutable.items())
    if not unchanged:raise RuntimeError('Immutable inputs changed during comparison collection')
    return safe_json({'status':'DESCRIPTIVE_FROZEN_COMPARISON_NO_FULL_OPTICAL_ACCEPTANCE',
        'created_utc':datetime.now(timezone.utc).isoformat(),'folder':str(folder),'proxy_folder':str(proxy_folder),
        'config':config,'config_fingerprint':fingerprint(config),'source_signature':source_signature,
        'selection':selection['selection'],'proxy_records':proxy,'Meep_records':records,
        'matched_Meep_pairs':pairs,'missing_comparisons':comparison_missing,'missing_evidence':missing,
        'mesh_evidence':mesh_evidence,'all_attempts':attempts,'frozen_case_checks':frozen_case_checks,
        'stage_metadata':[{'folder':stage['folder'],'duplicates':stage['duplicates'],
            **{key:stage['ledger'].get(key) for key in ('status','first_started_utc','absolute_deadline_utc','updated_utc',
                'elapsed_s','phase_selections','subsequent_phase_selection','subsequent_phase_runs','adaptive_selections')}} for stage in stages],
        'budget':{'status':budget_status,'checks':budget_checks,'limits':caps,'prior_attempts':prior_count,
            'additional_attempts':new_count,'cumulative_attempts':prior_count+new_count,'additional_elapsed_s':elapsed,
            'recorded_elapsed_s':recorded_elapsed,'elapsed_basis':elapsed_basis,'missing_evidence':budget_missing},
        'authorization_receipt':receipt,'prior_stage_linkage':prior,'input_sha256':immutable,
        'ledger_snapshot_sha256':snapshots,'immutable_inputs_unchanged':unchanged,
        'NN_advantage_claimed':False,'full_family_optical_acceptance_claimed':False,
        'limitations':['NN and non-neural direct-table inverse design are optimization methods; independent Meep is their forward validator.',
            'Field-derived baseline appears only as a separately named relief comparator; it is never substituted for direct.',
            'H40 and H100 were training heights. H75 was validation-observed and influenced checkpoint/seed selection.',
            'Proxy and Maxwell use different forward models; conservative Maxwell eta is not a proxy prediction.',
            'Matched comparison requires identical mesh/PML/padding/duration; unmatched records remain mesh history only.',
            'All numerical gates remain frozen. Missing or failed evidence is retained. No NN advantage is established.',
            'No global diffraction optimum, all-angle/ISO M2, MFD-tolerance robustness, experimental or fabrication acceptance is established.']})


def _band(ax,x,xy,color,style,label,marker=None):
    """Plot measured x/y mean and range, with missing values as real gaps."""
    mean=[sum(v)/2 if len(v)==2 and all(finite(a) for a in v) else np.nan for v in xy]
    low=[min(v) if len(v)==2 and all(finite(a) for a in v) else np.nan for v in xy]
    high=[max(v) if len(v)==2 and all(finite(a) for a in v) else np.nan for v in xy]
    ax.plot(x,mean,style,color=color,label=label,marker=marker,ms=4,lw=1.5)
    ax.fill_between(x,low,high,color=color,alpha=.1,linewidth=0)


def render(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'NN':'#2166ac','direct':'#d6604d','baseline':'#777777'}
    names={'NN':'NN inverse design','direct':'Non-neural direct-table inverse design','baseline':'Field-derived baseline'}
    proxy={(row['method'],row['H_um']):row for row in data['proxy_records']}
    records={row['identity']:row for row in data['Meep_records']}
    pairs={(row['H_um'],row['resolution']):row for row in data['matched_Meep_pairs']}
    output=Path(output)
    def pair_row(H,res,method):
        pair=pairs.get((H,res));return records.get(pair[method+'_identity']) if pair else None
    def save(fig,path):
        with path.open('xb') as stream:
            fig.savefig(stream,format='png',dpi=190,metadata={'Description':data['status'],
                'Provenance':'comparison.json; actual hashes, missing evidence and all attempts retained'})
        plt.close(fig)
    with plt.rc_context({'font.size':9,'legend.fontsize':7,'axes.titlesize':10}):
        fig,axes=plt.subplots(3,2,figsize=(13.2,11.4))
        for i,H in enumerate(HEIGHTS):
            relief,diameter=axes[i]
            for method in ('baseline','NN','direct'):
                row=proxy.get((method,H))
                if row:relief.stairs(row['heights_um'],row['radial_edges_um'],color=colors[method],
                                    ls=':' if method=='baseline' else '-',lw=1 if method=='baseline' else 1.5,label=names[method])
            for method in METHODS:
                row=proxy.get((method,H))
                if row:
                    _band(diameter,[p['z_um'] for p in row['planes']],
                          [[p['diameter_x_um'],p['diameter_y_um']] for p in row['planes']],
                          colors[method],'--',f'Proxy: {method}')
                for res in (24.,32.):
                    measured=pair_row(H,res,method)
                    if measured:
                        _band(diameter,[p['z_um'] for p in measured['planes']],
                              [[p['diameter_x_um'],p['diameter_y_um']] for p in measured['planes']],
                              colors[method],'-' if res==24 else ':',f'Meep matched {res:g}: {method}',marker='o' if res==24 else 'D')
            if (H,24.) not in pairs:
                diameter.text(.03,.95,'Matched r24 Meep pair unavailable',transform=diameter.transAxes,va='top',fontsize=8,color='#555555')
            domain='validation-observed' if H==75 else 'training height'
            relief.set(title=f'H={H:g} µm ({domain}) — frozen top relief',xlabel='Radius (µm)',ylabel='Relief above cylinder (µm)')
            diameter.set(title=f'H={H:g} µm — 1/e² widths (mean x/y; band spans x/y)',
                         xlabel='Propagation after output monitor z=H+4 µm (µm)',ylabel='1/e² intensity diameter (µm)')
            for ax in (relief,diameter):ax.grid(alpha=.2);ax.legend(loc='best',frameon=False)
        fig.suptitle('NN and non-neural direct-table inverse design: geometry and beam propagation\n'
                     'Dashed: one-way proxy. Solid/circles: independent Meep r24. Dotted/diamonds: matched r32.',fontsize=12)
        fig.text(.5,.018,'Baseline is a separately named field-derived seed. Missing direct evidence remains blank. '
                 'H75 influenced validation selection; it is not a fresh held-out case.',ha='center',fontsize=8)
        fig.subplots_adjust(top=.91,bottom=.07,left=.07,right=.985,hspace=.35,wspace=.22)
        save(fig,output/'geometry_and_propagation.png')
        fig,axes=plt.subplots(2,2,figsize=(13.2,9.5));eta,angle,C,Q=axes.flat
        for method in METHODS:
            color=colors[method]
            eta.plot(HEIGHTS,[100*proxy[(method,H)]['eta'] if (method,H) in proxy and finite(proxy[(method,H)]['eta']) else np.nan
                              for H in HEIGHTS],'--',color=color,marker='s',label=f'Proxy {method}')
            _band(angle,HEIGHTS,[proxy.get((method,H),{}).get('core_half_angles_deg',[None,None]) for H in HEIGHTS],
                  color,'--',f'Proxy {method}',marker='s')
            for res in (24.,32.):
                rows=[pair_row(H,res,method) for H in HEIGHTS]
                if any(row and finite(row['eta']) for row in rows):
                    eta.plot(HEIGHTS,[100*row['eta'] if row and finite(row['eta']) else np.nan for row in rows],
                             '-' if res==24 else ':',color=color,marker='o' if res==24 else 'D',label=f'Meep matched {res:g}: {method}')
                if any(row and all(finite(v) for v in row['core_half_angles_deg']) for row in rows):
                    _band(angle,HEIGHTS,[row['core_half_angles_deg'] if row else [None,None] for row in rows],color,
                          '-' if res==24 else ':',f'Meep matched {res:g}: {method}',marker='o' if res==24 else 'D')
        if not data['matched_Meep_pairs']:
            for ax in (eta,angle):ax.text(.03,.06,'No matched Meep pair yet: proxy curves only',
                                          transform=ax.transAxes,fontsize=8,color='#555555')
        height_colors={40.:'#1b9e77',75.:'#7570b3',100.:'#d95f02'}
        for H in HEIGHTS:
            for method in METHODS:
                rows=sorted([row for row in data['Meep_records'] if row['H_um']==H and row['method']==method],key=lambda r:r['resolution'])
                if not rows:continue
                for ax,key in ((C,'C'),(Q,'Q')):
                    ax.plot([row['resolution'] for row in rows],[row[key] if finite(row[key]) else np.nan for row in rows],
                            '-' if method=='NN' else '--',color=height_colors[H],marker='o' if method=='NN' else 's',
                            label=f'H{H:g}: {method}',lw=1.3)
        gates=data['config']['natural_training']['investigation_gates']
        cap=gates['max_abs_normalized_curvature'];C.axhspan(-cap,cap,color='0.95',zorder=-1)
        C.axhline(cap,color='firebrick',ls=':',lw=1);C.axhline(-cap,color='firebrick',ls=':',lw=1)
        Q.axhline(gates['max_angular_second_moment_ratio'],color='firebrick',ls=':',lw=1)
        eta.axhline(100*gates['useful_efficiency_min'],color='firebrick',ls=':',lw=1)
        eta.set(title='Useful efficiency: proxy vs conservative Meep',xlabel='Cylinder height H (µm)',ylabel='Useful modal efficiency (%)')
        angle.set(title='Fitted Gaussian-core half-angle (mean x/y; band spans x/y)',xlabel='Cylinder height H (µm)',ylabel='Core divergence half-angle (degrees)')
        C.set(title='Saved curvature/mesh evidence; all failures retained',xlabel='Resolution (cells/µm)',ylabel='Authoritative normalized curvature C')
        Q.set(title='Retained-band angular second-moment ratio',xlabel='Resolution (cells/µm)',ylabel='Q (not full-angle M²)')
        for ax in axes.flat:ax.grid(alpha=.2);ax.legend(ncol=2,loc='best',frameon=False)
        eta.set_xticks(HEIGHTS);angle.set_xticks(HEIGHTS);C.set_xticks([8,16,24,32]);Q.set_xticks([8,16,24,32])
        fig.suptitle('Frozen comparison: descriptive synthetic evidence, not full optical acceptance',fontsize=13)
        fig.text(.5,.018,'Methods are compared only at identical Meep settings. Unmatched meshes appear in histories only. '
                 'No NN superiority, global diffraction optimum, ISO M² or experimental validation is claimed.',ha='center',fontsize=8)
        fig.subplots_adjust(top=.92,bottom=.1,left=.075,right=.985,hspace=.32,wspace=.25)
        save(fig,output/'efficiency_angles_and_mesh.png')


def human_report(data):
    def formatted(value,places):return f'{value:.{places}f}' if finite(value) else 'NA'
    lines=['FROZEN NN / NON-NEURAL DIRECT-TABLE COMPARISON',data['status'],'',
        'Independent Meep is the forward validator, not the non-neural optimization method.',
        'Field-derived baseline is a separate seed comparator; no baseline/direct substitution.',
        'H40/H100 were training heights; H75 influenced validation checkpoint/seed selection.','',
        'FROZEN PROXY METRICS (not independent Maxwell acceptance)',
        'H(um) method   eta(%)        C           Q       output_Dx/Dy(um)   core_x/y(deg)']
    for row in data['proxy_records']:
        if row['method'] not in METHODS:continue
        first=row['planes'][0] if row['planes'] else {}
        eta=100*row['eta'] if finite(row['eta']) else None
        width=f"{formatted(first.get('diameter_x_um'),4)}/{formatted(first.get('diameter_y_um'),4)}"
        angles='/'.join(formatted(value,4) for value in row['core_half_angles_deg'])
        lines.append(f"{row['H_um']:5g} {row['method']:6} {formatted(eta,5):>10} {formatted(row['C'],6):>11} "
                     f"{formatted(row['Q'],6):>10} {width:>19} {angles}")
    lines.extend(['','MATCHED INDEPENDENT MEEP PAIRS (identical mesh/PML/padding/duration)'])
    for row in data['matched_Meep_pairs']:
        difference=100*row['eta_direct_minus_NN'] if finite(row['eta_direct_minus_NN']) else None
        lines.append(f"H{row['H_um']:g}, mesh{row['resolution']:g}: native both={row['native_both_pass']}; "
                     f"eta direct-minus-NN={formatted(difference,5)} percentage points; superiority not established")
    for row in data['missing_comparisons']:lines.append(f"MISSING H{row['H_um']:g} mesh{row['resolution']:g}: {row['reason']}")
    lines.extend(['','MESH EVIDENCE (does not replace boundary/duration/full-family acceptance)'])
    for row in data['mesh_evidence']:
        lines.append(f"H{row['H_um']:g} {row['method']} {row['resolutions']}: {row['comparison']['status']}; native both={row['native_both_pass']}")
        for failure in row['comparison'].get('failed_checks',[]):lines.append('  Failed: '+failure)
    budget=data['budget'];lines.extend(['','ADDITIONAL BUDGET',
        f"{budget['status']}: new attempts={budget['additional_attempts']}; prior={budget['prior_attempts']}; cumulative={budget['cumulative_attempts']}",
        f"Additional elapsed seconds={budget['additional_elapsed_s']}; limits={budget['limits']}",'',
        'Missing/unverified record details and exact hashes are retained in comparison.json.',''])
    lines.extend(data['limitations']);return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--proxy-folder')
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if not folder.is_dir():raise RuntimeError('Stage2 must be prepared before comparison output')
    if not output.is_relative_to(folder/'comparison') or output==folder/'comparison':
        raise ValueError('--output must be a new version directory below stage2/comparison/')
    if output.exists():raise FileExistsError('Preserve prior comparison; choose a fresh version')
    data=collect(folder,args.proxy_folder)
    data['comparison_utility_sha256']=sha(Path(__file__))
    data['comparison_function_source_sha256']=sha(ROOT/'scripts/report_height_family_maxwell_v2.py')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'comparison.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(human_report(data))
    render(data,output)
    print(json.dumps({'status':data['status'],'output':str(output),'matched_Meep_pairs':len(data['matched_Meep_pairs']),
                      'missing_comparisons':len(data['missing_comparisons']),'additional_attempts':data['budget']['additional_attempts']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
