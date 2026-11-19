"""Manuscript figures from saved JSON only; no optical reconstruction/training."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
ASSESS=CAMPAIGN/'maxwell_validation_v3/assessment/recovery_final_verification_v2/assessment.json'
COMPARISON=CAMPAIGN/'maxwell_validation_v3/forward_comparison/recovery_final_v1/comparison.json'
ASSESS_SHA='e120ec628e1add2eb61dffb23dc15f7ec7ac0d2c1b86bb19ea657bb246dd4e8b'
GRAPH_MANIFEST_SHA='f602c5e0ec7157c81fc7b7f4ebf7d4226e1f784d67088ae21ef5391dbb9122f1'
SELECTED={40.:'maxwell_validation_v1:43',75.:'maxwell_validation_v3:69',100.:'maxwell_validation_v3:83'}
HEIGHT_COLORS={40.:'#0072B2',75.:'#D55E00',100.:'#009E73'}
METHOD_COLORS={'NN':'#0072B2','direct':'#D55E00'}
INPUTS={}


def local_path(value):
    s=str(value)
    if os.name=='nt' and s.startswith('/mnt/') and len(s)>7 and s[6]=='/':s=s[5].upper()+':/'+s[7:]
    elif os.name!='nt' and len(s)>2 and s[1]==':':s='/mnt/'+s[0].lower()+'/'+s[3:].replace('\\','/')
    return Path(s).resolve()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path,expected=None):
    p=local_path(path);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and digest!=expected:raise ValueError('Saved source SHA differs: '+str(p))
    if str(p) in INPUTS and INPUTS[str(p)]!=digest:raise ValueError('Changed source during figure assembly')
    INPUTS[str(p)]=digest
    return json.loads(raw.decode('utf-8-sig'))


def write_csv(path,rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def collect():
    train=document(CAMPAIGN/'training_result.json')
    config=document(CAMPAIGN/'config.json')
    learning=document(CAMPAIGN/'maxwell_validation_v2/learning_evidence_v1/evidence.json')
    assert learning['H75_is_validation_observed'] and learning['all_inputs_unchanged']
    historical_inputs={str(local_path(p)):v for p,v in learning['input_sha256'].items()}
    for p in (CAMPAIGN/'training_result.json',CAMPAIGN/'config.json'):
        assert historical_inputs[str(local_path(p))]==INPUTS[str(local_path(p))]
    training=[];validation=[];seeds=[]
    for method in ('NN','direct'):
        for seed in (11,29,47):
            history_path=CAMPAIGN/f'{method}_seed{seed}/history.json'
            val_path=CAMPAIGN/f'{method}_seed{seed}/validation.json'
            history=document(history_path,historical_inputs[str(local_path(history_path))])
            values=document(val_path,historical_inputs[str(local_path(val_path))])
            assert len(history)==100 and len(values)==10
            report=next(r for r in train['reports'] if r['method']==method and r['seed']==seed)
            best=min(values,key=lambda r:r['key']);assert best['key']==report['validation_key']
            selected=train['selection'][method]['seed']==seed
            for r in history:
                training.append({'method':method,'seed':seed,'selected_arm':selected,'evaluated_before_update':r['evaluated_before_update'],
                    'sampled_H_um':r['H_um'],'continuous_proxy_objective_L':r['metrics']['loss'],'source_file':str(history_path),'source_sha256':INPUTS[str(local_path(history_path))]})
            for r in values:
                hs=sorted(m['H_um'] for m in r['metrics']);assert hs==config['natural_training']['validation_H_um']
                mean=sum(m['loss'] for m in r['metrics'])/len(r['metrics'])
                worst=min(m['declared_useful_efficiency'] for m in r['metrics'])
                assert abs(mean-r['key'][2])<1e-12 and abs(worst+r['key'][1])<1e-12
                validation.append({'method':method,'seed':seed,'selected_arm':selected,'step':r['step'],'quantized_mean_proxy_L':mean,
                    'quantized_worst_useful_eta_fraction':worst,'summed_normalized_gate_violations':r['key'][0],
                    'best_checkpoint_for_seed':r['step']==best['step'],'source_file':str(val_path),'source_sha256':INPUTS[str(local_path(val_path))]})
            seeds.append({'method':method,'seed':seed,'selected_arm':selected,'selected_checkpoint_step':best['step'],
                'worst_validation_eta_fraction':-report['validation_key'][1],'mean_validation_L':report['validation_key'][2],
                'objective_calls':report['objective_calls'],'source_file':str(CAMPAIGN/'training_result.json'),'source_sha256':INPUTS[str(local_path(CAMPAIGN/'training_result.json'))]})
    contribution=[]
    records=train['records'];assert len(records)==54
    for H in sorted({r['H_um'] for r in records}):
        group={r['method']:r for r in records if r['H_um']==H};assert set(group)=={'baseline','NN','direct'}
        for method in ('baseline','NN','direct'):
            r=group[method];m=r['metrics']
            contribution.append({'H_um':H,'height_domain':r['height_domain'],'method':method,'proxy_useful_eta_fraction':m['declared_useful_efficiency'],
                'proxy_objective_L':m['loss'],'proxy_C':m['normalized_curvature'],'proxy_retained_band_Q':m['angular_second_moment_ratio'],
                'eta_gain_over_physics_seed_percentage_points':100*(m['declared_useful_efficiency']-group['baseline']['metrics']['declared_useful_efficiency']),
                'loss_change_from_physics_seed':m['loss']-group['baseline']['metrics']['loss'],
                'source_file':str(CAMPAIGN/'training_result.json'),'source_sha256':INPUTS[str(local_path(CAMPAIGN/'training_result.json'))]})
    assessment=document(ASSESS,ASSESS_SHA)
    manifest=document(COMPARISON.parent/'delivery_manifest.json',GRAPH_MANIFEST_SHA)
    comparison=document(COMPARISON,manifest['files']['comparison.json']['sha256'])
    assert {r['H_um']:r['Meep']['identity'] for r in comparison['rows']}==SELECTED
    limits=document(CAMPAIGN/'maxwell_validation_v1/maxwell_validation_proposal.json')['numerical_acceptance']
    mesh=[]
    for r in comparison['NN_nominal_histories']:
        H=r['H_um']
        if H not in SELECTED:continue
        selected=next(x for x in assessment['NN_heights'] if x['H_um']==H)['Meep']
        assert selected['identity']==SELECTED[H] and r['design_sha256']==selected['design_sha256']
        assert r['job_type'] not in {'pml','padding','duration','reserve_duration'}
        raw=document(r['record_file'],r['record_sha256']);ref=document(r['reference_record_file'],r['reference_record_sha256'])
        closure=max(raw['power_balance_error'],ref['power_balance_error'])
        assert raw['settings']==ref['settings']==r['settings']
        mesh.append({'H_um':H,'identity':r['identity'],'resolution_cells_per_um':r['resolution'],'selected_latest':r['identity']==SELECTED[H],
            'useful_eta_fraction':r['eta'],'C':r['C'],'retained_band_Q':r['Q'],'device_power_closure_fraction':raw['power_balance_error'],
            'reference_power_closure_fraction':ref['power_balance_error'],'maximum_native_closure_fraction':closure,
            'native_checks_all_pass':r['native_all_pass'],'duration_meep_units':r['settings']['duration'],
            'PML_um':r['settings']['pml_um'],'padding_um':r['settings']['padding_um'],
            'record_file':r['record_file'],'record_sha256':r['record_sha256'],'analysis_file':r['analysis_file'],'analysis_sha256':r['analysis_sha256'],
            'design_sha256':r['design_sha256']})
    pairs=[]
    for p in assessment['all_prior_mesh_pairs']:
        if p['H_um'] not in SELECTED or not p['adjacent_in_current_sequence']:continue
        m=p['comparison']['measurements']
        pairs.append({'H_um':p['H_um'],'coarse_cells_per_um':p['resolutions'][0],'fine_cells_per_um':p['resolutions'][1],
            'prior_identity':p['prior_identity'],'fine_identity':p['fine_identity'],'declared_mesh_check_status':p['comparison']['status'],
            'maximum_processing_D_relative_change':m['maximum_diameter_relative_change'],
            'eta_relative_change':m['useful_efficiency_relative_change'],'absolute_C_change':m['curvature_absolute_change'],
            'retained_band_Q_absolute_change_diagnostic':m['retained_band_angular_ratio_absolute_change'],
            'source_file':str(ASSESS),'source_sha256':ASSESS_SHA})
    controls=[]
    for c in assessment['boundary_duration_checks']:
        m=c['comparison']['measurements']
        controls.append({'kind':c['job_type'],'H_um':c['H_um'],'resolution_cells_per_um':c['resolution'],
            'required_original_sampled_control':c['required_original_sampled_control'],'nominal_identity':c['nominal_identity'],'variation_identity':c['variation_identity'],
            'numerical_comparison_status':c['comparison']['status'],'maximum_D_relative_change':m['maximum_diameter_relative_change'],
            'eta_relative_change':m['useful_efficiency_relative_change'],'absolute_C_change':m['curvature_absolute_change'],
            'Q_nominal':c['Q_prior'],'Q_variant':c['Q_variant'],'Q_gate_crossing':c['Q_gate_crossing'],'scope':c['scope']})
    assert len(mesh)==11 and len(pairs)==8 and len(controls)==4
    return dict(training=training,validation=validation,seeds=seeds,contribution=contribution,mesh=mesh,pairs=pairs,controls=controls,limits=limits)


def plot(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10.4,'axes.titlesize':10.6,
        'xtick.labelsize':9.6,'ytick.labelsize':9.6,'legend.fontsize':9.2,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.8,'savefig.facecolor':'white'})
    def panel(ax,letter,title):
        ax.set_title(r'$\bf{'+letter+r'}$  '+title,pad=8,loc='left')
        ax.grid(alpha=.16,linewidth=.6)
    def save(fig,stem):
        for ext in ('png','pdf','svg'):fig.savefig(out/f'{stem}.{ext}',dpi=300)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6))
    for method in ('NN','direct'):
        for seed in (11,29,47):
            chosen=next(r for r in data['seeds'] if r['method']==method and r['seed']==seed)['selected_arm']
            h=[r for r in data['training'] if r['method']==method and r['seed']==seed]
            v=[r for r in data['validation'] if r['method']==method and r['seed']==seed]
            style=dict(color=METHOD_COLORS[method],lw=1.3 if chosen else .75,alpha=1 if chosen else .3,zorder=3 if chosen else 2)
            axes[0,0].plot([r['evaluated_before_update'] for r in h],[r['continuous_proxy_objective_L'] for r in h],**style)
            axes[0,1].plot([r['step'] for r in v],[r['quantized_mean_proxy_L'] for r in v],**style)
            if chosen:
                b=next(r for r in v if r['best_checkpoint_for_seed'])
                axes[0,1].scatter(b['step'],b['quantized_mean_proxy_L'],s=70,marker='*',color=METHOD_COLORS[method],edgecolor='white',lw=.4,zorder=5)
            s=next(r for r in data['seeds'] if r['method']==method and r['seed']==seed)
            axes[1,0].scatter(seed+(-1.2 if method=='NN' else 1.2),100*s['worst_validation_eta_fraction'],marker='*' if chosen else 'o',
                s=90 if chosen else 32,color=METHOD_COLORS[method],edgecolor='white',lw=.5,zorder=3)
        p=[r for r in data['contribution'] if r['method']==method]
        axes[1,1].plot([r['H_um'] for r in p],[r['eta_gain_over_physics_seed_percentage_points'] for r in p],
            color=METHOD_COLORS[method],lw=1.2,marker='o',ms=3.4)
    panel(axes[0,0],'a','Sampled training objective')
    axes[0,0].set(xlabel='Evaluation before update',ylabel='Continuous proxy objective $L$',xlim=(0,101))
    panel(axes[0,1],'b','Fixed validation heights')
    axes[0,1].set(xlabel='Training update',ylabel='Validation mean objective $L$',xlim=(8,102))
    axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    panel(axes[1,0],'c','Best checkpoint by seed')
    axes[1,0].set(xlabel='Random seed',ylabel=r'Worst validation $\eta$ (%)',xticks=[11,29,47],xlim=(7,51))
    axes[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    panel(axes[1,1],'d','Gain beyond the physics seed')
    axes[1,1].axhline(0,color='#666666',ls=':',lw=1)
    axes[1,1].set(xlabel='Cylinder height $H$ (µm)',ylabel=r'$\Delta\eta$ (percentage points)',xticks=[40,60,75,100])
    handles=[Line2D([],[],color=METHOD_COLORS['NN'],lw=1.5,label='Physics-seeded NN'),Line2D([],[],color=METHOD_COLORS['direct'],lw=1.5,label='Direct table optimizer'),
        Line2D([],[],color='#444444',marker='*',ls='',ms=8,label='Selected seed arm')]
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.52,.995),columnspacing=1.2,handlelength=2)
    fig.subplots_adjust(left=.12,right=.98,bottom=.1,top=.875,hspace=.57,wspace=.5)
    save(fig,'fig02_learning_and_proxy_contribution')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.6))
    for H,color in HEIGHT_COLORS.items():
        rows=sorted([r for r in data['mesh'] if r['H_um']==H],key=lambda r:r['resolution_cells_per_um'])
        xs=[r['resolution_cells_per_um'] for r in rows]
        for ax,key,factor in [(axes[0,0],'useful_eta_fraction',100),(axes[0,1],'C',1),(axes[1,0],'retained_band_Q',1)]:
            ax.plot(xs,[factor*r[key] for r in rows],color=color,lw=1.3,marker='o',ms=4)
            fine=next(r for r in rows if r['selected_latest'])
            ax.scatter(fine['resolution_cells_per_um'],factor*fine[key],color=color,marker='*',s=80,edgecolor='white',lw=.45,zorder=4)
        changes=sorted([r for r in data['pairs'] if r['H_um']==H],key=lambda r:r['fine_cells_per_um'])
        axes[1,1].plot([r['fine_cells_per_um'] for r in changes],[100*r['maximum_processing_D_relative_change'] for r in changes],color=color,lw=1.3,marker='o',ms=4)
        axes[1,1].plot(xs,[100*r['maximum_native_closure_fraction'] for r in rows],color=color,lw=1.1,ls=':',marker='x',ms=4)
    panel(axes[0,0],'a','Direct Maxwell efficiency')
    axes[0,0].set(ylabel=r'Useful efficiency $\eta$ (%)',ylim=(93.2,95.9))
    panel(axes[0,1],'b','Wavefront curvature')
    for y in (-.1,.1):axes[0,1].axhline(y,color='#777777',ls='--',lw=.8)
    axes[0,1].set(ylabel='Normalized curvature $C$',ylim=(-.12,.115))
    panel(axes[1,0],'c','Retained angular band')
    axes[1,0].axhline(1.1,color='#777777',ls='--',lw=.8)
    axes[1,0].set(ylabel='Retained-band ratio $Q$',ylim=(.98,1.56))
    panel(axes[1,1],'d','Width change and power closure')
    for y,style in ((2,'--'),(5,':')):axes[1,1].axhline(y,color='#888888',ls=style,lw=.8)
    axes[1,1].set(ylabel='Width change / closure (%)',ylim=(-.15,6.25))
    axes[1,1].legend(handles=[Line2D([],[],color='#333333',marker='o',lw=1.2,label='Max. width change'),Line2D([],[],color='#333333',ls=':',marker='x',lw=1.2,label='Native power closure')],loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=9.2,handlelength=1.6)
    for ax in axes.flat:ax.set(xlabel='Mesh resolution (cells/µm)',xticks=[8,16,24,32,40],xlim=(6.5,41.5))
    fig.legend(handles=[Line2D([],[],color=c,marker='o',lw=1.3,label=f'$H={H:g}$ µm') for H,c in HEIGHT_COLORS.items()]+
        [Line2D([],[],color='#444444',marker='*',ls='',ms=8,label='Finest saved result')],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.51,.995),columnspacing=1.1,handlelength=1.5)
    fig.subplots_adjust(left=.12,right=.98,bottom=.1,top=.875,hspace=.57,wspace=.5)
    save(fig,'fig05_mesh_and_numerical_verification')


CAPTIONS='''Figure 2. Optimization and the contribution beyond the physics-derived seed (proxy results only). (a) Raw continuous-relief objective values, evaluated before each update at sampled training heights; different points may use different H and are not a fixed-H learning curve. Three random seeds are retained for each method; bold curves identify the selected NN seed47 and direct-table seed11. (b) Mean quantized proxy objective over the same six validation heights at each checkpoint. Stars mark the selected checkpoints of the selected seed arms. (c) Worst validation useful efficiency at the best checkpoint of each seed; stars mark the selected arms. Checkpoint and seed selection used the full lexicographic validation key, not efficiency alone. (d) Change in proxy useful efficiency relative to the physics seed, after50nm height quantization. Both methods use the same height-specific field-derived seed; the NN predicts residual geometry rather than optical fields. H75 participated in validation and is not an untouched test. These small proxy differences do not establish a matched independent-Maxwell advantage for the NN.\n\nFigure 5. Independent-Maxwell numerical refinement of the same frozen50nm-quantized NN profile within each height. (a) Useful efficiency, (b) normalized curvature C, and (c) retained-band angular ratio Q for every available nominal mesh atH40,75,100µm, including earlier failed checks. Stars show the finest saved results: record43/r32,69/r32,83/r40. The H75 sequence starts at24cells/µm because no coarserH75 field was acquired. Dashed lines in(b,c) show the frozen nominal limits |C|≤0.1 andQ≤1.10; Q-change itself is diagnostic rather than a mesh-convergence gate. (d) Solid curves give the maximum relative x/y diameter change over all five processing planes between successive available meshes (plotted at the fine resolution); dotted curves give the larger device/reference native power-balance error. The frozen limits are2% and5%, respectively. The complete last-pair criteria also include useful-efficiency change, fitted-core divergence, |ΔC|≤0.005, absolute curvature and remaining curvature margin. Their original failed and final accepted pair values are retained in the source-data CSV. The original PML/padding/duration perturbations were sampled separately at16cells/µm, not at each finest mesh; padding also changes the native aperture. These sampled checks do not establish all-angle, finest-grid temporal/boundary or ISO-M² convergence. Record83 was recovered from complete saved fields; its original process exit/finish was unrecorded and its analysis was completed later. Scientific checks are separate from workflow-clock acceptance.\n'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path(__file__).parent);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('fig02_*')) or any(out.glob('fig05_*')):raise FileExistsError('Use a fresh figure output version')
    own=sha(__file__);data=collect()
    for key,name in [('training','fig02_training_objectives.csv'),('validation','fig02_validation_checkpoints.csv'),('seeds','fig02_seed_selection.csv'),
        ('contribution','fig02_proxy_seed_NN_direct.csv'),('mesh','fig05_nominal_mesh_records.csv'),('pairs','fig05_successive_mesh_checks.csv'),('controls','fig05_original_coarse_controls.csv')]:
        write_csv(out/name,data[key])
    write_csv(out/'fig02_requested_heights_proxy.csv',[r for r in data['contribution'] if r['H_um'] in SELECTED])
    (out/'fig02_fig05_captions.txt').write_text(CAPTIONS,encoding='utf-8')
    plot(data,out)
    assert sha(__file__)==own and all(sha(p)==digest for p,digest in INPUTS.items())
    manifest={'status':'SAVED_JSON_PUBLICATION_FIGURES_NO_NEW_OPTICAL_CALCULATIONS','created_utc':datetime.now(timezone.utc).isoformat(),
        'script_file':str(Path(__file__).resolve()),'script_sha256':own,'input_sha256':INPUTS,
        'selected_finest_records':SELECTED,'fig02_scope':'Proxy optimization and proxy comparison only; no NN independent-Maxwell superiority claim',
        'fig05_scope':'Same frozen NN profile within each height, full available nominal history; sampled coarse controls separate',
        'height75_validation_observed':True,'global_diffraction_or_full_M2_or_experiment_claimed':False,'new_optical_solves':0,'training_calls':0,'field_reconstructions':0,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file() and (p.name.startswith('fig02_') or p.name.startswith('fig05_'))}}
    with (out/'fig02_fig05_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'manifest_sha256':sha(out/'fig02_fig05_manifest.json'),'source_sha256':own,'JSON_inputs':len(INPUTS)}))


if __name__=='__main__':main()
