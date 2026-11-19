"""Compare two forward models of the same frozen NN geometry; launch no optics.

The NN generates geometry. Field/beam curves are a proxy forward prediction
of that geometry versus a direct independent Maxwell simulation of it.
--output must be a fresh stage2/forward_comparison/version directory.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_height_family_nn_direct_v2 import (
    HEIGHTS,OPTICAL_KEYS,EXCLUDED_VARIANTS,collect,read,sha,finite,close,
    local_path,safe_json,_band)
from report_height_family_maxwell_v2 import compare_analyses


def gaussian_diameters(D,zs,wavelength,n):
    """Paraxial declared Gaussian, waist D at the output reference; not a bound."""
    if not all(finite(v) and v>0 for v in (D,wavelength,n)):
        raise ValueError('Positive finite diameter, wavelength and ambient index required')
    if not all(finite(z) and z>=0 for z in zs):raise ValueError('Finite nonnegative downstream distances required')
    zr=math.pi*n*(D/2)**2/wavelength
    return [D*math.sqrt(1+(z/zr)**2) for z in zs]


def status(checks,missing=()):
    return {'status':'UNRESOLVED' if missing else ('PASS' if all(checks.values()) else 'FAIL'),
            'checks':checks,'failed_checks':[k for k,v in checks.items() if not v],
            'missing_evidence':list(missing)}


def build_comparison(data,limits):
    """Descriptive selection, never a full-family or boundary acceptance gate."""
    config=data['config'];gates=config['natural_training']['investigation_gates']
    proxy={r['H_um']:r for r in data['proxy_records'] if r['method']=='NN'}
    histories=[r for r in data['Meep_records'] if r['method']=='NN' and r['job_type'] not in EXCLUDED_VARIANTS]
    inputs=dict(data['input_sha256']);analyses={}
    def analysis(row):
        name=row['analysis_file']
        if name not in analyses:
            if sha(name)!=row['analysis_sha256']:raise ValueError('Saved analysis changed after collection')
            analyses[name]=read(name)
        return analyses[name]
    rows=[]
    for H in HEIGHTS:
        prediction=proxy.get(H)
        candidates=[r for r in histories if close(r['H_um'],H)]
        # Never filter on native/nominal PASS before selecting the finest field.
        measured=max(candidates,key=lambda r:(r['resolution'],histories.index(r))) if candidates else None
        row={'H_um':H,'height_use':'validation-observed; influenced seed/checkpoint selection' if H==75 else 'training height',
             'proxy':prediction,'Meep':measured,'ideal_declared_Gaussian':[],
             'selection_rule':'Finest completed, provenance-verified NN record; latest ledger order breaks mesh ties; no passing-record fallback',
             'same_geometry':False,'unfinished_finer_attempts':[]}
        if prediction:
            zs=config['evaluation']['processing_planes_um']
            ds=gaussian_diameters(prediction['declared_target_diameter_um'],zs,config['wavelength_um'],config['materials']['ambient']['n'])
            row['ideal_declared_Gaussian']=[{'z_um':z,'diameter_um':d} for z,d in zip(zs,ds)]
        if measured is None:
            row.update(native=status({},['Completed verified NN Maxwell field missing']),
                       nominal_optical=status({},['Completed verified NN Maxwell analysis missing']),
                       selected_mesh=status({},['Completed verified NN Maxwell mesh pair missing']))
            rows.append(row);continue
        row['same_geometry']=prediction is not None and measured['design_sha256']==prediction['design_sha256']
        if not row['same_geometry']:raise ValueError('Proxy and Maxwell must describe identical frozen NN geometry')
        raw=analysis(measured)
        missing=[*measured.get('native_missing_evidence',[])]
        row['native']=status(measured['native_checks'],missing)
        # Recompute all nominal flags from finite observables as well as
        # retaining saved flags; a stale True must not conceal NaN data.
        flags=measured['optical_gates']
        absent=['Stored optical gate '+key for key in OPTICAL_KEYS if flags.get(key) is None]
        values={'transmission':raw.get('conservative_T'),'declared_overlap':raw.get('declared_overlap'),
                'useful_efficiency':measured.get('eta'),'curvature':measured.get('C'),
                'angular_second_moment':measured.get('Q'),'sidelobe':raw.get('max_sidelobe_fraction')}
        absent.extend('Finite nominal observable '+key for key,value in values.items() if not finite(value))
        crossings=bool(measured['planes']) and all(finite(p.get(key)) and p[key]>0
            for p in measured['planes'] for key in ('diameter_x_um','diameter_y_um'))
        if not crossings:absent.append('Finite positive processing-plane widths')
        targets=raw.get('metrics',{}).get('target_widths_um',[])
        target0=targets[0] if targets else []
        first=measured['planes'][0] if measured['planes'] else {}
        widths=[first.get('diameter_x_um'),first.get('diameter_y_um')]
        target_valid=len(target0)==2 and all(finite(v) and v>0 for v in target0)
        if not target_valid:absent.append('Frozen vector target widths')
        computed={'transmission':finite(values['transmission']) and values['transmission']>=gates['transmission_min'],
            'declared_overlap':finite(values['declared_overlap']) and values['declared_overlap']>=gates['declared_Gaussian_overlap_min'],
            'useful_efficiency':finite(values['useful_efficiency']) and values['useful_efficiency']>=gates['useful_efficiency_min'],
            'curvature':finite(values['curvature']) and abs(values['curvature'])<=gates['max_abs_normalized_curvature'],
            'angular_second_moment':finite(values['angular_second_moment']) and 0<=values['angular_second_moment']<=gates['max_angular_second_moment_ratio'],
            'sidelobe':finite(values['sidelobe']) and 0<=values['sidelobe']<=gates['max_sidelobe_fraction'],
            'valid_width_crossings':crossings,'diameter':False}
        if target_valid and all(finite(v) and v>0 for v in widths):
            computed['diameter']=max(abs(w/t-1) for w,t in zip(widths,target0))<=gates['output_diameter_relative_tolerance']
        checks={'stored_'+key:flags.get(key) is True for key in OPTICAL_KEYS}
        checks.update(computed)
        checks['stored_flags_consistent_with_observables']=not absent and all(flags.get(key)==value for key,value in computed.items())
        row['nominal_optical']=status(checks,absent)
        coarser=[r for r in candidates if r['resolution']<measured['resolution']
            and r['design_sha256']==measured['design_sha256']
            and all(close(r['settings'].get(k),measured['settings'].get(k)) for k in ('H_um','pml_um','padding_um','duration'))]
        if coarser:
            prior=max(coarser,key=lambda r:(r['resolution'],histories.index(r)))
            comparison=compare_analyses(analysis(prior),raw,config,limits,require_collimation=True)
            missing=[*comparison['missing_evidence'],*prior.get('native_missing_evidence',[]),*measured.get('native_missing_evidence',[])]
            row['selected_mesh']={**status({'both_native_references':prior['native_all_pass'] and measured['native_all_pass'],
                'declared_last_pair':comparison['status']=='PASS'},missing),
                'resolutions':[prior['resolution'],measured['resolution']],'prior_identity':prior['identity'],
                'fine_identity':measured['identity'],'comparison':comparison}
        else:row['selected_mesh']=status({},['Coarser completed field with identical geometry/PML/padding/duration required'])
        if measured['resolution']<24.:
            row['selected_mesh']['status']='UNRESOLVED'
            row['selected_mesh']['missing_evidence'].append('Declared NN fine mesh >=24 cells/um is missing')
        verified={r['identity'] for r in candidates}
        for attempt in data['all_attempts']:
            if (attempt.get('method')=='NN' and close(attempt.get('H_um'),H)
                and attempt.get('job_type','nominal') not in EXCLUDED_VARIANTS
                and finite(attempt.get('resolution')) and attempt['resolution']>=measured['resolution']
                and all(close(attempt.get(k),measured['settings'].get(k)) for k in ('pml_um','padding_um','duration'))
                and attempt['identity'] not in verified):row['unfinished_finer_attempts'].append(attempt)
        if row['unfinished_finer_attempts']:
            row['selected_mesh']['status']='UNRESOLVED'
            row['selected_mesh']['missing_evidence'].append('A same-settings finest/finer attempt is unfinished or unverified')
        rows.append(row)
    if not all(sha(path)==digest for path,digest in inputs.items()):raise RuntimeError('Immutable comparison input changed while reading')
    return safe_json({'status':'DESCRIPTIVE_SAME_NN_GEOMETRY_PROXY_VS_DIRECT_MAXWELL',
        'rows':rows,'NN_nominal_histories':histories,'all_attempts':data['all_attempts'],
        'config':config,'numerical_limits':limits,'budget':data['budget'],
        'selected_source_signature':data['source_signature'],'selected_seeds':data['selection'],
        'geometry_generation':'Physics-derived height-specific field seed plus NN residual logits, followed by fabrication-height quantization',
        'input_sha256':inputs,'ledger_snapshot_sha256':data['ledger_snapshot_sha256'],
        'missing_or_unverified_evidence':data['missing_evidence'],
        'labels':{'proxy':'Proxy forward prediction of physics-seeded NN geometry',
                  'Meep':'Direct Maxwell simulation of same geometry',
                  'ideal':'Ideal declared Gaussian (paraxial reference)'},
        'full_optical_acceptance_claimed':False,'NN_superiority_claimed':False,
        'global_diffraction_optimum_claimed':False,'experimental_validation_claimed':False,
        'limitations':['The physics-seeded NN generates top-relief geometry; it does not generate the plotted optical fields.',
            'Each height uses its field-derived physics seed plus NN residual logits and final height quantization; this is not an unassisted height-to-lens prediction.',
            'Downstream z is measured from the declared output monitor at H+4 um: H100 uses absolute z104 um, not the facet or relief base.',
            'This plot uses existing native analyses unchanged; it does not resample the saved fields or establish sampling convergence.',
            'Direct Maxwell means forward simulation of that NN geometry, not the separately named non-neural direct-table optimizer.',
            'Maxwell curves use the finest completed verified mesh at each height; meshes can differ across heights. Failures remain plotted and labeled.',
            'A selected mesh PASS covers only the displayed same-settings pair, not all boundary/duration checks or full-family acceptance.',
            'Band spans x/y beam widths, not an uncertainty interval. Missing data remain gaps.',
            'The ideal curve is a declared paraxial Gaussian reference that still diffracts; it is not a proven global bound for the aperture.',
            'H75 was validation-observed and influenced seed/checkpoint selection; it is not held out.',
            'Meep useful efficiency uses conservative native/field normalization; proxy efficiency is a different forward-model prediction.',
            'Synthetic Gaussian source and assumed lossless materials remain; no full-angle M2, experimental or fabrication acceptance.']})


def render(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);labels=data['labels'];blue='#2166ac';orange='#d6604d'
    with plt.rc_context({'font.size':9,'legend.fontsize':7,'axes.titlesize':10}):
        fig,axes=plt.subplots(2,3,figsize=(15,9.5))
        for ax,row in zip(axes[0],data['rows']):
            for key,color,style,marker in [('proxy',blue,'--',None),('Meep',orange,'-','o')]:
                record=row[key]
                if record:
                    _band(ax,[p['z_um'] for p in record['planes']],
                          [[p['diameter_x_um'],p['diameter_y_um']] for p in record['planes']],color,style,labels[key],marker)
            ideal=row['ideal_declared_Gaussian']
            if ideal:ax.plot([p['z_um'] for p in ideal],[p['diameter_um'] for p in ideal],':',color='#333333',lw=1.5,label=labels['ideal'])
            measured=row['Meep']
            if measured:
                pair=row['selected_mesh'].get('resolutions')
                mesh_label=f"{pair[0]:g}→{pair[1]:g}" if pair else 'missing pair'
                text=f"Meep r{measured['resolution']:g} cells/µm\nNominal optical: {row['nominal_optical']['status']}\nNative/reference: {row['native']['status']}\nMesh {mesh_label}: {row['selected_mesh']['status']}"
            else:text='Meep missing; no inferred curve'
            ax.text(.025,.97,text,transform=ax.transAxes,va='top',fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'0.8','alpha':.9,'pad':4})
            domain='validation-observed' if row['H_um']==75 else 'training height'
            ax.set(title=f"H={row['H_um']:g} µm ({domain})",xlabel='Distance after output monitor at z=H+4 µm (µm)',
                   ylabel='1/e² intensity diameter (µm)')
            ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=True)
        gates=data['config']['natural_training']['investigation_gates']
        for ax,key,title,ylabel in zip(axes[1],('eta','C','Q'),
            ('Useful efficiency','Normalized wavefront curvature','Retained-band angular moment'),
            ('Useful modal efficiency (%)','C','Q (not full-angle M²)')):
            scale=100 if key=='eta' else 1
            for arm,color,style,marker in [('proxy',blue,'--','s'),('Meep',orange,'None','o')]:
                values=[scale*row[arm][key] if row[arm] and finite(row[arm].get(key)) else np.nan for row in data['rows']]
                ax.plot(HEIGHTS,values,linestyle=style,marker=marker,color=color,label=labels[arm],ms=6)
            for row in data['rows']:
                measured=row['Meep']
                if measured and finite(measured.get(key)):
                    ax.annotate(f"r{measured['resolution']:g}",(row['H_um'],scale*measured[key]),
                                xytext=(4,6),textcoords='offset points',fontsize=8,color=orange)
            if key=='eta':ax.axhline(100*gates['useful_efficiency_min'],color='firebrick',ls=':',lw=1,label='Frozen nominal gate')
            if key=='C':
                cap=gates['max_abs_normalized_curvature'];ax.axhspan(-cap,cap,color='0.95',zorder=-2)
                for value in (-cap,cap):ax.axhline(value,color='firebrick',ls=':',lw=1)
            if key=='Q':ax.axhline(gates['max_angular_second_moment_ratio'],color='firebrick',ls=':',lw=1,label='Frozen nominal gate')
            ax.set(title=title,xlabel='Cylinder height H (µm)',ylabel=ylabel,xticks=HEIGHTS)
            ax.grid(alpha=.2);ax.legend(loc='best',frameon=False)
        fig.suptitle('Frozen physics-seeded NN geometry: proxy prediction versus direct Maxwell forward simulation\n'
                     'Same geometry per height; finest completed Meep mesh shown even when failed or unconverged',fontsize=13)
        fig.text(.5,.045,'The NN generates geometry, not these fields. Bands span x/y widths. Different heights may use different Meep meshes. '
                 'H75 influenced validation selection.',ha='center',fontsize=8)
        fig.text(.5,.022,'No full-family acceptance, NN superiority, global diffraction optimum, full-angle M² or experimental validation is claimed.',
                 ha='center',fontsize=8)
        fig.subplots_adjust(top=.89,bottom=.13,left=.065,right=.985,hspace=.4,wspace=.26)
        with (output/'nn_proxy_vs_direct_maxwell.png').open('xb') as stream:
            fig.savefig(stream,format='png',dpi=190,metadata={'Description':data['status'],'Provenance':'comparison.json'})
        plt.close(fig)


def human_report(data):
    lines=['SAME FROZEN PHYSICS-SEEDED NN GEOMETRY: TWO FORWARD MODELS',data['status'],'',
           data['labels']['proxy']+' versus '+data['labels']['Meep'],
           'H(um) selected_Meep_mesh native nominal_optical selected_mesh eta(%) C Q']
    for row in data['rows']:
        m=row['Meep']
        if m:
            def f(key,places):return f"{m[key]:.{places}f}" if finite(m.get(key)) else 'NA'
            eta=f"{100*m['eta']:.5f}" if finite(m.get('eta')) else 'NA'
            lines.append(f"{row['H_um']:5g} {m['resolution']:g} {row['native']['status']} {row['nominal_optical']['status']} "
                         f"{row['selected_mesh']['status']} {eta} {f('C',6)} {f('Q',6)}")
        else:lines.append(f"{row['H_um']:5g} MISSING")
        for area in ('native','nominal_optical','selected_mesh'):
            for reason in row[area]['failed_checks']:lines.append(f'  {area} failed: '+reason)
            for reason in row[area]['missing_evidence']:lines.append(f'  {area} missing: '+reason)
    return '\n'.join(lines+['',*data['limitations']])+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve previous output; choose a fresh version')
    if not output.is_relative_to(folder/'forward_comparison') or output==folder/'forward_comparison':
        raise ValueError('Output must be below stage2/forward_comparison/version')
    data=collect(folder,methods=('NN',))
    proposal=local_path(data['prior_stage_linkage']['folder'])/'maxwell_validation_proposal.json'
    if sha(proposal)!=data['input_sha256'][str(proposal.resolve())]:raise ValueError('Frozen numerical proposal changed')
    result=build_comparison(data,read(proposal)['numerical_acceptance'])
    result['utility_sha256']=sha(Path(__file__))
    output.mkdir(parents=True,exist_ok=False)
    import json
    with (output/'comparison.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write(human_report(result))
    render(result,output)
    manifest={'status':'DESCRIPTIVE_GRAPH_DELIVERY_NO_OPTICAL_SOLVES',
        'utility_sha256':sha(Path(__file__)),'input_sha256':result['input_sha256'],
        'files':{name:{'sha256':sha(output/name),'bytes':(output/name).stat().st_size}
                 for name in ('comparison.json','REPORT.txt','nn_proxy_vs_direct_maxwell.png')},
        'optical_solves_launched':0,'training_runs_launched':0}
    with (output/'delivery_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'status':result['status'],'optical_solves_launched':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
