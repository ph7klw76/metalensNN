"""Plot saved nominal height-family Maxwell analyses; never start a solver.

The PNG is descriptive, not an acceptance report. Its .data.json sidecar
records the actual hashes, values, skipped entries and missing data. Only
completed nominal NN analyses and approved same-geometry reserve-mesh 32
analyses with verified provenance are plotted. Duration variants are excluded.
Optional proxy efficiency comes from the frozen campaign's NN_height_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def matching(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def inside(folder, name):
    if not isinstance(name, str) or not name:
        raise ValueError('Missing ledger path')
    result=(folder/name).resolve()
    if not result.is_relative_to(folder.resolve()):
        raise ValueError('Ledger path escapes validation folder')
    return result


def collect(folder, proxy_summary=None):
    """Read and hash existing evidence, leaving unavailable values as None."""
    folder=Path(folder).resolve()
    config_path=folder/'config.json'; ledger_path=folder/'ledger.json'
    config=read_json(config_path); ledger=read_json(ledger_path)
    entries=ledger if isinstance(ledger,list) else ledger.get('entries',ledger.get('solves',[]))
    if not isinstance(entries,list):raise ValueError('Ledger entries must be a list')
    proposal_path=folder/'maxwell_validation_proposal.json'
    proposal=read_json(proposal_path) if proposal_path.is_file() else {}
    heights=[float(x) for x in proposal.get('nominal_heights_um',config['natural_training']['train_H_um'])]
    nominal_meshes=[float(x) for x in proposal.get('meshes_cells_per_um',[8,16,24])]
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    hashes={str(config_path):sha256(config_path),str(ledger_path):sha256(ledger_path)}
    if proposal_path.is_file():hashes[str(proposal_path)]=sha256(proposal_path)
    reserve_approval={'approved':False,'reason':'Saved approval/amendment not available'};reserve_cases={}
    try:
        receipt_path=folder/'approval_receipt.json';amendment_path=folder/'cumulative_amendment.json'
        receipt=read_json(receipt_path);amendment=read_json(amendment_path)
        hashes[str(receipt_path)]=sha256(receipt_path);hashes[str(amendment_path)]=sha256(amendment_path)
        checks={'user_approval':receipt.get('status')=='APPROVED_BY_USER',
                'proposal_hash':proposal_path.is_file() and receipt.get('proposal_sha256')==sha256(proposal_path),
                'ledger_approval_hash':isinstance(ledger,dict) and ledger.get('approval_sha256')==sha256(receipt_path)}
        reserve_approval={'approved':all(checks.values()),'checks':checks}
        reserve_cases={float(case['H_um']):case for case in amendment.get('cases',[]) if case.get('method')=='NN'}
    except (OSError,ValueError,KeyError,TypeError) as error:reserve_approval['reason']=str(error)
    has_reserve_attempt=any(e.get('method')=='NN' and e.get('job_type')=='reserve_mesh' for e in entries)
    meshes=sorted(set(nominal_meshes+([32.] if reserve_approval['approved'] or has_reserve_attempt else [])))
    nominal24={float(e['H_um']):e for e in entries if e.get('method')=='NN'
        and e.get('job_type','nominal')=='nominal' and matching(e.get('resolution'),24.)
        and finite(e.get('H_um')) and e.get('status')=='completed'}
    records={}; skipped=[]; missing=[]
    for entry in entries:
        kind=entry.get('job_type','nominal')
        if entry.get('method')!='NN' or kind not in ('nominal','reserve_mesh'):continue
        label={'id':entry.get('id'),'H_um':entry.get('H_um'),'resolution':entry.get('resolution'),
               'job_type':kind,'status':entry.get('status'),'analysis_status':entry.get('analysis_status')}
        if entry.get('status')!='completed' or entry.get('analysis_status') not in (None,'completed'):
            skipped.append({**label,'reason':'Solve/analysis not completed'});continue
        H=entry.get('H_um'); resolution=entry.get('resolution')
        valid_mesh=matching(resolution,32.) if kind=='reserve_mesh' else finite(resolution) and float(resolution) in nominal_meshes
        if not finite(H) or not valid_mesh or float(H) not in heights:
            skipped.append({**label,'reason':'Undeclared or missing height/mesh'});continue
        try:
            if kind=='reserve_mesh':
                if not reserve_approval['approved']:raise ValueError('Reserve-mesh saved approval linkage is unresolved')
                base=nominal24.get(float(H));case=reserve_cases.get(float(H))
                if base is None or case is None:raise ValueError('Reserve mesh requires a listed frozen case and completed nominal24')
                if not all(matching(entry.get(k),base.get(k)) for k in ('H_um','pml_um','padding_um','duration')):
                    raise ValueError('Reserve mesh changes duration or other controlled settings')
                design_path=inside(folder,case.get('design_file'));actual_design_sha=sha256(design_path)
                hashes[str(design_path)]=actual_design_sha
                if actual_design_sha!=entry.get('design_sha256') or actual_design_sha!=base.get('design_sha256') or actual_design_sha!=case.get('design_sha256'):
                    raise ValueError('Reserve mesh frozen geometry hash differs from nominal24/amendment')
            analysis_path=inside(folder,entry.get('analysis_file'))
            raw_path=inside(folder,entry.get('record_file'))
            analysis=read_json(analysis_path);raw=read_json(raw_path)
            actual_analysis_sha=sha256(analysis_path);actual_raw_sha=sha256(raw_path)
            hashes[str(analysis_path)]=actual_analysis_sha;hashes[str(raw_path)]=actual_raw_sha
            recorded=entry.get('analysis_sha256')
            if not isinstance(recorded,str) or recorded!=actual_analysis_sha:
                raise ValueError('Missing or mismatched ledger analysis_sha256')
            if entry.get('record_sha256') is not None and entry['record_sha256']!=actual_raw_sha:
                raise ValueError('Raw record differs from recorded SHA256')
            settings=raw.get('settings',{})
            if not all(matching(entry.get(k),settings.get(k)) for k in ('H_um','resolution','pml_um','padding_um','duration')):
                raise ValueError('Ledger settings differ from raw record')
            if analysis.get('settings')!=settings:
                raise ValueError('Analysis settings differ from raw record')
            name=str(analysis.get('record','')).replace('\\','/').rsplit('/',1)[-1]
            if name!=raw_path.name:raise ValueError('Analysis names a different raw record')
            if raw.get('config_fingerprint')!=fingerprint:
                raise ValueError('Raw record config fingerprint differs from saved config')
        except (OSError,ValueError,TypeError) as error:
            skipped.append({**label,'reason':str(error)});continue
        row={**label,'H_um':float(H),'resolution':float(resolution),
             'analysis_file':str(analysis_path.relative_to(folder)),
             'analysis_sha256':actual_analysis_sha,'record_file':str(raw_path.relative_to(folder)),
             'record_sha256':actual_raw_sha,'C':None,'Q':None,'useful_efficiency':None,
             'series_type':'approved_reserve_mesh' if kind=='reserve_mesh' else 'nominal',
             'plot_marker':'D' if kind=='reserve_mesh' else 'o',
             'core_half_angles_deg':[None,None],
             'native_reconstruction_pass':analysis.get('reconstruction_pass'),
             'synthetic_optical_pass':analysis.get('synthetic_optical_pass')}
        for name,key in [('C','normalized_curvature'),('Q','angular_second_moment_ratio'),
                         ('useful_efficiency','conservative_useful_efficiency')]:
            value=analysis.get(key)
            if finite(value):row[name]=float(value)
            else:missing.append({**label,'metric':name,'reason':'Absent or nonfinite saved metric'})
        fit=analysis.get('metrics',{}).get('fitted_core_divergence',{})
        axes=fit.get('axes',[])
        for axis in range(2):
            angle=axes[axis].get('half_angle_rad') if len(axes)>axis else None
            if fit.get('valid') is True and finite(angle) and angle>0:
                row['core_half_angles_deg'][axis]=math.degrees(angle)
            else:missing.append({**label,'metric':f'core_half_angle_{axis}','reason':'Absent or invalid saved core fit'})
        key=(float(H),float(resolution))
        if key in records:
            skipped.append({**{k:records[key][k] for k in ('id','H_um','resolution')},
                            'reason':'Superseded by later completed nominal analysis in ledger order'})
        records[key]=row
    proxy_path=Path(proxy_summary).resolve() if proxy_summary else folder.parent/'NN_height_summary.csv'
    proxy=[]
    if proxy_path.is_file():
        hashes[str(proxy_path)]=sha256(proxy_path)
        with proxy_path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                try:H=float(row['H_um']);eta=float(row['useful_efficiency'])
                except (KeyError,ValueError,TypeError):continue
                if H in heights and math.isfinite(eta):proxy.append({'H_um':H,'useful_efficiency':eta})
    else:missing.append({'metric':'proxy_efficiency','reason':'Proxy summary CSV absent','path':str(proxy_path)})
    absent=[{'H_um':H,'resolution':r} for H in heights for r in nominal_meshes if (H,r) not in records]
    absent_reserve=[{'H_um':H,'resolution':32.} for H in heights if (H,32.) not in records] if 32. in meshes else []
    values=list(records.values());values.sort(key=lambda row:(row['H_um'],row['resolution']))
    nominal_values=[row for row in values if row['job_type']=='nominal']
    reserve_values=[row for row in values if row['job_type']=='reserve_mesh']
    gates=config['natural_training']['investigation_gates']
    return {'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Completed nominal and approved same-geometry reserve-mesh synthetic NN analyses; descriptive plots, not acceptance',
        'complete_nominal_grid':not absent and not [item for item in missing if item.get('job_type')!='reserve_mesh'],
        'expected_nominal_points':len(heights)*len(nominal_meshes),'plotted_nominal_points':len(nominal_values),
        'plotted_reserve_mesh_points':len(reserve_values),'plotted_total_points':len(values),
        'nominal_meshes_cells_per_um':nominal_meshes,'reserve_mesh_approval':reserve_approval,
        'heights_um':heights,'meshes_cells_per_um':meshes,'records':values,'proxy_efficiency':proxy,
        'absent_nominal_points':absent,'absent_reserve_mesh_points':absent_reserve,
        'missing_metrics':missing,'skipped_entries':skipped,
        'input_sha256':hashes,'config_fingerprint':fingerprint,
        'wavelength_um':config.get('wavelength_um'),'mfd_um':config.get('mfd_um'),
        'polymer_index':config.get('materials',{}).get('polymer',{}).get('n'),
        'gates':{'C_absolute':gates['max_abs_normalized_curvature'],
                 'Q_maximum':gates['max_angular_second_moment_ratio'],
                 'useful_efficiency_minimum':gates['useful_efficiency_min']},
        'limitations':['Different top profile for each cylinder height.',
            'Missing or invalid values are gaps; nominal data are not convergence acceptance.',
            'Reserve 32 diamonds retain prior nominal points; duration variants are excluded to avoid mixed numerics.',
            'Q is a retained-band angular second-moment ratio, not a full all-angle or ISO M2 measure.',
            'Gaussian source and assumed phase index/loss; no experimental or full MFD-tolerance validation.']}


def render(data, output):
    """Render the scientific figure using Matplotlib only at plot time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.exists():raise FileExistsError('Refusing to overwrite '+str(output))
    heights=data['heights_um'];meshes=data['meshes_cells_per_um']
    nominal_meshes=data.get('nominal_meshes_cells_per_um',meshes)
    records={(row['H_um'],row['resolution']):row for row in data['records']}
    def value(H,r,key,axis=None):
        row=records.get((H,r),{})
        return row.get(key,[None,None])[axis] if axis is not None else row.get(key)
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'legend.fontsize':8,'savefig.dpi':180}):
        fig,axes=plt.subplots(2,2,figsize=(13.0,9.5))
        C,Q,eta,angle=axes.flat
        palette=plt.get_cmap('viridis')
        for k,H in enumerate(heights):
            color=palette(k/max(1,len(heights)-1))
            for ax,key in [(C,'C'),(Q,'Q')]:
                ax.plot(nominal_meshes,[value(H,r,key) for r in nominal_meshes],'-o',color=color,lw=1.4,ms=4,label=f'H={H:g} µm')
                if 32. in meshes and value(H,32.,key) is not None:
                    ax.plot([24.,32.],[value(H,24.,key),value(H,32.,key)],':',color=color,lw=1)
                    ax.plot([32.],[value(H,32.,key)],'D',color=color,ms=6)
        if data.get('plotted_reserve_mesh_points',0):
            from matplotlib.lines import Line2D
            for ax in (C,Q):
                ax.add_line(Line2D([],[],color='black',marker='D',ls='None',label='32 cells/µm reserve mesh'))
        gate=data['gates']['C_absolute']
        C.axhspan(-gate,gate,color='0.95',zorder=-1)
        C.axhline(gate,color='firebrick',ls='--',lw=1);C.axhline(-gate,color='firebrick',ls='--',lw=1)
        C.set(title='(a) Phase-current curvature',xlabel='Mesh resolution (cells/µm)',ylabel='Normalized curvature C')
        Q.axhline(data['gates']['Q_maximum'],color='firebrick',ls='--',lw=1,label='Q investigation gate')
        Q.set(title='(b) Retained-band angular ratio',xlabel='Mesh resolution (cells/µm)',ylabel='Angular second-moment ratio Q')
        mesh_colors=['#2166ac','#d6604d','#4d9221','#762a83']
        for k,r in enumerate(meshes):
            color=mesh_colors[k%len(mesh_colors)]
            reserve=matching(r,32.) and r not in nominal_meshes
            marker='D' if reserve else 'o';suffix=' (reserve mesh)' if reserve else ''
            eta.plot(heights,[100*value(H,r,'useful_efficiency') if value(H,r,'useful_efficiency') is not None else None
                              for H in heights],(':' if reserve else '-')+marker,color=color,label=f'Maxwell {r:g} cells/µm{suffix}',ms=4)
            for axis,style in [(0,'-'),(1,'--')]:
                angle.plot(heights,[value(H,r,'core_half_angles_deg',axis) for H in heights],style+marker,
                           color=color,ms=3,label=f'{r:g}{" reserve" if reserve else ""} cells/µm; {"x" if axis==0 else "y"}')
        proxy={row['H_um']:row['useful_efficiency'] for row in data['proxy_efficiency']}
        if proxy:
            eta.plot(heights,[100*proxy[H] if H in proxy else None for H in heights],':s',color='black',
                     ms=4,label='Frozen one-way proxy')
        eta.axhline(100*data['gates']['useful_efficiency_minimum'],color='firebrick',ls='--',lw=1)
        eta.set(title='(c) Useful efficiency',xlabel='Cylinder height H (µm)',ylabel='Useful efficiency (%)')
        angle.set(title='(d) Fitted Gaussian-core divergence',xlabel='Cylinder height H (µm)',ylabel='Core divergence half-angle (degrees)')
        for ax in axes.flat:
            ax.grid(alpha=.2);ax.legend(ncol=2,loc='best');ax.set_axisbelow(True)
        C.set_xticks(meshes);Q.set_xticks(meshes);eta.set_xticks(heights);angle.set_xticks(heights)
        state='Complete saved nominal grid' if data['complete_nominal_grid'] else 'INCOMPLETE saved nominal grid'
        fig.suptitle('Height-specific NN metalenses: nominal synthetic Maxwell',fontsize=15,y=.985)
        fig.text(.5,.946,f"{state}: {data['plotted_nominal_points']}/{data['expected_nominal_points']} nominal + "
                 f"{data.get('plotted_reserve_mesh_points',0)} reserve analyses  |  "
                 f"λ={data['wavelength_um']} µm, MFD={data['mfd_um']} µm, polymer n={data['polymer_index']}",
                 ha='center',fontsize=10)
        fig.text(.5,.025,'Different profile at each H. Missing values remain gaps. Diamonds: approved 32 reserve mesh. Duration variants excluded.\n'
                 'Dashed red lines are investigation gates. '
                 'Descriptive data; numerical acceptance is separate. Q/core fits do not establish full all-angle M² or experimental performance.',
                 ha='center',va='center',fontsize=9)
        fig.subplots_adjust(left=.085,right=.98,bottom=.11,top=.9,hspace=.3,wspace=.25)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:
            fig.savefig(stream,format='png',metadata={'Description':data['scope'],'Nominal analyses':str(data['plotted_nominal_points']),
                                        'Reserve-mesh analyses':str(data.get('plotted_reserve_mesh_points',0)),
                                        'Provenance':'See adjacent .data.json for actual input SHA256 and missing records'})
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--proxy-summary',type=Path)
    args=parser.parse_args();folder=args.folder.resolve();output=args.output.resolve()
    if output.suffix.lower()!='.png':raise ValueError('--output must end in .png')
    manifest=output.with_suffix('.data.json')
    for path in (output,manifest):
        if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
        if path.is_relative_to(folder) and any(p in ('records','analyses','raw','frozen_designs','source_snapshots')
                                              for p in path.relative_to(folder).parts[:-1]):
            raise ValueError('Plot output cannot enter raw/provenance directories')
    data=collect(folder,args.proxy_summary)
    if not all(Path(path).is_file() and sha256(path)==value for path,value in data['input_sha256'].items()):
        raise RuntimeError('Input changed while reading; rerun after the campaign is stable')
    render(data,output)
    with manifest.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'PNG':str(output),'data':str(manifest),'nominal_points':data['plotted_nominal_points'],
                      'reserve_mesh_points':data['plotted_reserve_mesh_points'],
                      'complete_nominal_grid':data['complete_nominal_grid']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
