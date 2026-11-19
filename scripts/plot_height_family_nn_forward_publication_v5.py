"""Height-specific publication exports of pinned v3 forward comparisons.

Layout only: two 7.2-inch figures, each in PNG (300 dpi), SVG and PDF.
Scientific selection, metrics, native/nominal/mesh checks and missing-data
handling come unchanged from the SHA-pinned v3 utility. No optics launches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCIENTIFIC_SOURCE=ROOT/'scripts/plot_height_family_nn_forward_v3.py'
EXPECTED_SCIENTIFIC_SHA='aafa34734fbdd6c959278d935e43bfe0ae7e81cf877041b6380ac658d3d3d043'
PREDECESSOR_LAYOUT_SHA='3265f9d447403d9772cf321a8317a146f6ad52336ee31701c1fee127c46be2e7'
FIGURE_WIDTH_IN=7.2
FIGURE_HEIGHT_IN=3.6
RASTER_DPI=300


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_science():
    if sha(SCIENTIFIC_SOURCE)!=EXPECTED_SCIENTIFIC_SHA:
        raise ValueError('Pinned scientific v3 source changed; publication layout must not change scientific selection')
    spec=importlib.util.spec_from_file_location('pinned_nn_forward_science_v3',SCIENTIFIC_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save_formats(fig,output,stem):
    """Keep physical dimensions fixed across raster and vector exports."""
    for extension in ('png','svg','pdf'):
        metadata=({'Title':stem,'Creator':Path(__file__).name,'Subject':'Height-specific physics-seeded NN lenses; same frozen pattern within each proxy/Meep pair'}
                  if extension=='pdf' else {'Description':'Patterns differ across heights; each proxy/Meep pair uses the same frozen NN pattern. comparison.json contains exact provenance.'})
        with (output/f'{stem}.{extension}').open('xb') as stream:
            fig.savefig(stream,format=extension,dpi=RASTER_DPI,facecolor='white',metadata=metadata)


def render_publication(data,output,science):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output=Path(output);blue='#2166ac';orange='#d6604d';gray='#444444';gate_color='#a12635'
    handles=[Line2D([],[],color=blue,ls='--',lw=1.4,label='Proxy forward model'),
             Line2D([],[],color=orange,marker='o',lw=1.3,ms=3.4,label='Direct Meep forward model'),
             Line2D([],[],color=gray,ls=':',lw=1.2,label='Declared Gaussian reference')]
    rc={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(rc):
        fig,axes=plt.subplots(1,3,figsize=(FIGURE_WIDTH_IN,FIGURE_HEIGHT_IN),sharey=True)
        for ax,row in zip(axes,data['rows']):
            for key,color,style,marker in [('proxy',blue,'--',None),('Meep',orange,'-','o')]:
                record=row[key]
                if record:
                    science._band(ax,[p['z_um'] for p in record['planes']],
                        [[p['diameter_x_um'],p['diameter_y_um']] for p in record['planes']],
                        color,style,'',marker)
            ideal=row['ideal_declared_Gaussian']
            if ideal:ax.plot([p['z_um'] for p in ideal],[p['diameter_um'] for p in ideal],':',color=gray,lw=1.2)
            m=row['Meep']
            if m:
                pair=row['selected_mesh'].get('resolutions')
                mesh_label=(f"Mesh {pair[0]:g} to {pair[1]:g}: " if pair else 'Mesh: ')+row['selected_mesh']['status']
                annotation=f"Meep r{m['resolution']:g}\nNative: {row['native']['status']}\nOptical: {row['nominal_optical']['status']}\n{mesh_label}"
            else:annotation='Meep data missing'
            lower=row['H_um']==100
            ax.text(.035,.035 if lower else .975,annotation,transform=ax.transAxes,va='bottom' if lower else 'top',fontsize=8.5,
                bbox={'facecolor':'white','edgecolor':'0.8','alpha':.92,'pad':2.5})
            domain='validation-observed' if row['H_um']==75 else 'training height'
            ax.set_title(f"H = {row['H_um']:g} µm\n{domain}",pad=5)
            ax.set_xticks([0,50,100]);ax.grid(alpha=.2);ax.tick_params(length=3)
        axes[0].set_ylabel('1/e² intensity diameter (µm)')
        fig.suptitle('Height-specific physics-seeded NN lenses: beam propagation\n'
                     'Different patterns across H; same frozen pattern within each proxy/Meep pair',fontsize=9.7,y=.985)
        fig.supxlabel('Distance after the output monitor (µm)',fontsize=9,y=.205)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,
                   frameon=False,handlelength=2.1,columnspacing=1.,handletextpad=.45)
        fig.text(.5,.06,'Output monitor: z = H + 4 µm. Bands span x/y widths; they are not uncertainty bars.',
                 ha='center',fontsize=8.5)
        fig.text(.5,.02,'Finest completed Meep fields are shown, including failures. Missing fields remain blank.',
                 ha='center',fontsize=8.5)
        fig.subplots_adjust(left=.085,right=.985,top=.78,bottom=.31,wspace=.12)
        save_formats(fig,output,'beam_propagation');plt.close(fig)

        fig,axes=plt.subplots(1,3,figsize=(FIGURE_WIDTH_IN,FIGURE_HEIGHT_IN))
        gates=data['config']['natural_training']['investigation_gates']
        for ax,key,title,ylabel in zip(axes,('eta','C','Q'),('Useful efficiency','Wavefront curvature','Angular ratio'),
            ('Useful modal efficiency (%)','Normalized curvature C','Q (retained band)')):
            scale=100 if key=='eta' else 1
            for arm,color,style,marker in [('proxy',blue,'--','s'),('Meep',orange,'None','o')]:
                ys=[scale*r[arm][key] if r[arm] and science.finite(r[arm].get(key)) else np.nan for r in data['rows']]
                ax.plot(science.HEIGHTS,ys,color=color,ls=style,marker=marker,ms=3.8,lw=1.4)
            for row in data['rows']:
                m=row['Meep']
                if not m or not science.finite(m.get(key)):continue
                offset=(-4,-12) if row['H_um']==100 else (4,5)
                if key=='C':offset=(-4,5) if row['H_um']==100 else (4,-12)
                if key=='Q' and row['H_um']==75:offset=(-4,-12)
                ax.annotate(f"r{m['resolution']:g}",(row['H_um'],scale*m[key]),xytext=offset,
                    textcoords='offset points',ha='right' if row['H_um']==100 or key=='Q' and row['H_um']==75 else 'left',fontsize=8.5,color=orange)
            if key=='eta':ax.axhline(100*gates['useful_efficiency_min'],color=gate_color,ls=':',lw=1.)
            if key=='C':
                cap=gates['max_abs_normalized_curvature'];ax.axhspan(-cap,cap,color='0.95',zorder=-2)
                for value in (-cap,cap):ax.axhline(value,color=gate_color,ls=':',lw=1.)
            if key=='Q':ax.axhline(gates['max_angular_second_moment_ratio'],color=gate_color,ls=':',lw=1.)
            ax.set(title=title,ylabel=ylabel,xticks=science.HEIGHTS,xlim=(35,105))
            ax.grid(alpha=.2);ax.tick_params(length=3);ax.margins(y=.12)
        metric_handles=[handles[0],Line2D([],[],color=orange,ls='none',marker='o',ms=3.8,label='Direct Meep (finest per H)'),
                        Line2D([],[],color=gate_color,ls=':',lw=1.,label='Frozen nominal threshold')]
        fig.suptitle('Height-specific physics-seeded NN lenses: forward-model metrics\n'
                     'Different patterns across H; same frozen pattern within each proxy/Meep pair',fontsize=9.7,y=.985)
        fig.supxlabel('Cylinder height H (µm)',fontsize=9,y=.205)
        fig.legend(handles=metric_handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,
                   frameon=False,handlelength=2.,columnspacing=1.,handletextpad=.45)
        fig.text(.5,.06,'H75 is validation-observed. Q is not M². Mesh sizes may differ between heights.',ha='center',fontsize=8.5)
        fig.text(.5,.02,'Nominal or mesh failures remain plotted. No full acceptance, NN advantage or global diffraction bound.',ha='center',fontsize=8.5)
        fig.subplots_adjust(left=.085,right=.985,top=.82,bottom=.31,wspace=.54)
        save_formats(fig,output,'forward_model_metrics');plt.close(fig)


def prepare_data(folder,science):
    collected=science.collect(folder,methods=('NN',))
    proposal=science.local_path(collected['prior_stage_linkage']['folder'])/'maxwell_validation_proposal.json'
    if science.sha(proposal)!=collected['input_sha256'][str(proposal.resolve())]:raise ValueError('Frozen proposal changed')
    data=science.build_comparison(collected,science.read(proposal)['numerical_acceptance'])
    data['utility_sha256']=EXPECTED_SCIENTIFIC_SHA
    data['publication_layout_sha256']=sha(__file__)
    data['publication_layout']={'figure_width_in':FIGURE_WIDTH_IN,'figure_height_in':FIGURE_HEIGHT_IN,
        'raster_dpi':RASTER_DPI,'formats':['png','svg','pdf'],'scientific_logic_changed':False,
        'scientific_source_sha256':EXPECTED_SCIENTIFIC_SHA,'predecessor_layout_sha256':PREDECESSOR_LAYOUT_SHA,
        'primary_font_pt':9,'minimum_font_pt':8.5,
        'geometry_comparison_scope':'Distinct frozen top patterns between heights; identical frozen pattern within each height-specific proxy/Meep comparison'}
    return data


CSV_COLUMNS=('H_um','height_use','forward_model','design_file','design_sha256','record_identity',
    'resolution_cells_per_um','output_reference_z_um','downstream_distance_um',
    'diameter_x_um','diameter_y_um','useful_efficiency_fraction','useful_efficiency_percent',
    'normalized_curvature_C','retained_band_angular_ratio_Q',
    'Gaussian_core_half_angle_x_deg','Gaussian_core_half_angle_y_deg',
    'native_status','nominal_optical_status','selected_mesh_status','selected_mesh_pair_cells_per_um',
    'source_training_record_file','source_training_record_sha256',
    'Meep_record_file','Meep_record_sha256','analysis_file','analysis_sha256')


def csv_rows(data,science):
    """Copy saved output-plane metrics; angles are already degrees in v3 data."""
    training=[(path,digest) for path,digest in data['input_sha256'].items() if Path(path).name=='training_result.json']
    if len(training)!=1:raise ValueError('One provenance-verified training record required for CSV')
    output=[]
    for row in data['rows']:
        for arm in ('proxy','Meep'):
            record=row[arm];item={key:'' for key in CSV_COLUMNS}
            item.update(H_um=row['H_um'],height_use=row['height_use'],
                forward_model='proxy_forward_prediction' if arm=='proxy' else 'direct_Meep_forward_simulation',
                downstream_distance_um=0.,source_training_record_file=training[0][0],source_training_record_sha256=training[0][1])
            if record is None:
                item.update(native_status='UNRESOLVED_MISSING',nominal_optical_status='UNRESOLVED_MISSING',selected_mesh_status='UNRESOLVED_MISSING')
                output.append(item);continue
            prediction=row['proxy']
            item.update(design_file=prediction['design_file'] if prediction else '',design_sha256=record['design_sha256'])
            plane=[p for p in record['planes'] if science.close(p.get('z_um'),0.)]
            plane=plane[0] if len(plane)==1 else {}
            item.update(diameter_x_um=plane.get('diameter_x_um'),diameter_y_um=plane.get('diameter_y_um'),
                useful_efficiency_fraction=record.get('eta'),
                useful_efficiency_percent=100*record['eta'] if science.finite(record.get('eta')) else None,
                normalized_curvature_C=record.get('C'),retained_band_angular_ratio_Q=record.get('Q'))
            angles=record.get('core_half_angles_deg',[])
            item.update(Gaussian_core_half_angle_x_deg=angles[0] if len(angles)>0 else None,
                        Gaussian_core_half_angle_y_deg=angles[1] if len(angles)>1 else None)
            measured=row['Meep']
            if measured:item['output_reference_z_um']=measured['settings'].get('output_z_um',row['H_um']+4.)
            else:item['output_reference_z_um']=row['H_um']+4.
            if arm=='proxy':
                item.update(native_status='NOT_APPLICABLE_PROXY',nominal_optical_status='NOT_ASSESSED_PROXY',
                            selected_mesh_status='NOT_APPLICABLE_PROXY')
            else:
                pair=row['selected_mesh'].get('resolutions',[])
                item.update(record_identity=record['identity'],resolution_cells_per_um=record['resolution'],
                    native_status=row['native']['status'],nominal_optical_status=row['nominal_optical']['status'],
                    selected_mesh_status=row['selected_mesh']['status'],
                    selected_mesh_pair_cells_per_um=' to '.join(f'{v:g}' for v in pair),
                    Meep_record_file=record['record_file'],Meep_record_sha256=record['record_sha256'],
                    analysis_file=record['analysis_file'],analysis_sha256=record['analysis_sha256'])
            output.append(item)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--overview',action='store_true',help='Also export the unchanged v3 overview PNG')
    args=parser.parse_args();science=load_science()
    folder=science.local_path(args.folder).resolve();output=science.local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Preserve publication exports; choose a fresh version')
    if not output.is_relative_to(folder/'forward_comparison') or output==folder/'forward_comparison':
        raise ValueError('Output must be a fresh stage2/forward_comparison/version directory')
    data=prepare_data(folder,science);output.mkdir(parents=True,exist_ok=False)
    with (output/'comparison.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:
        stream.write('PUBLICATION LAYOUT ONLY: TWO 7.2-INCH FIGURES, PNG300/SVG/PDF\n'
                     'Scientific selection and checks use unchanged SHA-pinned v3.\n'
                     'Patterns differ between heights. Proxy and Meep use the same frozen geometry WITHIN each height.\n'
                     'CSV: six model/height rows; Dx/Dy at distance0 from output monitor z=H+4um; eta fraction/percent; '
                     'C and retained-band Q; existing fitted Gaussian-core HALF-angles in DEGREES.\n'
                     'CSV values copy existing verified data without resampling or rounded display values. Proxy native/mesh checks are not applicable.\n\n'+science.human_report(data))
    with (output/'proxy_vs_selected_Meep.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_COLUMNS,lineterminator='\n');writer.writeheader();writer.writerows(csv_rows(data,science))
    render_publication(data,output,science)
    if args.overview:science.render(data,output)
    if sha(SCIENTIFIC_SOURCE)!=EXPECTED_SCIENTIFIC_SHA:raise RuntimeError('Pinned scientific source changed during export')
    if not all(science.sha(path)==digest for path,digest in data['input_sha256'].items()):raise RuntimeError('Scientific inputs changed during export')
    files=[path for path in output.iterdir() if path.is_file()]
    manifest={'status':'LAYOUT_ONLY_PUBLICATION_EXPORT_NO_OPTICAL_SOLVES',
        'scientific_source_sha256':EXPECTED_SCIENTIFIC_SHA,'publication_utility_sha256':sha(__file__),
        'predecessor_layout_sha256':PREDECESSOR_LAYOUT_SHA,
        'input_sha256':data['input_sha256'],'scientific_logic_changed':False,
        'files':{path.name:{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(files)},
        'optical_solves_launched':0,'training_runs_launched':0,'immutable_inputs_unchanged':True}
    with (output/'delivery_manifest.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'status':manifest['status'],'exports':len(files),'optical_solves_launched':0}))
    return 0


if __name__=='__main__':raise SystemExit(main())
