"""Figure3: exact frozen50nm profiles and saved forward-field reconstruction.

Descriptive postprocessing only: no Meep, GPU training, controller or acceptance
run. The unchanged analyzer helpers provide the interpolation, Yee correction,
forward decomposition and matched incident normalization. Phase is Ex phase,
with one global piston removed, and an explicitly recorded bright-field mask.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
STAGE=CAMPAIGN/'maxwell_validation_v3'
CONFIG=STAGE/'config.json'
COMPARISON=STAGE/'forward_comparison/recovery_final_v1/comparison.json'
ANALYZER=ROOT/'scripts/height_family_maxwell_analysis_v2.py'
PINS={CONFIG:'ace2ec9cd90d72d31d469bf6c1372291b7b0ac80612bdc3a76b8335854d9a734',
 COMPARISON:'8d54aad5ea03ddf3de21f35d0062a22464008ed5cd7919c02922bf93de0f778d',
 ANALYZER:'4d1c2cd04c539660928f29d66bce3e701ef41a115512931070dd0f11a6dfc55b',
 ROOT/'src/fiber_metalens_nn/collimation_optics.py':'5381f4c88a5452862907284bf5fb405849ea39d3739bed7c8b29b8cc0abf0692',
 ROOT/'src/fiber_metalens_nn/collimation_dft_power.py':'888a0c13d2af4667ab75c282685fa78b36b07002a33dab86ef125f4a8455f385',
 ROOT/'src/fiber_metalens_nn/collimation_metrics.py':'fa6a4e99c834e68fc2067a539a641679a6c462413982f977cfb8a12ef9d772cd',
 ROOT/'src/fiber_metalens_nn/metrics.py':'ed29c34205306e3b84aec3ad49339c861972d91f1c4cde88590e028f06af62a9'}
SOURCES={40:(CAMPAIGN/'maxwell_validation_v1/frozen_designs/NN_H40.json','16a765787ce3dd50ad34bc6e8f3dac89e0fd0e6c3723ec6460cfa8bbe1b0b362','maxwell_validation_v1:43'),
 75:(STAGE/'frozen_designs/NN_H75.json','7861a7d4a8409026b1d57ed20bb5fcb9e30d2dbe7b4a68b8214ae1ee9a458c69','maxwell_validation_v3:69'),
 100:(STAGE/'frozen_designs/NN_H100.json','0e83c24f7978f764c937b74f15b2d6e6f71da96d3ce320677e941b8cc94b3772','maxwell_validation_v3:83')}
STEM='Figure3_profiles_and_fields_50nm'
def local(value):
    value=str(value)
    if sys.platform=='win32' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif sys.platform!='win32' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()
def sha(path):
    value=hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(local(path).read_text(encoding='utf-8-sig'))
def write_new(path,data):
    with path.open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
def load_analyzer():
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('frozen_field_display_analyzer',ANALYZER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def reconstruct(row,design_path,design_sha,analyzer,config,protected,known):
    def protect(path,expected=None):
        path=local(path);digest=sha(path)
        if expected is None:expected=known.get(path)
        if expected is None or digest!=expected:raise ValueError('Saved plotting source SHA differs: '+str(path))
        protected[str(path)]=digest;return path
    protect(design_path,design_sha);design=read(design_path)
    record_path=protect(row['record_file'],row['record_sha256']);saved_analysis=read(protect(row['analysis_file'],row['analysis_sha256']))
    record=read(record_path);reference_path=protect(record['reference_file'],row['reference_record_sha256']);reference=read(reference_path)
    if (record['settings']!=reference['settings'] or record['config_fingerprint']!=analyzer.fingerprint(config)
            or reference['config_fingerprint']!=analyzer.fingerprint(config) or local(record['design_file'])!=local(design_path)):
        # Selected40 also has byte-identical frozen copies; allow the original
        # worker design path if it has the exact selected frozen design SHA.
        if (record['settings']!=reference['settings'] or record['config_fingerprint']!=analyzer.fingerprint(config)
                or reference['config_fingerprint']!=analyzer.fingerprint(config) or sha(local(record['design_file']))!=design_sha):
            raise ValueError('Saved device/reference/design/config binding differs')
    protect(record['design_file'],design_sha)
    settings=record['settings'];resolution=float(settings['resolution']);dr=1/resolution;wavelength=float(config['wavelength_um'])
    ns=float(config['materials']['substrate']['n']);na=float(config['materials']['ambient']['n']);count=512;window=128.
    with (np.load(protect(record['fields_file']),allow_pickle=False) as device,
          np.load(protect(reference['fields_file']),allow_pickle=False) as incident_fields,
          np.load(protect(record['power_audit_fields_file']),allow_pickle=False) as audit_device,
          np.load(protect(reference['power_audit_fields_file']),allow_pickle=False) as audit_reference):
        for data,label in [(audit_device,'device'),(audit_reference,'reference')]:
            analyzer.require_symmetric_axial_collapse(data['pair_weights'],data['plane_z_um'],label)
        courant=float(settings.get('Courant',audit_device['Courant']))
        if abs(float(audit_reference['Courant'])-courant)>1e-12:raise ValueError('Matched Courant differs')
        if float(device['z_um'][0])!=float(settings['output_z_um']) or float(incident_fields['z_um'][0])!=float(settings['reflection_z_um']):
            raise ValueError('Wrong saved output/incident monitor plane')
        def convert(data,medium):
            x,dx,field=analyzer.centered_lp_to_cartesian(data['r_um'],data['fields'][0],dr,count,window)
            return x,dx,analyzer.paraxial_power_equivalent_fields(field,medium,wavelength,resolution,courant,symmetric_axial_collapse=True)
        _,dx,reference_cart=convert(incident_fields,ns)
        _,_,incident=analyzer.forward_field_with_power(reference_cart,dx,wavelength,ns)
        x,dx,device_cart=convert(device,na)
        _,field,power=analyzer.forward_field_with_power(device_cart,dx,wavelength,na)
    original=saved_analysis['reconstruction'];relative_T=power/incident
    agreement=dict(incident_relative_error=abs(incident/original['incident_field_units']-1),
        forward_transmission_absolute_error=abs(relative_T-original['field_forward_T']))
    if max(agreement.values())>1e-9:raise ValueError('Field display does not reproduce existing frozen reconstruction normalization')
    unit=field/np.sqrt(incident);electric=np.sum(np.abs(unit[:3])**2,axis=0);ex=unit[0]
    xx,yy=np.meshgrid(x,x,indexing='xy');domain=np.hypot(xx,yy)<=float(record['grid_metadata']['monitor_radius_um'])
    mask=domain&(electric>=.01*electric.max())&(np.abs(ex)**2>=.01*np.abs(ex).max()**2)
    if not mask.any():raise ValueError('Empty phase-valid bright-field mask')
    piston=float(np.angle(np.sum(np.abs(ex[mask])**2*ex[mask]/np.abs(ex[mask]))))
    phase=np.angle(ex*np.exp(-1j*piston));phase=np.where(mask,phase,np.nan)
    crop=np.flatnonzero(abs(x)<=25.);sl=np.ix_(crop,crop)
    return dict(H_um=design['H_um'],design=design,record=row,settings=settings,x_um=x[crop],dx_um=dx,
        EH_unit_incident=unit[:,crop[:,None],crop],electric_unit_incident=electric[sl],phase_rad=phase[sl],phase_valid_mask=mask[sl],
        native_domain_mask=domain[sl],phase_piston_removed_rad=piston,incident_field_power_units=incident,
        output_forward_power_per_incident=relative_T,normalization_replay=agreement,full_grid_count=count,full_window_um=window)
def draw(records,output,common_peak,phase_limit,dpi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(7.8,8.0));grid=fig.add_gridspec(3,4,width_ratios=[1,1,1,.055],height_ratios=[.65,1,1],
        left=.09,right=.91,bottom=.13,top=.85,wspace=.30,hspace=.38)
    intensity_cmap=plt.get_cmap('magma').copy();intensity_cmap.set_bad('#eeeeee')
    phase_cmap=plt.get_cmap('RdBu_r').copy();phase_cmap.set_bad('#e8e8e8')
    letters='abcdefghi';intensity_image=phase_image=None
    for j,data in enumerate(records):
        H=data['H_um'];h=np.asarray(data['design']['heights_um']);r=np.linspace(0,data['design']['cylinder_diameter_um']/2,len(h)+1)
        ax=fig.add_subplot(grid[0,j]);ax.stairs(h,r,color='#244f70',linewidth=1.4,fill=True,alpha=.2)
        ax.stairs(h,r,color='#244f70',linewidth=1.2);ax.set(xlim=(0,25),ylim=(0,3.15),xticks=[0,10,20,25],yticks=[0,1,2,3],xlabel=r'$r$ ($\mu$m)')
        ax.set_title(rf'$H={H:g}$ $\mu$m'+'\n'+rf'{data["settings"]["resolution"]:g} cells/$\mu$m',pad=8)
        if j==0:ax.set_ylabel(r'Relief $h(r)$ ($\mu$m)')
        ax.grid(alpha=.18,linewidth=.4);ax.text(-.18,1.04,f'({letters[j]})',transform=ax.transAxes,weight='bold',fontsize=10)
        for i in (1,2):
            ax=fig.add_subplot(grid[i,j]);x=data['x_um'];extent=[x[0]-data['dx_um']/2,x[-1]+data['dx_um']/2]*2
            if i==1:
                values=np.log10(np.maximum(data['electric_unit_incident']/common_peak,1e-4))
                values=np.ma.array(values,mask=~data['native_domain_mask'])
                intensity_image=ax.imshow(values,origin='lower',extent=extent,cmap=intensity_cmap,vmin=-4,vmax=0,interpolation='nearest',rasterized=True)
            else:
                phase_image=ax.imshow(np.ma.masked_invalid(data['phase_rad']),origin='lower',extent=extent,cmap=phase_cmap,
                    vmin=-phase_limit,vmax=phase_limit,interpolation='nearest',rasterized=True)
                ax.contour(x,x,data['phase_valid_mask'].astype(float),levels=[.5],colors=['#444444'],linewidths=.35)
            ax.set(xlim=(-25,25),ylim=(-25,25),xticks=[-20,0,20],yticks=[-20,0,20],xlabel=r'$x$ ($\mu$m)')
            if j==0:ax.set_ylabel(r'$y$ ($\mu$m)')
            ax.text(-.18,1.025,f'({letters[3*i+j]})',transform=ax.transAxes,weight='bold',fontsize=10)
    color=fig.colorbar(intensity_image,cax=fig.add_subplot(grid[1,3]),ticks=[-4,-3,-2,-1,0]);color.set_label(r'$\log_{10}(I_E/I_{E,\mathrm{max}})$',labelpad=7)
    color=fig.colorbar(phase_image,cax=fig.add_subplot(grid[2,3]));color.set_label(r'Piston-removed $\arg(E_x)$ (rad)',labelpad=7)
    fig.suptitle('Frozen 50 nm relief profiles and reconstructed output fields',fontsize=12,y=.982)
    fig.text(.09,.947,r'Each field plane: $z=H+4$ $\mu$m. Matched incident-power normalization; common display scales.',fontsize=8.5)
    fig.text(.09,.038,'Phase: gray pixels fail the 1% bright-field mask; only global piston is removed. No defocus correction.',fontsize=8)
    for suffix in ('png','svg','pdf'):fig.savefig(output/(STEM+'.'+suffix),dpi=dpi,facecolor='white')
    plt.close(fig)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dpi',type=int,default=350);args=parser.parse_args()
    names=[STEM+'.'+x for x in ('png','svg','pdf')]+['Figure3_profiles_50nm.csv','Figure3_plotted_fields_50nm.npz','Figure3_provenance.json','Figure3_caption.txt']
    if args.dpi<300:raise ValueError('Publication raster must be at least300dpi')
    if any((args.output/name).exists() for name in names):raise FileExistsError('Preserve existing Figure3; choose a new output folder/version')
    protected={str(p.resolve()):digest for p,digest in PINS.items()}
    for path,digest in protected.items():
        if sha(path)!=digest:raise ValueError('Frozen reconstruction/config/comparison source changed: '+path)
    protected[str(Path(__file__).resolve())]=sha(__file__)
    config=read(CONFIG);comparison=read(COMPARISON);known={local(p):d for p,d in comparison['input_sha256'].items()};analyzer=load_analyzer();records=[]
    for H,(path,digest,identity) in SOURCES.items():
        row=next(r['Meep'] for r in comparison['rows'] if r['H_um']==H)
        if row['identity']!=identity or row['design_sha256']!=digest:raise ValueError('Wrong selected frozen profile')
        records.append(reconstruct(row,path,digest,analyzer,config,protected,known))
    common_peak=max(float(d['electric_unit_incident'].max()) for d in records)
    max_phase=max(float(np.nanmax(abs(d['phase_rad']))) for d in records);phase_limit=max(.05,math.ceil(max_phase/.05)*.05)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'Figure3_profiles_50nm.csv').open('x',encoding='utf-8',newline='') as stream:
        columns=['H_um','annulus_index','r_inner_um','r_outer_um','r_mid_um','relief_h_um','absolute_top_z_um','height_quantization_nm','source_design_sha256'];writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for data in records:
            design=data['design'];edges=np.linspace(0,design['cylinder_diameter_um']/2,len(design['heights_um'])+1)
            for i,h in enumerate(design['heights_um']):writer.writerow(dict(H_um=data['H_um'],annulus_index=i,r_inner_um=edges[i],r_outer_um=edges[i+1],r_mid_um=(edges[i]+edges[i+1])/2,
                relief_h_um=h,absolute_top_z_um=data['H_um']+h,height_quantization_nm=50,source_design_sha256=SOURCES[int(data['H_um'])][1]))
    arrays={'component_order':np.asarray(['Ex','Ey','Ez','Hx_Z0scaled','Hy_Z0scaled','Hz_Z0scaled']),
        'common_electric_intensity_peak_unit_incident':np.asarray(common_peak),'common_phase_limit_rad':np.asarray(phase_limit),
        'display_intensity_log10_floor':np.asarray(-4.),'phase_mask_relative_threshold':np.asarray(.01)}
    summaries=[]
    for data in records:
        prefix=f'H{int(data["H_um"]):03d}_'
        for key in ('x_um','EH_unit_incident','electric_unit_incident','phase_rad','phase_valid_mask','native_domain_mask'):arrays[prefix+key]=data[key]
        arrays[prefix+'intensity_relative_common_peak']=data['electric_unit_incident']/common_peak
        summaries.append({key:data[key] for key in ('H_um','settings','dx_um','phase_piston_removed_rad','incident_field_power_units','output_forward_power_per_incident','normalization_replay','full_grid_count','full_window_um')})
    np.savez_compressed(args.output/'Figure3_plotted_fields_50nm.npz',**arrays)
    draw(records,args.output,common_peak,phase_limit,args.dpi)
    caption=("Figure3. Frozen 50 nm-quantized height-specific NN profiles and descriptive reconstruction of their archived Maxwell fields. "
        "Columns correspond to H=40,75,100 µm, using selected records43/69/83. The first row gives the exact radial relief height above each cylinder. "
        "The second row shows total electric intensity I_E=|Ex|²+|Ey|²+|Ez|² after normalization to the matched incident-reference power; "
        "a single peak across all three panels defines the common logarithmic color scale, displayed over four decades. "
        "The third row gives arg(Ex) with one intensity-weighted global piston removed, without curvature/defocus subtraction. "
        "Phase is displayed only where both total electric intensity and Ex intensity exceed1% of their respective lens-specific peaks; "
        "the native35µm-radius monitor domain also applies. Gray indicates invalid/unshown pixels. "
        "All maps are at z=H+4µm in air and share x/y coordinates and color limits. The unchanged512×512,128µm-window reconstruction uses "
        "the frozen centered-LP, paraxial Yee power-equivalent and retained-band forward-field conventions; the figure crops x,y to±25µm. "
        "This is saved-data visualization, not a new simulation, acceptance test,500nm-quantized redesign or experimental field measurement.\n")
    with (args.output/'Figure3_caption.txt').open('x',encoding='utf-8') as stream:stream.write(caption)
    if any(sha(p)!=digest for p,digest in protected.items()):raise RuntimeError('Plot source changed during postprocessing')
    write_new(args.output/'Figure3_provenance.json',dict(status='SAVED_FIELDS_DESCRIPTIVE_MANUSCRIPT_FIGURE_NO_NEW_OPTICS',created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=protected,records=summaries,common_intensity_peak_unit_incident=common_peak,common_phase_color_limits_rad=[-phase_limit,phase_limit],
        raster_dpi=args.dpi,figure_size_inches=[7.8,8.0],display_xy_limits_um=[-25,25],
        intensity_definition='sum(abs(unit-incident-power forward E_xyz)^2); one common global peak, four-decade log display',
        phase_definition='arg(Ex*exp(-i*piston)); circular mean piston weighted byabs(Ex)^2 over phase-valid pixels',
        phase_mask='native collection domain AND total I_E>=0.01 own peak AND abs(Ex)^2>=0.01 own peak; no defocus removal',
        original_height_quantization_um=.05,plotted_designs_are_500nm_quantized=False,source_execution_clock_not_repaired=True,new_optical_or_training_runs=0,
        files={name:dict(sha256=sha(args.output/name),bytes=(args.output/name).stat().st_size) for name in names if name!='Figure3_provenance.json'}))
    print(json.dumps(dict(status='FIGURE3_COMPLETE_NO_NEW_OPTICS',output=str(args.output),phase_limit_rad=phase_limit,source_records=[r['record']['identity'] for r in records])))
if __name__=='__main__':main()
