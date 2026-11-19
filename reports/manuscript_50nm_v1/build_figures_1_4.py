"""Publication schematics and saved-data plots only; no optical solve or training."""
from pathlib import Path
import csv, hashlib, json, os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

ROOT=HERE.parents[1]
CAMP=ROOT/'artifacts/1550nm_height_family_n15263_v2'
COMP=CAMP/'maxwell_validation_v3/forward_comparison/recovery_final_v1/comparison.json'
COARSE=ROOT/'exports/height_family_cylinder_attached_slc_500nm_v5'
OUT=HERE/'figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,
 'axes.titlesize':10.5,'legend.fontsize':8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
inputs={}
def read(path):
    inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8-sig'))
def save(fig,stem):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/(stem+'.'+ext),dpi=360,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def schematic():
    fig=plt.figure(figsize=(7.2,5.3));gs=fig.add_gridspec(1,2,width_ratios=[.92,1.15],wspace=.18)
    ax=fig.add_subplot(gs[0,0]);ax.set_xlim(-39,39);ax.set_ylim(-28,106);ax.axis('off')
    ax.text(-37,104,'a  Physical configuration',weight='bold',fontsize=10.5)
    ax.add_patch(Rectangle((-35,-25),70,25,fc='#dce9f2',ec='#405f77',lw=1))
    ax.add_patch(Rectangle((-4,-25),8,25,fc='#aac3d5',ec='#405f77',lw=.7))
    ax.text(0,-18,'SMF Gaussian input\nMFD 10.4 µm',fontsize=8,ha='center')
    ax.add_patch(Rectangle((-25,0),50,75,fc='#d6e9ed',ec='#35647a',lw=1.2))
    rr=np.array([-25,-19,-16,-13,-10,-6,0,6,10,13,16,19,25])
    hh=np.array([0,0,.5,1,1.6,2.5,3,2.5,1.6,1,.5,0,0])*2
    ax.fill_between(rr,75,75+hh,step='mid',color='#dca34f',edgecolor='#a46516',lw=.7)
    ax.axhline(0,color='#555',lw=.7,ls=':');ax.text(18,-5,'Facet z = 0',fontsize=7.8,ha='center')
    ax.plot([-29,29],[85,85],color='#222',ls='--',lw=.9)
    ax.text(0,89,'Output monitor z = H + 4 µm',ha='center',fontsize=7.8)
    # Beam envelope is explanatory, not a simulated field.
    for sign in (-1,1):
        zz=np.linspace(0,74,100);envelope=sign*(5.2+4.1*(zz/75)**1.35)
        ax.plot(envelope,zz,color='#0072B2',lw=1.2)
        ax.plot([sign*9.3,sign*9.5],[82,102],color='#0072B2',lw=1.1)
    ax.annotate('',xy=(0,100),xytext=(0,87),arrowprops=dict(arrowstyle='->',color='#0072B2'))
    ax.annotate('',xy=(-31,75),xytext=(-31,0),arrowprops=dict(arrowstyle='<->',color='black',lw=.8))
    ax.text(-33,37,'H',ha='right',va='center',fontsize=10)
    ax.annotate('',xy=(25,13),xytext=(-25,13),arrowprops=dict(arrowstyle='<->',color='black',lw=.8))
    ax.text(0,17,'D = 50 µm',ha='center',fontsize=8.5)
    ax.text(0,49,'Solid polymer\nn = 1.5263\nat 1550 nm',ha='center',fontsize=9)
    ax.text(0,65,'Top relief 0 to 3 µm',ha='center',fontsize=8)
    ax.text(0,-31,'Schematic envelope; relief exaggerated\nSeparate patterns at H = 40, 75 and 100 µm',ha='center',fontsize=7.5)
    ax=fig.add_subplot(gs[0,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(0,1,'b  Physics seeded inverse design',weight='bold',fontsize=10.5)
    def box(x,y,w,h,label,fc='#f1f3f5',fs=8.5):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=fc,ec='#555',lw=.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.8,color='#444'))
    box(.28,.87,.43,.07,'Cylinder height H')
    box(.02,.70,.44,.10,'Cylinder field\nphase seed',fc='#e4eef3')
    box(.54,.70,.44,.10,'Height conditioned NN\n3 × 64 tanh → 50 logits',fc='#e4eef3',fs=7.9)
    arrow((.4,.86),(.24,.81));arrow((.6,.86),(.76,.81))
    box(.21,.53,.58,.10,'Seed plus NN residual\nbounded continuous relief')
    arrow((.24,.69),(.42,.64));arrow((.76,.69),(.58,.64))
    box(.02,.32,.43,.11,'Differentiable finite\nthickness optical model',fc='#e8f1ed',fs=8)
    box(.55,.32,.43,.11,'50 nm quantization\nvalidation selection',fc='#f6ebd9',fs=8)
    arrow((.37,.52),(.24,.44));arrow((.64,.52),(.76,.44))
    ax.text(.22,.25,'Training loss and gradients',ha='center',fontsize=7.2)
    box(.47,.08,.51,.12,'Frozen pattern\nindependent Meep validation',fc='#e8f1ed',fs=8)
    arrow((.76,.31),(.76,.21))
    ax.text(.03,.10,'Continuous training\nQuantized selection\nNo Meep gradients',fontsize=7.5,linespacing=1.5)
    fig.subplots_adjust(left=.04,right=.99,bottom=.08,top=.97)
    save(fig,'figure_1_device_and_workflow')

def performance():
    data=read(COMP);fig,axes=plt.subplots(2,3,figsize=(7.2,5.55),gridspec_kw={'hspace':.58,'wspace':.40})
    rows=data['rows'];blue='#0072B2';red='#D55E00'
    csvrows=[]
    for j,row in enumerate(rows):
        ax=axes[0,j]
        for arm,color,ls,marker in [('proxy',blue,'--','s'),('Meep',red,'-','o')]:
            planes=row[arm]['planes'];z=np.array([p['z_um'] for p in planes]);dx=np.array([p['diameter_x_um'] for p in planes]);dy=np.array([p['diameter_y_um'] for p in planes])
            ax.fill_between(z,np.minimum(dx,dy),np.maximum(dx,dy),color=color,alpha=.17)
            ax.plot(z,.5*(dx+dy),color=color,ls=ls,marker=marker,ms=3,lw=1.25)
            for p in planes:csvrows.append(dict(H_um=row['H_um'],forward_model=arm,**p))
        target=row['ideal_declared_Gaussian'];ax.plot([p['z_um'] for p in target],[p['diameter_um'] for p in target],color='#555',ls=':',lw=1.25)
        ax.set_title(f"{'abc'[j]}  H = {row['H_um']:g} µm")
        ax.set_xlim(-3,103);ax.set_ylim(10.7,21.4);ax.set_xticks([0,50,100]);ax.grid(alpha=.18)
        ax.set_xlabel('Distance from monitor (µm)')
        if j==0:ax.set_ylabel('1/e² intensity diameter (µm)')
        ax.text(.05,.95,f"Meep {row['Meep']['resolution']:g} cells/µm",transform=ax.transAxes,va='top',fontsize=7.7)
    hs=[r['H_um'] for r in rows]
    for j,(key,title,ylabel) in enumerate([('eta','Useful modal efficiency','Efficiency (%)'),('C','Wavefront curvature','Normalized curvature C'),('Q','Retained angular ratio','Q')]):
        ax=axes[1,j];factor=100 if key=='eta' else 1
        for arm,color,ls,marker in [('proxy',blue,'--','s'),('Meep',red,'-','o')]:
            ax.plot(hs,[factor*r[arm][key] for r in rows],color=color,ls=ls,marker=marker,ms=4,lw=1.25)
        ax.set_title(f"{'def'[j]}  {title}",fontsize=9.3);ax.set_xlabel('Cylinder height H (µm)');ax.set_ylabel(ylabel);ax.set_xticks(hs);ax.grid(alpha=.18)
        if key=='eta':ax.set_ylim(94.85,95.75)
        if key=='C':ax.axhline(0,color='#888',lw=.6);ax.set_ylim(-.052,.046)
        if key=='Q':ax.axhline(1,color='#888',ls=':',lw=.8);ax.set_ylim(.980,1.045)
    fig.legend(handles=[Line2D([],[],color=blue,ls='--',marker='s',ms=4,label='Design proxy'),Line2D([],[],color=red,marker='o',ms=4,label='Direct Meep'),Line2D([],[],color='#555',ls=':',label='Declared Gaussian reference')],loc='lower center',ncol=3,bbox_to_anchor=(.50,-.01),frameon=False,fontsize=8)
    fig.subplots_adjust(left=.095,right=.99,top=.95,bottom=.12)
    save(fig,'figure_4_collimation_performance')
    with (OUT/'figure_4_widths.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)

def coarse_conversion():
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.65),sharex=True,sharey=True)
    for ax,H in zip(axes,[40,75,100]):
        m=read(COARSE/f'H{H:03d}/slicing_change_manifest.json');r=np.arange(51)*.5
        old=np.r_[m['original_heights_um'],0];new=np.r_[m['exported_heights_um'],0]
        ax.step(r,old,where='post',lw=1.2,color='#0072B2',label='Original 50 nm')
        ax.step(r,new,where='post',lw=1.2,color='#D55E00',label='500 nm fabrication')
        ax.set_title(f'H = {H} µm');ax.set_xlabel('Radius (µm)');ax.set_xlim(0,25);ax.set_ylim(-.08,3.15);ax.set_xticks([0,10,20,25]);ax.grid(alpha=.18)
    axes[0].set_ylabel('Relief height (µm)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.02))
    fig.subplots_adjust(left=.08,right=.99,bottom=.28,top=.87,wspace=.15)
    save(fig,'figure_S1_quantization_conversion')

if __name__=='__main__':
    schematic();performance();coarse_conversion()
    (OUT/'root_figures_manifest.json').write_text(json.dumps({'input_sha256':inputs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_optical_solves':0,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('figure_*') if p.is_file() and p.name.startswith(('figure_1','figure_4','figure_S1'))}},indent=2)+'\n')
    print('Saved Figures 1 and 4 and supplementary quantization profile plot')
