"""Freeze the Corning-informed height-specific inverse-design protocol.

Preparation and analytic Gaussian references only; no GPU or Maxwell solve.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2'
CONFIG=ROOT/'configs/1550nm_height_family_n15263_v2.json'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    if FOLDER.exists() or CONFIG.exists():
        raise FileExistsError('Keep previous study intact; choose a new version')
    cfg={
        'schema_version':2,'result_label':'CORNING_GAUSSIAN_SURROGATE_N15263_HEIGHT_SPECIFIC_PROXY_STUDY',
        'wavelength_um':1.55,'mfd_um':10.4,'fiber_nominal_NA':.14,
        'input_mode_provenance':{
            'manufacturer':'Corning SMF-28 Ultra','source_url':'https://www.corning.com/media/worldwide/coc/documents/Fiber/product-information-sheets/PI-1424-AEN.pdf',
            'sheet_issue':'July 2025','mfd_1550_um':10.4,'mfd_tolerance_um':.5,
            'nominal_NA':.14,'NA_measurement':'one percent power level of one-dimensional far-field scan at1310nm',
            'cladding_diameter_um':125.,'cladding_tolerance_um':.7,
            'user_core_label_um':8.,'core_label_is_not_MFD':True,
            'group_index_1550':1.4682,'group_index_not_used_as_phase_index':True,
            'mode_model':'Gaussian transverse amplitude exp(-r^2/w0^2), waist at facet; not measured/eigenmode vector field',
            'source_phase_index':1.45,'source_phase_index_is_assumed':True,
            'fiber_radial_index_profile_supplied':False},
        'materials':{'substrate':{'n':1.45,'kappa':0.},'polymer':{'n':1.5263,'kappa':0.},'ambient':{'n':1.,'kappa':0.}},
        'material_provenance':'Polymer real index supplied by user at1550nm; input phase index1.45 and zero losses assumed. No measured resin absorption or complete fiber index profile.',
        'architecture':'solid_cylinder_with_top_annular_relief',
        'design_mode':'DIFFERENT_PROFILE_FOR_EACH_H; latest user correction supersedes earlier fixed-profile preference',
        'cylinder_diameter_um':50.,'annuli':50,'height_max_um':3.,
        'fabrication':{'height_step_um':.05,'min_width_um':.5,'max_relief_aspect_ratio':8.,'limits_are_assumed':True},
        'proxy':{'grid_n':192,'window_um':64.,'padding':2,'pedestal_dz_um':.5,'relief_slices':24,'output_clearance_um':1.},
        'loss':{'version':'height_conditioned_n15263_v2','negative_useful_power':1.,'width':.1,'phase':.5,'angular':.25,'sidelobe':.2,'roughness':.0002,'transmission_floor':.9,'low_power_penalty':10.},
        'evaluation':{'processing_planes_um':[0.,25.,50.,75.,100.],'diagnostic_planes_um':[200.],
                      'beam_target':'H-specific natural-footprint flat-phase Gaussian at H+4um; physically diffracting downstream',
                      'interpretation':'near-declared-Gaussian collimation under bounded relief; not a global lower bound for all50um aperture optics; no ISO M2 claim'},
        'natural_training':{
            'train_H_um':[40.,50.,60.,70.,80.,90.,100.],
            'validation_H_um':[45.,55.,65.,75.,85.,95.],
            'heldout_H_um':[42.5,57.5,72.5,87.5,97.5],
            'seeds':[11,29,47],'steps':100,'pilot_steps':10,'validation_interval':10,
            'learning_rate':.004,'direct_learning_rate':.025,
            'normalization_center_um':70.,'normalization_half_range_um':30.,
            'selection':'quantized validation only: summed normalized violations, worst useful efficiency, mean loss',
            'direct_baseline':'7x50 residual-logit table at training heights; linear interpolation of residuals at otherH; no heldout fitting',
            'network':'3 hidden64-unit tanh layers;50 zero-initialized residual-logit outputs conditioned onH',
            'investigation_gates':{'transmission_min':.9,'declared_Gaussian_overlap_min':.98,'useful_efficiency_min':.9,'output_diameter_relative_tolerance':.1,'max_sidelobe_fraction':.02,'max_abs_normalized_curvature':.1,'max_angular_second_moment_ratio':1.1}},
        'campaign_limits':{'pilot_timeout_s':300,'training_timeout_s':900,'bundle_cache_max':32,'one_gpu_worker':True,
                           'GPU_delta_max_mib':6144,'GPU_free_min_mib':384,
                           'historical_cumulative_Maxwell_used':13,'historical_cumulative_Maxwell_ceiling':13,
                           'new_Maxwell_runs_authorized':0,'new_Maxwell_amendment_required':True},
        'resource_policy':{'host_ram_gate':None,'project_rss_gate':None,'wsl_ram_gate':None,'GPU_controls':'retain','checkpointing':True},
        'scope':'RTX4060 inverse-design family and proxy evidence; independent Maxwell acceptance deferred until a new explicit bounded allocation. No SLC manufacturing acceptance implied.'}
    FOLDER.mkdir(parents=True)
    write(CONFIG,cfg);write(FOLDER/'config.json',cfg)
    protected={}
    for relative in ['artifacts/natural_inverse_design_candidate_registry_v1.json','results.md','acceptance-repair.md',
                     'artifacts/1550nm_natural_nn_v1/designs/NN.json','exports/cylinder_attached_slc_v1/C01.slc']:
        p=ROOT/relative
        protected[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    n=cfg['materials']['polymer']['n'];wave=cfg['wavelength_um'];w0=cfg['mfd_um']/2;zr=math.pi*n*w0*w0/wave
    rows=[];seeds={}
    for H in cfg['natural_training']['train_H_um']:
        w=w0*math.sqrt(1+(H/zr)**2);K=n*H/(H*H+zr*zr)
        rows.append({'H_um':H,'Gaussian_top_diameter_um':2*w,'polymer_wavefront_radius_um':H+zr*zr/H,
                     'phase_power_K_per_um':K,'equivalent_phase_f_um':1/K,
                     'same_footprint_flat_Gaussian_half_angle_rad':wave/(math.pi*w),
                     'same_footprint_flat_Gaussian_half_angle_deg':wave/(math.pi*w)*180/math.pi,
                     'flat_Gaussian_diameter_after100um':2*w*math.sqrt(1+(100/(math.pi*w*w/wave))**2)})
        seeds[str(H)]=[max(0.,min(3.,3-K*((j+.5)*25/50)**2/(2*(n-1)))) for j in range(50)]
    write(FOLDER/'analytic_feasibility.json',{'status':'ANALYTIC_REFERENCE_ONLY','polymer_Rayleigh_um':zr,
        'full_2pi_relief_um':wave/(n-1),'phase_span_max_rad':2*math.pi*(n-1)*3/wave,
        'rows':rows,'analytic_height_seeds_um':seeds,
        'limitations':'Homogeneous Gaussian and zero-thickness phase reference; finite-cylinder/vector/reflection and finite-relief effects require proxy and independent Maxwell checks.'})
    write(FOLDER/'provenance.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                                  'protected_before':protected,'new_optical_Maxwell_runs':0,'request':'Different optimized top profile for each H40..100, confirmed latest user steering'})
    print(json.dumps({'folder':str(FOLDER),'MFD_um':10.4,'polymer_n':n,'design_mode':cfg['design_mode'],'analytic_rows':rows},indent=2))

if __name__=='__main__':main()
