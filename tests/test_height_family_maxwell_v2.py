"""Pure setup and saved-field mathematics checks; no Meep solve is launched."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fiber_metalens_nn.collimation_optics import vector_spectrum, propagate, flux, fingerprint

ROOT=Path(__file__).resolve().parents[1]


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


worker=load('height_family_setup_test','height_family_maxwell_worker_v2.py')
analysis=load('height_family_analysis_test','height_family_maxwell_analysis_v2.py')


def config():
    return {
        'result_label':'UNIT_TEST_SYNTHETIC_FIELDS_NO_SOLVE',
        'wavelength_um':1.55,'mfd_um':10.4,'cylinder_diameter_um':50.,
        'height_max_um':3.,'annuli':50,
        'proxy':{'output_clearance_um':1.},
        'materials':{'substrate':{'n':1.45,'kappa':0.},
                     'polymer':{'n':1.5263,'kappa':0.},'ambient':{'n':1.,'kappa':0.}},
        'evaluation':{'processing_planes_um':[0.,25.,50.,75.,100.],
                      'diagnostic_planes_um':[200.]},
        'natural_training':{'investigation_gates':{
            'transmission_min':.9,'declared_Gaussian_overlap_min':.98,
            'useful_efficiency_min':.9,'output_diameter_relative_tolerance':.1,
            'max_sidelobe_fraction':.02,'max_abs_normalized_curvature':.1,
            'max_angular_second_moment_ratio':1.1}},
    }


def gaussian_spectrum(count=256,window=128.,wavelength=1.31,diameter=14.,radius=None):
    dx=window/count;x=(np.arange(count)-count/2+.5)*dx
    xx,yy=np.meshgrid(x,x,indexing='xy');r2=xx*xx+yy*yy
    phase=0 if radius is None else np.pi*r2/(wavelength*radius)
    scalar=np.exp(-r2/(diameter/2)**2+1j*phase)
    return vector_spectrum(np.fft.fft2(scalar),dx,wavelength),dx


def test_setup_accepts_new_config_without_legacy_maxwell_block():
    cfg=config();s=worker.configured_dimensions(cfg,100.,16.)
    assert 'maxwell' not in cfg  # No mutation or legacy config substitution.
    assert s['monitor_radius_um']==35.
    assert s['cell_radial_extent_um']==37.
    assert s['output_z_um']==104. and s['crosscheck_z_um']==109.
    assert s['top_um']==113. and s['cell_axial_extent_um']==119.
    assert s['duration']==335.
    assert s['wavelength_um']==1.55 and s['mfd_um']==10.4


def test_tall_duration_is_a_lower_bound_and_explicit_extension_is_preserved():
    cfg=config()
    assert worker.effective_duration(cfg,40.)==160.
    assert worker.effective_duration(cfg,70.)==243.
    assert worker.effective_duration(cfg,100.,160.)==335.
    assert worker.effective_duration(cfg,100.,503.)==503.
    for invalid in [0.,-1.,float('nan')]:
        with pytest.raises(ValueError):worker.effective_duration(cfg,invalid)


def test_domains_follow_height_and_actual_padding_without_changing_config():
    cfg=config();s=worker.configured_dimensions(cfg,57.5,12.,padding=15.,pml=3.)
    assert s['monitor_radius_um']==40.
    assert s['cell_radial_extent_um']==43.
    assert s['output_z_um']==61.5
    assert s['crosscheck_z_um']==66.5
    assert 'maxwell' not in cfg
    with pytest.raises(ValueError):worker.configured_dimensions(cfg,40.,16.,pml=4.)


def test_source_envelope_has_declared_facet_waist_and_input_phase_index():
    r=np.array([0.,2.,5.2])
    actual=worker.source_envelope(r,0.,5.2,1.55,1.45)
    np.testing.assert_allclose(actual,np.exp(-(r/5.2)**2),atol=1e-14)
    a=worker.source_envelope(r,-2.,5.2,1.55,1.45)
    b=worker.source_envelope(r,-2.,5.2,1.31,1.5263)
    assert not np.allclose(a,b,rtol=1e-6,atol=1e-9)
    with pytest.raises(ValueError):worker.source_envelope(r,float('nan'),5.2,1.55,1.45)


def test_design_mismatch_rejected_before_meep_import():
    cfg=config();design={'units':'um','H_um':70.,'cylinder_diameter_um':50.,
                         'heights_um':[1.]*50}
    assert worker.validate_design(design,cfg,70.).shape==(50,)
    for key,value in [('units','mm'),('H_um',35.),('cylinder_diameter_um',48.),
                      ('heights_um',[4.]*50),('heights_um',[1.]*24)]:
        bad={**design,key:value}
        with pytest.raises(ValueError):worker.validate_design(bad,cfg,70.)


def test_actual_centered_stencil_handles_off_grid_height_without_assuming_bottom_origin():
    positions,weights=worker.centered_stencil_from_probe(100.1,[100.0625,100.125],.0625)
    np.testing.assert_allclose(positions,[100.0625,100.125],atol=1e-12)
    np.testing.assert_allclose(weights,[.4,.6],atol=1e-12)
    assert np.dot(positions,weights)==pytest.approx(100.1,abs=1e-12)
    assert worker.centered_stencil_from_probe(100.125,[100.125],.0625)[1]==[1.,0.]
    for bad in [[100.], [100.,100.125], [100.,100.0625,100.125]]:
        with pytest.raises(ValueError):worker.centered_stencil_from_probe(100.1,bad,.0625)


def test_dry_run_writes_no_solver_artifact_or_imports_meep(tmp_path):
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config()))
    output=tmp_path/'never_created.json'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/height_family_maxwell_worker_v2.py'),
        '--config',str(cfg),'--output',str(output),'--H','100','--resolution','16','--dry-run'],
        check=True,text=True,capture_output=True)
    info=json.loads(result.stdout)
    assert info['status']=='SETUP_ONLY_NO_SOLVE'
    assert info['settings']['duration']==335.
    assert not output.exists()
    assert sorted(q.name for q in tmp_path.iterdir())==['config.json']


@pytest.mark.parametrize('n',[1.,1.45,1.5263])
def test_forward_reference_decomposition_uses_actual_index_and_impedance(n):
    count=32;dx=.75
    field=np.zeros((6,count,count),complex);field[0]=1.;field[4]=n
    _,actual,power=analysis.forward_field_with_power(field,dx,1.31,n)
    np.testing.assert_allclose(actual,field,atol=1e-12)
    assert power==pytest.approx(.5*n*(count*dx)**2,rel=1e-12)


def test_gaussian_metrics_are_declared_target_based_and_diffraction_consistent():
    cfg=config();cfg['wavelength_um']=1.31
    spectrum,dx=gaussian_spectrum(wavelength=1.31)
    incident=float(flux(propagate(spectrum,dx,1.31,0)).sum()*dx**2)
    m=analysis.spectrum_metrics(spectrum,dx,1.31,14.,cfg,incident)
    assert m['declared_overlap']==pytest.approx(1.,abs=1e-12)
    assert m['transmission']==pytest.approx(1.,abs=1e-12)
    assert abs(m['normalized_curvature'])<1e-10
    assert m['angular_second_moment_ratio']==pytest.approx(1.,abs=1e-12)
    assert m['all_processing_crossings_valid']
    np.testing.assert_allclose(m['diameters_output_um'],m['target_widths_um'][0],atol=1e-12)
    assert m['fitted_core_divergence']['valid']
    expected=1.31/(math.pi*7)
    for axis in m['fitted_core_divergence']['axes']:
        assert axis['half_angle_rad']==pytest.approx(expected,rel=.02)
    plane=next(row for row in m['planes'] if row['z_um']==100.)
    expected_ratio=math.sqrt(1+(100/(math.pi*7**2/1.31))**2)
    assert plane['diameter_x_um']/m['diameters_output_um'][0]==pytest.approx(expected_ratio,rel=.01)
    assert m['M2'] is None and m['physical_validation'] is False
    assert 'Independent vector Maxwell' in m['model']


def test_phase_current_recovers_known_paraxial_curvature_at_nondefault_wavelength():
    spectrum,dx=gaussian_spectrum(wavelength=1.31,radius=500.)
    measured=analysis.phase_current_curvature(spectrum,dx,1.31,14.)
    assert measured['inverse_R_per_um']==pytest.approx(1/500.,rel=.02)
    assert measured['normalized_curvature']==pytest.approx(math.pi*7**2/(1.31*500),rel=.02)


def test_mismatched_configuration_is_rejected_before_field_files_are_read(tmp_path):
    cfg=config();settings={'m':1,'resolution':16.}
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps({'config_fingerprint':'wrong','settings':settings}))
    device=tmp_path/'device.json'
    device.write_text(json.dumps({'reference_file':str(reference),'settings':settings,
                                  'config_fingerprint':fingerprint(cfg)}))
    with pytest.raises(ValueError,match='configuration'):
        analysis.analyze(device,cfg,14.)


def test_wrong_azimuthal_record_is_not_silently_reconstructed(tmp_path):
    cfg=config();settings={'m':-1,'resolution':16.};fp=fingerprint(cfg)
    reference=tmp_path/'reference.json'
    reference.write_text(json.dumps({'config_fingerprint':fp,'settings':settings}))
    device=tmp_path/'device.json'
    device.write_text(json.dumps({'reference_file':str(reference),'settings':settings,
                                  'config_fingerprint':fp}))
    with pytest.raises(ValueError,match=r'm=\+1'):
        analysis.analyze(device,cfg,14.)


def test_symmetric_collapse_guard_validates_finite_shape_and_all_planes():
    analysis.require_symmetric_axial_collapse([[.5,.5],[.5,.5]],[44.,49.],'device')
    for weights in ([[1.,0.],[.5,.5]], [[.5,.5],[.4,.6]],
                    [[.5,.5],[float('nan'),.5]], [[.5,.5]]):
        with pytest.raises(ValueError):
            analysis.require_symmetric_axial_collapse(weights,[44.,49.],'device')


@pytest.mark.parametrize('bad_record',['device','reference'])
def test_analysis_rejects_asymmetric_saved_plane_before_reconstruction(tmp_path,bad_record):
    """Even the last reference plane, unused for incident conversion, is checked."""
    cfg=config();settings={'m':1,'resolution':16.};fp=fingerprint(cfg)
    field_file=tmp_path/'unused-fields.npz'
    np.savez(field_file,stub=np.empty(0))
    records={}
    for label,planes in (('device',[44.,49.]),('reference',[-1.,44.,49.])):
        weights=np.full((len(planes),2),.5)
        if label==bad_record:weights[-1]=[.4,.6]
        audit_file=tmp_path/f'{label}-audit.npz'
        np.savez(audit_file,pair_weights=weights,plane_z_um=planes)
        records[label]={'config_fingerprint':fp,'settings':settings,
                        'fields_file':str(field_file),'power_audit_fields_file':str(audit_file)}
    reference=tmp_path/'reference.json';reference.write_text(json.dumps(records['reference']))
    device=tmp_path/'device.json'
    device.write_text(json.dumps({**records['device'],'reference_file':str(reference)}))
    with pytest.raises(ValueError,match=f'{bad_record}: symmetric axial collapse'):
        analysis.analyze(device,cfg,14.)
