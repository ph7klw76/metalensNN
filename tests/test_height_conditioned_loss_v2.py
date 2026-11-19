"""Small CPU checks for unequal-index, independently conditioned height loss.

These tests do not launch Meep or a training campaign. A real CPU JAX backend
is required for proxy tests; all computation uses deliberately small grids.
"""
import copy
import json
import os

# Set before importing any package module that could initialize a JAX backend.
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fiber_metalens_nn.height_conditioned_loss_v2 import (
    HeightConditionedLoss, normal_interface_power, selection_key,
)


def config():
    return {
        'wavelength_um': 1.55, 'mfd_um': 10.4, 'fiber_nominal_NA': .14,
        'fiber_NA_interpretation': 'Datasheet 1%-power far-field NA at1310nm; not a mode cutoff',
        'mfd_provenance': 'Corning PI1424:10.4+/-0.5um at1550nm; Gaussian surrogate',
        'input_phase_index_is_assumed': True,
        'materials': {'substrate': {'n':1.45,'kappa':0.},
                      'polymer': {'n':1.5263,'kappa':0.},
                      'ambient': {'n':1.,'kappa':0.}},
        'material_provenance': 'Polymer index supplied by user; input phase index1.45 assumed',
        'cylinder_diameter_um': 50., 'annuli': 50, 'height_max_um': 3.,
        'proxy': {'grid_n':32,'window_um':64.,'padding':2,
                  'pedestal_dz_um':5.,'relief_slices':8,'output_clearance_um':1.},
        'fabrication': {'height_step_um':.05},
        'evaluation': {'processing_planes_um':[0.,25.,100.]},
        'loss': {'negative_useful_power':1.,'width':.05,'phase':.1,
                 'angular':.05,'sidelobe':.1,'roughness':.0002,
                 'transmission_floor':.9,'low_power_penalty':10.},
    }


def test_unequal_index_fresnel_power_identity_requires_impedance_factor():
    result = normal_interface_power(1.45, 1.5263)
    expected_r = ((1.45-1.5263)/(1.45+1.5263))**2
    assert result['power_reflection'] == pytest.approx(expected_r)
    assert result['power_transmission'] == pytest.approx(1-expected_r, abs=1e-14)
    assert result['power_closure_error'] < 1e-14
    assert result['electric_amplitude_transmission']**2 != pytest.approx(result['power_transmission'])
    reverse = normal_interface_power(1.5263, 1.45)
    assert reverse['power_transmission'] == pytest.approx(result['power_transmission'])


def test_index_matched_limit_and_invalid_indices():
    result = normal_interface_power(1.5263, 1.5263)
    assert result['power_transmission'] == 1
    assert result['power_reflection'] == 0
    for invalid in [0., -1., float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            normal_interface_power(invalid, 1.5263)


def test_unequal_index_bundle_and_metadata_are_explicit_and_json_safe():
    loss = HeightConditionedLoss(config())
    bundle = loss.bundle(40.)
    assert len(bundle) == 6
    assert loss.proxy is loss.p
    assert bundle[3].shape == (50,)
    assert bundle[2].shape == (3,2)
    assert np.isfinite(np.asarray(bundle[0])).all()
    assert np.all(np.asarray(bundle[3]) >= 0) and np.all(np.asarray(bundle[3]) <= 3)
    metadata = loss.metadata['40.0']
    assert metadata['materials']['indices_are_matched'] is False
    assert metadata['materials']['input_n'] == 1.45
    assert metadata['materials']['polymer_n'] == 1.5263
    assert metadata['input_mode']['mfd_um'] == 10.4
    assert metadata['input_mode']['input_phase_index_is_assumed']
    assert metadata['target_output_z_um'] == 44.
    assert 'eigenmode' in metadata['input_mode']['model']
    json.dumps(loss.metadata, allow_nan=False)
    assert loss.bundle(40) is bundle


def test_declared_sweep_can_exceed_old_eleven_height_cache():
    loss = HeightConditionedLoss(config())
    for H in range(40,101,5):
        loss.bundle(H)
    assert len(loss.bundle_cache) == 13
    assert set(loss.metadata) == {str(float(H)) for H in range(40,101,5)}


def test_changing_mfd_rebuilds_input_and_natural_target():
    nominal = HeightConditionedLoss(config())
    changed_cfg = config()
    changed_cfg['mfd_um'] = 10.9
    changed = HeightConditionedLoss(changed_cfg)
    a, b = nominal.bundle(40), changed.bundle(40)
    assert abs(float(a[4])-float(b[4])) > 1e-3
    assert changed.metadata['40.0']['input_mode']['mfd_um'] == 10.9


def test_nested_datasheet_provenance_is_preserved_without_using_group_index():
    cfg = config()
    cfg.pop('mfd_provenance')
    cfg.pop('fiber_NA_interpretation')
    cfg.pop('input_phase_index_is_assumed')
    cfg['input_mode_provenance'] = {
        'manufacturer':'Corning SMF-28 Ultra',
        'mfd_1550_um':10.4,'mfd_tolerance_um':.5,
        'source_url':'https://example.invalid/unit-test-datasheet',
        'NA_measurement':'one percent far-field power at1310nm',
        'source_phase_index':1.45,'source_phase_index_is_assumed':True,
        'group_index_1550':1.4682,'group_index_not_used_as_phase_index':True,
    }
    cfg['campaign_limits'] = {'bundle_cache_max':32}
    loss = HeightConditionedLoss(cfg)
    loss.bundle(40.)
    mode = loss.metadata['40.0']['input_mode']
    assert mode['datasheet_nominal_mfd_um'] == 10.4
    assert mode['datasheet_mfd_tolerance_um'] == .5
    assert mode['input_phase_index'] == 1.45
    assert mode['input_phase_index_is_assumed']
    assert mode['manufacturer_provenance']['group_index_1550'] == 1.4682
    assert '1310nm' in mode['fiber_NA_interpretation']
    assert loss.bundle_limit == 32
    json.dumps(loss.metadata,allow_nan=False)


def test_cpu_loss_and_sampled_gradient_are_finite_and_match_finite_difference():
    assert jax.devices()[0].platform == 'cpu'
    loss = HeightConditionedLoss(config())
    bundle = loss.bundle(70.)
    heights = jnp.full((50,),1.2,dtype=jnp.float32)
    objective = lambda h: loss.terms(h,bundle)['loss']
    value, gradient = jax.value_and_grad(objective)(heights)
    assert np.isfinite(float(value)) and np.isfinite(np.asarray(gradient)).all()
    # r=sqrt(1^2+3^2)um is an illuminated sampled point on this tiny grid,
    # lying in zero-based annulus6. Avoid a vacuous derivative in an empty bin.
    delta = np.zeros(50,np.float32);delta[6]=.001
    fd = float((objective(heights+delta)-objective(heights-delta))/.002)
    ad = float(gradient[6])
    assert abs(ad) > 1e-7
    assert abs(fd-ad) < .003 + .1*max(abs(fd),abs(ad))
    metrics = loss.terms(heights,bundle)
    assert metrics['weighted_terms'].shape == (8,)
    assert float(metrics['loss']) == pytest.approx(float(metrics['weighted_terms'].sum()))
    assert 0 <= float(metrics['declared_overlap']) <= 1.00001
    with pytest.raises(ValueError):
        loss.terms(jnp.ones(49),bundle)


def test_invalid_height_and_loss_configuration_are_rejected_before_work():
    loss = HeightConditionedLoss(config())
    for H in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError):
            loss.bundle(H)
    bad=config();bad['loss']['phase']=-1
    with pytest.raises(ValueError):
        HeightConditionedLoss(bad)
    bad=config();bad['evaluation']['processing_planes_um']=[25.,100.]
    with pytest.raises(ValueError):
        HeightConditionedLoss(bad)


def test_selection_preserves_throughput_and_does_not_reward_low_power_core():
    criteria = {'transmission_min':.9,'declared_Gaussian_overlap_min':.98,
                'useful_efficiency_min':.9,'max_sidelobe_fraction':.02,
                'max_abs_normalized_curvature':.1,
                'max_angular_second_moment_ratio':1.1,
                'output_diameter_relative_tolerance':.1}
    good = {'transmission':.95,'declared_overlap':.995,
            'declared_useful_efficiency':.945,'max_sidelobe_fraction':.01,
            'normalized_curvature':.01,'angular_second_moment_ratio':1.,
            'widths_um':[[12.,12.]]*3,'target_widths_um':[[12.,12.]]*3,
            'valid_crossings':[[True,True]]*3,'loss':-.94}
    bad = copy.deepcopy(good)
    bad.update(transmission=.1,declared_useful_efficiency=.1,loss=-100.)
    assert selection_key([good],criteria) < selection_key([bad],criteria)
