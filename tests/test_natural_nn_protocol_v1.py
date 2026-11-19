"""Fast protocol/selection regressions; no optical solve or GPU allocation."""
import copy,json
from pathlib import Path
import numpy as np
from fiber_metalens_nn.natural_nn_loss_v1 import selection_key
BASE=Path(__file__).resolve().parents[1]/'artifacts/1550nm_natural_nn_v1'

def metric():
    return dict(transmission=.95,declared_overlap=.995,declared_useful_efficiency=.945,
      max_sidelobe_fraction=.012,normalized_curvature=.02,angular_second_moment_ratio=1.01,
      widths_um=[[10,10]]*5,target_widths_um=[[10,10]]*5,valid_crossings=[[True,True]]*5,loss=-.94)

def config():return json.loads((BASE/'config.json').read_text())

def test_low_power_narrow_core_cannot_win_selection_by_loss_or_overlap():
    c=config()['natural_training']['investigation_gates'];good=metric();bad=metric();bad.update(transmission=.1,declared_overlap=1.,declared_useful_efficiency=.1,loss=-1000.)
    assert selection_key([good],c)<selection_key([bad],c)

def test_invalid_or_wrong_sized_profile_cannot_win():
    c=config()['natural_training']['investigation_gates'];good=metric();bad=copy.deepcopy(good);bad['widths_um'][0]=[2,2]
    assert selection_key([good],c)<selection_key([bad],c)
    bad=copy.deepcopy(good);bad['valid_crossings'][0]=[False,False]
    assert selection_key([good],c)<selection_key([bad],c)

def test_feasible_selection_prefers_worst_height_useful_efficiency():
    c=config()['natural_training']['investigation_gates'];good=metric();bad=metric();bad['declared_useful_efficiency']=.93;bad['loss']=-100.
    assert selection_key([good],c)<selection_key([bad],c)

def test_disjoint_splits_and_heldout_excluded_from_all_training_target_caches():
    c=config()['natural_training'];train=set(c['train_H_um']);val=set(c['validation_H_um']);held=set(c['heldout_H_um'])
    assert not train&val and not train&held and not val&held and held=={35}
    for name in ['pilot_frozen_training_targets.json','train_frozen_training_targets.json']:
        assert '35.0' not in json.loads((BASE/name).read_text())

def test_all_methods_have_three_seeds_and_equal_actual_budget_and_recovery():
    result=json.loads((BASE/'training_result.json').read_text())
    assert len(result['reports'])==6
    assert all(r['steps']==100 and r['objective_calls']==142 and r['trained_recovery']['passed'] for r in result['reports'])

def test_approved_allocation_preserves_three_true_devices_and_old_ledger():
    p=json.loads((BASE/'allocation_and_protocol.json').read_text())
    assert p['approved_cumulative_unique_ceiling']==13 and p['historical_count']==10
    assert p['new_slot_allocation']==['untouched_NN_H35','field_derived_realizable_baseline_H35','matched_non_neural_direct_H35']
    assert len(p['deferred_actual_device_cases'])==2
