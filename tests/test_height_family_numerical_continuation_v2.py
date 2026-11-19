"""Pure scheduling checks; no Meep/JAX import or child process is needed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1]/'scripts/continue_height_family_numerical_checks_v2.py'
spec = importlib.util.spec_from_file_location('height_family_conservative_continuation_test', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def ledger(entries=None):
    return dict(first_started_utc='2026-11-12T09:17:00Z', elapsed_s=100.,
                entries=[] if entries is None else entries)


def test_absolute_elapsed_includes_downtime_and_never_resets_recorded_time():
    assert module.absolute_elapsed(ledger(), '2026-11-12T11:17:00Z') == 7200.
    prior = ledger();prior['elapsed_s'] = 7300.
    assert module.absolute_elapsed(prior, '2026-11-12T11:17:00Z') == 7300.
    assert module.absolute_elapsed(ledger(), '2026-11-12T13:17:00Z') == 14400.
    with pytest.raises(ValueError, match='original start'):
        module.absolute_elapsed(ledger(), '2026-11-12T09:16:59Z')
    with pytest.raises(ValueError, match='Timezone-aware'):
        module.absolute_elapsed(dict(ledger(), first_started_utc='2026-11-12T09:17:00'),
                                '2026-11-12T11:17:00Z')


def test_selection_is_exact_original_plan_pairs_only():
    original = module.runner.make_plan()
    chosen = module.select_pairs(original, ['pml_H40', 'padding_H70', 'duration_H100'])
    assert [pair['name'] for pair in chosen] == ['pml_H40', 'padding_H70', 'duration_H100']
    for pair in chosen:
        assert [job['method'] for job in pair['jobs']] == ['reference', 'NN']
        assert all(job in original for job in pair['jobs'])
    for names in [[], ['pml_H50'], ['nominal_H40'], ['pml_H40', 'pml_H40'], ['reserve_H100']]:
        with pytest.raises(ValueError):
            module.select_pairs(original, names)


def test_insufficient_original_time_or_solve_slots_prevents_pair():
    pair = module.select_pairs(module.runner.make_plan(), ['pml_H40'])[0]
    assert module.pair_fits(pair, ledger(), 14000., 200.)
    assert not module.pair_fits(pair, ledger(), 14000., 220.)
    assert not module.pair_fits(pair, ledger(), 14220., 0.)
    # Both jobs are new settings;59 prior attempts leave only one solve slot.
    old = dict(pair['jobs'][0], pml_um=2., status='completed')
    assert not module.pair_fits(pair, ledger([old]*59), 100., 10.)


def test_failed_settings_are_refused_and_live_prior_pid_blocks():
    pair = module.select_pairs(module.runner.make_plan(), ['pml_H40'])[0]
    failed = dict(pair['jobs'][0], status='timeout', owned_pid=123)
    with pytest.raises(ValueError, match='retries are forbidden'):
        module.require_unfailed_selected_settings(ledger([failed]), [pair])
    with pytest.raises(RuntimeError, match='still exists'):
        module.require_no_live_owned_child(ledger([failed]), alive=lambda pid: pid == 123)
    module.require_no_live_owned_child(ledger([failed]), alive=lambda pid: False)


def test_pair_estimate_uses_measured_nominal16_domain_and_25percent_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(module.runner, 'FOLDER', tmp_path)
    pair = module.select_pairs(module.runner.make_plan(), ['pml_H40'])[0]
    entries = []
    for i, (method, cost) in enumerate([('reference', 100.), ('NN', 120.)], 1):
        record = tmp_path/f'nominal-{i}.json'
        record.write_text(json.dumps(dict(settings=dict(cell_radial_extent_um=37., cell_axial_extent_um=59.))))
        entries.append(dict(method=method, H_um=40., resolution=16., pml_um=2., padding_um=10.,
                            duration=160., job_type='nominal', status='completed', id=i,
                            wrapper_wall_s=cost, record_file=record.name, record_sha256=module.runner.sha(record)))
    cost = module.estimated_pair_cost(pair, ledger(entries))
    assert cost['estimated_total_s'] == pytest.approx(1.25*(100+120)*(38/37)*(60/59)+120)
    assert all(row['margin'] == 1.25 for row in cost['measurements'])
    assert [row['nominal_entry_id'] for row in cost['measurements']] == [1, 2]


def test_solver_and_analysis_caps_share_truthful_absolute_reserve_boundary(monkeypatch):
    calls = []
    fake_processes = SimpleNamespace(run=lambda *args, **kwargs: calls.append(kwargs['timeout']))
    monkeypatch.setattr(module.runner, 'subprocess', fake_processes)
    validator = SimpleNamespace(elapsed=lambda: 14210.)
    with module.supervision_caps(validator):
        assert module.runner.check_capacity(59, 1.) == 5.
        module.runner.subprocess.run(['analysis-only-stub'], timeout=120.)
        assert calls == [5.]
        with pytest.raises(RuntimeError, match='solve limit'):
            module.runner.check_capacity(60, 1.)
    assert module.runner.subprocess is fake_processes
    assert validator.elapsed() == 14210.  # Reserve never inflates elapsed time.


def test_pairs_without_explicit_run_do_not_launch_or_touch_ledger(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['helper', '--pairs', 'pml_H40,padding_H70'])
    monkeypatch.setattr(module, 'run_continuation', lambda names: pytest.fail('No implicit continuation'))
    monkeypatch.setattr(module.runner, 'Validation', lambda: pytest.fail('No implicit Validation initialization'))
    module.main()
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'INSPECTION_ONLY_NO_LAUNCH'
    assert value['optical_solves_launched'] == 0
    assert [pair['name'] for pair in value['selected_pairs']] == ['pml_H40', 'padding_H70']


def adaptive_fixture(folder):
    folder=Path(folder)
    config={'synthetic_pure_fixture':True}
    (folder/'config.json').write_text(json.dumps(config))
    (folder/'maxwell_validation_proposal.json').write_text(json.dumps(
        dict(numerical_acceptance=dict(curvature_last_pair_delta_max=.005))))
    design=folder/'frozen-H40.json';design.write_text(json.dumps(dict(H_um=40.,heights_um=[0.]*50)))
    cases=[dict(method='NN',H_um=40.,design_file=design.name,design_sha256=module.runner.sha(design))]
    entries=[]
    for identity,method,resolution,cost,current in [
            (1,'NN',16.,50.,-.046851),(2,'reference',24.,100.,None),(3,'NN',24.,120.,-.031241)]:
        settings=dict(H_um=40.,resolution=resolution,pml_um=2.,padding_um=10.,duration=160.,
                      cell_radial_extent_um=37.,cell_axial_extent_um=59.)
        record=folder/f'record-{identity}.json'
        record.write_text(json.dumps(dict(settings=settings,
            config_fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())))
        entry=dict(id=identity,method=method,H_um=40.,resolution=resolution,pml_um=2.,padding_um=10.,
                   duration=160.,job_type='nominal',status='completed',wrapper_wall_s=cost,
                   record_file=record.name,record_sha256=module.runner.sha(record))
        if method=='NN':
            analysis=folder/f'analysis-{identity}.json'
            analysis.write_text(json.dumps(dict(settings=settings,reconstruction_pass=True,normalized_curvature=current)))
            entry.update(analysis_status='completed',analysis_file=analysis.name,
                         analysis_sha256=module.runner.sha(analysis),design_sha256=cases[0]['design_sha256'])
        entries.append(entry)
    return ledger(entries),cases


def test_only_exact_original_adaptive_H40_mesh_reserve_is_selected():
    pair=module.select_pairs(module.runner.make_plan(),[module.ADAPTIVE_NAME])[0]
    assert pair['selection_rule']=='approved_original_adaptive_mesh_reserve'
    assert pair['in_nominal_58_plan'] is False
    assert [job['method'] for job in pair['jobs']]==['reference','NN']
    for job in pair['jobs']:
        assert module.runner.cache_identity(job)==(40.,32.,2.,10.,160.)
        assert job['job_type']=='reserve_mesh'
    for name in ['reserve_mesh_H70_r32','reserve_mesh_H40_r24','reserve_mesh_H40_r40']:
        with pytest.raises(ValueError):module.select_pairs(module.runner.make_plan(),[name])


def test_adaptive_selection_requires_observed_failure_and_measured24_cost(tmp_path,monkeypatch):
    monkeypatch.setattr(module.runner,'FOLDER',tmp_path)
    data,cases=adaptive_fixture(tmp_path)
    pair=module.select_pairs(module.runner.make_plan(),[module.ADAPTIVE_NAME])[0]
    evidence=module.adaptive_selection_evidence(pair,data,cases)
    assert evidence['observed_H40_nonconverged'] is True
    assert evidence['curvature_absolute_delta']==pytest.approx(.01561)
    assert evidence['unchanged_curvature_delta_limit']==.005
    estimate=module.estimated_pair_cost(pair,data)
    assert estimate['estimated_total_s']==pytest.approx(1.15*(100+120)*(32/24)**3+120)
    assert all(row['measured_resolution']==24 and row['margin']==1.15 for row in estimate['measurements'])
    assert module.pair_fits(pair,data,13000.,estimate['estimated_total_s'])
    assert not module.pair_fits(pair,data,14000.,estimate['estimated_total_s'])


def test_adaptive_selection_rejects_changed_hashes_and_converged_pair(tmp_path,monkeypatch):
    monkeypatch.setattr(module.runner,'FOLDER',tmp_path)
    data,cases=adaptive_fixture(tmp_path)
    pair=module.select_pairs(module.runner.make_plan(),[module.ADAPTIVE_NAME])[0]
    entry=data['entries'][-1];path=tmp_path/entry['analysis_file']
    analysis=json.loads(path.read_text());analysis['normalized_curvature']=-.046
    path.write_text(json.dumps(analysis))
    with pytest.raises(ValueError,match='hashes changed'):
        module.adaptive_selection_evidence(pair,data,cases)
    entry['analysis_sha256']=module.runner.sha(path)
    with pytest.raises(ValueError,match='curvature delta'):
        module.adaptive_selection_evidence(pair,data,cases)


def test_adaptive_reserve_refuses_prior_attempts_and_never_exceeds_two(tmp_path,monkeypatch):
    monkeypatch.setattr(module.runner,'FOLDER',tmp_path)
    data,cases=adaptive_fixture(tmp_path)
    pair=module.select_pairs(module.runner.make_plan(),[module.ADAPTIVE_NAME])[0]
    first=dict(pair['jobs'][0],status='timeout')
    data['entries'].append(first)
    with pytest.raises(ValueError,match='no prior reserve'):
        module.adaptive_selection_evidence(pair,data,cases)
    module.require_adaptive_attempt_ceiling(data)
    data['entries'].append(dict(pair['jobs'][1],status='started'))
    module.require_adaptive_attempt_ceiling(data)
    data['entries'].append(dict(pair['jobs'][1],status='failed'))
    with pytest.raises(ValueError,match='at most two'):
        module.require_adaptive_attempt_ceiling(data)


def test_adaptive_name_without_run_is_inspection_only(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['helper','--pairs',module.ADAPTIVE_NAME])
    monkeypatch.setattr(module,'run_continuation',lambda names:pytest.fail('No implicit reserve launch'))
    module.main()
    output=json.loads(capsys.readouterr().out)
    assert output['status']=='INSPECTION_ONLY_NO_LAUNCH'
    assert output['selected_pairs'][0]['in_nominal_58_plan'] is False
