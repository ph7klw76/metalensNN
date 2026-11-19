"""Pure boundary tests: these never import Meep or launch an optical solve."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


path=Path(__file__).resolve().parents[1]/"scripts/run_height_family_maxwell_v2.py"
spec=importlib.util.spec_from_file_location("height_family_runner_v2",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_approved_plan_does_not_reset_cumulative_budget():
    plan=module.make_plan()
    devices={(j["method"],j["H_um"]) for j in plan if j["method"]!="reference"}
    assert len(plan)==58
    assert len(devices)==11
    assert 13+len(devices)==24
    assert len(plan)+2==60


def test_every_device_has_prior_exact_settings_reference():
    references=set()
    for job in module.make_plan():
        identity=module.cache_identity(job)
        if job["method"]=="reference":
            references.add(identity)
        else:
            assert identity in references


def test_numeric_representations_share_cache_but_physics_does_not():
    job=dict(H_um=40,resolution=16,pml_um=2,padding_um=10,duration=160)
    floats={k:float(v) for k,v in job.items()}
    assert module.cache_identity(job)==module.cache_identity(floats)
    assert module.cache_identity(job)!=module.cache_identity(dict(job,H_um=50))
    assert module.cache_identity(job)!=module.cache_identity(dict(job,pml_um=3))


def test_stop_limits_and_remaining_deadline():
    assert module.check_capacity(0,0)==1800
    assert module.check_capacity(59,14398)==2
    with pytest.raises(RuntimeError,match="solve limit"):
        module.check_capacity(60,0)
    with pytest.raises(RuntimeError,match="wall-time"):
        module.check_capacity(59,14399)


def test_actual_tall_roundtrip_cannot_inherit_short_duration():
    assert module.duration(40)==160
    assert module.duration(100)==335
    assert module.duration(100)*1.5==502.5


@pytest.mark.parametrize('previous_status',[None,'failed'])
def test_completed_solver_recovers_analysis_only_and_preserves_attempts(tmp_path,monkeypatch,previous_status):
    monkeypatch.setattr(module,'FOLDER',tmp_path)
    job=dict(method='NN',H_um=40.,resolution=16.,pml_um=2.,padding_um=10.,duration=160.,job_type='nominal')
    fields=tmp_path/'fields.npz';fields.write_bytes(b'preserved raw fields')
    audit=tmp_path/'audit.npz';audit.write_bytes(b'preserved native flux')
    record=tmp_path/'record.json'
    record.write_text(json.dumps(dict(fields_file=str(fields),power_audit_fields_file=str(audit))))
    design=tmp_path/'design.json';design.write_text(json.dumps(dict(target_diameter_um=14.)))
    entry=dict(job,id=1,status='completed',record_file=record.name,record_sha256=module.sha(record))
    old_log=None
    if previous_status:
        entry['analysis_status']=previous_status
        entry['analysis_attempts']=[dict(number=1,status=previous_status)]
        (tmp_path/'logs').mkdir()
        old_log=tmp_path/'logs/record.analysis-attempt-01.log';old_log.write_text('original failure evidence')
    validation=module.Validation.__new__(module.Validation)
    validation.ledger={'entries':[entry]};validation.env={};validation.active=None
    validation.cases=[dict(method='NN',H_um=40.,design_file=design.name)]
    validation.verify=lambda:None
    validation.elapsed=lambda:100.
    validation.save=lambda:None
    commands=[]
    def fake_analysis(command,**kwargs):
        assert str(module.ANALYZER) in command
        assert kwargs['timeout']<=14400-validation.elapsed()
        commands.append(command)
        out=Path(command[command.index('--output')+1])
        out.write_text(json.dumps(dict(synthetic_optical_pass=True,reconstruction_pass=True,
                                      normalized_curvature=.01,conservative_useful_efficiency=.95)))
        kwargs['stdout'].write('saved-field analysis only')
        return SimpleNamespace(returncode=0)
    def forbidden_solve(*args,**kwargs):
        raise AssertionError('Recovery must not launch a Meep subprocess')
    monkeypatch.setattr(module.subprocess,'run',fake_analysis)
    monkeypatch.setattr(module.subprocess,'Popen',forbidden_solve)
    result=validation.execute(job)
    assert result is entry and len(validation.ledger['entries'])==1
    assert entry['analysis_status']=='completed' and entry['C']==.01
    assert module.sha(tmp_path/entry['analysis_file'])==entry['analysis_sha256']
    assert len(commands)==1
    assert validation.execute(job) is entry
    assert len(commands)==1  # Successful analysis cache launches neither process.
    if old_log:
        assert old_log.read_text()=='original failure evidence'
        assert entry['analysis_attempts'][-1]['number']==2


def test_analysis_recovery_obeys_existing_wall_budget(tmp_path,monkeypatch):
    validation=module.Validation.__new__(module.Validation)
    validation.verify=lambda:None;validation.elapsed=lambda:14400.;validation.save=lambda:None
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('No analysis after wall budget'))
    with pytest.raises(RuntimeError,match='Wall budget exhausted before analysis'):
        validation.analyze_entry(dict(method='NN'))
