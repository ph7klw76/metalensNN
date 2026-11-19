"""Record ROOT's first specific adaptive choice; this launches no worker."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v3'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    ledger=read(FOLDER/'ledger.json')
    if len(ledger['entries'])!=14 or any(e['status']!='completed' for e in ledger['entries']):
        raise ValueError('All14 primary solves must be completed before this specific choice')
    report=FOLDER/'assessment/primary_final_v1/assessment.json'
    assessment=read(report)
    if any(assessment[k]['status']!='PASS' for k in ['authorization','budget','primary_execution']):
        raise ValueError('Reviewed full primary assessment requires execution/protocol/budget PASS')
    utility=ROOT/'scripts/assess_height_family_stage3_v2.py'
    if sha(utility)!='96dd3849f2d3c5997d6c05524ac5503452694cd78819619cee766a878d1f7b4c':
        raise ValueError('Reviewed assessment utility changed')
    byid={e['id']:e for e in ledger['entries']}
    scale=(40./32.)**3
    forecast=sum(byid[i]['wrapper_wall_s']*scale*1.15 for i in (76,77))+120.
    job=dict(H_um=60.,resolution=40.,pml_um=2.,padding_um=10.,duration=213.)
    request=dict(
        reason='ROOT selects H60 r40 as the first adaptive pair because it is the only remaining '
               'declared mesh failure after all14 primaries. The unchanged NN geometry at r32 has '
               'native and nominal PASS, but its24-to32 curvature change0.013246714 exceeds0.005. '
               'Increase only resolution against completed NN77/reference76; preserve all old failures. '
               'The second adaptive pair remains unselected until this result is assessed.',
        fine_time_priority_review='At the latest fine grids, H100 Q=1.0436246058031338 and H75 '
               'Q=1.0311679786023833 are below1.10 by0.0563754 and0.0688320. These margins are '
               'larger than the observed coarse H100 time drift0.003522181559. That drift is not '
               'transferred as a correction or an added gate. First resolve the actual H60 spatial '
               'curvature failure; consider remaining numerical/temporal uncertainty after the pair. '
               'No fully-steady, full-angle or global-diffraction-limit claim is made.',
        primary_assessment_file=str(report.resolve()),primary_assessment_sha256=sha(report),
        primary_assessment_ledger_sha256=sha(FOLDER/'ledger.json'),
        primary_assessment_utility_file=str(utility.resolve()),primary_assessment_utility_sha256=sha(utility),
        pairs=[dict(pair_id='H60_r40_pair1',
                    reason='Only unresolved declared primary curvature convergence: isolate r32-to40 '
                           'on the frozen H60 profile, with a new matched reference.',
                    baseline=dict(stage_id='stage3',record_id=77),changed_variable='resolution',
                    forecast_ceiling_s=math.ceil(forecast/60.)*60.+600.,
                    jobs=[dict(method='reference',**job),dict(method='NN',**job)])],
    )
    output=FOLDER/'adaptive_requests/H60_r40_pair1_v1.json'
    output.parent.mkdir(exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(request,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='ROOT_SPECIFIC_REQUEST_WRITTEN_NO_WORKERS',file=str(output),
                          sha256=sha(output),measured_forecast_h=forecast/3600,
                          recorded_utc=datetime.now(timezone.utc).isoformat(),
                          second_pair_selected=False),indent=2))

if __name__=='__main__':main()
