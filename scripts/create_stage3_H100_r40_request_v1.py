"""Record ROOT's final adaptive choice; no solver is imported or launched."""
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
    if len(ledger['entries'])!=16 or any(e['status']!='completed' for e in ledger['entries']):
        raise ValueError('Exactly16 completed stage3 solves are required for this final pair')
    if len(ledger.get('adaptive_pair_runs',[]))!=1 or ledger['adaptive_pair_runs'][0]['status']!='completed':
        raise ValueError('The H60 adaptive pair must finish before the final selection')
    report=FOLDER/'assessment/after_H60_r40_adaptive_v1/assessment.json'
    assessment=read(report)
    if any(assessment[k]['status']!='PASS' for k in ('authorization','budget','primary_execution','adaptive_execution',
                                                    'all_eight_nominal_heights_acceptance')):
        raise ValueError('The completed first-pair assessment must verify all8 nominal heights')
    utility=ROOT/'scripts/assess_height_family_stage3_v2.py'
    expected='96dd3849f2d3c5997d6c05524ac5503452694cd78819619cee766a878d1f7b4c'
    if sha(utility)!=expected:raise ValueError('Reviewed assessment utility changed')
    byid={e['id']:e for e in ledger['entries']}
    forecast=sum(byid[i]['wrapper_wall_s']*(40./32.)**3*1.15 for i in (66,67))+120.
    job=dict(H_um=100.,resolution=40.,pml_um=2.,padding_um=10.,duration=335.)
    request=dict(
        reason='All8 heights now pass the frozen native/nominal/core-mesh protocol after the H60 correction. '
               'ROOT selects the final2 approved calls to strengthen numerical evidence at the user-requested '
               'maximum height100um. Its retained-band Q changed from1.103364771550854 at24 to '
               '1.0436246058031338 at32, and saved-field bins locate most of that change in faint wide-angle '
               'content. Q mesh drift is not a frozen convergence gate, so this new40-grid pair is a prospective '
               'diagnostic of unresolved angular-moment spatial sensitivity, not a new acceptance threshold '
               'or geometry optimization. Increase only resolution against NN67/reference66. This is the '
               'second and final pair; the18/83 ceiling remains exact.',
        fine_time_priority_review='Latest H100/H75 Q margins to1.10 are0.0563754/0.0688320, larger than '
               'the observed coarse H100 time drift0.003522181559. Hence the conditional close-Q temporal '
               'priority is not triggered. Existing coarse numerical-duration checks pass, but do not prove '
               'fine-grid temporal or all-angle convergence. Isolate spatial resolution at the unchanged335 '
               'duration, transfer no coarse correction, add no Q convergence gate, and claim no fully-steady, '
               'ISO-M2 or global diffraction optimum.',
        primary_assessment_file=str(report.resolve()),primary_assessment_sha256=sha(report),
        primary_assessment_ledger_sha256=sha(FOLDER/'ledger.json'),
        primary_assessment_utility_file=str(utility.resolve()),primary_assessment_utility_sha256=sha(utility),
        pairs=[dict(pair_id='H100_r40_pair2',
                    reason='Unresolved retained-band angular-moment mesh sensitivity at the maximum requested '
                           'height; compare the same frozen NN100 profile at32 and40 without other changes.',
                    baseline=dict(stage_id='stage3',record_id=67),changed_variable='resolution',
                    forecast_ceiling_s=math.ceil(forecast/60.)*60.+600.,
                    jobs=[dict(method='reference',**job),dict(method='NN',**job)])],
    )
    output=FOLDER/'adaptive_requests/H100_r40_pair2_v1.json';output.parent.mkdir(exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(request,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='ROOT_FINAL_PAIR_REQUEST_WRITTEN_NO_WORKERS',file=str(output),
                          sha256=sha(output),measured_forecast_h=forecast/3600,
                          recorded_utc=datetime.now(timezone.utc).isoformat(),
                          projected_new_attempts=18,projected_cumulative_attempts=83),indent=2))

if __name__=='__main__':main()
