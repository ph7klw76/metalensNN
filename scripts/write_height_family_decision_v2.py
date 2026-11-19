"""Readable decision and latest-measured table from completed saved records."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    output=FOLDER/'DECISION.txt';table_path=FOLDER/'NN_Maxwell_latest.csv'
    if output.exists() or table_path.exists():raise FileExistsError('Preserve the existing decision.')
    ledger=read(FOLDER/'ledger.json');summary=read(FOLDER/'summary.json')
    if 'RUNNING' in ledger['status']:raise RuntimeError('Wait for the optical worker to stop.')
    assessments={row['H_um']:row for row in summary['NN_heights']}
    rows=[]
    for H in (40,50,60,70,80,90,100):
        entries=[e for e in ledger['entries'] if e['method']=='NN' and e['H_um']==H
                 and e['job_type'] in ('nominal','reserve_mesh')
                 and e['status']=='completed' and e.get('analysis_status')=='completed']
        if not entries:continue
        entry=max(entries,key=lambda e:e['resolution']);a=read(FOLDER/entry['analysis_file'])
        metrics=a['metrics'];angles=metrics['fitted_core_divergence']['axes']
        assessment=assessments[H]
        measurements=assessment.get('comparison',{}).get('measurements',{})
        rows.append(dict(H_um=H,resolution_cells_per_um=entry['resolution'],
            output_Dx_um=metrics['diameters_output_um'][0],output_Dy_um=metrics['diameters_output_um'][1],
            core_half_angle_x_deg=angles[0]['half_angle_rad']*180/math.pi,
            core_half_angle_y_deg=angles[1]['half_angle_rad']*180/math.pi,
            useful_efficiency=a['conservative_useful_efficiency'],normalized_curvature=a['normalized_curvature'],
            retained_band_Q=a['angular_second_moment_ratio'],nominal_optical_pass=a['synthetic_optical_pass'],
            native_reconstruction_pass=a['reconstruction_pass'],
            declared_mesh_assessment=assessment['status'],
            last_pair_C_absolute_change=measurements.get('curvature_absolute_change'),
            failed_nominal_gates=','.join(k for k,v in a['optical_gates'].items() if not v),
            record_file=entry['record_file'],analysis_file=entry['analysis_file']))
    with table_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    nominal16=[e for e in ledger['entries'] if e['method']=='NN' and e['job_type']=='nominal'
               and e['resolution']==16 and e.get('analysis_status')=='completed']
    passed16=sum(e['nominal_optical_pass'] for e in nominal16)
    completed=sum(e['status']=='completed' for e in ledger['entries'])
    lines=['HEIGHT-SPECIFIC METALENS: BOUNDED VALIDATION DECISION','',summary['status'],'',
        f"{completed} completed solves from {len(ledger['entries'])} attempted solves. All attempts remain charged.",
        'The original user limit was 11 additional physical device cases (cumulative 24), 60 solves, four hours, one worker and no RAM gates.',
        'The original runner was interrupted at its owned H = 100 um, resolution 24 reference after a measured remaining-time forecast; the interrupted attempt is preserved.',
        'The approved two-solve reserve was assigned prospectively to the unchanged H = 40 um profile at resolution 32.',
        'Additional predeclared boundary checks ran only when their measured-cost estimate fit the original deadline.',
        '',f'At resolution 16, {passed16}/{len(nominal16)} NN heights passed nominal optical gates.',
        'Nominal optical PASS, selected mesh convergence and full family acceptance are different assessments.',
        '', 'LATEST COMPLETED NN MEASUREMENTS',
        'H(um) mesh  useful(%)       C          Q       nominal   mesh assessment']
    for row in rows:
        lines.append(f"{row['H_um']:5g} {row['resolution_cells_per_um']:4g}  {100*row['useful_efficiency']:8.4f}  {row['normalized_curvature']:+.7f}  {row['retained_band_Q']:.6f}  {str(row['nominal_optical_pass']):7s}   {row['declared_mesh_assessment']}")
    lines+=['','WHAT THE FAILURE DIAGNOSIS ESTABLISHES',
        'For unchanged H = 40 um geometry, C changed from -0.04685101 at mesh 16 to -0.03124102 at mesh 24: delta 0.01560999 exceeds the frozen 0.005 limit.',
        'Independent saved-field calculations, finer Cartesian resampling, radial-current and quadratic-phase diagnostics reproduced a changing wavefront.',
        'No extraction, normalization, coordinate, configuration or geometry-hash bug was found. No code correction or threshold relaxation is justified by this evidence.',
        'The mesh 32 result is assessed against mesh 24 in summary.json, preserving the earlier failed pairs. Temporal convergence remains separately unresolved where not tested.',
        '', 'SCOPE AND UNMET REQUIREMENTS',
        'These are different frozen top profiles for H = 40/50/60/70/80/90/100 um, on a 50 um diameter solid cylinder, with 0-3 um top relief.',
        'The source is a nominal Gaussian MFD = 10.4 um surrogate at 1550 nm, with assumed phase index 1.45; polymer real index 1.5263 is supplied by the user. Material losses are assumed zero.',
        'The naturally expanded output remains approximately 12-17 um in diameter; the earlier 30 um expansion aspiration has not been demonstrated.',
        'The declared Gaussian footprint is a reference, not a proven global optimum over every 50 um aperture design.',
        'Complete family convergence, full MFD-tolerance robustness, ISO M2, experimental validation and NN superiority are not established.',
        'No new SLC fabrication release or printing was performed.',
        '', 'EVIDENCE',
        'REPORT.txt and summary.json: declared gates, missing checks, histories and budget.',
        'NN_Maxwell_latest.csv and nominal_maxwell_trends.png: actual completed measurements; gaps remain explicit.',
        'diagnosis_h40_curvature_saved_fields_v1.json and source_mode_audit_final.json: saved-field diagnosis and source fidelity.',
        'approval_receipt.json, cumulative_amendment.json, ledger.json, controlled_stop/ and supervision/: scope and execution accounting.']
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],rows=len(rows),decision=str(output))))


if __name__=='__main__':main()
