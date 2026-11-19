"""Summarize saved height-family Maxwell records without launching computation.

--folder is the validation directory containing ledger.json. Every ledger
entry counts as an attempted solve, including failed attempts. Device entries
identify their homogeneous reference using reference_id. Raw records and
analyses are read only; missing, failed or mismatched evidence is unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


OPTICAL_KEYS = ('transmission', 'declared_overlap', 'useful_efficiency', 'diameter',
                'sidelobe', 'curvature', 'angular_second_moment', 'valid_width_crossings')
NATIVE_KEYS = ('native_quadrature', 'paired_native_ratio', 'monitor_alignment',
               'aperture_fraction_disagreement', 'filter_net_loss', 'crosscheck', 'power_closure')
SETTING_KEYS = ('H_um', 'resolution', 'pml_um', 'padding_um', 'duration')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def assessed(checks, missing=()):
    missing = list(missing)
    return {'status': 'UNRESOLVED' if missing else ('PASS' if all(checks.values()) else 'FAIL'),
            'checks': checks, 'failed_checks': [k for k, v in checks.items() if not v],
            'missing_evidence': missing}


def json_safe(value):
    if isinstance(value, dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value, list):return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):return None
    return value


def path_from_entry(folder, name):
    if not isinstance(name, str) or not name:
        raise ValueError('Missing relative file path')
    candidate = (folder / name).resolve()
    if not candidate.is_relative_to(folder.resolve()):
        raise ValueError('Ledger record/analysis path escapes validation directory')
    return candidate


def compare_analyses(prior, fine, config, limits, *, require_collimation=True):
    """Compare every declared processing-plane width and final metric pair."""
    checks = {}; missing = []; measurements = {}
    planes = [float(z) for z in config['evaluation']['processing_planes_um']]
    def plane_map(analysis):
        rows = analysis.get('metrics', {}).get('planes', [])
        return {float(row['z_um']): row for row in rows if finite(row.get('z_um'))}
    old_planes, new_planes = plane_map(prior), plane_map(fine)
    deltas = []
    for z in planes:
        old, new = old_planes.get(z), new_planes.get(z)
        if old is None or new is None:
            missing.append(f'processing-plane widths at z={z:g}'); continue
        valid = old.get('valid_crossings') == [True, True] and new.get('valid_crossings') == [True, True]
        values = [old.get('diameter_x_um'), old.get('diameter_y_um'),
                  new.get('diameter_x_um'), new.get('diameter_y_um')]
        if not valid or not all(finite(v) and v > 0 for v in values):
            missing.append(f'finite valid positive processing-plane widths at z={z:g}'); continue
        change = [abs(values[q+2]/values[q]-1) for q in range(2)]
        deltas.append({'z_um': z, 'xy_relative_changes': change})
    measurements['diameter_changes_all_processing_planes'] = deltas
    if len(deltas) == len(planes):
        largest = max(max(row['xy_relative_changes']) for row in deltas)
        measurements['maximum_diameter_relative_change'] = largest
        checks['all_processing_diameters'] = largest < limits.get('diameter_relative_mesh_change_max', .02)
    else: checks['all_processing_diameters'] = False
    eta0, eta1 = prior.get('conservative_useful_efficiency'), fine.get('conservative_useful_efficiency')
    if finite(eta0) and eta0 > 0 and finite(eta1) and eta1 >= 0:
        change = abs(eta1/eta0-1); measurements['useful_efficiency_relative_change'] = change
        checks['useful_efficiency'] = change < limits.get('eta_relative_mesh_change_max', .05)
    else:
        missing.append('finite useful-efficiency pair'); checks['useful_efficiency'] = False
    def core_angles(a):
        fit = a.get('metrics', {}).get('fitted_core_divergence', {})
        if fit.get('valid') is not True: return None
        axes = fit.get('axes', [])
        if len(axes) != 2: return None
        result = [axis.get('half_angle_rad') for axis in axes]
        return result if all(finite(v) and v > 0 for v in result) else None
    old_angles, new_angles = core_angles(prior), core_angles(fine)
    if old_angles and new_angles:
        changes = [abs(x-y) for x,y in zip(new_angles, old_angles)]
        allowed = [max(.02*abs(v), .001) for v in old_angles]
        checks['fitted_core_divergence'] = all(x <= y for x,y in zip(changes, allowed))
        measurements['core_half_angle_changes_rad'] = changes
        measurements['core_half_angle_allowed_changes_rad'] = allowed
    else:
        missing.append('valid fitted Gaussian-core angle pair'); checks['fitted_core_divergence'] = False
    c0, c1 = prior.get('normalized_curvature'), fine.get('normalized_curvature')
    if finite(c0) and finite(c1):
        cap = limits.get('curvature_absolute_max', .1); delta = abs(c1-c0); margin = cap-abs(c1)
        checks['curvature_last_pair_delta']=delta <= limits.get('curvature_last_pair_delta_max', .005)
        if require_collimation:
            checks.update(curvature_absolute=abs(c1) <= cap,
                          curvature_change_below_remaining_margin=delta < margin)
        measurements.update(curvature_prior=c0, curvature_fine=c1,
                            curvature_absolute_change=delta, curvature_remaining_margin=margin)
    else:
        missing.append('finite phase-current curvature pair')
        checks['curvature_last_pair_delta']=False
        if require_collimation:checks.update(curvature_absolute=False,curvature_change_below_remaining_margin=False)
    d0, d1 = prior.get('declared_target_diameter_um'), fine.get('declared_target_diameter_um')
    checks['same_frozen_declared_target'] = close(d0, d1) and d0 > 0
    if not finite(d0) or not finite(d1): missing.append('declared physical Gaussian target pair')
    q0, q1 = prior.get('angular_second_moment_ratio'), fine.get('angular_second_moment_ratio')
    measurements['retained_band_angular_ratio_absolute_change'] = abs(q1-q0) if finite(q0) and finite(q1) else None
    result = assessed(checks, missing); result['measurements'] = measurements
    result['collimation_absolute_and_remaining_margin_required']=require_collimation
    result['scope'] = 'Declared processing-plane widths, useful efficiency, core fit and phase-current curvature; retained-band angular change is diagnostic'
    return result


def summarize(folder, config, proposal, ledger=None):
    folder=Path(folder).resolve(); ledger=read_json(folder/'ledger.json') if ledger is None else ledger
    entries = ledger if isinstance(ledger, list) else ledger.get('entries', ledger.get('solves', []))
    if not isinstance(entries, list): raise ValueError('Ledger entries/solves must be a list')
    limits=proposal.get('numerical_acceptance', {}); closure_limit=limits.get('power_closure_error_max', .05)
    expected_fingerprint=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    by_id={}; duplicates=[]; source_hashes={}; loaded={}; protected_paths=set()
    for entry in entries:
        identity=str(entry.get('id', ''))
        if not identity or identity in by_id: duplicates.append(identity)
        else: by_id[identity]=entry
    def load_entry(entry):
        identity=str(entry.get('id', ''))
        if identity in loaded: return loaded[identity]
        missing=[]; checks={}; record=None; analysis=None
        if entry.get('status') != 'completed': missing.append('completed solve status')
        try:
            path=path_from_entry(folder, entry.get('record_file')); protected_paths.add(path)
            record=read_json(path); source_hashes[str(path)]=digest(path)
            if entry.get('record_sha256') is not None:
                checks['raw_record_matches_recorded_sha256']=source_hashes[str(path)]==entry['record_sha256']
        except (OSError, ValueError, TypeError) as e: missing.append('readable raw record: '+str(e))
        if record is not None:
            settings=record.get('settings', {})
            checks['ledger_matches_actual_settings']=all(close(entry.get(k),settings.get(k)) for k in SETTING_KEYS)
            checks['actual_config_fingerprint']=record.get('config_fingerprint') == expected_fingerprint
            power=record.get('power_balance_error')
            checks['native_control_surface_power_closure']=finite(power) and 0 <= power < closure_limit
            if not finite(power): missing.append('finite native control-surface closure')
            if entry.get('method') == 'reference':
                checks['positive_native_incident_power']=finite(record.get('incident_power')) and record['incident_power'] > 0
            else:
                if entry.get('analysis_status') not in (None, 'completed'):
                    missing.append('completed independent field analysis')
                try:
                    ap=path_from_entry(folder, entry.get('analysis_file')); protected_paths.add(ap)
                    analysis=read_json(ap); source_hashes[str(ap)]=digest(ap)
                    if entry.get('analysis_sha256') is not None:
                        checks['analysis_matches_recorded_sha256']=source_hashes[str(ap)]==entry['analysis_sha256']
                except (OSError, ValueError, TypeError) as e: missing.append('readable independent analysis: '+str(e))
                if analysis is not None:
                    checks['analysis_matches_actual_settings']=analysis.get('settings') == settings
                    actual_name=str(analysis.get('record', '')).replace('\\','/').rsplit('/',1)[-1]
                    checks['analysis_names_actual_raw_record']=actual_name == Path(entry['record_file']).name
                    native=analysis.get('numerical_checks', {})
                    for key in NATIVE_KEYS:
                        checks['native_'+key]=native.get(key) is True
                        if key not in native: missing.append('native analysis check '+key)
        result={'entry':entry,'record':record,'analysis':analysis,'evidence':assessed(checks,missing)}
        loaded[identity]=result; return result
    def reference_for(device):
        entry=device['entry']; rid=entry.get('reference_id')
        if rid is None or str(rid) not in by_id:
            return {'status':'UNRESOLVED','missing_evidence':['matching reference_id'], 'reference_id':rid}
        ref=load_entry(by_id[str(rid)]); same=all(close(entry.get(k),ref['entry'].get(k)) for k in SETTING_KEYS)
        correct=ref['entry'].get('method')=='reference'
        match_name=False
        if device['record'] is not None and device['record'].get('reference_file'):
            name=str(device['record']['reference_file']).replace('\\','/').rsplit('/',1)[-1]
            match_name=name==Path(ref['entry'].get('record_file','')).name
        checks={'same_height_and_numerical_settings':same,'reference_method':correct,
                'raw_device_names_reference':match_name,'reference_native_evidence':ref['evidence']['status']=='PASS'}
        checks['actual_full_settings_match']=(device['record'] is not None and ref['record'] is not None
                                              and device['record'].get('settings')==ref['record'].get('settings'))
        missing=list(ref['evidence']['missing_evidence'])
        if device['record'] is not None and not device['record'].get('reference_file'):
            missing.append('raw device reference_file provenance')
        result=assessed(checks,missing); result.update(reference_id=str(rid),record_file=ref['entry'].get('record_file'),
                                                      native_power_balance_error=(ref['record'] or {}).get('power_balance_error'))
        return result
    def nominal_candidates(method,H,res):
        return [entry for entry in entries if entry.get('method')==method and close(entry.get('H_um'),H)
                and close(entry.get('resolution'),res)
                and ((not close(res,32.) and entry.get('job_type','nominal') in ('nominal','reserve_duration'))
                     or (method=='NN' and close(res,32.) and entry.get('job_type')=='reserve_mesh'))]
    def pick(candidates):
        completed=[e for e in candidates if e.get('status')=='completed']
        if not completed: return None
        # An explicit same-geometry duration extension may supersede a failed
        # closure attempt; retain every attempt and its native outcome below.
        def rank(e):
            a=load_entry(e);ref=reference_for(a)
            return (a['evidence']['status']=='PASS' and ref['status']=='PASS',float(e.get('duration',0)),str(e.get('id','')))
        return load_entry(max(completed,key=rank))
    def nominal_pair(method,H,coarse,fine):
        prior=pick(nominal_candidates(method,H,coarse)); last=pick(nominal_candidates(method,H,fine))
        missing=[]
        if prior is None: missing.append(f'{method} H{H:g} completed resolution{coarse:g}')
        if last is None: missing.append(f'{method} H{H:g} completed resolution{fine:g}')
        result={'method':method,'H_um':H,'last_pair_resolutions':[coarse,fine]}
        if missing: result.update(assessed({},missing)); return result
        refs=[reference_for(a) for a in (prior,last)]
        optical=last['analysis'].get('optical_gates',{}) if last['analysis'] else {}
        gate_checks={key:optical.get(key) is True for key in OPTICAL_KEYS}
        nominal_optical=assessed(gate_checks,[key for key in OPTICAL_KEYS if key not in optical])
        if prior['analysis'] and last['analysis']:
            comparison=compare_analyses(prior['analysis'],last['analysis'],config,limits,require_collimation=method=='NN')
        else: comparison=assessed({},['two completed independent analyses'])
        checks={'prior_native_evidence':prior['evidence']['status']=='PASS',
                'fine_native_evidence':last['evidence']['status']=='PASS',
                'both_reference_native_closures':all(r['status']=='PASS' for r in refs),
                'last_pair_numerical_convergence':comparison['status']=='PASS',
                'same_PML_and_padding':all(close(prior['entry'].get(k),last['entry'].get(k))
                                            for k in ('pml_um','padding_um')),
                'same_duration':close(prior['entry'].get('duration'),last['entry'].get('duration'))}
        if not checks['same_duration']:
            missing.append('isolated mesh comparison requires identical duration; mixed-duration stability is not established')
        hashes=[a['entry'].get('design_sha256') for a in (prior,last)]
        checks['same_frozen_device_geometry']=all(isinstance(h,str) and len(h)==64 for h in hashes) and hashes[0]==hashes[1]
        if any(not isinstance(h,str) or len(h)!=64 for h in hashes):missing.append('device design_sha256 pair')
        for a in (prior,last): missing.extend(a['evidence']['missing_evidence'])
        for ref in refs: missing.extend(ref.get('missing_evidence',[]))
        missing.extend(comparison['missing_evidence'])
        if method=='NN':
            checks['nominal_optical_gates']=nominal_optical['status']=='PASS'
            missing.extend(nominal_optical['missing_evidence'])
        result.update(assessed(checks,missing))
        result.update(prior_id=prior['entry']['id'],fine_id=last['entry']['id'],
                      prior_record_file=prior['entry']['record_file'],fine_record_file=last['entry']['record_file'],
                      references=refs,nominal_optical=nominal_optical,comparison=comparison,
                      fine_metrics={key:last['analysis'].get(key) if last['analysis'] else None for key in
                           ['conservative_T','conservative_useful_efficiency','declared_overlap','normalized_curvature',
                            'max_sidelobe_fraction','angular_second_moment_ratio']})
        return result
    requested=[float(h) for h in proposal.get('nominal_heights_um', config['natural_training']['train_H_um'])]
    nn=[]
    for H in requested:
        available={res:pick(nominal_candidates('NN',H,res)) for res in (8.,16.,24.,32.)}
        history=[]
        if available[8.] is not None or available[16.] is not None:
            history.append(nominal_pair('NN',H,8.,16.))
        nominal=nominal_pair('NN',H,16.,24.)
        history.append(nominal)
        # The original frozen runner's approved reserve is one additional
        # resolution32 pair. Started/failed32 attempts cannot replace completed
        #24 evidence, and remain visible in all_attempts.
        if available[32.] is not None:
            nominal=nominal_pair('NN',H,24.,32.)
            history.append(nominal)
        row=dict(nominal)
        row['mesh_history']=history
        completed=[res for res,item in available.items() if item is not None]
        finest=max(completed) if completed else None
        row['finest_completed_resolution']=finest
        if finest is not None:
            actual=available[finest]
            optical=(actual['analysis'] or {}).get('optical_gates',{})
            row['nominal_optical']=assessed({key:optical.get(key) is True for key in OPTICAL_KEYS},
                                            [key for key in OPTICAL_KEYS if key not in optical])
            row['nominal_optical_resolution']=finest
            row['nominal_optical_id']=actual['entry']['id']
            row['fine_metrics']={key:(actual['analysis'] or {}).get(key) for key in
                ['conservative_T','conservative_useful_efficiency','declared_overlap','normalized_curvature',
                 'max_sidelobe_fraction','angular_second_moment_ratio']}
        nn.append(row)
    controls=[nominal_pair(case['method'],float(case['H_um']),8.,16.) for case in proposal.get('device_cases',[])
              if case['method'] not in ('NN','reference')]
    boundaries=[]
    boundary_specs=[('pml',40.,'pml_um',3.),('pml',100.,'pml_um',3.),
                    ('padding',70.,'padding_um',15.),('duration',100.,'duration',None)]
    for kind,H,setting,minimum in boundary_specs:
        base=pick(nominal_candidates('NN',H,16.))
        candidates=[e for e in entries if e.get('method')=='NN' and close(e.get('H_um'),H)
                    and close(e.get('resolution'),16.) and e.get('job_type')==kind]
        variation=pick(candidates); row={'job_type':kind,'H_um':H,'resolution':16.}
        if base is None or variation is None:
            row.update(assessed({},['completed nominal/variation pair and matching references']));boundaries.append(row);continue
        refs=[reference_for(a) for a in (base,variation)]
        if base['analysis'] and variation['analysis']:
            comparison=compare_analyses(base['analysis'],variation['analysis'],config,limits)
        else: comparison=assessed({},['two completed independent analyses'])
        minimum=1.5*base['entry']['duration'] if kind=='duration' else minimum
        checks={'declared_setting_increased':finite(variation['entry'].get(setting)) and variation['entry'][setting]>=minimum-1e-9,
                'other_controlled_settings_unchanged':all(close(base['entry'].get(k),variation['entry'].get(k))
                    for k in SETTING_KEYS if k!=setting),
                'nominal_native_evidence':base['evidence']['status']=='PASS',
                'variation_native_evidence':variation['evidence']['status']=='PASS',
                'both_reference_native_closures':all(r['status']=='PASS' for r in refs),
                'same_geometry_numerical_stability':comparison['status']=='PASS'}
        hashes=[a['entry'].get('design_sha256') for a in (base,variation)]
        checks['same_frozen_device_geometry']=all(isinstance(h,str) and len(h)==64 for h in hashes) and hashes[0]==hashes[1]
        missing=[*base['evidence']['missing_evidence'],*variation['evidence']['missing_evidence'],
                 *comparison['missing_evidence'],*(v for r in refs for v in r.get('missing_evidence',[]))]
        if any(not isinstance(h,str) or len(h)!=64 for h in hashes):missing.append('device design_sha256 pair')
        row.update(assessed(checks,missing));row.update(nominal_id=base['entry']['id'],variation_id=variation['entry']['id'],
                    references=refs,comparison=comparison);boundaries.append(row)
    for entry in entries:
        if entry.get('record_file'): load_entry(entry)
    attempt_rows=[]
    for entry in entries:
        data=loaded.get(str(entry.get('id','')))
        attempt_rows.append({'id':entry.get('id'),'method':entry.get('method'),'H_um':entry.get('H_um'),
            'resolution':entry.get('resolution'),'job_type':entry.get('job_type','nominal'),'status':entry.get('status'),
            'record_file':entry.get('record_file'),'analysis_file':entry.get('analysis_file'),
            'native_power_balance_error':(data['record'] or {}).get('power_balance_error') if data else None,
            'evidence_status':data['evidence']['status'] if data else 'UNRESOLVED',
            'evidence':data['evidence'] if data else assessed({},['completed raw solve/analysis'])})
    known_cases={(case['method'],float(case['H_um'])) for case in proposal.get('device_cases',[])}
    actual_cases={(e.get('method'),float(e['H_um'])) for e in entries if e.get('method')!='reference' and finite(e.get('H_um'))}
    total=len(entries); cap=proposal.get('maximum_total_CPU_solves_including_references_and_retries',60)
    elapsed=next((ledger[k] for k in ('elapsed_s','wall_s','total_wall_s')
                  if isinstance(ledger,dict) and finite(ledger.get(k))),None)
    budget_checks={'all_attempts_within_solve_cap':total<=cap,
                   'listed_unique_device_cases_only':actual_cases<=known_cases,
                   'unique_device_ceiling':len(actual_cases)<=proposal.get('new_unique_device_ceiling',11),
                   'ledger_ids_unique':not duplicates}
    budget_missing=[]
    if elapsed is not None: budget_checks['wall_time_within_cap']=elapsed<=proposal.get('maximum_total_wall_time_s',14400)
    else: budget_missing.append('ledger elapsed_s/wall_s/total_wall_s for wall-budget verification')
    budget=assessed(budget_checks,budget_missing);budget.update(attempted_solves=total,maximum_attempts=cap,
                    elapsed_s=elapsed,unique_device_cases=len(actual_cases),duplicate_ids=duplicates,
                    status_counts=dict(Counter(str(e.get('status','unknown')) for e in entries)))
    missing=[]
    for row in nn+boundaries: missing.extend(row.get('missing_evidence',[]))
    checks={'all_requested_NN_heights':len(nn)==len(requested) and bool(nn),
            'all_NN_nominal_and_mesh_checks':all(r['status']=='PASS' for r in nn),
            'sampled_PML_padding_duration_checks':len(boundaries)==4 and all(r['status']=='PASS' for r in boundaries),
            'ledger_ids_unique':not duplicates}
    family=assessed(checks,missing)
    unchanged=all(path.exists() and digest(path)==sha for path,sha in
                  ((Path(name),value) for name,value in source_hashes.items()))
    if not unchanged:
        family['status']='UNRESOLVED';family['missing_evidence'].append('Immutable raw input files changed during report')
    if family['status']=='PASS':
        overall='SYNTHETIC_NN_FAMILY_PASS' if budget['status']=='PASS' else 'SYNTHETIC_NN_CHECKS_PASS_PROTOCOL_UNRESOLVED'
    else:overall='UNRESOLVED_OR_FAILED_SYNTHETIC_NN_ACCEPTANCE'
    if budget['status']=='FAIL':overall='PROTOCOL_BUDGET_FAILURE'
    result={'status':overall,
        'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only assessment of existing independent Maxwell records',
        'NN_family_acceptance':family,'NN_heights':nn,'control_comparisons':controls,
        'control_optical_gates_are_diagnostic':True,'boundary_duration_checks':boundaries,'budget':budget,
        'all_attempts':attempt_rows,'record_analysis_sha256':source_hashes,'raw_inputs_unchanged_during_report':unchanged,
        'config_fingerprint':expected_fingerprint,'physical_experimental_validation':False,
        'full_all_angle_or_ISO_M2_claimed':False,'NN_superiority_claimed':False,
        'limitations':['Gaussian source surrogate and assumed input phase index; supplied polymer real index and zero losses.',
                       'Nominal gates, numerical convergence, reference native closure and control comparisons are separate.',
                       'Core-fit and retained-band quantities do not establish complete all-angle or ISO M2 behavior.',
                       'Nominal MFD validation does not establish full manufacturer-tolerance robustness.',
                       'This module starts no solver, optimization, external job, fabrication export or printer action.']}
    return json_safe(result)


def human_report(result):
    lines=['HEIGHT FAMILY INDEPENDENT MAXWELL ASSESSMENT', '', result['status'], '',
           'NN acceptance requires all requested heights and the sampled PML/padding/duration checks.',
           'Control optical gates are diagnostic; flat cylinder is not required to collimate.', '',
           'H(um)  last mesh pair  assessment  eta_fine  C_fine']
    for row in result['NN_heights']:
        values=row.get('fine_metrics',{});eta=values.get('conservative_useful_efficiency');c=values.get('normalized_curvature')
        lines.append(f"{row['H_um']:5g}  {row['last_pair_resolutions']}  {row['status']}  {eta!s}  {c!s}")
        for item in row.get('missing_evidence',[]):lines.append('  Missing: '+item)
        for item in row.get('failed_checks',[]):lines.append('  Failed: '+item)
        for item in row.get('comparison',{}).get('failed_checks',[]):lines.append('  Pair failed: '+item)
        lines.append(f"  Nominal optical gates at actual finest completed mesh: {row.get('nominal_optical_resolution')!s}")
        for prior in row.get('mesh_history',[]):
            lines.append(f"  Retained mesh history {prior['last_pair_resolutions']}: {prior['status']}")
            for item in prior.get('failed_checks',[]):lines.append('    Failed: '+item)
            for item in prior.get('comparison',{}).get('failed_checks',[]):lines.append('    Pair failed: '+item)
            for item in prior.get('missing_evidence',[]):lines.append('    Missing: '+item)
    lines.extend(['','BOUNDARY AND DURATION CHECKS'])
    for row in result['boundary_duration_checks']:
        lines.append(f"{row['job_type']} H{row['H_um']:g}: {row['status']}")
        for item in row.get('missing_evidence',[]):lines.append('  Missing: '+item)
        for item in row.get('failed_checks',[]):lines.append('  Failed: '+item)
        for item in row.get('comparison',{}).get('failed_checks',[]):lines.append('  Pair failed: '+item)
    lines.extend(['','CONTROLS: NUMERICAL COMPARISONS; OPTICAL GATES DIAGNOSTIC'])
    for row in result['control_comparisons']:
        lines.append(f"{row['method']} H{row['H_um']:g}: numerical {row['status']}; optical {row.get('nominal_optical',{}).get('status','UNRESOLVED')}")
        for item in row.get('failed_checks',[]):lines.append('  Failed: '+item)
        for item in row.get('missing_evidence',[]):lines.append('  Missing: '+item)
    budget=result['budget'];lines.extend(['','BUDGET AND COVERAGE',
        f"Attempts including failures: {budget['attempted_solves']}/{budget['maximum_attempts']}",
        f"Unique listed device cases: {budget['unique_device_cases']}; budget assessment: {budget['status']}",
        f"Elapsed wall seconds: {budget['elapsed_s']!s}",
        'All reference filenames and native closure values are retained in summary.json.',
        'Missing evidence remains unresolved; optical PASS alone is not numerical convergence.',
        'No complete all-angle, ISO M2, experimental, manufacturing or NN-superiority claim is made.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--proposal',type=Path)
    parser.add_argument('--output-json',type=Path);parser.add_argument('--output-text',type=Path)
    parser.add_argument('--overwrite-summary',action='store_true')
    args=parser.parse_args();folder=args.folder.resolve()
    def locate(explicit,name):
        if explicit:return explicit.resolve()
        for p in [folder/name,folder.parent/name]:
            if p.is_file():return p
        raise FileNotFoundError(name)
    config_path=locate(args.config,'config.json');proposal_path=locate(args.proposal,'maxwell_validation_proposal.json')
    result=summarize(folder,read_json(config_path),read_json(proposal_path))
    outputs=[args.output_json or folder/'summary.json',args.output_text or folder/'REPORT.txt']
    protected={config_path.resolve(),proposal_path.resolve(),(folder/'ledger.json').resolve(),
               *(Path(p).resolve() for p in result['record_analysis_sha256'])}
    if outputs[0].resolve()==outputs[1].resolve():raise ValueError('JSON and text outputs must differ')
    for output in outputs:
        resolved=output.resolve()
        if resolved in protected or any(name in resolved.relative_to(folder).parts[:-1] for name in ('records','analyses','raw')):
            raise ValueError('Report output cannot overwrite or enter raw records/analyses')
        if output.exists() and not args.overwrite_summary:raise FileExistsError('Preserve existing report or use --overwrite-summary')
    for output in outputs:output.parent.mkdir(parents=True,exist_ok=True)
    outputs[0].write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    outputs[1].write_text(human_report(result),encoding='utf-8')
    print(json.dumps({'status':result['status'],'JSON':str(outputs[0]),'text':str(outputs[1])},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
