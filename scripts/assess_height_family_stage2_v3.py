"""Assess saved full-family stage1/stage2 evidence without launching optics.

Keep nominal gates, native E/H/reference checks, selected mesh convergence,
boundary/duration checks, requested comparisons and authorization independent.
Use a NEW --output version directory under the prepared stage2 assessment/.
No Meep, JAX, training, geometry edit or receipt/ledger mutation is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_height_family_nn_direct_v2 import collect,read,sha,local_path,finite,close,contained,safe_json,OPTICAL_KEYS,NUMERICS
from report_height_family_maxwell_v2 import compare_analyses

FAMILY=(40.,50.,60.,70.,80.,90.,100.)
COLLECT_HEIGHTS=tuple(sorted(set(FAMILY+(75.,))))
VARIANTS={'pml','padding','duration','reserve_duration'}


def assessed(checks,missing=()):
    missing=list(dict.fromkeys(missing))
    return {'status':'UNRESOLVED' if missing else ('PASS' if all(checks.values()) else 'FAIL'),
        'checks':checks,'failed_checks':[key for key,value in checks.items() if not value],'missing_evidence':missing}


def utc(value):
    if not isinstance(value,str):raise ValueError('Timestamp text required')
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('Timezone-aware UTC timestamp required')
    return parsed.astimezone(timezone.utc)


def same_job(a,b):
    return (a.get('method')==b.get('method') and a.get('job_type','nominal')==b.get('job_type','nominal')
            and all(close(a.get(key),b.get(key)) for key in NUMERICS))


class Authorization:
    """Read prospective phase receipts; never create or infer a selection."""
    def __init__(self,folder,data):
        self.folder=Path(folder).resolve();self.data=data;self.hashes={};self.cache={};self.clock_anchors={}
        self.prior=local_path(data['prior_stage_linkage']['folder']).resolve()
        self.metadata={Path(row['folder']).resolve():row for row in data['stage_metadata']}
        self.newmeta=self.metadata[self.folder]
        self.prep=self.document(self.folder/'preparation_manifest.json')
        self.preparation_checks=[]
        for name,expected in self.prep.items():
            path=contained(self.folder,name);actual=self.protect(path)
            self.preparation_checks.append({'file':name,'expected_sha256':expected,'actual_sha256':actual,'match':actual==expected})
            if actual!=expected:raise ValueError('Stage2 preparation file changed: '+name)
        self.plan=self.document(self.folder/'plan.json')
        self.source_manifest=self.document(self.folder/'source_manifest.json')
        self.frozen_cases=self.document(self.folder/'frozen_cases.json')['cases']
        self.oldplan=self.document(self.prior/'plan.json')
        self.limits=self.document(self.prior/'maxwell_validation_proposal.json').get('numerical_acceptance',{})
        self.oldapproval=self.document(self.prior/'approval_receipt.json')
        self.old_approved=self.oldapproval.get('status')=='APPROVED_BY_USER'
        self.registry=[]
        for key in ('phase_selections','subsequent_phase_runs','adaptive_selections'):
            value=self.newmeta.get(key)
            if isinstance(value,list):self.registry.extend(value)
        value=self.newmeta.get('subsequent_phase_selection')
        if isinstance(value,dict):self.registry.append(value)
    def protect(self,path):
        path=local_path(path).resolve();value=sha(path);self.hashes[str(path)]=value;return value
    def document(self,path):self.protect(path);return read(path)
    def receipt(self,attempt):
        explicit=attempt.get('selection_receipt_file');expected=attempt.get('selection_receipt_sha256')
        registry=[row for row in self.registry if isinstance(row,dict)]
        if explicit:
            candidates=[row for row in registry if row.get('receipt_file')==explicit and row.get('receipt_sha256')==expected]
            if not candidates:return assessed({},['Entry receipt is not linked to the snapshotted ledger selection registry'])
        else:candidates=registry
        reasons=[]
        for linked in reversed(candidates):
            try:
                path=contained(self.folder,linked['receipt_file'])
                if self.protect(path)!=linked['receipt_sha256']:raise ValueError('Selection receipt hash mismatch')
                receipt=self.document(path)
                if receipt.get('status') not in ('PROSPECTIVE_MODE_SELECTION_BEFORE_SUBSEQUENT_DATA',
                                                'PROSPECTIVE_ADAPTIVE_SELECTION_BEFORE_NEW_DATA'):
                    raise ValueError('Unsupported or missing prospective selection status')
                checks={'approval_hash':receipt.get('approval_sha256')==self.protect(self.folder/'approval_receipt.json'),
                    'preparation_hash':receipt.get('preparation_manifest_sha256')==self.protect(self.folder/'preparation_manifest.json'),
                    'plan_hash':receipt.get('frozen_plan_sha256')==self.protect(self.folder/'plan.json'),
                    'unchanged_deadline':receipt.get('unchanged_deadline') is True,
                    'unchanged_thresholds':receipt.get('unchanged_thresholds') is True,
                    'physics_unchanged':receipt.get('physics_source_changes') is False}
                checks['frozen_runner_hash']=receipt.get('frozen_stage_runner_sha256')==self.source_manifest.get('scripts/run_height_family_continuation_v3.py')
                jobs=receipt.get('selected_jobs',[])
                checks['exact_selected_job']=any(same_job(attempt,job) for job in jobs)
                mode=receipt.get('selected_mode');origin=receipt.get('decision_origin')
                checks['decision_origin']=(origin=='EXPLICIT_USER_REPLY'
                    or mode in ('repair_only','adaptive','common_requested_H75') and origin=='EXPLICIT_ROOT_SELECTION_REASON'
                    or mode=='original_controls_from_request' and origin=='EXPLICIT_USER_REQUEST')
                decision=receipt.get('decision_text_exact')
                import hashlib
                checks['decision_text_hash']=isinstance(decision,str) and bool(decision.strip()) and hashlib.sha256(decision.encode()).hexdigest()==receipt.get('decision_text_sha256')
                drivers=[candidate for candidate in path.parent.glob('*.py') if self.protect(candidate)==receipt.get('driver_sha256')]
                checks['driver_snapshot']=len(drivers)==1 and receipt.get('driver_sha256')==linked.get('driver_sha256')
                prior_snapshot=path.parent/'prior_ledger.json'
                checks['prospective_ledger_snapshot']=prior_snapshot.is_file() and self.protect(prior_snapshot)==receipt.get('prior_ledger_snapshot_sha256')
                if checks['prospective_ledger_snapshot']:
                    earlier=self.document(prior_snapshot)
                    checks['attempt_absent_before_selection']=not any(row.get('id')==attempt.get('id') for row in earlier.get('entries',[]))
                    try:
                        utc(earlier['first_started_utc']);utc(earlier['absolute_deadline_utc'])
                        checks['original_execution_clock_in_snapshot']=True
                    except (KeyError,TypeError,ValueError):checks['original_execution_clock_in_snapshot']=False
                missing=[]
                try:checks['receipt_precedes_attempt']=utc(receipt['created_utc'])<=utc(attempt['started_utc'])
                except (KeyError,TypeError,ValueError):checks['receipt_precedes_attempt']=False;missing.append('Receipt/attempt prospective timestamps')
                # The common repair selection authorizes H70 r32 only. Final
                # adaptive attempts are never inferred from its reserved slots.
                if mode=='repair_only':
                    checks['repair_only_scope']=attempt.get('method') in ('NN','reference') and close(attempt.get('H_um'),70.) and close(attempt.get('resolution'),32.) and attempt.get('job_type')=='refinement'
                if mode=='common_requested_H75':
                    expected_numerics={'H_um':75.,'resolution':24.,'pml_um':2.,'padding_um':10.,'duration':259.}
                    def common75_job(job):
                        return (job.get('method') in ('NN','reference') and job.get('job_type')=='nominal'
                            and job.get('phase')=='COMMON_REQUESTED_H75' and all(close(job.get(k),v) for k,v in expected_numerics.items()))
                    checks['common75_exact_execution_scope']=receipt.get('execution_scope')=='COMMON_REQUESTED_FROZEN_NN75_FORWARD_ONLY_PRIORITY_AND_DIRECT_MEANING_PENDING'
                    checks['common75_exact_reference_NN_pair']=len(jobs)==2 and [job.get('method') for job in jobs]==['reference','NN'] and all(common75_job(job) for job in jobs)
                    checks['common75_attempt_scope']=common75_job(attempt)
                    prepared=self.plan.get('prospective_phase_B',[])
                    checks['common75_matches_unchanged_prepared_pair']=len(jobs)==2 and all(any(same_job(job,original) for original in prepared) for job in jobs)
                    checks['common75_entry_receipt_link']=bool(explicit) and expected==linked['receipt_sha256']
                    checks['common75_entry_mode_scope']=attempt.get('selection_mode')==mode and attempt.get('selection_scope')==receipt.get('execution_scope')
                    flags={'priority_selected':False,'direct_calculation_meaning_selected':False,
                        'separately_optimized_direct_launched':False,'original_control_deferral_selected':False,
                        'no_original_controls_deferred_by_this_selection':True,'adaptive_launch_authorized_by_this_receipt':False}
                    checks['common75_no_branch_or_direct_selection']=all(receipt.get(key) is value for key,value in flags.items())
                    effect={'all_common_projected_attempts':14,'comparison_remaining':4,'comparison_projected_total':18,
                        'comparison_adaptive_remaining':2,'original_controls_remaining':6,'original_projected_total':20,
                        'original_adaptive_remaining':0}
                    recorded=receipt.get('common75_budget_effect',{})
                    checks['common75_both_budget_options_recorded']=all(type(recorded.get(key)) is int and recorded[key]==value for key,value in effect.items())
                    if attempt.get('method')=='NN':
                        selected=[case for case in self.frozen_cases if case.get('method')=='NN' and close(case.get('H_um'),75.)]
                        checks['common75_frozen_NN_geometry']=len(selected)==1 and attempt.get('design_sha256')==selected[0].get('design_sha256')
                if mode in ('original_controls_after_common','original_controls_from_request'):
                    canonical=[]
                    for kind,H,pml,padding,duration in [('pml',100.,3.,10.,335.),
                            ('padding',70.,2.,15.,243.),('duration',100.,2.,10.,502.5)]:
                        for method in ('reference','NN'):
                            canonical.append(dict(method=method,H_um=H,resolution=16.,pml_um=pml,
                                padding_um=padding,duration=duration,job_type=kind))
                    from_request=mode=='original_controls_from_request'
                    scope=('ORIGINAL_EXPLICIT_USER_REQUEST_REMAINING_SIX_CONTROLS_SAME_NN_FORWARD_NO_ADAPTIVE_RESERVE'
                        if from_request else 'USER_SELECTED_ORIGINAL_CONTROLS_AFTER_COMMON_NO_ADAPTIVE_RESERVE')
                    checks['original_controls_exact_decision_origin']=origin==('EXPLICIT_USER_REQUEST' if from_request else 'EXPLICIT_USER_REPLY')
                    checks['original_controls_exact_scope']=receipt.get('execution_scope')==scope
                    checks['original_controls_exact_six_jobs']=len(jobs)==6 and all(same_job(job,expected_job)
                        and job.get('phase')=='ORIGINAL_CONTROLS_AFTER_COMMON' for job,expected_job in zip(jobs,canonical))
                    checks['original_controls_match_sealed_original_plan']=len(jobs)==6 and all(
                        any(same_job(job,original) for original in self.oldplan.get('jobs',[])) for job in jobs)
                    checks['original_controls_attempt_phase']=attempt.get('phase')=='ORIGINAL_CONTROLS_AFTER_COMMON'
                    checks['original_controls_entry_link']=bool(explicit) and expected==linked['receipt_sha256']
                    checks['original_controls_entry_mode_scope']=attempt.get('selection_mode')==mode and attempt.get('selection_scope')==scope
                    flags={'adaptive_launch_authorized_by_this_receipt':False,'separate20cap_selection_after_common':True,
                        'frozen18cap_driver_unchanged':True,'separately_optimized_direct_launched':False}
                    flags.update({'frozen_v1_actual_reply_driver_unchanged':True} if from_request
                        else {'direct_calculation_interpretation_claimed':False})
                    checks['original_controls_no_adaptive_or_direct_claim']=all(receipt.get(key) is value for key,value in flags.items())
                    effect={'common_completed_attempts':14,'original_control_calls':6,'projected_new_attempts':20,
                        'projected_cumulative_attempts':65,'adaptive_remaining':0}
                    recorded=receipt.get('original_controls_budget_effect',{})
                    checks['original_controls_exact20_without_reserve']=all(type(recorded.get(key)) is int and recorded[key]==value
                        for key,value in effect.items()) and type(receipt.get('adaptive_reserve')) is int and receipt['adaptive_reserve']==0
                    checks['original_controls_frozen_cases_hash']=receipt.get('frozen_cases_sha256')==self.protect(self.folder/'frozen_cases.json')
                    if from_request:
                        approval=self.document(self.folder/'approval_receipt.json')
                        immutable_request=self.folder/'user_authorization_exact.txt'
                        request_copy=contained(path.parent,receipt.get('original_user_request_file'))
                        interpretation_copy=contained(path.parent,receipt.get('root_scope_interpretation_file'))
                        original_hash=self.protect(immutable_request);request_hash=self.protect(request_copy)
                        checks['original_request_exact_immutable_authority']=(approval.get('status')=='APPROVED_BY_USER'
                            and decision==approval.get('user_message_exact')
                            and receipt.get('decision_text_sha256')==receipt.get('original_user_request_sha256')==approval.get('user_message_sha256')==original_hash==request_hash
                            and immutable_request.read_bytes()==decision.encode('utf-8')==request_copy.read_bytes())
                        checks['original_request_fixed_sidecar_names']=(receipt.get('original_user_request_file')=='original_user_request.txt'
                            and receipt.get('root_scope_interpretation_file')=='root_scope_interpretation.txt')
                        interpretation=receipt.get('root_scope_interpretation')
                        checks['root_interpretation_separately_attributed']=(isinstance(interpretation,str) and bool(interpretation.strip())
                            and receipt.get('root_scope_interpretation_origin')=='ROOT_SCOPE_INTERPRETATION_NOT_USER_QUOTE'
                            and receipt.get('root_scope_interpretation_is_user_quote') is False
                            and self.protect(interpretation_copy)==receipt.get('root_scope_interpretation_sha256')
                            and interpretation_copy.read_bytes()==interpretation.encode('utf-8'))
                        request_flags={'new_priority_reply_received':False,'clarification_pending':True,
                            'elapsed_used_as_authorization':False,'user_direct_calculation_meaning_claimed':False}
                        checks['original_request_no_invented_reply_or_meaning']=all(receipt.get(key) is value for key,value in request_flags.items())
                        checks['original_request_same_NN_forward_scope']=(receipt.get('direct_calculation_assumption')=='INDEPENDENT_MEEP_FORWARD_OF_SAME_FROZEN_NN_GEOMETRY'
                            and receipt.get('separate_non_neural_comparison_scope')=='PROXY_ONLY_UNLESS_USER_STEERS')
                    required_common=[*self.plan.get('phase_A',[]),*self.plan.get('prospective_phase_C',[]),
                        *(job for job in self.plan.get('prospective_phase_B',[]) if job.get('method') in ('reference','NN') and close(job.get('H_um'),75.))]
                    common_verified=len(required_common)==14 and checks['prospective_ledger_snapshot']
                    if common_verified:
                        for job in required_common:
                            matching=[row for row in earlier.get('entries',[]) if same_job(row,job)]
                            if len(matching)!=1 or matching[0].get('status')!='completed':common_verified=False;break
                            completed=matching[0]
                            if self.protect(contained(self.folder,completed['record_file']))!=completed.get('record_sha256'):
                                common_verified=False;break
                            if completed['method']=='NN' and (completed.get('analysis_status')!='completed'
                                    or self.protect(contained(self.folder,completed['analysis_file']))!=completed.get('analysis_sha256')):
                                common_verified=False;break
                    checks['original_controls_all14_common_complete_before_selection']=common_verified
                    if attempt.get('method')=='NN':
                        selected=[case for case in self.frozen_cases if case.get('method')=='NN' and close(case.get('H_um'),attempt.get('H_um'))]
                        checks['original_controls_unchanged_NN_geometry']=len(selected)==1 and attempt.get('design_sha256')==selected[0].get('design_sha256')
                if close(attempt.get('resolution'),40.) or mode=='adaptive':
                    checks['explicit_adaptive_launch']=receipt.get('adaptive_launch_authorized_by_this_receipt') is True
                result=assessed(checks,missing)
                if result['status']=='PASS':
                    self.clock_anchors[str(prior_snapshot)]={'first_started_utc':earlier['first_started_utc'],
                        'absolute_deadline_utc':earlier['absolute_deadline_utc'],'elapsed_s':earlier.get('elapsed_s'),
                        'snapshot_sha256':receipt['prior_ledger_snapshot_sha256']}
                    return {**result,'receipt_file':str(path),'receipt_sha256':linked['receipt_sha256'],'mode':mode,
                        'decision_origin':origin,'execution_scope':receipt.get('execution_scope'),
                        **({'root_scope_interpretation':receipt['root_scope_interpretation'],
                            'root_scope_interpretation_is_user_quote':False,'new_priority_reply_received':False,
                            'clarification_pending':True,'elapsed_used_as_authorization':False}
                            if mode=='original_controls_from_request' else {})}
                reasons.extend(result['failed_checks']+result['missing_evidence'])
            except (OSError,KeyError,TypeError,ValueError) as error:reasons.append(str(error))
        return assessed({},reasons or ['No valid prospective selection receipt covers this job'])
    def attempt(self,attempt):
        identity=attempt['identity']
        if identity in self.cache:return self.cache[identity]
        stage=Path(attempt['stage_folder']).resolve();kind=attempt.get('job_type','nominal')
        if stage==self.prior:
            if kind=='reserve_mesh':
                checks={'old_approval':self.old_approved,'historical_H40_only':close(attempt.get('H_um'),40.),
                        'historical_resolution32':close(attempt.get('resolution'),32.)}
                result={**assessed(checks),'basis':'Historical stage1 reserve; not the new refinement allocation'}
            else:
                result={**assessed({'old_approval':self.old_approved,'original_plan_job':any(same_job(attempt,job) for job in self.oldplan.get('jobs',[]))}),
                        'basis':'Authenticated original stage1 plan and approval'}
        elif stage==self.folder:
            if attempt.get('phase')=='A' and any(same_job(attempt,job) for job in self.plan.get('phase_A',[])):
                result={**assessed({'prepared_phase_A_only':self.plan.get('status')=='ONLY_PHASE_A_EXECUTABLE'}),
                        'basis':'Original prepared phase-A approval'}
            else:result=self.receipt(attempt)
        else:result=assessed({},['Attempt origin is outside the authenticated two stages'])
        self.cache[identity]=result;return result


def assess_evidence(folder,data,*,now=None):
    folder=local_path(folder).resolve();now=datetime.now(timezone.utc) if now is None else now
    auth=Authorization(folder,data);attempts={row['identity']:row for row in data['all_attempts']}
    records=data['Meep_records'];config=data['config'];native_by_record={}
    def record_authorization(row):
        attempt=attempts.get(row['identity'])
        if attempt is None:return assessed({},['Saved record has no snapshotted ledger entry'])
        result=auth.attempt(attempt)
        matches=[]
        for item in data['all_attempts']:
            if item.get('method')!='reference' or not item.get('record_file'):continue
            try:path=contained(Path(item['stage_folder']),item['record_file'])
            except (ValueError,TypeError):continue
            if str(path)==row.get('reference_record_file'):matches.append(item)
        ref=auth.attempt(matches[0]) if len(matches)==1 else assessed({},['One authenticated matching reference attempt required'])
        missing=[*result['missing_evidence'],*ref['missing_evidence']]
        return {**assessed({'device_authorization':result['status']=='PASS','reference_authorization':ref['status']=='PASS'},missing),
                'device':result,'reference':ref}
    def analysis(row):
        path=Path(row['analysis_file'])
        if sha(path)!=row['analysis_sha256']:raise ValueError('Saved analysis changed after collection')
        return read(path)
    def nominal_assessment(row):
        saved=analysis(row);gates=config['natural_training']['investigation_gates']
        metrics={'transmission':saved.get('conservative_T'),'declared_overlap':saved.get('declared_overlap'),
            'useful_efficiency':row.get('eta'),'sidelobe':saved.get('max_sidelobe_fraction'),
            'curvature':row.get('C'),'angular_second_moment':row.get('Q')}
        missing=[f'Finite nominal observable {key}' for key,value in metrics.items() if not finite(value)]
        for key in ('sidelobe','angular_second_moment'):
            if finite(metrics[key]) and metrics[key]<0:missing.append('Nonnegative nominal observable '+key)
        target=saved.get('metrics',{}).get('target_widths_um',[])
        target0=target[0] if target else []
        if len(target0)!=2 or not all(finite(v) and v>0 for v in target0):missing.append('Frozen vector target output widths')
        first=row['planes'][0] if row['planes'] else {}
        widths=[first.get('diameter_x_um'),first.get('diameter_y_um')]
        crossings=all(finite(p.get(key)) and p[key]>0 for p in row['planes'] for key in ('diameter_x_um','diameter_y_um')) and bool(row['planes'])
        if not crossings:missing.append('Finite valid processing-plane 1/e2 widths')
        if not all(finite(value) and value>0 for value in row['core_half_angles_deg']):missing.append('Valid fitted Gaussian-core half-angles')
        computed={'transmission':finite(metrics['transmission']) and metrics['transmission']>=gates['transmission_min'],
            'declared_overlap':finite(metrics['declared_overlap']) and metrics['declared_overlap']>=gates['declared_Gaussian_overlap_min'],
            'useful_efficiency':finite(metrics['useful_efficiency']) and metrics['useful_efficiency']>=gates['useful_efficiency_min'],
            'sidelobe':finite(metrics['sidelobe']) and 0<=metrics['sidelobe']<=gates['max_sidelobe_fraction'],
            'curvature':finite(metrics['curvature']) and abs(metrics['curvature'])<=gates['max_abs_normalized_curvature'],
            'angular_second_moment':finite(metrics['angular_second_moment']) and 0<=metrics['angular_second_moment']<=gates['max_angular_second_moment_ratio'],
            'valid_width_crossings':crossings,'diameter':False}
        if len(target0)==2 and all(finite(v) and v>0 for v in target0+widths):
            computed['diameter']=max(abs(width/target_width-1) for width,target_width in zip(widths,target0))<=gates['output_diameter_relative_tolerance']
        stored=row['optical_gates'];missing.extend('Stored nominal gate '+key for key in OPTICAL_KEYS if stored.get(key) is None)
        checks={'stored_'+key:stored.get(key) is True for key in OPTICAL_KEYS}
        checks.update({'recomputed_'+key:bool(value) for key,value in computed.items()})
        checks['stored_flags_consistent_with_observables']=not missing and all(stored.get(key)==value for key,value in computed.items())
        return {**assessed(checks,missing),'saved_flags':stored,'recomputed_flags':computed,'saved_scalar_observables':metrics}
    for row in records:native_by_record[row['identity']]=record_authorization(row)
    nominal=[row for row in records if row['job_type'] not in VARIANTS]
    def pair_check(prior,fine,require_collimation=True):
        isolation={key:close(prior['settings'].get(key),fine['settings'].get(key)) for key in ('H_um','pml_um','padding_um','duration')}
        comparison=compare_analyses(analysis(prior),analysis(fine),config,auth.limits,require_collimation=require_collimation)
        authorizations=[native_by_record[row['identity']] for row in (prior,fine)]
        missing=[*comparison['missing_evidence'],*(reason for item in authorizations for reason in item['missing_evidence']),
                 *(reason for row in (prior,fine) for reason in row.get('native_missing_evidence',[]))]
        checks={'identical_frozen_geometry':prior['design_sha256']==fine['design_sha256'],
            'same_H_PML_padding_duration':all(isolation.values()),'native_and_reference_both':all(row['native_all_pass'] for row in (prior,fine)),
            'both_authorizations':all(item['status']=='PASS' for item in authorizations),'declared_convergence':comparison['status']=='PASS'}
        return {**assessed(checks,missing),'resolutions':[prior['resolution'],fine['resolution']],
            'prior_identity':prior['identity'],'fine_identity':fine['identity'],'comparison':comparison,
            'controlled_settings_checks':isolation,'authorizations':authorizations,
            'curvature_delta':abs(fine['C']-prior['C']) if finite(prior['C']) and finite(fine['C']) else None}
    heights=[];all_histories=[]
    for H in FAMILY:
        candidates=[row for row in nominal if row['method']=='NN' and close(row['H_um'],H)]
        if not candidates:
            heights.append({'H_um':H,**assessed({},['Completed verified nominal NN evidence required'])});continue
        latest=max(candidates,key=lambda row:(row['resolution'],data['Meep_records'].index(row)))
        nominal_gate=nominal_assessment(latest)
        grouped=defaultdict(dict)
        for row in candidates:
            key=(row['design_sha256'],*(row['settings'].get(key) for key in ('pml_um','padding_um','duration')))
            grouped[key][row['resolution']]=row
        histories=[]
        for group in grouped.values():
            ordered=sorted(group)
            for i,coarse in enumerate(ordered):
                for j in range(i+1,len(ordered)):
                    item=pair_check(group[coarse],group[ordered[j]])
                    item['adjacent_in_current_sequence']=j==i+1
                    item['scope']='Recomputed comparison of unchanged saved records; not an additional solve'
                    histories.append(item)
        # Do not select an older passing branch if the finest completed field
        # has failed native checks or lacks a same-settings coarser record.
        selected=[row for row in histories if row['fine_identity']==latest['identity'] and row['adjacent_in_current_sequence']]
        selected=selected[-1] if selected else assessed({},['A coarser completed mesh with identical geometry/PML/padding/duration is required'])
        if latest['resolution']<24.:
            selected={**selected,'status':'UNRESOLVED','missing_evidence':selected.get('missing_evidence',[])+['Declared NN fine mesh>=24 is missing']}
        native=assessed(latest['native_checks'],latest.get('native_missing_evidence',[]))
        authorization=native_by_record[latest['identity']]
        verified={row['identity'] for row in candidates}
        unfinished=[]
        for attempt in data['all_attempts']:
            if attempt.get('method')!='NN' or not close(attempt.get('H_um'),H) or attempt.get('job_type') in VARIANTS:continue
            if not finite(attempt.get('resolution')) or attempt['resolution']<latest['resolution']:continue
            if not all(close(attempt.get(k),latest['settings'].get(k)) for k in ('pml_um','padding_um','duration')):continue
            if attempt['identity'] not in verified:unfinished.append(attempt)
        checks={'nominal_optical':nominal_gate['status']=='PASS','native_reference':native['status']=='PASS',
                'selected_numerical_pair':selected['status']=='PASS','authorization':authorization['status']=='PASS',
                'no_unfinished_or_unverified_finer_attempt':not unfinished}
        missing=[*nominal_gate['missing_evidence'],*selected.get('missing_evidence',[]),*authorization['missing_evidence'],*native['missing_evidence']]
        missing.extend('Finest/finer attempt unresolved: '+attempt['identity'] for attempt in unfinished)
        row={'H_um':H,**assessed(checks,missing),'latest_identity':latest['identity'],'latest_resolution':latest['resolution'],
             'latest_record':{key:latest.get(key) for key in ('record_file','record_sha256','analysis_file','analysis_sha256','eta','C','Q','core_half_angles_deg','planes')},
             'nominal_optical':nominal_gate,'native_reference':native,'authorization':authorization,
             'selected_numerics':selected,'all_prior_pairs':histories,'unfinished_finer_attempts':unfinished,
             'nominal_measurement_history':[{key:r.get(key) for key in ('identity','resolution','job_type','eta','C','Q',
                 'native_all_pass','optical_gates','record_file','record_sha256','analysis_file','analysis_sha256')}
                 for r in sorted(candidates,key=lambda r:r['resolution'])]}
        heights.append(row);all_histories.extend({'H_um':H,**item} for item in histories)
    boundaries=[]
    for kind,H,changed,minimum in [('pml',40.,'pml_um',3.),('pml',100.,'pml_um',3.),
                                   ('padding',70.,'padding_um',15.),('duration',100.,'duration',None)]:
        variations=[r for r in records if r['method']=='NN' and close(r['H_um'],H) and r['job_type'] in (kind,'reserve_duration' if kind=='duration' else kind)]
        output={'job_type':kind,'H_um':H,
            'variation_scope':('Combined radial-domain and output/control-aperture perturbation; '
                'the frozen worker increases monitor radius with padding. This is not a fixed-aperture boundary-only comparison.'
                if kind=='padding' else 'Declared '+kind+' perturbation of the frozen numerical setup')}
        if kind=='padding':
            output.update(fixed_aperture_boundary_only_established=False,
                fixed_aperture_followup='When both saved fields exist, quantify a common35um radius using saved E/H separately; the padding15 run has no native35um flux monitor. Retain original full-aperture native analyses and gates.')
        if not variations:
            boundaries.append({**output,**assessed({},['Completed verified variation and matching native reference required'])});continue
        variation=max(variations,key=lambda r:(r['resolution'],records.index(r)))
        bases=[r for r in nominal if r['method']=='NN' and close(r['H_um'],H) and close(r['resolution'],variation['resolution'])]
        if not bases:
            boundaries.append({**output,**assessed({},['Nominal evidence at the variation mesh required'])});continue
        controls=[key for key in NUMERICS if key!=changed]
        base=max(bases,key=lambda r:(sum(close(r['settings'].get(k),variation['settings'].get(k)) for k in controls),records.index(r)))
        comparison=compare_analyses(analysis(base),analysis(variation),config,auth.limits,require_collimation=True)
        required=1.5*base['settings']['duration'] if kind=='duration' else minimum
        authorizations=[native_by_record[r['identity']] for r in (base,variation)]
        checks={'only_intended_numerical_setting_changed':all(close(base['settings'].get(k),variation['settings'].get(k)) for k in controls),
            'declared_variation_reached':finite(variation['settings'].get(changed)) and variation['settings'][changed]>=required-1e-9,
            'same_frozen_geometry':base['design_sha256']==variation['design_sha256'],
            'native_and_references':base['native_all_pass'] and variation['native_all_pass'],
            'both_authorizations':all(r['status']=='PASS' for r in authorizations),'numerical_stability':comparison['status']=='PASS'}
        missing=[*comparison['missing_evidence'],*(m for r in authorizations for m in r['missing_evidence']),
                 *(m for r in (base,variation) for m in r.get('native_missing_evidence',[]))]
        boundaries.append({**output,**assessed(checks,missing),'resolution':variation['resolution'],
            'nominal_identity':base['identity'],'variation_identity':variation['identity'],
            'saved_monitor_radii_um':{'nominal':base['settings'].get('monitor_radius_um'),
                'variation':variation['settings'].get('monitor_radius_um')},
            'comparison':comparison,'authorizations':authorizations})
    controls=[]
    for method in ('baseline','direct'):
        for H in COLLECT_HEIGHTS:
            rows=[r for r in nominal if r['method']==method and close(r['H_um'],H)]
            if not rows:continue
            grouped=defaultdict(dict)
            for row in rows:grouped[tuple(row['settings'].get(k) for k in ('pml_um','padding_um','duration'))][row['resolution']]=row
            history=[]
            for group in grouped.values():
                ordered=sorted(group)
                for coarse,fine in zip(ordered,ordered[1:]):history.append(pair_check(group[coarse],group[fine],require_collimation=False))
            controls.append({'method':method,'H_um':H,'all_numerical_pairs':history,
                'scope':'Diagnostic field-derived baseline' if method=='baseline' else 'Matched non-neural direct-table optimizer; not Meep itself'})
    return finish(folder,data,auth,now,heights,boundaries,controls,all_histories,native_by_record)


def finish(folder,data,auth,now,heights,boundaries,controls,histories,native_by_record):
    newmeta=auth.newmeta;limits=data['authorization_receipt']['limits']
    new=[a for a in data['all_attempts'] if Path(a['stage_folder']).resolve()==folder]
    old=[a for a in data['all_attempts'] if Path(a['stage_folder']).resolve()==auth.prior]
    new_authorizations=[{'identity':item['identity'],**auth.attempt(item)} for item in new]
    budget_checks={'old45_preserved':len(old)==45,'approved_prior45':limits.get('prior_attempts')==45,
        'old_IDs_1_through45':sorted(row['id'] for row in old)==list(range(1,46)),
        'new_IDs_continue46': [row['id'] for row in new]==list(range(46,46+len(new))),
        'new20_cap':len(new)<=20 and limits.get('additional_CPU_solves')==20,
        'cumulative65_cap':len(old)+len(new)<=65 and limits.get('cumulative_CPU_solves')==65,
        '48h_amendment':limits.get('additional_wall_s')==172800}
    missing=[];elapsed=None;deadline=None;clock_elapsed=None;earliest=None
    lower_bounds=[value for value in [newmeta.get('elapsed_s'),*[a.get('elapsed_s') for a in auth.clock_anchors.values()]] if finite(value) and value>=0]
    try:
        start=utc(newmeta['first_started_utc']);deadline=utc(newmeta['absolute_deadline_utc'])
        budget_checks['fixed_absolute_deadline']=deadline==start+timedelta(seconds=172800)
        clock_elapsed=(now-start).total_seconds();elapsed=max([clock_elapsed,*lower_bounds])
        budget_checks['wall_clock_including_idle_gaps']=clock_elapsed>=0 and elapsed<=172800 and now<=deadline
        times=[]
        for attempt in new:
            try:times.append(utc(attempt['started_utc']))
            except (KeyError,TypeError,ValueError):missing.append('New attempt start timestamp: '+attempt['identity'])
        if times:
            earliest=min(times);budget_checks['first_start_not_after_first_attempt']=start<=earliest<=now
        if auth.clock_anchors:
            budget_checks['first_start_matches_immutable_snapshots']=all(utc(anchor['first_started_utc'])==start for anchor in auth.clock_anchors.values())
            budget_checks['deadline_matches_immutable_snapshots']=all(utc(anchor['absolute_deadline_utc'])==deadline for anchor in auth.clock_anchors.values())
        if not times and not auth.clock_anchors:missing.append('No started-attempt or immutable snapshot execution-clock anchor')
    except (KeyError,TypeError,ValueError):missing.append('Original stage2 absolute first-start/deadline required')
    frozen=auth.document(folder/'frozen_cases.json');cases=frozen['cases']
    reserved=[case for case in cases if case.get('new_case') is True]
    charged=set();unknown=[]
    for item in new:
        if item.get('method')=='reference':continue
        if not any(item.get('method')==case.get('method') and close(item.get('H_um'),case.get('H_um'))
                   and item.get('design_sha256')==case.get('design_sha256') for case in cases):unknown.append(item['identity'])
        for case in reserved:
            if item.get('case_id')==case.get('case_id') or item.get('method')==case.get('method') and close(item.get('H_um'),case.get('H_um')):
                charged.add(case.get('case_id',f"{case['method']}_H{case['H_um']}"))
    budget_checks.update(reserved_new4=len(reserved)==4 and limits.get('additional_unique')==4,
        unique_ceiling28=limits.get('cumulative_unique')==28 and frozen.get('cumulative_unique_ceiling')==28 and 24+len(charged)<=28,
        only_listed_device_cases=not unknown,
        no_duplicate_stage_ids=all(not stage['duplicates'] for stage in data['stage_metadata']))
    budget=assessed(budget_checks,missing)
    if not all(budget_checks.values()):budget['status']='FAIL'  # Known resource violations take precedence over missing diagnostics.
    budget.update(prior_attempts=len(old),additional_attempts=len(new),
        cumulative_attempts=len(old)+len(new),absolute_elapsed_s_including_idle=elapsed,
        elapsed_from_current_start_s=clock_elapsed,recorded_elapsed_lower_bound_s=max(lower_bounds) if lower_bounds else None,
        earliest_attempt_started_utc=earliest.isoformat() if earliest else None,immutable_clock_anchors=auth.clock_anchors,
        absolute_deadline_utc=deadline.isoformat() if deadline else None,
        reserved_additional_unique=4,charged_new_unique=len(charged),charged_new_case_ids=sorted(charged),
        actual_cumulative_unique=None if unknown else 24+len(charged),known_charged_cumulative_unique=24+len(charged),
        reserved_cumulative_unique=28,unregistered_device_attempts=unknown)
    # Inspect authorization of every new attempt, including failed references.
    auth_missing=[reason for row in new_authorizations for reason in row['missing_evidence']]
    protocol=assessed({'all_preparation_hashes':all(row['match'] for row in auth.preparation_checks),
        'all_new_attempts_prospectively_authorized':all(row['status']=='PASS' for row in new_authorizations)},auth_missing)
    family_missing=[reason for row in heights+boundaries for reason in row.get('missing_evidence',[])]
    family=assessed({'all_seven_nominal_native_mesh_rows':len(heights)==7 and all(row['status']=='PASS' for row in heights),
        'all_four_boundary_duration_checks':len(boundaries)==4 and all(row['status']=='PASS' for row in boundaries),
        'budget':budget['status']=='PASS','protocol':protocol['status']=='PASS'},
        [*family_missing,*budget['missing_evidence'],*protocol['missing_evidence']])
    records={row['identity']:row for row in data['Meep_records']}
    comparisons=[row for row in data['matched_Meep_pairs'] if row['H_um'] in (40.,75.,100.)
        and all(records[row[method+'_identity']]['job_type'] not in VARIANTS for method in ('NN','direct'))]
    required=[row for row in comparisons if close(row['resolution'],24.)]
    comparison_missing=[f'Matched r24 NN/direct at H{H:g}' for H in (40.,75.,100.)
        if not any(close(row['H_um'],H) for row in required)]
    required_records=[records[row[method+'_identity']] for row in required for method in ('NN','direct')]
    coverage=True
    for row in required_records:
        comparison_missing.extend(row.get('native_missing_evidence',[]))
        comparison_missing.extend(native_by_record[row['identity']]['missing_evidence'])
        valid=(all(finite(row.get(k)) for k in ('eta','C','Q')) and all(finite(v) for v in row['core_half_angles_deg'])
               and all(finite(p.get(k)) and p[k]>0 for p in row['planes'] for k in ('diameter_x_um','diameter_y_um')))
        coverage=coverage and valid
        if not valid:comparison_missing.append('Required comparison metric/crossing/core-fit missing: '+row['identity'])
    comparison=assessed({'matched24_at_each_requested_H':not any(item.startswith('Matched r24') for item in comparison_missing),
        'both_native_references_for_each_pair':bool(required) and all(row['native_both_pass'] for row in required),
        'all_pair_authorizations':bool(required_records) and all(native_by_record[row['identity']]['status']=='PASS' for row in required_records),
        'requested_metric_coverage':coverage},comparison_missing)
    unchanged=all(sha(path)==digest for path,digest in {**data['input_sha256'],**auth.hashes}.items())
    if not unchanged:raise RuntimeError('Immutable assessment inputs changed while reading')
    return safe_json({'status':'SAVED_FULL_FAMILY_ASSESSMENT','created_utc':now.isoformat(),
        'full_family_acceptance':family,'NN_heights':heights,'boundary_duration_checks':boundaries,
        'all_prior_mesh_pairs':histories,'diagnostic_controls':controls,
        'requested_NN_direct_comparisons':{**comparison,'matched_pairs':comparisons,
            'required_for_full_family_acceptance':False,
            'scope_clarification':'Legacy JSON key: separate NN versus non-neural optimizer comparisons. These are optional under the stated direct=independent Meep forward of the same frozen NN interpretation; their absence is not a required optical failure.',
            'same_NN_proxy_Meep_evidence_directory':str(folder/'forward_comparison') if (folder/'forward_comparison').is_dir() else None,
            'same_NN_forward_evidence_generated_here':False,
            'H75_note':'Validation-observed and used in seed/checkpoint selection; outside original seven-height aggregate'},
        'protocol':protocol,'new_attempt_authorizations':new_authorizations,'preparation_checks':auth.preparation_checks,
        'budget':budget,'all_attempts':data['all_attempts'],'missing_or_unverified_evidence':data['missing_evidence'],
        'input_sha256':{**data['input_sha256'],**auth.hashes},'ledger_snapshot_sha256':data['ledger_snapshot_sha256'],
        'immutable_inputs_unchanged':unchanged,'source_contract':{'wavelength_um':data['config']['wavelength_um'],
            'MFD_um':data['config']['mfd_um'],'input_phase_index_assumed':data['config']['materials']['substrate']['n'],
            'polymer_index_user_supplied':data['config']['materials']['polymer']['n'],
            'losses_assumed_zero':all(m.get('kappa')==0 for m in data['config']['materials'].values())},
        'NN_advantage_claimed':False,'full_M2_or_global_diffraction_optimum_claimed':False,'experimental_validation_claimed':False,
        'scope':['Nominal optical gates, native/reference checks, selected same-settings mesh evidence, boundary/duration and comparisons are separate.',
            'Old failed pairs remain visible. New refinement jobs are not the historical two-solve reserve.',
            'All attempted solves remain charged. Four additional physical cases are reserved; actual simulated new-case use is counted separately. No fresh-root reset.',
            'Synthetic Gaussian source and assumed phase index/losses remain; no MFD-tolerance, full M2, global optimum or experiment claim.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if not folder.is_dir():raise RuntimeError('Stage2 must already be prepared')
    if output.exists():raise FileExistsError('Choose a fresh assessment version')
    if not output.is_relative_to(folder/'assessment') or output==folder/'assessment':raise ValueError('Output must be below stage2/assessment/version')
    data=collect(folder,heights=COLLECT_HEIGHTS,methods=('NN','direct','baseline'),include_variants=True)
    result=assess_evidence(folder,data)
    result['utility_sha256']=sha(Path(__file__))
    output.mkdir(parents=True,exist_ok=False)
    with (output/'assessment.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['SAVED FULL-FAMILY ASSESSMENT',f"NN family: {result['full_family_acceptance']['status']}",
        f"Protocol: {result['protocol']['status']}; budget: {result['budget']['status']}",'',
        'H(um) finest_mesh eta(%) C Q selected_pair curvature_delta row_status']
    for row in result['NN_heights']:
        latest=row.get('latest_record',{});pair=row.get('selected_numerics',{})
        def f(value,n):return f'{value:.{n}f}' if finite(value) else 'NA'
        lines.append(f"{row['H_um']:5g} {row.get('latest_resolution','NA')} {f(100*latest['eta'],5) if finite(latest.get('eta')) else 'NA'} "
            f"{f(latest.get('C'),6)} {f(latest.get('Q'),6)} {pair.get('resolutions','missing')} {f(pair.get('curvature_delta'),6)} {row['status']}")
        for reason in row.get('missing_evidence',[]):lines.append('  Missing: '+reason)
        for failure in row.get('failed_checks',[]):lines.append('  Failed: '+failure)
        for failure in pair.get('comparison',{}).get('failed_checks',[]):lines.append('  Mesh pair failed: '+failure)
        for failure in row.get('native_reference',{}).get('failed_checks',[]):lines.append('  Native failed: '+failure)
    lines.extend(['','BOUNDARY / DURATION'])
    for row in result['boundary_duration_checks']:
        lines.append(f"{row['job_type']} H{row['H_um']:g}: {row['status']}")
        lines.append('  Scope: '+row['variation_scope'])
        for reason in row['missing_evidence']:lines.append('  Missing: '+reason)
        for failure in row['failed_checks']:lines.append('  Failed: '+failure)
        for failure in row.get('comparison',{}).get('failed_checks',[]):lines.append('  Pair failed: '+failure)
    lines.extend(['',f"Separate NN/non-neural optimizer comparisons: {result['requested_NN_direct_comparisons']['status']}",
        result['requested_NN_direct_comparisons']['scope_clarification'],
        'Required same-NN geometry proxy/Meep forward evidence is reported separately in forward_comparison; it does not establish full-family acceptance.',
        'H75 was validation-observed, not a fresh held-out test.','',*result['scope']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'NN_family':result['full_family_acceptance']['status'],
        'protocol':result['protocol']['status'],'additional_attempts':result['budget']['additional_attempts']},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
