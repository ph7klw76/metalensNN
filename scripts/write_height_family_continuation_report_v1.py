"""Prepare a final plain-text report from completed reviewed saved evidence.

Standard library only. Inert without explicit CLI arguments. Never invokes
inspect/seal/process queries, optical workers, field reconstruction or training.
The final assessment, figures and audits must already exist and match18 jobs.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import textwrap

ROOT=Path(__file__).resolve().parents[1]
HEIGHTS=(40.,50.,60.,70.,75.,80.,90.,100.)
SEAL_SHA='44581b0e0d91adf61136a105be9a0f5649e7af6f138c5608d67d388958b6263e'
REFS={
    'prior':'maxwell_validation_v2/assessment/final_v1/assessment.json',
    'primary':'maxwell_validation_v3/assessment/primary_final_v1/assessment.json',
    'first16':'maxwell_validation_v3/assessment/after_H60_r40_adaptive_v1/assessment.json',
    'learning':'maxwell_validation_v2/learning_evidence_v1/evidence.json',
    'replay':'maxwell_validation_v2/model_replay.json',
    'q_diagnosis':'maxwell_validation_v3/diagnostics/H100_r32_saved_fields_v1/diagnostic.json',
    'q_diagnosis_audit':'stage3_H100_saved_diagnosis_independent_review_v1/review.json',
    'clock_tests':'maxwell_validation_v2/software_checks/comparison_clock_v1/evidence.json',
    'collector_tests':'maxwell_validation_v3/software_checks/adaptive_analysis_tools_v1/source_manifest.json',
    'alias_tests':'maxwell_validation_v3/software_checks/publication_training_alias_v1/source_manifest.json'}
HISTORICAL_PINS={
    'q_diagnosis':'42720a69eec4ed9cbf3eb0617166fd50f554063cc425d5b6315f7e771a17de28',
    'q_diagnosis_audit':'b135d959bb79228cc13418b51d0c18d20f5d26a9d49f5227873f87e92220ae7f',
    'collector_tests':'615d73fd70f09003997c61fe0e7d777c9d634ec4ae62915ae5c36beaa8818252',
    'alias_tests':'f95b222a041638465a81dfe90046cd5cb4a98433a0601917db18724f8417e0d5'}
CONTEXT_RECEIPTS={
    'primary':('adaptive_selections/20261114T164034245691Z/selection_receipt.json','1e61f8e5536123577062da39f099a0d189201ca75dfd3c337079c6bdaa7f2e46'),
    'first16':('adaptive_selections/20261114T190445844863Z/selection_receipt.json','6c78928cabe3e28df1b615b772d631bb1c8941a989187759451a4a9320143872')}


def local_path(value):
    value=str(value)
    if os.name=='nt' and re.match(r'^/mnt/[a-z]/',value):value=value[5].upper()+':/'+value[7:]
    elif os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
    return Path(value).resolve()


def sha(path):return hashlib.sha256(local_path(path).read_bytes()).hexdigest()
def read(path):return json.loads(local_path(path).read_text(encoding='utf-8-sig'))
def row(report,H):return next(r for r in report['NN_heights'] if r['H_um']==H)
def words(text):return len(text.split())
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def display(value,style,scale=1.):return format(scale*value,style) if finite(value) else 'NA'
def paragraph(text):
    # Keep height/mesh symbols such as H75/r32 intact while separating prose
    # and physical units. Evidence paths are appended separately, untouched.
    for old,new in [('35-to40um','35 to 40 um'),('335-to502.5','335 to 502.5'),
                    ('r24-to32','r24-to-r32'),('16-to24','16-to-24'),
                    ('24-to32','24-to-32'),('32-to40','32-to-40'),('0.98nk0','0.98 n k0')]:
        text=text.replace(old,new)
    prefixes=r'The|the|at|above|below|before|after|earlier|with|are|is|than|from|to|span|approximately|optimal|coarse|separate|hidden|versus|exceeded|NA|RTX'
    text=re.sub(r'\b('+prefixes+r')(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=\d)(?=um\b|nm\b|hours\b|matched\b|declared\b|clock\b|collector\b|alias\b|degrees\b|annuli\b)',' ',text)
    text=re.sub(r"'s(?=\d)","'s ",text)
    text=re.sub(r',(?=\d)',', ',text)
    return textwrap.fill(text,90,break_long_words=False,break_on_hyphens=False)


class ProtectedInputs:
    def __init__(self,initial=None):
        self.values={}
        for p,digest in (initial or {}).items():self.expect(p,digest)
    def expect(self,p,digest):
        key=str(local_path(p))
        if key in self.values and self.values[key]!=digest:raise ValueError('Protected input changed between authenticated reads: '+key)
        self.values[key]=digest
    def document(self,p,expected=None):
        p=local_path(p);payload=p.read_bytes();digest=hashlib.sha256(payload).hexdigest()
        if expected is not None and digest!=expected:raise ValueError('Pinned historical context bytes changed: '+str(p))
        self.expect(p,digest)
        return json.loads(payload.decode('utf-8-sig'))
    def unchanged(self):return all(sha(p)==digest for p,digest in self.values.items())


def validate_finished(report,ledger):
    """Completion permits a truthful FAIL report; it never forces optics PASS."""
    if len(ledger.get('entries',[]))!=18 or [e.get('id') for e in ledger['entries']]!=list(range(66,84)):
        raise ValueError('Final report requires all18 actual attempts66..83; no interim report')
    if any(e.get('status')!='completed' or e.get('method')=='NN' and e.get('analysis_status')!='completed' for e in ledger['entries']):
        raise ValueError('Selected optical execution/analysis remains incomplete')
    attempts=report.get('all_attempts',[])
    if (len(attempts)!=83 or sorted(a.get('id',-1) for a in attempts)!=list(range(1,84))
            or any(a.get('status')!='completed' for a in attempts if a['id']!=41)
            or not isinstance(next(a for a in attempts if a['id']==41).get('status'),str)
            or not next(a for a in attempts if a['id']==41).get('status')
            or next(a for a in attempts if a['id']==41).get('status')=='completed'):
        raise ValueError('All83 charged attempts and the preserved interruption41 are required')
    for key in ('authorization','budget','primary_execution','adaptive_execution','continuation_execution'):
        if report.get(key,{}).get('status')!='PASS':raise ValueError('Final execution/provenance incomplete: '+key)
    budget=report['budget']
    if any(budget.get(k)!=v for k,v in [('additional_attempts',18),('cumulative_attempts',83),('actual_cumulative_cases',25),('reserved_case_ceiling',28)]):
        raise ValueError('Final83-attempt/25-case accounting differs')
    if report['primary_execution'].get('completed_verified')!=14 or report['adaptive_execution'].get('pending_current_selected_pairs'):
        raise ValueError('Primary or selected adaptive work remains pending')
    if len(report.get('selected_adaptive_diagnostics',[]))!=2:raise ValueError('Both selected diagnostic outcomes required')
    if len(report.get('NN_heights',[]))!=8 or sorted(r['H_um'] for r in report['NN_heights'])!=list(HEIGHTS):
        raise ValueError('All eight actual height outcomes required')
    for r in report['NN_heights']:
        if not r.get('Meep'):raise ValueError('Verified final saved field missing')
        if any(r.get(k,{}).get('status') not in ('PASS','FAIL','UNRESOLVED') for k in ('native','nominal_optical','selected_mesh')):
            raise ValueError('Every native/nominal/mesh outcome must remain explicit')
    controls=[r for r in report.get('boundary_duration_checks',[]) if r.get('required_original_sampled_control')]
    if len(controls)!=4 or {(r.get('job_type'),r.get('H_um')) for r in controls}!={('pml',40.),('pml',100.),('padding',70.),('duration',100.)}:
        raise ValueError('All four unchanged original sampled controls are required')
    for name,selected in [('all_eight_nominal_heights_acceptance',report['NN_heights']),
                          ('original_seven_height_family_acceptance',[r for r in report['NN_heights'] if r['H_um']!=75.])]:
        if report.get(name,{}).get('status')=='PASS' and (
                any(r[k]['status']!='PASS' for r in selected for k in ('native','nominal_optical','selected_mesh'))
                or any(r.get('status')!='PASS' for r in controls)):
            raise ValueError('Family PASS contradicts a selected outcome or required original control')
    if report.get('execution_completion_is_not_optical_acceptance') is not True:raise ValueError('Execution/optical distinction missing')
    if any(report.get(k) is not False for k in ('NN_superiority_claimed','global_diffraction_optimum_claimed','full_M2_claimed','experimental_validation_claimed')):
        raise ValueError('Unsupported final scientific claim')


def audit_binds_ledger(audit,ledger_path,digest):
    if not str(audit.get('status','')).startswith('PASS'):return False
    proofs=[audit.get('ledger_sha256'),audit.get('stage3_ledger_sha256')]
    for key in ('ledger_snapshot_sha256','input_sha256'):
        inventory=audit.get(key)
        if isinstance(inventory,dict):proofs.extend(v for k,v in inventory.items() if local_path(k)==local_path(ledger_path))
    return digest in proofs


def audit_binds_graph(audit,manifest_path,digest):
    if not str(audit.get('status','')).startswith('PASS'):return False
    if digest in (audit.get('graph_delivery_manifest_sha256'),audit.get('delivery_manifest_sha256')):return True
    for key in ('input_sha256','output_sha256','file_sha256'):
        inventory=audit.get(key)
        if isinstance(inventory,dict):
            for p,value in inventory.items():
                if local_path(p)==local_path(manifest_path) and value==digest:return True
    return False


def historical_marker(report,prior_ledger):
    source=[a for a in prior_ledger['entries'] if a['id']==41]
    observed=[a for a in report['all_attempts'] if a['id']==41]
    if len(source)!=len(observed) or len(source)!=1 or source[0].get('status')=='completed' or observed[0].get('status')!=source[0].get('status'):
        raise ValueError('Historical interruption41 must retain its exact immutable legacy status')
    return source[0]['status']


def require_unsealed(stage):
    if any((local_path(stage)/name).exists() for name in ('delivery_manifest.json','final_closure','final_source_snapshots')):
        raise ValueError('Sealed or closing stage3 tree must remain immutable')


def bind_context_to_receipt(stage,context_path,context,receipt_path,receipt_sha,inputs,protected):
    receipt_path=local_path(receipt_path);context_path=local_path(context_path)
    if inputs.get(receipt_path)!=receipt_sha:raise ValueError('Historical adaptive receipt is absent from authenticated final inputs')
    receipt=protected.document(receipt_path,receipt_sha)
    digest=receipt['primary_assessment_sha256']
    if sha(context_path)!=digest:raise ValueError('Historical assessment does not match prospective receipt')
    copied=protected.document(receipt_path.parent/'primary_assessment.json',digest)
    if copied!=context:raise ValueError('Historical assessment copy differs')
    request_path=receipt_path.parent/'selection_request_exact.json'
    if inputs.get(request_path)!=receipt['request_sha256']:raise ValueError('Historical request proof absent from final inputs')
    request=protected.document(request_path,receipt['request_sha256'])
    snapshots={local_path(p):value for p,value in context['ledger_snapshot_sha256'].items()}
    if (local_path(request['primary_assessment_file'])!=context_path or request['primary_assessment_sha256']!=digest
            or receipt['primary_assessment_ledger_sha256']!=snapshots[local_path(stage/'ledger.json')]
            or context['assessment_utility_sha256']!=receipt['primary_assessment_utility_sha256']):
        raise ValueError('Historical assessment/request/utility/ledger lineage differs')


def render_report(report,ledger,context,paths):
    """All current outcomes/values come from the supplied final assessment."""
    budget=report['budget'];cfg=context['config'];learning=context['learning'];q=context['q_diagnosis']
    first=context['first16'];prior=context['prior'];primary=context['primary']
    complete=sum(x.get('status')=='completed' for x in report['all_attempts'])
    first_status=first['all_eight_nominal_heights_acceptance']['status']
    final_status=report['all_eight_nominal_heights_acceptance']['status']
    original=report['original_seven_height_family_acceptance']['status']
    problems=[str(int(r['H_um'])) for r in report['NN_heights'] if any(r[k]['status']!='PASS' for k in ('native','nominal_optical','selected_mesh'))]
    diagnostic_summary='; '.join(f"{d['pair_id']}: native {d['native_status']}, numerical {d['numerical_status']}, target nominal {d['target_nominal_status']}" for d in report['selected_adaptive_diagnostics'])
    model=learning['model'];w=learning['objective_weights']
    requested=[row(report,H)['Meep'] for H in (40.,75.,100.)]
    diameters=[p[k] for m in requested for p in (m.get('planes') or []) if p.get('z_um')==0
               for k in ('diameter_x_um','diameter_y_um') if finite(p.get(k)) and p[k]>0]
    diameter_text=(f"Available output x/y diameters span {min(diameters):.2f}-{max(diameters):.2f}um; unavailable values remain NA. "
                   if diameters else "No finite output width is available for the requested three heights. ")
    diameter_text+="The earlier approximately30um expansion aspiration is not established by this natural-size study. "
    p90=row(prior,90.)['selected_mesh']['comparison']['measurements']['maximum_diameter_relative_change']
    old60=row(primary,60.)['selected_mesh']['comparison']['measurements']['curvature_absolute_change']
    refined60=row(first,60.)['selected_mesh']['comparison']['measurements']['curvature_absolute_change']
    wide=next(x for x in q['Q_reduction_decomposition']['groups'] if x['label']=='20_deg_and_above_retained')
    controls=[r for r in report['boundary_duration_checks'] if r.get('required_original_sampled_control')]
    duration=next(r for r in controls if r['job_type']=='duration' and r['H_um']==100.)
    cp=context['clock_tests']['successful_check']['passed'];ap=context['collector_tests']['pure_tests']['WSL']['count'];tp=context['alias_tests']['pure_tests']['native']
    paragraphs=[
        f"The completed continuation has declared original-seven-height outcome {original} and all-eight-height outcome {final_status}. At the earlier16-call snapshot, the all-eight classification was {first_status}; the actual final values below supersede that snapshot. "
        +(f"Failed or unresolved selected native/nominal/mesh outcomes remain at H={', '.join(problems)}um. " if problems else "Every selected per-height native, nominal and declared last-mesh check passes. ")
        +"These classifications cover the frozen synthetic protocol. Computational completion is separate from optical acceptance; no global diffraction optimum, ISO M2, full-angle convergence or experiment is established.",
        f"All {budget['additional_attempts']} new Meep attempts completed: {budget['cumulative_attempts']} cumulative attempts, {complete} completed calculations and preserved interrupted attempt41. Its immutable legacy status '{context['historical41_status']}' remains historical evidence, not a current worker. Actual physical-case use is {budget['actual_cumulative_cases']}; {budget['reserved_case_ceiling']} is the reserved ceiling. "
        f"The assessment records {budget['absolute_elapsed_s_including_idle']/3600:.3f}hours from {budget['original_first_started_utc']}, including idle and approval gaps; the unchanged deadline is {budget['absolute_deadline_utc']}. The18-call amendment did not reset time. One local CPU optical worker ran at a time, with no host/process/WSL RAM gates. Initial proxy training used RTX4060; this continuation added no training updates. Later packaging time does not authorize later solves.",
        f"The model uses vacuum wavelength {cfg['wavelength_um']:g}um, lossless polymer n={cfg['materials']['polymer']['n']:g}, ambient air and a {cfg['cylinder_diameter_um']:g}um solid cylinder. Heights span40-100um; each height has a different frozen annular top pattern. The50annuli are0.5um wide, with0-3um relief quantized at0.05um. The manufacturer-nominal Gaussian MFD is {cfg['mfd_um']:g}um; the user's8um core description is separate. Nominal NA0.14 is not a hard Gaussian angular cutoff. The assumed input phase index is1.45, not a solved fiber eigenmode. Output and cross-check monitors are at H+4 and H+9um. Finite Gaussian-pulse excitation and Fourier extraction at1550nm are used, rather than a continuous-wave calculation.",
        f"The NN is a height-conditioned geometry generator with {len(model['hidden_layers'])} hidden64-unit tanh layers and {model['outputs']} residual logits. Each inference also requires an independently computed cylinder-field phase seed. Residuals modify that seed before bounded decoding and fabrication quantization. The loss rewards useful mode power, with width/phase/angular/sidelobe/roughness weights {w['width']}/{w['phase']}/{w['angular']}/{w['sidelobe']}/{w['roughness']}. Continuous-relief training and quantized-validation checkpoint selection differ. H75 influenced validation and selection; it is observed, not held out.",
        f"Finite-difference checks, nonzero-optimizer-state recovery and bitwise selected-checkpoint decoder replay support that optimization and geometry generation ran. The audited {learning['matched_training_validation_recovery_objective_calls']}matched training/validation/recovery objective calls exclude pilots, per-height seed/target construction and final reporting. Aggregate proxy validation improved, but quantized loss worsened at H75/H100 and H100 efficiency decreased relative to its seed. Three separately optimized non-neural patterns remain proxy-only. Neither their small proxy differences nor the present forward-model agreement establishes NN superiority."]
    lines=['FINAL CONTINUATION REPORT','',*('\n\n'.join(paragraph(x) for x in paragraphs).splitlines()),'',
        'Final frozen NN results: eta is conservative useful-mode efficiency; C is normalized',
        'curvature; Q is the original retained-band angular-moment ratio. Rounded display values',
        'never determine gates; full-precision JSON and saved-field hashes do.',
        'H(um) cells/um eta(%)    C          Q          native     nominal    mesh/pair']
    for r in report['NN_heights']:
        m=r['Meep'];pair=r['selected_mesh'].get('resolutions',[])
        pairtext='->'.join(f'{v:g}' for v in pair) or 'missing'
        lines.append(f"{r['H_um']:5g} {display(m.get('resolution'),'g'):>8} {display(m.get('eta'),'.5f',100):>9} {display(m.get('C'),'+.6f'):>9} {display(m.get('Q'),'.6f'):>10} {r['native']['status']:<10} {r['nominal_optical']['status']:<10} {r['selected_mesh']['status']} {pairtext}")
    after=[
        "At 40/75/100 um, the figures compare a proxy forward prediction with direct Meep forward simulation of the SAME frozen NN-generated geometry within each height; patterns differ between heights. This is not a non-neural optimizer comparison. "
        +diameter_text+"Distances start at H+4, not the facet. Gaussian-core HALF-angles are in degrees; x/y diameter bands are not uncertainty bars. Finite Gaussian beams still diffract downstream.",
        f"Earlier failures remain: H80/H100 r24 angular ratios were {row(prior,80.)['Meep']['Q']:.8f}/{row(prior,100.)['Meep']['Q']:.8f}, above1.10; H75 lacked a mesh pair. H90's16-to24 maximum width change was {100*p90:.6f}%, above2%. H60's24-to32 curvature change {old60:.8f} exceeded0.005; unchanged-geometry 32-to-40 reduced it to {refined60:.8f}. Refinement did not relax thresholds or redesign failed lenses. The separate980nm focus failure,6.373um versus7.8um, motivated validated connected-Prism, finite-thickness and extraction checks; those historical repairs were not offsets applied to1550nm results.",
        f"H100's historical r24-to32 saved-field diagnosis allocated {100*wide['signed_share_of_total_Q_reduction']:.3f}% of the net Q reduction to retained angles above20degrees, offset by increased lower-angle contributions. Their weak power fell {100*wide['r24_power_fraction']:.8f}% to {100*wide['r32_power_fraction']:.8f}%. Tested Cartesian sampling/window alternatives shifted r32 Q by at most {q['r32_max_abs_Q_sampling_shift']:.8f}, much less than that native-mesh change. This locates weighted contributions without proving their physical cause. The original 0.98 n k0 cutoff remains; no halo trimming, all-angle/M2 inference or transfer of this diagnosis to the final finer field is justified.",
        f"The four original coarse16-cell sampled PML/domain/duration controls have {sum(r['status']=='PASS' for r in controls)}/{len(controls)}declared passes. The H70 padding check also changes monitor radius from 35 to 40 um, so it combines domain and aperture effects. H100 duration from 335 to 502.5 changed coarse Q by {duration['Q_variant']-duration['Q_prior']:+.8f}; both coarse angular flags remain failed. This is not a correction for a finer mesh or proof of complete temporal accuracy. Selected adaptive outcomes are retained separately: {diagnostic_summary}. Additional fine results never erase older failed pairs.",
        f"Software corrections made pending/stopped clocks count idle time, bound reference IDs/origins, immutable configurations, exact worker argv/output and frozen geometry, and authenticated prospective adaptive requests. A CSV failure arose from two identical training-result copies; a verified canonical-path projection fixes CSV provenance while preserving both full-inventory aliases. Evidence includes {cp}clock, {ap}collector and {tp}alias counterexample tests, plus independent raw/CSV audits. An initial native render-fixture error was missing Matplotlib; the existing WSL environment passed, without an optical or geometry change.",
        "The evidence therefore supports only the final table's declared synthetic outcomes and explicitly listed diagnostics. It does not validate arbitrary fiber MFD/NA/material inputs, continuous height tolerances, unobserved generalization, a globally optimal50um-aperture beam, new-height fabrication, printer exposure or experimental performance. The historical H35/C01 assembly remains separate. All source, checkpoint, field, geometry, failure and elapsed-time records remain reviewable; no additional optical work is implied by this report."]
    lines.extend(['','\n\n'.join(paragraph(x) for x in after),'','Evidence:'])
    for key in ('assessment','graphs','audits','learning','replay','q_diagnosis'):
        lines.append(f"{key}: {paths[key]}")
    text='\n'.join(lines)+'\n'
    if not 800<=words(text)<=1200:raise ValueError(f'Report length {words(text)} outside800-1200words')
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment',required=True);parser.add_argument('--graphs',required=True)
    parser.add_argument('--audit',action='append',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();assessment=local_path(args.assessment);stage=assessment.parent.parent.parent
    output=local_path(args.output);graphs=local_path(args.graphs)
    require_unsealed(stage)
    if output.exists() or Path(str(output)+'.provenance.json').exists():raise FileExistsError('Preserve previous report; choose a new output')
    if stage.name!='maxwell_validation_v3' or not output.is_relative_to(stage) or output.suffix!='.txt':raise ValueError('Use a fresh TXT output under the actual stage3 folder')
    source=ROOT/'scripts/seal_height_family_stage3_v1.py'
    if sha(source)!=SEAL_SHA:raise ValueError('Reviewed pure final assessment/graph guard changed')
    spec=importlib.util.spec_from_file_location('reviewed_pure_report_guards',source)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    protected=ProtectedInputs({str(Path(__file__).resolve()):sha(__file__),str(source):SEAL_SHA})
    document=protected.document
    ledger=document(stage/'ledger.json');report=document(assessment);validate_finished(report,ledger)
    guard.verify_assessment(stage,assessment,ledger,{},at=datetime.now(timezone.utc))
    for p,digest in guard.normalized_inventory(report['input_sha256']).items():
        protected.expect(p,digest)
    graph_proof=guard.verify_graph_artifact(stage,graphs/'delivery_manifest.json',ledger,report)
    protected.expect(graphs/'delivery_manifest.json',graph_proof['delivery_manifest_sha256'])
    audit_paths=[local_path(p) for p in args.audit];audits=[document(p) for p in audit_paths]
    ledger_digest=protected.values[str(stage/'ledger.json')]
    if not audits or not all(audit_binds_ledger(a,stage/'ledger.json',ledger_digest) for a in audits):raise ValueError('Every final independent audit must PASS and bind this exact final ledger')
    if not any(audit_binds_graph(a,graphs/'delivery_manifest.json',graph_proof['delivery_manifest_sha256']) for a in audits):raise ValueError('An independent final audit must explicitly bind the actual final graph delivery SHA')
    campaign=stage.parent;context={key:document(campaign/name,HISTORICAL_PINS.get(key)) for key,name in REFS.items()};context['config']=document(stage/'config.json')
    if not str(context['learning'].get('status','')).startswith('PASS') or context['replay'].get('status')!='PASS_FROZEN_DECODER_REPLAY':raise ValueError('Verified learning/replay evidence required')
    if context['first16']['budget']['additional_attempts']!=16:raise ValueError('Historical first16 snapshot differs')
    final_inputs=guard.normalized_inventory(report['input_sha256'])
    for key,(name,digest) in CONTEXT_RECEIPTS.items():
        bind_context_to_receipt(stage,campaign/REFS[key],context[key],stage/name,digest,final_inputs,protected)
    prior=document(stage/'prior_stages.json');old=next(s for s in prior['stages'] if s['stage_id']=='stage2')
    original=next(s for s in prior['stages'] if s['stage_id']=='stage1')
    original_ledger=document(original['ledger_file'])
    if sha(original['ledger_file'])!=original['ledger_sha256']:raise ValueError('Historical stage1 ledger changed')
    context['historical41_status']=historical_marker(report,original_ledger)
    if sha(campaign/REFS['prior'])!=old['final_assessment_sha256']:raise ValueError('Historical final stage2 assessment differs')
    old_manifest=document(local_path(old['folder'])/'delivery_manifest.json')
    if sha(local_path(old['folder'])/'delivery_manifest.json')!=old['delivery_manifest_sha256']:raise ValueError('Historical stage2 seal differs')
    for key,name in REFS.items():
        ref=campaign/name
        if ref.is_relative_to(local_path(old['folder'])):
            relative=ref.relative_to(local_path(old['folder'])).as_posix()
            if sha(ref)!=old_manifest['files'][relative]['sha256']:raise ValueError('Sealed historical evidence changed: '+key)
    paths={'assessment':str(assessment),'graphs':str(graphs),'audits':', '.join(map(str,audit_paths)),
        **{key:str(campaign/REFS[key]) for key in ('learning','replay','q_diagnosis')}}
    text=render_report(report,ledger,context,paths)
    if not protected.unchanged():raise RuntimeError('Protected report input changed')
    require_unsealed(stage)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:stream.write(text)
    metadata={'status':'SAVED_FINAL_CONTINUATION_REPORT_NO_OPTICAL_WORK','report_file':str(output),'report_sha256':sha(output),
        'report_created_utc':datetime.now(timezone.utc).isoformat(),'assessment_created_utc':report.get('created_utc'),
        'word_count':words(text),'input_sha256':protected.values,'assessment_sha256':sha(assessment),'graph_proof':graph_proof,
        'source_sha256':sha(__file__),'all_current_outcomes_derived_from_final_JSON':True,
        'zero_new_optical_or_training_or_field_reconstruction_calls':True,'no_process_scan_inspect_or_seal_called':True}
    with Path(str(output)+'.provenance.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'word_count':words(text),'sha256':metadata['report_sha256'],'optical_calls':0}))


if __name__=='__main__':main()
