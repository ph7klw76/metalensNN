"""Record the actual user amendment; no numerical solver is imported or run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/1550nm_height_family_n15263_v2'
DEST = CAMPAIGN / 'maxwell_validation_v3/authorization'
QUESTION = ('The approved 20 additional Meep solves are exhausted. Do you approve up to 18 more local solves '
            '(14 mesh-refinement calls plus 4 adaptive checks; cumulative ceiling 83) to resolve the remaining heights? '
            'The prepared proposal keeps the original deadline of 15 November, 16:59 Malaysia time, one CPU worker, '
            'no RAM gates, and unchanged designs and acceptance thresholds.')
REPLY = 'Approve up to 18 additional solves'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    prior = CAMPAIGN / 'maxwell_validation_v2'
    proposal = CAMPAIGN / 'prepared_maxwell_continuation_v4/revisions/post65_v1/PROPOSAL.json'
    deadline = datetime.fromisoformat('2026-11-15T08:59:13.107420+00:00')
    now = datetime.now(timezone.utc)
    if now >= deadline:
        raise RuntimeError('The retained original deadline has elapsed')
    if sha(prior / 'ledger.json') != '00ce8db6903e941f05139453f203beb7e0f0f8a580309066f81dec2d9d075623':
        raise RuntimeError('Prior sealed ledger changed')
    if sha(proposal) != '49f5301f2d7167d039bab39a086d4866f413e3e3274e49e1aff6e2d3accb90b0':
        raise RuntimeError('Prospective proposal changed')
    DEST.mkdir(parents=True, exist_ok=False)
    for name, value in [('question_exact.txt', QUESTION), ('user_reply_exact.txt', REPLY)]:
        with (DEST / name).open('x', encoding='utf-8', newline='') as stream:
            stream.write(value)
    receipt = dict(
        status='APPROVED_BY_USER', decision_origin='EXPLICIT_USER_REPLY',
        recorded_utc=now.isoformat(), user_reply_exact=REPLY,
        user_reply_sha256=sha(DEST / 'user_reply_exact.txt'), question_exact=QUESTION,
        question_sha256=sha(DEST / 'question_exact.txt'),
        source_question_item_id='["request_user_input_async","call_024ee36c6e054a89b093e00d272e4ea3",0]',
        prior_cumulative_attempts=65, additional_attempt_ceiling=18, cumulative_attempt_ceiling=83,
        primary_attempts=14, adaptive_attempt_ceiling=4,
        original_first_started_utc='2026-11-13T08:59:13.107420+00:00',
        original_absolute_deadline_utc=deadline.isoformat(),
        time_extension_authorized=False, time_remaining_at_recording_s=(deadline-now).total_seconds(),
        idle_and_approval_wait_count=True, one_local_CPU_optical_worker=True, all_RAM_gates=False,
        unchanged_physics=True, unchanged_designs=True, unchanged_acceptance_thresholds=True,
        new_physical_cases_authorized=0, actual_prior_physical_cases=25, reserved_case_ceiling=28,
        adaptive_specific_jobs_require_prospective_selection=True,
        retraining_authorized=False, external_compute_authorized=False,
        printing_authorized=False, system_changes_authorized=False,
        prior_ledger_sha256=sha(prior / 'ledger.json'),
        prior_delivery_manifest_sha256=sha(prior / 'delivery_manifest.json'),
        prior_archive_sha256=sha(CAMPAIGN / 'height_specific_Maxwell_continuation_v2_evidence.zip'),
        approved_proposal_file=str(proposal.relative_to(ROOT)), approved_proposal_sha256=sha(proposal),
        root_note='Approval changes the solve ceiling only. All work remains bounded by the original absolute deadline. '
                  'The full old 31.137-hour forecast no longer fits; primary jobs and any selected adaptive pair must fit '
                  'a recomputed remaining-time budget. Every failed or interrupted launched attempt is charged.',
        elapsed_used_as_approval=False, worker_launches=0,
    )
    with (DEST / 'approval_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], file=str(DEST/'approval_receipt.json'),
                          sha256=sha(DEST/'approval_receipt.json'),
                          remaining_h=receipt['time_remaining_at_recording_s']/3600), indent=2))

if __name__ == '__main__':
    main()
