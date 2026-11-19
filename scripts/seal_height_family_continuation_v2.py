"""Seal completed local validation evidence; never imports an optical solver."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / 'artifacts/1550nm_height_family_n15263_v2'
FOLDER = CAMPAIGN / 'maxwell_validation_v2'
PRIOR = CAMPAIGN / 'maxwell_validation_v1'
ARCHIVE = CAMPAIGN / 'height_specific_Maxwell_continuation_v2_evidence.zip'
VERIFICATION = CAMPAIGN / 'height_specific_Maxwell_continuation_v2_evidence.verification.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def contained(parent, name):
    parent = Path(parent).resolve()
    path = (parent / name).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('Outside intended evidence directory: ' + str(name))
    return path


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    manifest_path = FOLDER / 'delivery_manifest.json'
    snapshots = FOLDER / 'final_source_snapshots'
    inputs = FOLDER / 'final_training_inputs'
    if any(path.exists() for path in (manifest_path, snapshots, inputs, ARCHIVE, VERIFICATION)):
        raise FileExistsError('Preserve the existing seal; choose a separately versioned workflow')
    ledger = read(FOLDER / 'ledger.json')
    if (len(ledger['entries']) != 20 or ledger['cumulative_attempted_solves'] != 65
            or [row['id'] for row in ledger['entries']] != list(range(46, 66))
            or any(row['status'] != 'completed' for row in ledger['entries'])
            or ledger['status'] != 'USER_SELECTED_ORIGINAL_CONTROLS_EXECUTED_REQUIRES_FINAL_ASSESSMENT'):
        raise ValueError('Exactly twenty completed, closed stage2 solves are required')
    assessment = read(FOLDER / 'assessment/final_v1/assessment.json')
    if assessment['protocol']['status'] != 'PASS' or assessment['budget']['status'] != 'PASS':
        raise ValueError('Verified protocol and resource evidence required')
    if read(FOLDER / 'audit/final_raw_v1/audit_completion.json')['status'] != 'PASS_INDEPENDENT_AUDIT_FINALIZED':
        raise ValueError('Final independent audit is not complete')
    report_hash = sha(FOLDER / 'CONTINUATION_REPORT.txt')
    if report_hash != '354ef03b86cfabd00a581e50016685eaf9e0fd02f22c58f32d2dd754d414f091':
        raise ValueError('Independently audited report bytes changed')

    prior_manifest = read(PRIOR / 'delivery_manifest.json')
    for name, item in prior_manifest['files'].items():
        if sha(contained(PRIOR, name)) != item['sha256']:
            raise ValueError('Sealed prior payload changed: ' + name)
    old_archive = CAMPAIGN / 'height_specific_Maxwell_validation_v1.zip'
    old_archive_hash = sha(old_archive)
    if old_archive_hash != 'a18dc723b2513946f6cadc6afabb794a2fcb529f697be997d5beb87c3b3305fd':
        raise ValueError('Original sealed archive changed')
    for name, expected in read(FOLDER / 'protected_historical.json').items():
        if sha(contained(ROOT, name)) != expected:
            raise ValueError('Protected historical input changed: ' + name)
    required_sources = dict(read(FOLDER / 'source_manifest.json'))
    replay = read(FOLDER / 'model_replay.json')
    required_sources.update(replay['source_hashes'])
    for name, expected in required_sources.items():
        if sha(contained(ROOT, name)) != expected:
            raise ValueError('Frozen execution/learning source changed: ' + name)

    copied = {}

    def copy(source, target, expected=None):
        source = Path(source).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError('Only project inputs may be copied: ' + str(source))
        original = sha(source)
        if expected is not None and original != expected:
            raise ValueError('Expected input SHA differs: ' + str(source))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != original or sha(source) != original:
            raise ValueError('Snapshot copy/source integrity differs')
        copied[str(target.relative_to(FOLDER).as_posix())] = {
            'original_project_path': str(source.relative_to(ROOT).as_posix()), 'sha256': original}

    source_names = set(required_sources)
    source_names.update(path.relative_to(ROOT).as_posix()
                        for path in (ROOT / 'src/fiber_metalens_nn').rglob('*.py'))
    for name in ('run_height_family_subsequent_phases_v3.py', 'run_height_family_common_H75_v1.py',
                 'run_height_family_original_controls_v1.py', 'run_height_family_original_controls_v2.py',
                 'assess_height_family_stage2_v3.py', 'compare_height_family_nn_direct_v2.py',
                 'report_height_family_maxwell_v2.py', 'plot_height_family_nn_forward_v3.py',
                 'plot_height_family_nn_forward_publication_v4.py',
                 'plot_height_family_nn_forward_publication_v5.py',
                 'verify_height_family_frozen_inference_v3.py', Path(__file__).name):
        source_names.add('scripts/' + name)
    for skill in ('metalens-phase-workflow', 'fiber-metalens-design-and-slc'):
        source_names.update(path.relative_to(ROOT).as_posix()
                            for path in (ROOT / 'skills' / skill).rglob('*')
                            if path.is_file() and path.suffix in ('.md', '.yaml', '.yml', '.txt', '.json'))
    for name in sorted(source_names):
        copy(contained(ROOT, name), snapshots / name, required_sources.get(name))
    for name in ('config.json', 'training_result.json'):
        copy(CAMPAIGN / name, inputs / name)
    for path in sorted((CAMPAIGN / 'designs').rglob('*.json')):
        copy(path, inputs / path.relative_to(CAMPAIGN))
    for directory in sorted(CAMPAIGN.glob('*_seed*')):
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.pkl', '.npz', '.txt'):
                    copy(path, inputs / path.relative_to(CAMPAIGN))
    for record in replay['checkpoints'].values():
        raw = record['path']
        local = Path(raw[5] + ':/' + raw[7:]) if raw.startswith('/mnt/') else Path(raw)
        if sha(local) != record['sha256']:
            raise ValueError('Selected learning checkpoint changed')
        if sha(inputs / local.relative_to(CAMPAIGN)) != record['sha256']:
            raise ValueError('Selected learning checkpoint was not copied exactly')
    write_new(FOLDER / 'final_snapshot_index.json', copied)

    files = {}
    for path in sorted(FOLDER.rglob('*')):
        if path.is_file():
            if not path.resolve().is_relative_to(FOLDER.resolve()):
                raise ValueError('Evidence contains an external link')
            files[path.relative_to(FOLDER).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest = {
        'status': 'SEALED_COMPLETED_BOUNDED_VALIDATION_OPTICAL_ACCEPTANCE_NOT_CLAIMED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'new_attempts_completed': 20, 'cumulative_attempts': 65, 'cumulative_completed': 64,
        'actual_cumulative_unique': 25, 'reserved_unique_ceiling': 28,
        'full_family_acceptance': assessment['full_family_acceptance']['status'],
        'ledger_sha256': sha(FOLDER / 'ledger.json'), 'report_sha256': report_hash,
        'original_first_started_utc': ledger['first_started_utc'],
        'original_absolute_deadline_utc': ledger['absolute_deadline_utc'],
        'remaining_authorized_Meep_calls': 0,
        'prior_delivery_manifest_sha256': sha(PRIOR / 'delivery_manifest.json'),
        'prior_original_archive_sha256': old_archive_hash,
        'prior_delivered_payloads_unchanged': len(prior_manifest['files']),
        'archive_scope': 'Old delivered payloads plus new validation evidence and selected project learning/source inputs; original provenance paths retained; virtual environments excluded',
        'files': files,
    }
    write_new(manifest_path, manifest)
    members = {}
    for name, item in prior_manifest['files'].items():
        members['maxwell_validation_v1/' + name] = (contained(PRIOR, name), item['sha256'])
    members['maxwell_validation_v1/delivery_manifest.json'] = (PRIOR / 'delivery_manifest.json', sha(PRIOR / 'delivery_manifest.json'))
    for name, item in files.items():
        members['maxwell_validation_v2/' + name] = (contained(FOLDER, name), item['sha256'])
    members['maxwell_validation_v2/delivery_manifest.json'] = (manifest_path, sha(manifest_path))
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, (path, expected) in sorted(members.items()):
            if sha(path) != expected:
                raise ValueError('Payload changed before packaging: ' + name)
            bundle.write(path, name)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(members) or bundle.testzip() is not None:
            raise ValueError('Archive membership/CRC verification failed')
        for name, (_, expected) in members.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError('Archive payload SHA differs: ' + name)
    if sha(old_archive) != old_archive_hash:
        raise ValueError('Prior archive changed during sealing')
    result = {'status': 'PASS_ARCHIVE_CRC_AND_EVERY_PAYLOAD_SHA',
              'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
              'archive_bytes': ARCHIVE.stat().st_size, 'members_verified': len(members),
              'new_stage_payload_files': len(files), 'prior_payload_files': len(prior_manifest['files']),
              'delivery_manifest_sha256': sha(manifest_path), 'optical_acceptance': 'FAIL',
              'Meep_solves_launched_by_sealing': 0}
    write_new(VERIFICATION, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
