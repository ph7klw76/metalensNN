"""Archive completed bounded validation evidence without changing any result."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/1550nm_height_family_n15263_v2'
FOLDER=CAMPAIGN/'maxwell_validation_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    archive=CAMPAIGN/'height_specific_Maxwell_validation_v1.zip'
    manifest_path=FOLDER/'delivery_manifest.json'
    if archive.exists() or manifest_path.exists():
        raise FileExistsError('Preserve the existing package; choose a new version.')
    ledger=json.loads((FOLDER/'ledger.json').read_text())
    summary=json.loads((FOLDER/'summary.json').read_text())
    if 'RUNNING' in ledger['status']:
        raise RuntimeError('Do not package a running optical stage.')
    for name in ('REPORT.txt','source_mode_audit_final.json','nominal_maxwell_trends.png'):
        if not(FOLDER/name).is_file():raise FileNotFoundError(name)
    post=FOLDER/'postprocessing_sources'
    post.mkdir(exist_ok=False)
    scripts=['report_height_family_maxwell_v2.py','plot_height_family_maxwell_v2.py',
             'audit_height_family_reference_mode_v2.py','diagnose_height_family_h40_curvature_v2.py',
             'continue_height_family_numerical_checks_v2.py','package_height_family_maxwell_v2.py',
             'write_height_family_decision_v2.py',
             'controlled_stop_height_family_v2.py']
    source_hashes={}
    for name in scripts:
        source=ROOT/'scripts'/name
        if source.is_file():
            shutil.copyfile(source,post/name)
            source_hashes['scripts/'+name]=sha(source)
    test_folder=FOLDER/'verification_sources'
    test_folder.mkdir(exist_ok=False)
    for path in sorted((ROOT/'tests').glob('test_height_family*py')):
        shutil.copyfile(path,test_folder/path.name)
        source_hashes['tests/'+path.name]=sha(path)
    for path in sorted(CAMPAIGN.glob('maxwell*tests*.xml')):
        shutil.copyfile(path,test_folder/path.name)
    (post/'source_manifest.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    now=datetime.now(timezone.utc)
    elapsed=(now-datetime.fromisoformat(ledger['first_started_utc'])).total_seconds()
    finalization={'created_utc':now.isoformat(),'optical_stage_elapsed_s':ledger['elapsed_s'],
                  'elapsed_through_report_packaging_s':elapsed,
                  'within_original_four_hour_window':elapsed<=14400,
                  'attempted_solves':len(ledger['entries']),
                  'completed_solves':sum(e['status']=='completed' for e in ledger['entries']),
                  'new_solves_in_postprocessing':0,
                  'further_compute_proposal_status':'AWAITING_USER_APPROVAL_NOT_STARTED'}
    (FOLDER/'finalization_receipt.json').write_text(json.dumps(finalization,indent=2)+'\n')
    files={}
    for path in sorted(FOLDER.rglob('*')):
        if path.is_file() and path.suffix!='.lock' and not path.name.endswith('.tmp'):
            files[path.relative_to(FOLDER).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    manifest={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'assessment_status':summary['status'],'ledger_status':ledger['status'],
        'scope':'Frozen height-specific candidates, actual independent Maxwell records, failures and selected numerical checks.',
        'limitations':['Synthetic Gaussian source and assumed input phase index; no experimental or ISO M2 acceptance.',
                       'Read REPORT.txt and summary.json for failed and missing checks; packaging is not optical PASS.',
                       'No SLC manufacturing release or printing is implied.',
                       'Original absolute paths in raw JSON preserve provenance; NPZ and JSON files are included at their validation-relative paths.'],
        'attempted_solves':len(ledger['entries']),
        'cumulative_unique_device_ceiling':24,'files':files,
    }
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for name in files:stream.write(FOLDER/name,arcname=name)
        stream.write(manifest_path,arcname=manifest_path.name)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:raise ValueError('Archive CRC failure')
        for name,item in files.items():
            if hashlib.sha256(stream.read(name)).hexdigest()!=item['sha256']:
                raise ValueError('Archive byte mismatch: '+name)
        assert json.loads(stream.read(manifest_path.name))==manifest
    receipt={'status':'PASS_ARCHIVE_INTEGRITY','assessment_status':summary['status'],
             'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
             'payload_files':len(files),'all_payload_hashes_verified':True,'optical_pass_implied':False,
             'created_utc':datetime.now(timezone.utc).isoformat()}
    (FOLDER/'delivery_integrity.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
