"""Read-only compact progress of the approved local continuation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v3'

def main():
    ledger = json.loads((FOLDER / 'ledger.json').read_text(encoding='utf-8-sig'))
    now = datetime.now(timezone.utc)
    entries = ledger['entries']
    latest = entries[-1] if entries else {}
    result = dict(
        utc=now.isoformat(), status=ledger['status'], new_attempts=len(entries),
        cumulative_attempts=65+len(entries), completed=sum(e['status']=='completed' for e in entries),
        remaining_authorized_attempts=18-len(entries),
        deadline_remaining_h=(datetime.fromisoformat(ledger['absolute_deadline_utc'])-now).total_seconds()/3600,
        latest={key:latest.get(key) for key in ('id','method','H_um','resolution','job_type','status',
                                                'analysis_status','started_utc','finished_utc','C','eta',
                                                'power_balance_error','error')},
        optical_workers_launched_by_inspection=0,
    )
    if latest.get('log_file'):
        path = FOLDER / latest['log_file']
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-16384))
                lines = stream.read().decode('utf-8',errors='replace').splitlines()
            progress = [line for line in lines if re.match(r'^Meep progress:', line)]
            result['latest_Meep_progress'] = progress[-1] if progress else None
            if latest.get('status') != 'completed':
                result['log_tail'] = lines[-3:]
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
