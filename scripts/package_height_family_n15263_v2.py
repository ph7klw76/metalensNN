"""Package completed proxy candidates and verify the delivered bytes; no optical run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "artifacts/1550nm_height_family_n15263_v2"


def main():
    archive = FOLDER / "height_specific_NN_proxy_candidates.zip"
    manifest_path = FOLDER / "delivery_manifest.json"
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("Preserve the existing delivery; use a separately versioned package.")
    gate = json.loads((FOLDER / "proxy_family_gate.json").read_text())
    if (gate["status"] != "PROXY_FAMILY_COMPLETE_INDEPENDENT_VALIDATION_PENDING"
            or gate["unique_quantized_requested_profiles"] != 7
            or gate["proxy_sampling_status"] != "PASS_PROXY_SAMPLING"
            or not gate["protected_evidence_unchanged"]):
        raise ValueError("Completed nominal proxy-family evidence is required.")
    names = [
        "REPORT.txt", "config.json", "provenance.json", "analytic_feasibility.json",
        "NN_height_profiles_um.csv", "NN_height_summary.csv", "height_family_proxy_results.png",
        "proxy_family_gate.json", "proxy_sampling.json", "proxy_sampling_attempt1_failure.json",
        "training_result.json", "selection_before_heldout.json", "all_target_metadata.json",
        "loss_tests.xml", "pilot_gradient.json", "pilot_gate.json",
        "pilot_execution_completed.json", "train_execution_completed.json",
        "pilot_source_manifest.json", "train_source_manifest.json",
        "maxwell_adapter_tests_final.xml", "MAXWELL_VALIDATION_PROPOSAL.txt",
        "maxwell_validation_proposal.json",
    ]
    names += [f"designs/H{height}/NN.json" for height in range(40, 101, 10)]
    names += [p.relative_to(FOLDER).as_posix() for p in sorted((FOLDER / "source_snapshots").rglob("*")) if p.is_file()]
    files = {}
    for name in names:
        data = (FOLDER / name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate["status"],
        "scope": "Seven distinct top-relief candidates for cylinder H = 40-100 um; scalar proxy evidence only.",
        "limitations": [
            "Independent Maxwell validation awaits an explicit cumulative budget amendment.",
            "No global diffraction optimum or full M2 acceptance is established.",
            "The frozen H100 profile fails the nominal curvature criterion at MFD 9.9 um.",
            "This archive is not an SLC exposure job or a calibrated fabrication release.",
        ],
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for name in names:
            stream.write(FOLDER / name, arcname=name)
        stream.write(manifest_path, arcname=manifest_path.name)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for name, item in files.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != item["sha256"]:
                raise ValueError(f"Archive content mismatch: {name}")
        assert json.loads(stream.read("delivery_manifest.json")) == manifest
    receipt = {
        "status": "PASS_ARCHIVE_INTEGRITY",
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "payload_files": len(files),
        "all_payload_hashes_verified": True,
        "physics_rerun": False,
    }
    (FOLDER / "delivery_integrity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
