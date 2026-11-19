"""Export and independently audit the desktop app's attached 500 nm design."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

import audit_metalens_slc as independent
import export_height_family_attached_slc_v1 as writer


LAYER_UM = 0.5
CYLINDER_DIAMETER_UM = 50.0
ANNULI = 50
AZIMUTHAL_SEGMENTS = 128
PACKAGED_BUILDER = writer.ROOT / 'vendor' / 'slc_builder_subset.py'
PACKAGED_BUILDER_SHA256 = 'b23d330788ee21bc1c24d49142362313ed8884d356ba7725cfd20128e22d1220'


def validate_height(height_um: float) -> float:
    """Keep the cylinder–relief interface and every step on the SLC grid."""
    try:
        height = float(height_um)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Enter a cylinder height between 40 and 100 µm.") from exc
    if not math.isfinite(height) or not 40 <= height <= 100:
        raise ValueError("Enter a cylinder height between 40 and 100 µm.")
    if not (height / LAYER_UM).is_integer():
        raise ValueError("Use a cylinder height in 0.5 µm increments to preserve uniform 500 nm SLC layers.")
    return height


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def _audit_json(value):
    """Normalize the independent audit's NumPy flags and failed measurements."""
    if isinstance(value, np.generic):
        return _audit_json(value.item())
    if isinstance(value, dict):
        return {key: _audit_json(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_audit_json(entry) for entry in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _design(height_um: float, heights_500nm_um: Sequence[float], provenance: dict | None) -> dict:
    height = validate_height(height_um)
    heights = np.asarray(heights_500nm_um, dtype=float)
    if heights.shape != (ANNULI,) or not np.isfinite(heights).all():
        raise ValueError("The predicted design must contain 50 finite annular heights.")
    if np.any(heights < 0) or np.any(heights > 3) or np.any(heights % LAYER_UM != 0):
        raise ValueError("The fabrication profile needs 0–3 µm relief on the 500 nm grid.")
    if heights[0] <= 0 or np.any(np.diff(heights) > 0):
        raise ValueError("The independent SLC audit requires a positive central relief with nonincreasing heights.")
    return {
        "kind": "desktop_nn_500nm_fabrication",
        "units": "um",
        "H_um": height,
        "cylinder_diameter_um": CYLINDER_DIAMETER_UM,
        "heights_um": heights.tolist(),
        "height_quantization_um": LAYER_UM,
        "inference_provenance": provenance,
    }


def _publish(files: Sequence[tuple[Path, bytes]]) -> None:
    """Publish sidecars first, avoiding overwrite even if another writer races."""
    created = []
    try:
        for path, data in files:
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except Exception:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def _keep_rejected(folder: Path, raw: bytes, source: bytes, audit: dict) -> Path:
    """Keep independent-audit failures identifiable and away from deliverables."""
    rejected = Path(tempfile.mkdtemp(prefix="rejected-metalens-", dir=folder))
    (rejected / "design.slc.rejected").write_bytes(raw)
    (rejected / "source.json").write_bytes(source)
    (rejected / "audit.json").write_bytes(_json_bytes(_audit_json(audit)))
    return rejected


def export_slc(
    destination: Path,
    height_um: float,
    heights_500nm_um: Sequence[float],
    *,
    inference_provenance: dict | None = None,
) -> dict:
    """Export one exact user-named SLC and two geometry verification sidecars."""
    design = _design(height_um, heights_500nm_um, inference_provenance)
    source = _json_bytes(design)
    destination = Path(destination).expanduser().absolute()
    if destination.suffix.lower() != ".slc":
        raise ValueError("Choose a filename ending in .slc.")
    if not destination.parent.is_dir():
        raise ValueError("Choose an existing folder for the SLC export.")
    audit_path = destination.with_suffix(".audit.json")
    manifest_path = destination.with_suffix(".manifest.json")
    for path in (destination, audit_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path.name} already exists. Choose a new SLC filename.")

    core, assembly = writer.load_helpers()
    portable = PACKAGED_BUILDER.is_file() or (writer.ROOT / 'runtime').is_dir()
    builder_path = PACKAGED_BUILDER if portable else writer.BUILDER
    expected_builder_sha = PACKAGED_BUILDER_SHA256 if portable else writer.BUILDER_SHA
    if not builder_path.is_file() or writer.sha(builder_path) != expected_builder_sha:
        raise ValueError("The installed SLC builder differs from the reviewed minimum-Z converter.")
    builder = core.load_builder(builder_path)
    if builder.source_sha256 != expected_builder_sha:
        raise ValueError("The SLC builder changed during loading.")

    with tempfile.TemporaryDirectory(prefix=".metalens-export-", dir=destination.parent) as temp_dir:
        staging = Path(temp_dir)
        source_path = staging / "source.json"
        source_path.write_bytes(source)
        # The native builder's short internal filename policy applies only in staging.
        manifest = writer.export_component(
            design, source_path, staging / "assembly", builder, core, assembly,
            "D01.slc", layer_um=LAYER_UM, make_preview=False,
        )
        raw = Path(manifest["final_slc"]).read_bytes()
        try:
            audit = independent.audit(raw, source, LAYER_UM, AZIMUTHAL_SEGMENTS, "assembly")
        except ValueError as exc:
            rejected = _keep_rejected(destination.parent, raw, source, {"status": "FAIL", "error": str(exc)})
            raise RuntimeError(f"SLC geometry audit could not read the export. Evidence: {rejected}") from exc
        if audit["status"] != "PASS":
            rejected = _keep_rejected(destination.parent, raw, source, audit)
            failed = ", ".join(audit["failed_checks"])
            raise RuntimeError(f"SLC geometry audit failed ({failed}). Evidence: {rejected}")

    slc_sha = hashlib.sha256(raw).hexdigest()
    audit["provenance"] = {
        "slc_sha256": slc_sha,
        "source_design_sha256": hashlib.sha256(source).hexdigest(),
        "source_design_manifest": manifest_path.name,
        "source_design_manifest_field": "source_design_exact_json_utf8",
        "builder_source_sha256": builder.source_sha256,
        "original_builder_source_sha256": writer.BUILDER_SHA,
        "binary_auditor_sha256": writer.sha(Path(independent.__file__)),
    }
    audit_bytes = _json_bytes(_audit_json(audit))
    manifest.pop("actual_builder_output")
    manifest.pop("source_design")
    manifest.pop("independent_audit_required")
    manifest.update({
        "final_slc": str(destination),
        "source_design_exact_json_utf8": source.decode("utf-8"),
        "source_design": design,
        "original_builder_source_sha256": writer.BUILDER_SHA,
        "independent_audit_status": "PASS",
        "independent_audit": audit_path.name,
        "independent_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
    })
    _publish(((audit_path, audit_bytes), (manifest_path, _json_bytes(manifest)), (destination, raw)))
    return {
        "slc_path": str(destination),
        "audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        "audit_status": "PASS",
        "audit_check_count": len(audit["checks"]),
        "slc_sha256": slc_sha,
        "layer_count": manifest["layer_count"],
        "layer_interval_um": LAYER_UM,
        "actual_slab_interval_min_um": manifest["actual_slab_interval_min_um"],
        "actual_slab_interval_max_um": manifest["actual_slab_interval_max_um"],
        "cylinder_height_um": manifest["cylinder_height_um"],
        "cylinder_diameter_um": manifest["cylinder_diameter_um"],
        "maximum_relief_um": manifest["maximum_relief_um"],
        "total_height_um": manifest["total_height_um"],
    }
