"""Metric extraction from saved cylindrical vector DFT slices."""
from pathlib import Path
import numpy as np
from .metrics import peak_position, width_metrics


def analyze_fields(report):
    if "fields_file" not in report:
        return report
    path = Path(report["fields_file"])
    with np.load(path) as source:
        data = {key:source[key] for key in source.files}
    er,ep,ez = data["Er"],data["Ep"],data["Ez"]
    # Public point DFTs at r=0 can return zeros at half-grid z positions
    # in cylindrical Meep. A nonempty radial slice remains well defined.
    # Its first centered Yee voxel approximates the axis within one cell.
    axial = abs(er[:,0])**2+abs(ez[:,0])**2
    focus,boundary = peak_position(data["focus_scan_um"],axial)
    index = int(np.argmax(axial))
    r = data["r_um"]
    x = np.concatenate([-r[:0:-1],r])
    ix,iy = abs(er[index])**2+abs(ez[index])**2,abs(ep[index])**2
    if "axis_point_intensity_raw" not in data:
        data["axis_point_intensity_raw"] = data["axis_intensity"].copy()
    data["axis_intensity"] = axial
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**data)
    temporary.replace(path)
    result = dict(report)
    if "raw_point_monitor_focal_um" not in result:
        result["raw_point_monitor_focal_um"] = report["focal_um"]
    result.update(focal_um=float(focus),focus_on_scan_boundary=bool(boundary),
                  focus_definition="maximum Er/Ez intensity in first centered radial Yee voxel; near-axis radius <= one cell",
                  near_axis_radius_upper_bound_um=1/report["settings"]["resolution"],
                  metric_analysis_version=2,
                  widths=width_metrics(x,np.concatenate([ix[:0:-1],ix]),np.concatenate([iy[:0:-1],iy])),
                  width_axial_plane_um=float(data["focus_scan_um"][index]),
                  sidelobe_peak_ratio_x=float(np.max(ix[r>=1.1])/max(np.max(ix),1e-30)),
                  sidelobe_peak_ratio_y=float(np.max(iy[r>=1.1])/max(np.max(iy),1e-30)))
    return result
