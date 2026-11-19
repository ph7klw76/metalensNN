"""Supported annular geometry in physical micrometres."""
import json
from pathlib import Path
import numpy as np


def load_config(path="configs/desktop_trial.json"):
    return json.loads(Path(path).read_text())


def complex_index(material):
    return complex(material["n"], material["kappa"])


def epsilon(material):
    return complex_index(material)**2


def edges(config):
    return np.linspace(0, config["aperture_diameter_um"]/2, config["annuli"]+1)


def project(heights, config):
    step = config["fabrication"]["height_step_um"]
    return np.clip(np.round(np.asarray(heights)/step)*step, 0, config["height_max_um"])


def feasible(heights, config):
    h = np.asarray(heights)
    width = np.diff(edges(config))
    fab = config["fabrication"]
    errors = []
    if h.shape != (config["annuli"],) or not np.all(np.isfinite(h)):
        return ["invalid height array"]
    if np.any(h < -1e-7) or np.any(h > config["height_max_um"]+1e-7):
        errors.append("height bounds")
    if np.any(width[h > 0] < fab["min_width_um"]):
        errors.append("minimum width")
    if np.any(h/width > fab["max_aspect_ratio"]+1e-7):
        errors.append("aspect ratio")
    step = fab["height_step_um"]
    if np.max(np.abs(h/step-np.round(h/step))) > 1e-4:
        errors.append("height grid")
    empty = h == 0
    start = None
    for i, zero in enumerate(np.r_[empty, False]):
        if zero and start is None:
            start = i
        elif not zero and start is not None:
            # Only interior empty runs are gaps between two printed rings.
            if start > 0 and i < len(h) and np.sum(width[start:i])+1e-7 < fab["min_gap_um"]:
                errors.append("minimum gap")
            start = None
    return errors


def fresnel_heights(focal_um, config, xp=np):
    focal_um = focal_um+config["training"].get("fresnel_seed_offset_um",0.0)
    r = xp.asarray((edges(config)[:-1]+edges(config)[1:])/2)
    n = config["materials"]["ambient"]["n"]
    contrast = config["materials"]["polymer"]["n"]-n
    if contrast <= 0:
        raise ValueError("Fresnel baseline requires positive declared index contrast")
    k0 = 2*xp.pi/config["wavelength_um"]
    phase = xp.mod(-k0*n*(xp.sqrt(r*r+focal_um*focal_um)-focal_um), 2*xp.pi)
    return xp.clip(phase/(k0*contrast), 0, config["height_max_um"])


def baselines(focal_um, config):
    return {"no_lens": np.zeros(config["annuli"]),
            "flat": np.full(config["annuli"], config["flat_height_um"]),
            "fresnel": project(fresnel_heights(focal_um, config), config)}


def profile_polygons(heights, config):
    """Exact r,z outlines of connected positive-height portions, without seams."""
    e = edges(config)
    result = []
    first = 0
    while first < len(heights):
        if heights[first] <= 0:
            first += 1
            continue
        end = first+1
        while end < len(heights) and heights[end] > 0:
            end += 1
        points = [(e[first],0),(e[end],0)]
        for i in range(end-1,first-1,-1):
            for point in [(e[i+1],heights[i]),(e[i],heights[i])]:
                if point != points[-1]:
                    points.append(point)
        # Remove collinear vertices, including adjacent equal-height annuli.
        changed = True
        while changed and len(points)>3:
            changed = False
            for i in range(len(points)):
                a,b,c = points[i-1],points[i],points[(i+1)%len(points)]
                if abs((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])) < 1e-12:
                    points.pop(i)
                    changed = True
                    break
        result.append(points)
        first = end
    return result


def epsilon_rz(heights, r, z, config):
    """Binary polymer/background plus explicit substrate and PEC stop mask."""
    h = np.asarray(heights)
    r, z = np.asarray(r), np.asarray(z)
    index = np.clip(np.searchsorted(edges(config), r, side="right")-1, 0, len(h)-1)
    inside = r < config["aperture_diameter_um"]/2
    polymer = inside[:,None] & (z[None,:] >= 0) & (z[None,:] < h[index,None])
    substrate = z[None,:] < 0
    eps = np.full((len(r),len(z)), epsilon(config["materials"]["ambient"]), dtype=complex)
    eps = np.where(substrate, epsilon(config["materials"]["substrate"]), eps)
    eps = np.where(polymer, epsilon(config["materials"]["polymer"]), eps)
    thickness = config["illumination"]["aperture_stop_thickness_um"]
    pec = (~inside[:,None]) & (z[None,:] >= -thickness) & (z[None,:] < 0)
    return eps, polymer, pec


def geometric_na(focal_um, config, substrate_plane=False):
    radius = config["aperture_diameter_um"]/2
    distance = focal_um+(config["exit_plane_z_um"] if substrate_plane else 0)
    return config["materials"]["ambient"]["n"]*radius/np.hypot(radius,distance)


def save_design(path, heights, focal_um, kind, config, **metadata):
    heights = project(heights, config)
    errors = feasible(heights, config)
    if errors:
        raise ValueError(errors)
    output = {"result_label":config["result_label"],"kind":kind,"units":"um",
              "target_focal_um":float(focal_um),"reference_plane_z_um":config["exit_plane_z_um"],
              "ring_edges_um":edges(config).tolist(),"heights_um":heights.tolist(),
              "feasible":True,"materials":config["materials"],**metadata}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(output,indent=2)+"\n")
    return output
