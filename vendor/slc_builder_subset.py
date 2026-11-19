"""SLC source subset extracted from the audited fiber builder. The app compiles these functions in a dedicated namespace."""
EXPORT_ORIGIN_MODE = "top_physical_end_center"

UM_TO_MM = 0.001

SLC_XY_MM_PER_UM = 0.24

SLC_Z_MM_PER_UM = 0.10

SLC_LAYER_THICKNESS_UM = 1.00

SLC_Z_ORIGIN = "preserve"

SLC_XY_ORIGIN = "preserve"

SLC_FIRST_LAYER_FRACTION = 0.5

SLC_MAX_LAYERS = 100000

SLC_MAX_TRIANGLES = 2000000

SLC_MAX_FILENAME_LENGTH = 15

def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value

def vecnorm(values, axis=None):
    """Euclidean norm without importing/using numpy.linalg.

    This intentionally stays in NumPy core ufuncs so a standalone build does
    not need to enter the LAPACK/BLAS-backed numpy.linalg path merely to take
    vector norms.
    """
    array = np.asarray(values, dtype=float)
    return np.sqrt(np.sum(array * array, axis=axis))

def slc_options(layer_um=None, z_origin=None, *, xy_mm_per_um=None,
                z_mm_per_um=None, xy_origin=None):
    values = {
        "layer_thickness_um": SLC_LAYER_THICKNESS_UM if layer_um is None else layer_um,
        "xy_mm_per_source_um": SLC_XY_MM_PER_UM if xy_mm_per_um is None else xy_mm_per_um,
        "z_mm_per_source_um": SLC_Z_MM_PER_UM if z_mm_per_um is None else z_mm_per_um,
        "z_origin": SLC_Z_ORIGIN if z_origin is None else z_origin,
        "xy_origin": SLC_XY_ORIGIN if xy_origin is None else xy_origin,
        "first_layer_fraction": SLC_FIRST_LAYER_FRACTION,
        "max_layers": SLC_MAX_LAYERS,
        "options_version": 2,
    }
    return slc_normalize_options(values)

def slc_normalize_options(options):
    options = dict(options)
    options.setdefault("xy_mm_per_source_um", SLC_XY_MM_PER_UM)
    options.setdefault("z_mm_per_source_um", SLC_Z_MM_PER_UM)
    options.setdefault("xy_origin", SLC_XY_ORIGIN)
    options.setdefault("z_origin", SLC_Z_ORIGIN)
    options.setdefault("layer_thickness_um", SLC_LAYER_THICKNESS_UM)
    options.setdefault("first_layer_fraction", SLC_FIRST_LAYER_FRACTION)
    options.setdefault("max_layers", SLC_MAX_LAYERS)
    for key in ("layer_thickness_um", "xy_mm_per_source_um", "z_mm_per_source_um"):
        if isinstance(options[key], (bool, np.bool_)):
            raise ValueError(key + " must be a positive numeric value, not a boolean.")
        options[key] = positive(options[key], key)
    phase = float(options["first_layer_fraction"])
    if not math.isfinite(phase) or not 0 < phase < 1:
        raise ValueError("SLC_FIRST_LAYER_FRACTION must lie strictly between 0 and 1.")
    options["first_layer_fraction"] = phase
    if options["z_origin"] not in {"preserve", "minimum"}:
        raise ValueError("SLC_Z_ORIGIN must be 'preserve' or 'minimum'.")
    if options["xy_origin"] not in {"preserve", "center", "minimum"}:
        raise ValueError("SLC_XY_ORIGIN must be 'preserve', 'center', or 'minimum'.")
    maximum = options["max_layers"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("SLC layer-count limit must be a positive integer.")
    options["options_version"] = 2
    return options

def slc_companion_names(stem):
    return (stem + ".slc", stem + "_Support.slc", stem + ".3dp",
            stem + "_slc_manifest.json", stem + ".slc.tmp")

def short_slc_path(path):
    """Reserve enough filename space for the exact _Support.slc suffix."""
    path = Path(path)
    maximum = SLC_MAX_FILENAME_LENGTH
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 13 <= maximum <= 15:
        raise ValueError("SLC_MAX_FILENAME_LENGTH must be 13 to 15 with _Support.slc.")
    budget = maximum - len("_Support.slc")
    stem = "".join(c for c in path.stem if c.isascii() and c.isalnum())
    prefix = stem[0].upper() if stem and stem[0].isalpha() else "F"
    reserved = {"CON", "PRN", "AUX", "NUL"}
    occupied = {p.name.casefold() for p in path.parent.iterdir()} if path.parent.exists() else set()

    def available(name):
        return (name.upper() not in reserved and
                not any(n.casefold() in occupied for n in slc_companion_names(name)))

    if stem and len(stem) <= budget and available(stem):
        return path.with_name(stem + ".slc")
    if budget == 1:
        candidates = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    else:
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        candidates = []
        for index in range(1, 36 ** (budget - 1)):
            value, suffix = index, ""
            for _ in range(budget - 1):
                value, remainder = divmod(value, 36)
                suffix = digits[remainder] + suffix
            candidates.append(prefix + suffix)
    for name in candidates:
        if available(name):
            return path.with_name(name + ".slc")
    raise FileExistsError("No unused short SLC/support/3DP filename remains in this folder.")

def slc_output_path(stl_path, options, output_folder=None):
    slc_normalize_options(options)
    path = Path(stl_path)
    folder = path.parent if output_folder is None else Path(output_folder)
    parts = path.stem.split("_")
    prefix = "F"
    if (len(parts) >= 3 and parts[0].startswith("T") and parts[1].startswith("B")
            and parts[0][1:].isascii() and parts[0][1:].isdigit()
            and parts[1][1:].isascii() and parts[1][1:].isdigit()
            and parts[2] in {"center", "left", "right"}):
        prefix = {"center": "C", "left": "L", "right": "R"}[parts[2]]
    return short_slc_path(folder / (prefix + "01.slc"))

def slc_transform_mesh(mesh, options):
    options = slc_normalize_options(options)
    factors = np.array([options["xy_mm_per_source_um"],
                        options["xy_mm_per_source_um"],
                        options["z_mm_per_source_um"]], dtype=np.float64) / UM_TO_MM
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = np.asarray(mesh["vertices"], dtype=np.float64) * factors
    if not np.isfinite(vertices).all():
        raise ValueError("SLC scaling overflow: reduce output scale or inspect STL units.")
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    translation = np.zeros(3, dtype=np.float64)
    if options["xy_origin"] == "center":
        translation[:2] = -(lower[:2] / 2 + upper[:2] / 2)
    elif options["xy_origin"] == "minimum":
        translation[:2] = -lower[:2]
    if options["z_origin"] == "minimum":
        translation[2] = -lower[2]
    vertices = vertices + translation
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if (not np.isfinite(vertices).all() or np.any(upper <= lower)
            or np.max(np.abs(vertices)) > np.finfo(np.float32).max):
        raise ValueError("Scaled SLC extents are invalid or exceed float32 range.")
    result = dict(mesh)
    result["vertices"] = vertices
    result["bounds_mm"] = [lower.tolist(), upper.tolist()]
    result["volume_mm3"] = float(mesh["volume_mm3"] * np.prod(factors))
    if not math.isfinite(result["volume_mm3"]) or result["volume_mm3"] <= 0:
        raise ValueError("Scaled SLC volume is invalid.")
    return result, factors, translation

def slc_load_stl_mm(path):
    path = Path(path)
    size = path.stat().st_size
    if size < 15 or size > 512 * 1024 * 1024:
        raise ValueError("STL is empty, too short, or exceeds the 512 MiB safety limit.")
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else 0
    if count and len(raw) == 84 + count * 50:
        if count > SLC_MAX_TRIANGLES:
            raise ValueError("STL triangle count exceeds SLC_MAX_TRIANGLES.")
        dtype = np.dtype([("normal", "<f4", (3,)), ("xyz", "<f4", (3, 3)), ("attr", "<u2")])
        triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)["xyz"].astype(np.float64)
        encoding = "binary_little_endian"
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("STL is neither valid-size binary nor UTF-8/ASCII STL.") from exc

        triangles, current = [], []
        state, solids = "outside", 0
        for line_number, line in enumerate(text.splitlines(), 1):
            words = line.strip().split()
            if not words:
                continue
            lower = [word.lower() for word in words]
            try:
                if state == "outside" and lower[0] == "solid":
                    state = "in_solid"; solids += 1
                elif state == "in_solid" and lower[0] == "endsolid":
                    state = "outside"
                elif state == "in_solid" and lower[:2] == ["facet", "normal"] and len(words) == 5:
                    if not all(math.isfinite(float(v)) for v in words[2:]):
                        raise ValueError("Nonfinite facet normal")
                    current = []; state = "loop"
                elif state == "loop" and lower == ["outer", "loop"]:
                    state = "vertices"
                elif state == "vertices" and lower[0] == "vertex" and len(words) == 4 and len(current) < 3:
                    current.append([float(v) for v in words[1:]])
                elif state == "vertices" and lower == ["endloop"] and len(current) == 3:
                    state = "endfacet"
                elif state == "endfacet" and lower == ["endfacet"]:
                    triangles.append(current); state = "in_solid"
                    if len(triangles) > SLC_MAX_TRIANGLES:
                        raise ValueError("Too many triangles")
                else:
                    raise ValueError("Unexpected token or malformed facet")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Malformed ASCII STL at line {line_number}: {exc}") from exc
        if state != "outside" or not solids or not triangles:
            raise ValueError("Incomplete or empty ASCII STL.")
        triangles = np.asarray(triangles, dtype=np.float64)
        encoding = "ascii"
    if len(triangles) < 4 or not np.isfinite(triangles).all():
        raise ValueError("STL must have at least four finite triangles.")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    spans = np.ptp(vertices, axis=0)
    if np.any(spans <= 0):
        raise ValueError("STL must enclose a nonzero 3-D volume.")
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(vecnorm(vectors, axis=1) <= 1e-22):
        raise ValueError("STL has a degenerate triangle; repair it before conversion.")
    directed = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    sorted_edges = np.sort(directed, axis=1)
    edges, edge_ids, uses = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(uses != 2):
        raise ValueError("STL is not closed two-manifold: each edge must belong to exactly two faces. No gap repair is performed.")
    winding = np.bincount(edge_ids, weights=np.where(directed[:, 0] < directed[:, 1], 1., -1.), minlength=len(edges))
    if np.any(winding != 0):
        raise ValueError("STL has inconsistent triangle winding. Fix normals/face order before slicing.")
    shifted = triangles - vertices.mean(axis=0)
    volume = float(np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6)
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("STL has nonpositive oriented volume. Inverted solids are not automatically repaired.")
    return {"vertices": vertices, "faces": faces, "edges": edges,
            "face_edge_ids": edge_ids.reshape(-1, 3), "volume_mm3": volume,
            "encoding": encoding, "bounds_mm": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]}

def slc_mesh_from_arrays(vertices, faces, volume_mm3=None):
    """Build the validated SLC mesh structure directly from in-memory geometry.

    This avoids writing a temporary STL and parsing/deduplicating it again.  The
    same closed-manifold, winding, degeneracy and positive-volume checks used by
    ``slc_load_stl_mm`` are retained.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("In-memory mesh vertices must have shape (N,3) with N >= 4.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("In-memory mesh faces must have shape (M,3) with M >= 4.")
    if len(faces) > SLC_MAX_TRIANGLES:
        raise ValueError("Mesh triangle count exceeds SLC_MAX_TRIANGLES.")
    if not np.isfinite(vertices).all():
        raise ValueError("In-memory mesh contains nonfinite vertices.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("In-memory mesh face index is outside the vertex array.")

    triangles = vertices[faces]
    spans = np.ptp(vertices, axis=0)
    if np.any(spans <= 0):
        raise ValueError("Mesh must enclose a nonzero 3-D volume.")
    vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(vecnorm(vectors, axis=1) <= 1e-22):
        raise ValueError("Mesh has a degenerate triangle.")

    directed = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    sorted_edges = np.sort(directed, axis=1)
    edges, edge_ids, uses = np.unique(sorted_edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(uses != 2):
        raise ValueError("Mesh is not closed two-manifold: each edge must belong to exactly two faces.")
    winding = np.bincount(
        edge_ids,
        weights=np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0),
        minlength=len(edges),
    )
    if np.any(winding != 0):
        raise ValueError("Mesh has inconsistent triangle winding.")

    shifted = triangles - vertices.mean(axis=0)
    calculated_volume = float(
        np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6
    )
    if not math.isfinite(calculated_volume) or calculated_volume <= 0:
        raise ValueError("Mesh has nonpositive oriented volume.")
    if volume_mm3 is not None and not math.isclose(
        calculated_volume, float(volume_mm3), rel_tol=2e-6, abs_tol=1e-15
    ):
        raise ValueError("In-memory mesh volume changed before SLC conversion.")

    return {
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "face_edge_ids": edge_ids.reshape(-1, 3),
        "volume_mm3": calculated_volume,
        "encoding": "in_memory_mesh",
        "bounds_mm": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }

def slc_mesh_sha256(mesh):
    """Fingerprint the exact in-memory mesh used for SLC conversion."""
    value = hashlib.sha256()
    for key, dtype in (("vertices", "<f8"), ("faces", "<i8")):
        array = np.ascontiguousarray(mesh[key], dtype=dtype)
        value.update(key.encode("ascii"))
        value.update(struct.pack("<II", array.ndim, array.shape[0]))
        value.update(struct.pack("<I", array.shape[1]))
        value.update(array.tobytes(order="C"))
    return value.hexdigest()

def slc_signed_area(loop):
    points = np.asarray(loop, float)
    points = points - points[0]
    return float(.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                              points[:, 1] * np.roll(points[:, 0], -1)))

def slc_point_inside(point, loop):
    p = np.asarray(loop, float)
    q = np.roll(p, -1, axis=0)
    x, y = point
    crossing = (p[:, 1] > y) != (q[:, 1] > y)
    if not crossing.any():
        return False
    pp, qq = p[crossing], q[crossing]
    xx = pp[:, 0] + (y - pp[:, 1]) * (qq[:, 0] - pp[:, 0]) / (qq[:, 1] - pp[:, 1])
    return bool(np.count_nonzero(xx > x) % 2)

def slc_check_planar_loops(loops):
    if not loops:
        return
    aa, bb, owners, indices = [], [], [], []
    for owner, loop in enumerate(loops):
        p = np.asarray(loop, dtype=np.float64)
        if len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("SLC loop has fewer than three finite, unique vertices.")
        q = np.roll(p, -1, axis=0)
        if np.any(np.all(p == q, axis=1)) or abs(slc_signed_area(p)) <= 1e-24:
            raise ValueError("SLC slice contains a zero edge or zero-area contour.")
        aa.extend(p); bb.extend(q); owners.extend([owner] * len(p)); indices.extend(range(len(p)))
    aa, bb = np.asarray(aa), np.asarray(bb)
    owners, indices = np.asarray(owners), np.asarray(indices)
    low, high = np.minimum(aa, bb), np.maximum(aa, bb)

    axis = int(np.argmax(np.ptp(np.vstack((aa, bb)), axis=0)))
    order = np.argsort(low[:, axis], kind="stable")
    aa, bb, low, high, owners, indices = [v[order] for v in (aa, bb, low, high, owners, indices)]
    sizes = np.array([len(p) for p in loops])
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    for i in range(len(aa) - 1):
        stop = int(np.searchsorted(low[:, axis], high[i, axis], side="right"))
        if stop <= i + 1:
            continue
        js = np.arange(i + 1, stop)
        hit = np.all(low[js] <= high[i], axis=1) & np.all(high[js] >= low[i], axis=1)
        adjacent = ((owners[js] == owners[i]) &
                    ((np.abs(indices[js] - indices[i]) == 1) |
                     (np.abs(indices[js] - indices[i]) == sizes[owners[i]] - 1)))
        js = js[hit & ~adjacent]
        if not len(js):
            continue
        u, v = bb[i] - aa[i], bb[js] - aa[js]
        s1, s2 = cross(u, aa[js] - aa[i]), cross(u, bb[js] - aa[i])
        s3, s4 = cross(v, aa[i] - aa[js]), cross(v, bb[i] - aa[js])
        opposite1 = ((s1 <= 0) & (s2 >= 0)) | ((s1 >= 0) & (s2 <= 0))
        opposite2 = ((s3 <= 0) & (s4 >= 0)) | ((s3 >= 0) & (s4 <= 0))
        if np.any(opposite1 & opposite2):
            raise ValueError("A slice contour self-intersects or touches/crosses another contour. No repair or gap stitching was applied.")

def slc_orient_and_quantize(loops):
    quantized, max_error = [], 0.
    for loop in loops:
        array = np.asarray(loop, np.float64)
        rounded = array.astype("<f4")
        if not np.isfinite(rounded).all():
            raise ValueError("SLC float32 coordinate overflow.")
        max_error = max(max_error, float(np.max(np.abs(rounded.astype(float) - array))))


        keep = np.r_[True, np.any(rounded[1:] != rounded[:-1], axis=1)]
        rounded = rounded[keep]
        if len(rounded) > 1 and np.array_equal(rounded[0], rounded[-1]):
            rounded = rounded[:-1]
        quantized.append(rounded)
    slc_check_planar_loops(quantized)
    oriented = []
    for i, loop in enumerate(quantized):
        depth = sum(slc_point_inside(loop[0], other) for j, other in enumerate(quantized) if i != j)
        area = slc_signed_area(loop)
        exterior = depth % 2 == 0
        if (area > 0) != exterior:
            loop = loop[::-1].copy()

        first = int(np.lexsort((loop[:, 1], loop[:, 0]))[0])
        loop = np.roll(loop, -first, axis=0)
        closed = np.vstack((loop, loop[0])).astype("<f4")
        oriented.append({"xy": closed, "depth": int(depth), "signed_area_mm2": slc_signed_area(loop)})
    oriented.sort(key=lambda item: (item["depth"], float(item["xy"][0, 0]), float(item["xy"][0, 1])))
    return oriented, max_error

def slc_slice_at_z(mesh, z):
    vertices, edges = mesh["vertices"], mesh["edges"]
    above = vertices[:, 2] > z
    crossing = above[edges[:, 0]] != above[edges[:, 1]]
    crossed = np.flatnonzero(crossing)
    if not len(crossed):
        return [], 0.
    a, b = vertices[edges[crossed, 0]], vertices[edges[crossed, 1]]
    fraction = (z - a[:, 2]) / (b[:, 2] - a[:, 2])
    xy = a[:, :2] + fraction[:, None] * (b[:, :2] - a[:, :2])
    node_ids = len(vertices) + crossed
    on_a, on_b = a[:, 2] == z, b[:, 2] == z
    node_ids[on_a] = edges[crossed[on_a], 0]
    node_ids[on_b] = edges[crossed[on_b], 1]
    edge_node = np.full(len(edges), -1, dtype=np.int64)
    edge_node[crossed] = node_ids
    coords = {int(node): value for node, value in zip(node_ids, xy)}
    triangle_crossings = crossing[mesh["face_edge_ids"]]
    counts = triangle_crossings.sum(axis=1)
    if np.any((counts != 0) & (counts != 2)):
        raise ValueError("Unexpected triangle/plane intersection topology.")
    selected = mesh["face_edge_ids"][counts == 2]
    selected_mask = crossing[selected]
    nodes = edge_node[selected[selected_mask]].reshape(-1, 2)
    nodes = nodes[nodes[:, 0] != nodes[:, 1]]
    nodes = np.unique(np.sort(nodes, axis=1), axis=0)
    adjacency = {}
    for ia, ib in nodes:
        a_id, b_id = int(ia), int(ib)
        adjacency.setdefault(a_id, []).append(b_id)
        adjacency.setdefault(b_id, []).append(a_id)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError(f"Open/branched slice at Z={z:.12g} mm. Change slice phase/thickness or inspect the mesh; no gap is invented.")
    unseen = set(adjacency)
    loops = []
    while unseen:
        first = min(unseen)
        current, previous, ids = first, None, []
        while True:
            if current not in unseen:
                if current == first:
                    break
                raise ValueError("Slice graph does not form disjoint closed loops.")
            unseen.remove(current); ids.append(current)
            neighbors = adjacency[current]
            after = neighbors[0] if neighbors[0] != previous else neighbors[1]
            previous, current = current, after
        if len(ids) < 3:
            raise ValueError("Slice has a degenerate polygon.")
        loops.append(np.asarray([coords[node] for node in ids]))
    return slc_orient_and_quantize(loops)

def slc_readback(path):
    raw = Path(path).read_bytes()
    end = raw.find(b"\r\n\x1a", 0, 2048)
    if end < 0:
        raise ValueError("SLC header terminator missing within 2048 bytes.")
    header = raw[:end].decode("ascii")
    if "-SLCVER 2.0" not in header or "-UNIT MM" not in header or "-TYPE PART" not in header:
        raise ValueError("Unexpected SLC version, unit, or contour type.")
    cursor = end + 3
    if len(raw) < cursor + 257:
        raise ValueError("Truncated SLC reserved/sample-table area.")
    cursor += 256
    ntable = raw[cursor]; cursor += 1
    def unpack(fmt):
        nonlocal cursor
        size = struct.calcsize(fmt)
        if cursor + size > len(raw):
            raise ValueError("Truncated SLC record.")
        value = struct.unpack_from(fmt, raw, cursor); cursor += size
        return value
    if ntable != 1:
        raise ValueError("This exporter uses exactly one constant-thickness sampling table entry.")
    sampling = unpack("<4f")
    if not all(math.isfinite(v) for v in sampling) or sampling[1] <= 0 or sampling[2:] != (0., 0.):
        raise ValueError("Invalid SLC sampling table.")
    layers = []
    while True:
        z, count = unpack("<fI")
        if not math.isfinite(z):
            raise ValueError("Nonfinite SLC slice height.")
        if count == 0xffffffff:
            if not layers or z <= layers[-1]["z_mm"] or cursor != len(raw):
                raise ValueError("Invalid terminal height or trailing bytes.")
            terminal_z = z; break
        if count > (len(raw) - cursor) // 8:
            raise ValueError("Implausible SLC contour count.")
        if layers and z <= layers[-1]["z_mm"]:
            raise ValueError("SLC heights must increase strictly.")
        loops = []
        for _ in range(count):
            nvertex, ngap = unpack("<II")
            if nvertex < 4 or ngap != 0 or nvertex * 8 > len(raw) - cursor:
                raise ValueError("Invalid closed PART contour record.")
            coords = np.frombuffer(raw, dtype="<f4", count=nvertex * 2, offset=cursor).reshape(-1, 2).copy()
            cursor += nvertex * 8
            if not np.isfinite(coords).all() or not np.array_equal(coords[0], coords[-1]):
                raise ValueError("SLC contour must be finite and explicitly closed.")
            loops.append(coords)
        open_loops = [p[:-1] for p in loops]
        slc_check_planar_loops(open_loops)
        for i, p in enumerate(open_loops):
            depth = sum(slc_point_inside(p[0], q) for j, q in enumerate(open_loops) if i != j)
            if (slc_signed_area(p) > 0) != (depth % 2 == 0):
                raise ValueError("Exterior/hole contour orientation is incorrect.")
        layers.append({"z_mm": z, "loops": loops})
    if layers[0]["z_mm"] != sampling[0]:
        raise ValueError("First sampling-table Z differs from first contour Z.")

    zs = np.asarray([v["z_mm"] for v in layers])
    expected = sampling[0] + np.arange(len(zs)) * sampling[1]
    allowance = max(1e-12, float(np.max(np.abs(np.spacing(zs.astype(np.float32))))) * 4)
    if not np.allclose(zs, expected, rtol=0, atol=allowance):
        raise ValueError("SLC heights do not match the constant-thickness sampling table.")
    return {"header": header, "sampling_entry": list(sampling), "layers": layers,
            "terminal_z_mm": terminal_z, "file_bytes": len(raw)}

def slc_layer_schedule(lower_z, upper_z, thickness, phase, max_layers):
    """Return SLC minimum-Z records, interior sampling heights, and terminal Z.

    SLC records describe slabs beginning at the stored Z. Sampling inside each
    slab avoids coplanar mesh intersections without moving the material boundary.
    """
    first, top = float(np.float32(lower_z)), float(np.float32(upper_z))
    if not all(math.isfinite(v) for v in (first, top, thickness, phase)):
        raise ValueError("Nonfinite SLC layer schedule.")
    if not first < top or thickness <= 0 or not 0 < phase < 1:
        raise ValueError("Invalid SLC layer extent, spacing, or sample fraction.")
    estimate = max(1, int(math.ceil((top - first) / thickness)))
    if estimate > max_layers + 1:
        raise ValueError(f"SLC would need about {estimate} layers; safety limit is {max_layers}.")
    planes = np.asarray(first + np.arange(estimate) * thickness, dtype=np.float32).astype(float)
    # Do not create a microscopic extra slab solely from float32 roundoff at
    # the top when the physical height is an exact multiple of the spacing.
    rounding = 2 * max(abs(float(np.spacing(np.float32(first)))),
                       abs(float(np.spacing(np.float32(top)))))
    planes = planes[(planes == first) | (planes < top - rounding)]
    if not len(planes) or np.any(np.diff(planes) <= 0) or len(planes) > max_layers:
        raise ValueError("SLC layer count/spacing cannot be represented at this scale and origin.")
    ends = np.r_[planes[1:], top]
    lows = np.maximum(planes, lower_z)
    highs = np.minimum(ends, upper_z)
    samples = lows + phase * (highs - lows)
    if np.any(highs <= lows) or np.any(samples <= lows) or np.any(samples >= highs):
        raise ValueError("Cannot resolve an interior contour sample for every SLC slab.")
    return planes, samples, top

def _convert_stl_to_slc_only(stl_path=None, slc_path=None, options=None, source_mesh=None):
    options = slc_normalize_options(slc_options() if options is None else options)
    layer_um = options["layer_thickness_um"]
    phase, origin, max_layers = (options["first_layer_fraction"], options["z_origin"], options["max_layers"])
    if source_mesh is None:
        stl_path = Path(stl_path).resolve()
        if stl_path.suffix.lower() != ".stl":
            raise ValueError("Input must be an ORIGINAL mm-unit .stl, not a previously scaled .slc.")
        target = (Path(slc_path).resolve() if slc_path is not None
                  else slc_output_path(stl_path, options).resolve())
        if target == stl_path or target.suffix.lower() != ".slc":
            raise ValueError("Output must be a separate .slc file; the STL will not be overwritten.")
        source_label = str(stl_path)
    else:
        if slc_path is None:
            raise ValueError("An explicit .slc output path is required for in-memory mesh conversion.")
        stl_path = None
        target = Path(slc_path).resolve()
        if target.suffix.lower() != ".slc":
            raise ValueError("Output must have the .slc suffix.")
        source_label = "in-memory validated rectangular mesh"
    target = short_slc_path(target)
    manifest_path = target.with_name(target.stem + "_slc_manifest.json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"SLC output already exists: {target}. Use a new output filename/folder.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"A temporary export already exists: {temporary}. Inspect it before retrying.")
    audit = {"status": "slicing", "source_stl": None if stl_path is None else str(stl_path),
             "source_mesh": source_label, "output_slc": str(target),
             "options": options, "slc_version": "2.0", "units": "mm", "slice_axis": "+Z",
             "input_unit_assumption": "Source mesh coordinates are millimetres; output-only scaling applied once",
             "coordinate_policy": {"xy": options["xy_origin"], "z": origin},
             "machine_compatibility_verified": False, "chitubox_import_tested": False,
             "repair_performed": False, "exposure_or_support_generation": False}
    try:
        if source_mesh is None:
            audit["source_sha256"] = sha256(stl_path)
            source_mesh = slc_load_stl_mm(stl_path)
            source_fingerprint = audit["source_sha256"]
        else:
            source_fingerprint = slc_mesh_sha256(source_mesh)
            audit["source_mesh_sha256"] = source_fingerprint
        audit["source_bounds_mm"] = source_mesh["bounds_mm"]
        audit["source_triangles"] = int(len(source_mesh["faces"]))
        audit["source_volume_mm3"] = source_mesh["volume_mm3"]
        audit["source_stl_encoding"] = source_mesh["encoding"]
        mesh, factors, translation = slc_transform_mesh(source_mesh, options)
        lower, upper = np.array(mesh["bounds_mm"], dtype=np.float64)
        audit.update(
            export_origin_mode=EXPORT_ORIGIN_MODE,
            export_origin_definition="centre of the top physical fiber end face is (0,0,0) before and after SLC scaling",
            scaling_mm_per_source_um={"x": options["xy_mm_per_source_um"],
                                      "y": options["xy_mm_per_source_um"],
                                      "z": options["z_mm_per_source_um"]},
            slc_scale_xyz=factors.tolist(), slc_translation_mm=translation.tolist(),
            slc_bounds_mm=mesh["bounds_mm"], slc_size_mm=(upper - lower).tolist(),
            transformed_volume_mm3=mesh["volume_mm3"],
            transform_order="scale source STL coordinates, then translate the SLC-only copy",
            forward_transform="SLC_mm[axis] = STL_mm[axis] * slc_scale_xyz[axis] + slc_translation_mm[axis]",
            inverse_transform="STL_mm[axis] = (SLC_mm[axis] - slc_translation_mm[axis]) / slc_scale_xyz[axis]",
            source_reference_origin_in_slc_mm=translation.tolist(),
            source_layer_thickness_um=layer_um,
            uniform_scaling=bool(factors[0] == factors[1] == factors[2]),
        )

        requested_thickness = layer_um * options["z_mm_per_source_um"]
        if not math.isfinite(requested_thickness) or requested_thickness > np.finfo(np.float32).max:
            raise ValueError("Scaled layer thickness exceeds float32 range.")
        thickness = float(np.float32(requested_thickness))
        if not math.isfinite(thickness) or thickness <= 0 or thickness > upper[2] - lower[2]:
            raise ValueError("SLC layer thickness must be positive and no greater than the scaled Z extent.")
        planes, sample_planes, top = slc_layer_schedule(
            lower[2], upper[2], thickness, phase, max_layers)

        extents = f"{lower[0]:.9g},{upper[0]:.9g} {lower[1]:.9g},{upper[1]:.9g} {lower[2]:.9g},{upper[2]:.9g}"
        header = ("-SLCVER 2.0 -UNIT MM -TYPE PART -PACKAGE SquareFiberScaled2 "
                  "-EXTENTS " + extents + "\r\n\x1a").encode("ascii")
        if len(header) > 2048:
            raise ValueError("SLC header exceeds 2048 bytes.")
        layer_info, max_xy_error = [], 0.
        with temporary.open("xb") as stream:
            stream.write(header)
            stream.write(bytes(256))
            stream.write(struct.pack("<B4f", 1, planes[0], thickness, 0., 0.))
            for number, z in enumerate(planes):
                sample_z = float(sample_planes[number])
                contours, quant_error = slc_slice_at_z(mesh, sample_z)
                max_xy_error = max(max_xy_error, quant_error)
                stream.write(struct.pack("<fI", z, len(contours)))
                for contour in contours:
                    xy = contour["xy"]
                    stream.write(struct.pack("<II", len(xy), 0))
                    stream.write(xy.astype("<f4").tobytes(order="C"))
                original_z_mm = float((z - translation[2]) / factors[2])
                area = float(sum(p["signed_area_mm2"] for p in contours))
                layer_info.append({"index": number, "z_mm": float(z),
                                   "sample_z_mm": sample_z,
                                   "upper_z_mm": float(planes[number + 1] if number + 1 < len(planes) else top),
                                   "z_original_mm": original_z_mm,
                                   "z_original_um": original_z_mm / UM_TO_MM,
                                   "contours": len(contours),
                                   "vertices_including_closure": sum(len(p["xy"]) for p in contours),
                                   "net_area_mm2": area,
                                   "net_area_original_mm2": area / (factors[0] * factors[1]),
                                   "hole_count": sum(p["depth"] % 2 for p in contours)})
            stream.write(struct.pack("<fI", top, 0xffffffff))
        if not any(p["contours"] for p in layer_info):
            raise ValueError("All slicing planes are empty. Choose a finer source slice thickness.")
        parsed = slc_readback(temporary)
        if len(parsed["layers"]) != len(layer_info):
            raise ValueError("SLC readback layer count mismatch.")
        if stl_path is not None:
            if sha256(stl_path) != source_fingerprint:
                raise RuntimeError("The source STL changed during conversion. No completed SLC is published.")
        elif slc_mesh_sha256(source_mesh) != source_fingerprint:
            raise RuntimeError("The in-memory source mesh changed during conversion.")
        temporary.replace(target)
        audit.update(status="success", source_unchanged=True,
                     layer_count=len(layer_info), contour_count=sum(p["contours"] for p in layer_info),
                     first_plane_mm=float(planes[0]), last_plane_mm=float(planes[-1]), terminal_z_mm=top,
                     requested_layer_thickness_mm=requested_thickness,
                     layer_thickness_mm=thickness, layers=layer_info,
                     layer_z_convention="minimum Z of slab; contours sampled in its interior",
                     contour_winding="outer/island CCW, holes CW when viewed from +Z; repeated first vertex",
                     max_xy_float32_quantization_error_mm=max_xy_error,
                     max_xy_float32_quantization_error_source_um=max_xy_error / options["xy_mm_per_source_um"],
                     first_plane_above_mesh_minimum_mm=float(planes[0] - lower[2]),
                     last_plane_below_mesh_maximum_mm=float(upper[2] - planes[-1]),
                     first_plane_above_mesh_minimum_source_um=float((planes[0] - lower[2]) / options["z_mm_per_source_um"]),
                     last_plane_below_mesh_maximum_source_um=float((upper[2] - planes[-1]) / options["z_mm_per_source_um"]),
                     slc_sha256=sha256(target), bytes=target.stat().st_size,
                     readback="header/table/records/order/closure/winding checked by bundled subset reader",
                     limitations=[
                         "Scaled SLC is not the original microscopic fabrication geometry; source geometry remains unchanged.",
                         "Different XY/Z factors change 3-D angles, lengths, and cross-section aspect ratios.",
                         "No machine compensation or material shrinkage calibration is established by these factors.",
                         "Stored Z values are slab lower bounds; contours are sampled inside each slab. Mesh and slice discretization remain.",
                         "Receiving software must accept SLC 2.0 PART, MM and the chosen origin, including negative Z when preserving the top-end centre.",
                         "CHITUBOX has not been run here; file readback is not certification of application compatibility.",
                         "No supports, scan vectors, exposure parameters, or printer jobs are generated.",
                         "Do not reapply scaling to an STL already enlarged for display."])
        dump_json(manifest_path, audit)
        source_kind_text = "source STL unchanged" if stl_path is not None else "in-memory source mesh unchanged"
        print(f"SLC created: {target}\n"
              f"  Scale on source geometry: X/Y x{factors[0]:g}; Z x{factors[2]:g}\n"
              f"  {len(layer_info)} planes; original spacing={layer_um:g} um -> SLC spacing={thickness:.9g} mm\n"
              f"  Output size: {upper[0]-lower[0]:.6f} x {upper[1]-lower[1]:.6f} x {upper[2]-lower[2]:.6f} mm\n"
              f"  XY origin={options['xy_origin']}; Z origin={origin}; {source_kind_text}")
        print("  Closed solid contours exported. Generate solid fill/hatching in the receiving software.")
        if lower[2] < 0:
            print("  This origin includes negative Z layers; the receiving software must retain and fill them.")
        return {"slc": str(target), "slc_manifest": str(manifest_path), "layers": len(layer_info),
                "layer_thickness_um": layer_um, "slc_layer_thickness_mm": thickness,
                "slc_scale_xyz": factors.tolist(), "slc_translation_mm": translation.tolist(),
                "slc_size_mm": (upper-lower).tolist(), "chitubox_import_tested": False}
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        audit.update(status="failed", error=str(exc),
                     source_stl_preserved=(stl_path.is_file() if stl_path is not None else None))
        dump_json(manifest_path, audit)
        source_note = "STL remains unchanged" if stl_path is not None else "in-memory source mesh was not modified"
        operation = "STL-to-SLC" if stl_path is not None else "Mesh-to-SLC"
        raise RuntimeError(f"{operation} conversion failed: {exc}. {source_note}. Audit: {manifest_path}") from exc
