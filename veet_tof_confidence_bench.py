"""veet_tof_confidence_bench.py -- bench re-derivation of TOF confidence thresholds.

Standalone, headless CLI. Given the TOF_Conf_Test bench data set (a flat
target -- paper notepad or phone, at 0deg or 30deg back tilt -- presented to
the sensor across a sweep of distances, one raw CSV per distance point), this
script:

  1. Loads each per-distance CSV (raw TOF rows, no INF line -- see
     load_tof_csv).
  2. Estimates the true measured distance to the object from raw Object 1
     (primary-return) data only (never through veet_tof_bands' gate -- see
     estimate_object_distance for why).
  3. Works out geometrically how many of the 64 zones the object should
     occupy at that distance (three rounding conventions: floor/nearest/
     ceil -- see compute_footprint).
  4. Renders a VEET_TOF_Viewer.py-style spatial view per distance point,
     with the three candidate footprints overlaid.
  5. Derives a recommended minimum raw confidence per point per rounding
     mode, separating "object" zone confidence from "phantom"
     (non-footprint, in-field) zone confidence.
  6. Treats every per-point recommendation (Object 1 per rounding mode,
     plus the Object 2 / wall recommendation) as an anchor at its measured
     distance, fits the same floored power law veet_tof_bands.py uses for
     its 10 mm gate curve (tb.fit_power_law), reports it next to the
     currently-loaded curve, and writes one candidate v2 thresholds JSON
     per rounding mode.

All gating/geometry constants come from veet_tof_bands.py -- this script
does not reimplement the band table, FOV constants, or gating math.
Deliberately does not import VEET_TOF_Viewer.py (which imports PySide6 at
module scope) so this stays runnable from a plain CLI with no Qt install.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import veet_tof_bands as tb

ROUND_MODES = ("floor", "nearest", "ceil")
_ROUND_FUNCS = {
    "floor": math.floor,
    "nearest": round,
    "ceil": math.ceil,
}

TOF_RETURN_COLS = 64  # zones per return
# time_stamp, TOF, Conf1 x64, Conf2 x64, Dist1 x64, Dist2 x64 -- same field
# order as VEET_TOF_Viewer.py's TOF_FIELDS_PER_ROW. Duplicated here rather
# than imported, per that module's own "stays small and standalone"
# precedent (it deliberately does not share its parser with
# VEET_Data_Charting.py either) -- importing VEET_TOF_Viewer.py would also
# pull in its module-level PySide6 import.
TOF_FIELDS_PER_ROW = 2 + 4 * TOF_RETURN_COLS


# --- Filename / discovery ---------------------------------------------------

_NOMINAL_MM_RE = re.compile(r"^(\d+)\s*m{0,2}(?=[_-])", re.IGNORECASE)


def parse_nominal_mm(path: str) -> "float | None":
    """Extract the nominal test distance (mm) from a filename stem.

    Tolerant of the data set's inconsistent naming: '51mm_Paper' -> 51,
    '448_paper' -> 448 (no 'mm'), '798m_paper' -> 798 (typo 'm' for 'mm').
    """
    stem = os.path.basename(path)
    m = _NOMINAL_MM_RE.match(stem)
    return float(m.group(1)) if m else None


def discover_files(data_dir: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> list[str]:
    """Select the CSVs for one object/tilt combination: filename (lowercased)
    must contain every token in `include` and none of `exclude`.

    Substring matching, not word-boundary matching -- callers must pick
    tokens that don't collide (e.g. selecting "0deg" phone files by
    excluding "30deg" rather than requiring "0deg", since "30deg" contains
    "0deg" as a substring and would otherwise false-match).
    """
    candidates = []
    for name in os.listdir(data_dir):
        if not name.lower().endswith(".csv"):
            continue
        lower = name.lower()
        if not all(tok in lower for tok in include):
            continue
        if any(tok in lower for tok in exclude):
            continue
        candidates.append(os.path.join(data_dir, name))

    dated = [(parse_nominal_mm(p), p) for p in candidates]
    missing = [p for mm, p in dated if mm is None]
    if missing:
        raise ValueError(f"Could not parse nominal distance from: {missing}")
    dated.sort(key=lambda t: t[0])

    files = [p for _, p in dated]
    print(f"Discovered {len(files)} files (include={list(include)}, exclude={list(exclude)}):")
    for mm, p in dated:
        print(f"  {mm:>6.0f} mm  {os.path.basename(p)}")
    if not files:
        print("WARNING: no files matched -- check --file-include/--file-exclude.", file=sys.stderr)
    return files


# --- Loading -----------------------------------------------------------------

@dataclass
class TofFrameData:
    times: np.ndarray            # (n,) float epoch seconds (or NaN)
    conf1: np.ndarray            # (n, 8, 8)
    conf2: np.ndarray            # (n, 8, 8)
    dist1: np.ndarray            # (n, 8, 8) mm
    dist2: np.ndarray            # (n, 8, 8) mm
    sample_period_s: float
    n_frames: int


def load_tof_csv(path: str) -> TofFrameData:
    """Parse a single per-distance-point CSV: raw TOF rows, no header, no
    INF line. Field layout matches VEET_TOF_Viewer.py's
    load_from_device_folder (confirmed against 150mm_Paper.csv): each row is
    timestamp, 'TOF', conf1 x64, conf2 x64, dist1 x64, dist2 x64.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))

    if not rows:
        raise ValueError(f"No data rows in {path}")

    fixed = []
    for r in rows:
        if len(r) < TOF_FIELDS_PER_ROW:
            r = r + [""] * (TOF_FIELDS_PER_ROW - len(r))
        fixed.append(r[:TOF_FIELDS_PER_ROW])
    arr = np.array(fixed, dtype=object)

    times = np.array(
        [float(v) if v not in ("", None) else np.nan for v in arr[:, 0]], dtype=float
    )

    def to_float(block):
        return np.array(
            [[float(v) if v not in ("", None) else np.nan for v in row] for row in block],
            dtype=float,
        )

    values = to_float(arr[:, 2:])
    n = values.shape[0]
    conf1 = values[:, 0 * TOF_RETURN_COLS: 1 * TOF_RETURN_COLS].reshape(n, 8, 8)
    conf2 = values[:, 1 * TOF_RETURN_COLS: 2 * TOF_RETURN_COLS].reshape(n, 8, 8)
    dist1 = values[:, 2 * TOF_RETURN_COLS: 3 * TOF_RETURN_COLS].reshape(n, 8, 8)
    dist2 = values[:, 3 * TOF_RETURN_COLS: 4 * TOF_RETURN_COLS].reshape(n, 8, 8)

    order = np.argsort(times)
    times, conf1, conf2, dist1, dist2 = (
        times[order], conf1[order], conf2[order], dist1[order], dist2[order]
    )

    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if n >= 2 and diffs.size:
        period = float(np.median(diffs))
    else:
        period = 1.0

    return TofFrameData(
        times=times, conf1=conf1, conf2=conf2, dist1=dist1, dist2=dist2,
        sample_period_s=period, n_frames=n,
    )


# --- Geometry ------------------------------------------------------------

def _centered_block(n: int, grid: int = 8) -> tuple[int, int]:
    """Centered [lo, hi) block of width n within a grid of size `grid`."""
    n = int(np.clip(n, 1, grid))
    lo = int(np.clip(math.floor(grid / 2.0 - n / 2.0 + 0.5), 0, grid - n))
    return lo, lo + n


def estimate_distance_in_mask(
    dist_arr: np.ndarray, mask: np.ndarray
) -> tuple["float | None", float, int]:
    """Robust first-pass distance estimate pooled over `mask` zones across
    all frames of `dist_arr` (n,8,8). Drops sub-floor/no-return samples
    (< tb.NO_RETURN_FLOOR_MM, which includes literal 0 = no second return).
    Median + MAD, so a stray phantom zone in the pool doesn't skew it.
    """
    window = dist_arr[:, mask].ravel()
    window = window[np.isfinite(window) & (window >= tb.NO_RETURN_FLOOR_MM)]
    if window.size == 0:
        return None, float("nan"), 0
    z_hat = float(np.median(window))
    mad = float(np.median(np.abs(window - z_hat)))
    return z_hat, mad, int(window.size)


def estimate_object_distance(
    dist1: np.ndarray, center_window: int = 2
) -> tuple["float | None", float, int]:
    """First-pass measured distance to Object 1, from raw dist1 only.

    Deliberately does NOT use tb.resolve_dual_return / tb.valid_mask: both
    depend on tb.min_confidence(z), i.e. on the very thresholds this bench
    test exists to re-derive. Using them here would make the new
    thresholds partly a function of the old (possibly wrong) ones. The
    geometric center zones are the only "which zones are the object"
    signal available before any threshold exists, so that's what's used.
    """
    row_lo, row_hi = _centered_block(center_window)
    col_lo, col_hi = _centered_block(center_window)
    mask = np.zeros(tb.GRID_SHAPE, dtype=bool)
    mask[row_lo:row_hi, col_lo:col_hi] = True
    return estimate_distance_in_mask(dist1, mask)


def compute_footprint(
    z_hat: float, obj_w_mm: float, obj_h_mm: float, mode: str
) -> dict:
    """Geometric footprint (rows x cols) of the object at distance z_hat,
    for one of the three rounding conventions."""
    if mode not in _ROUND_FUNCS:
        raise ValueError(f"Unknown rounding mode: {mode}")
    round_fn = _ROUND_FUNCS[mode]

    pitch_v_deg, pitch_h_deg = tb.ZONE_PITCH_DEG  # (vertical=rows, horizontal=cols)
    mm_per_zone_row = 2.0 * z_hat * math.tan(math.radians(pitch_v_deg) / 2.0)
    mm_per_zone_col = 2.0 * z_hat * math.tan(math.radians(pitch_h_deg) / 2.0)

    raw_rows = obj_h_mm / mm_per_zone_row  # short axis (125mm) -> rows
    raw_cols = obj_w_mm / mm_per_zone_col  # long axis (200mm) -> cols

    n_rows = int(np.clip(round_fn(raw_rows), 1, 8))
    n_cols = int(np.clip(round_fn(raw_cols), 1, 8))
    row_lo, row_hi = _centered_block(n_rows)
    col_lo, col_hi = _centered_block(n_cols)

    mask = np.zeros(tb.GRID_SHAPE, dtype=bool)
    mask[row_lo:row_hi, col_lo:col_hi] = True

    return {
        "mask": mask, "n_rows": n_rows, "n_cols": n_cols,
        "row_lo": row_lo, "row_hi": row_hi, "col_lo": col_lo, "col_hi": col_hi,
        "mm_per_zone_row": mm_per_zone_row, "mm_per_zone_col": mm_per_zone_col,
    }


# --- Threshold derivation -------------------------------------------------

def zone_conf_samples(
    conf1: np.ndarray, dist1: np.ndarray, mask: np.ndarray, z_hat: float, tol_frac: float
) -> np.ndarray:
    """Raw confidence samples from `mask` zones whose measured distance is
    within tolerance of the estimated object distance z_hat."""
    tol = max(tol_frac * z_hat, 20.0)
    near = np.abs(dist1 - z_hat) <= tol
    sel = near & mask[None, :, :]
    vals = conf1[sel]
    return vals[np.isfinite(vals)]


def recommend_threshold(
    object_samples: np.ndarray, phantom_samples: np.ndarray,
    p_obj: float, p_phantom: float,
) -> dict:
    if object_samples.size == 0:
        return {
            "recommended": None, "p_lo": None, "p_hi": None,
            "flag": "no_object_samples", "n_object": 0, "n_phantom": int(phantom_samples.size),
        }

    p_lo = float(np.percentile(object_samples, p_obj))

    if phantom_samples.size == 0:
        recommended = math.ceil(p_lo)
        flag = "no_phantom_observed"
        p_hi = None
    else:
        p_hi = float(np.percentile(phantom_samples, p_phantom))
        if p_hi + 1 <= p_lo:
            recommended = math.ceil(p_hi) + 1
            flag = "clean_separation"
        else:
            recommended = math.ceil(p_lo)
            flag = f"overlap (gap={p_hi + 1 - p_lo:.0f})"

    recommended = int(np.clip(recommended, 0, tb.CONF_MAX))
    return {
        "recommended": recommended, "p_lo": p_lo, "p_hi": p_hi, "flag": flag,
        "n_object": int(object_samples.size), "n_phantom": int(phantom_samples.size),
    }


# --- Per-point analysis -----------------------------------------------------

_WALL_PARTITION_MODE = "nearest"  # which Object-1 footprint mode defines the Object-2/wall split


@dataclass
class PointResult:
    file: str
    nominal_mm: float
    measured_mm: "float | None"
    mad_mm: float
    n_center_samples: int
    band_idx: "int | None"
    band_name: str
    n_frames: int
    dist1_median: np.ndarray  # (8,8) for rendering
    conf1_median: np.ndarray  # (8,8) for rendering
    dist2_median: np.ndarray  # (8,8) for rendering
    conf2_median: np.ndarray  # (8,8) for rendering
    resolved_dist: np.ndarray   # (8,8) one-object-per-zone result, current thresholds
    resolved_conf: np.ndarray   # (8,8)
    resolved_source: np.ndarray  # (8,8) int: 0=none, 1=Object 1, 2=Object 2
    obj1_valid: np.ndarray       # (8,8) bool, current thresholds -- zones where Object 2 is moot
    footprints: dict          # mode -> compute_footprint() result
    recs: dict                # mode -> recommend_threshold() result
    current_min_raw: "int | None"   # loaded threshold curve at measured_mm
    # Object 2 / wall: uses the Object-1 footprint (_WALL_PARTITION_MODE) as a
    # known partition rather than an independent geometric guess -- see
    # analyze_point for why.
    wall_mask: "np.ndarray | None"       # (8,8) region where the wall should be unoccluded
    measured2_mm: "float | None"
    mad2_mm: float
    band2_idx: "int | None"
    band2_name: str
    current2_min_raw: "int | None"
    rec2: dict


def analyze_point(
    path: str, obj_w_mm: float, obj_h_mm: float, center_window: int,
    p_obj: float, p_phantom: float, tol_frac: float,
) -> PointResult:
    nominal_mm = parse_nominal_mm(path)
    data = load_tof_csv(path)

    z_hat, mad, n_center = estimate_object_distance(data.dist1, center_window)

    footprints: dict = {}
    recs: dict = {}
    band_idx = None
    band_name = "n/a"
    current_min_raw = None

    if z_hat is not None:
        band_idx = int(tb.band_index(np.array([z_hat]))[0])
        band_name = tb.BANDS[band_idx].name
        current_min_raw = int(tb.min_confidence(np.array([z_hat]))[0])

        for mode in ROUND_MODES:
            fp = compute_footprint(z_hat, obj_w_mm, obj_h_mm, mode)
            object_samples = zone_conf_samples(data.conf1, data.dist1, fp["mask"], z_hat, tol_frac)
            phantom_mask = tb.ANALYSIS_FIELD & ~fp["mask"]
            phantom_samples = zone_conf_samples(data.conf1, data.dist1, phantom_mask, z_hat, tol_frac)
            rec = recommend_threshold(object_samples, phantom_samples, p_obj, p_phantom)
            footprints[mode] = fp
            recs[mode] = rec
    else:
        for mode in ROUND_MODES:
            footprints[mode] = None
            recs[mode] = {
                "recommended": None, "p_lo": None, "p_hi": None,
                "flag": "no_valid_center_return", "n_object": 0, "n_phantom": 0,
            }

    # --- Object 2 / wall -----------------------------------------------
    # Object 2 in this data set is usually the wall behind the target, not
    # the target itself (the target occludes the wall wherever it's
    # actually present, so dist2 reads sub-floor/0 inside the target's own
    # footprint -- see the 700mm spatial PNGs). Rather than guess a wall
    # "footprint" geometrically (a wall isn't a 200x125mm object), this
    # reuses the already-derived Object-1 footprint as ground truth for
    # where the wall should/shouldn't be visible: the complement of the
    # footprint is where the wall should read cleanly; the footprint
    # itself is where the wall should be occluded, so any confident
    # wall-distance reading leaking through there is genuine crosstalk.
    wall_mask = None
    measured2_mm = None
    mad2 = float("nan")
    band2_idx = None
    band2_name = "n/a"
    current2_min_raw = None
    rec2 = {"recommended": None, "p_lo": None, "p_hi": None,
            "flag": "no_object1_footprint", "n_object": 0, "n_phantom": 0}

    fp1 = footprints.get(_WALL_PARTITION_MODE)
    if fp1 is not None:
        wall_mask = tb.ANALYSIS_FIELD & ~fp1["mask"]
        occluded_mask = fp1["mask"]
        z_hat2, mad2, _n_wall = estimate_distance_in_mask(data.dist2, wall_mask)
        if z_hat2 is not None:
            measured2_mm = z_hat2
            band2_idx = int(tb.band_index(np.array([z_hat2]))[0])
            band2_name = tb.BANDS[band2_idx].name
            current2_min_raw = int(tb.min_confidence(np.array([z_hat2]))[0])
            wall_samples = zone_conf_samples(data.conf2, data.dist2, wall_mask, z_hat2, tol_frac)
            leak_samples = zone_conf_samples(data.conf2, data.dist2, occluded_mask, z_hat2, tol_frac)
            rec2 = recommend_threshold(wall_samples, leak_samples, p_obj, p_phantom)
        else:
            rec2 = {"recommended": None, "p_lo": None, "p_hi": None,
                    "flag": "no_wall_return", "n_object": 0, "n_phantom": 0}

    dist1_median = np.nanmedian(data.dist1, axis=0)
    conf1_median = np.nanmedian(data.conf1, axis=0)
    dist2_median = np.nanmedian(data.dist2, axis=0)
    conf2_median = np.nanmedian(data.conf2, axis=0)

    resolved_dist, resolved_conf, resolved_source, obj1_valid = resolve_single_object(
        dist1_median, conf1_median, dist2_median, conf2_median)

    return PointResult(
        file=os.path.basename(path), nominal_mm=nominal_mm, measured_mm=z_hat,
        mad_mm=mad, n_center_samples=n_center, band_idx=band_idx, band_name=band_name,
        n_frames=data.n_frames, dist1_median=dist1_median, conf1_median=conf1_median,
        dist2_median=dist2_median, conf2_median=conf2_median,
        resolved_dist=resolved_dist, resolved_conf=resolved_conf,
        resolved_source=resolved_source, obj1_valid=obj1_valid,
        footprints=footprints, recs=recs, current_min_raw=current_min_raw,
        wall_mask=wall_mask, measured2_mm=measured2_mm, mad2_mm=mad2,
        band2_idx=band2_idx, band2_name=band2_name, current2_min_raw=current2_min_raw,
        rec2=rec2,
    )


# --- Rendering ---------------------------------------------------------------

def _savefig(fig, out_path: str, **kwargs) -> None:
    """fig.savefig with a short retry -- OneDrive-synced output folders can
    transiently lock a just-written file (observed as OSError [Errno 22] on
    Windows), unrelated to the plot content. Retrying is always safe since
    savefig is idempotent."""
    last_err = None
    for attempt in range(5):
        try:
            fig.savefig(out_path, **kwargs)
            return
        except OSError as e:
            last_err = e
            time.sleep(0.4 * (attempt + 1))
    raise last_err


_MODE_STYLE = {
    "floor": dict(edgecolor="#00e5ff", linestyle="dashed", linewidth=2.0),
    "nearest": dict(edgecolor="#39ff14", linestyle="solid", linewidth=2.2),
    "ceil": dict(edgecolor="#ff9d00", linestyle="dotted", linewidth=2.4),
}


def _overlay_analysis_field(ax):
    ax.axhline(0.5, color="orange", linewidth=2)
    ax.axhline(6.5, color="orange", linewidth=2)


def _overlay_invalid(ax, invalid_mask):
    for r in range(8):
        for c in range(8):
            if invalid_mask[r, c]:
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                    hatch="xx", edgecolor="red", linewidth=1.0))


def _overlay_footprints(ax, footprints):
    for mode in ROUND_MODES:
        fp = footprints.get(mode)
        if fp is None:
            continue
        style = _MODE_STYLE[mode]
        pad = {"floor": 0.12, "nearest": 0.0, "ceil": -0.12}[mode]
        ax.add_patch(mpatches.Rectangle(
            (fp["col_lo"] - 0.5 + pad, fp["row_lo"] - 0.5 + pad),
            (fp["col_hi"] - fp["col_lo"]) - 2 * pad,
            (fp["row_hi"] - fp["row_lo"]) - 2 * pad,
            fill=False, **style))


def _overlay_wall_partition(ax, fp1):
    """Boundary between the wall-visible region and the Object-1 footprint
    (where the wall should be occluded) -- reuses fp1's rectangle bounds."""
    if fp1 is None:
        return
    ax.add_patch(mpatches.Rectangle(
        (fp1["col_lo"] - 0.5, fp1["row_lo"] - 0.5),
        fp1["col_hi"] - fp1["col_lo"], fp1["row_hi"] - fp1["row_lo"],
        fill=False, edgecolor="magenta", linewidth=2.2, linestyle="solid"))


def _overlay_moot_x(ax, moot_mask):
    """Bold X over zones where Object 2 is moot: this sensor's use case
    assumes only one object per zone, so once Object 1 clears the (current,
    loaded) confidence gate there, Object 2's reading there is discarded."""
    for r in range(8):
        for c in range(8):
            if moot_mask[r, c]:
                ax.plot([c - 0.35, c + 0.35], [r - 0.35, r + 0.35],
                        color="black", linewidth=2.4, solid_capstyle="round")
                ax.plot([c - 0.35, c + 0.35], [r + 0.35, r - 0.35],
                        color="black", linewidth=2.4, solid_capstyle="round")


def resolve_single_object(dist1, conf1, dist2, conf2):
    """One object per zone: Object 1 wins if it clears the (current, loaded)
    confidence gate; otherwise Object 2 wins if IT clears the gate;
    otherwise the zone has no trusted reading. Uses tb.valid_mask directly
    (the currently-loaded thresholds) -- this is a "what does the gate do
    right now" view for eyeballing/threshold tuning, not part of the
    percentile-based derivation math elsewhere in this script (which stays
    on raw dist1/conf1 to avoid circularity -- see estimate_object_distance).

    Returns (resolved_dist, resolved_conf, source) where source is
    0 = no trusted reading, 1 = Object 1, 2 = Object 2.
    """
    valid1 = tb.valid_mask(dist1, conf1)
    valid2 = tb.valid_mask(dist2, conf2)
    resolved_dist = np.where(valid1, dist1, np.where(valid2, dist2, np.nan))
    resolved_conf = np.where(valid1, conf1, np.where(valid2, conf2, np.nan))
    source = np.where(valid1, 1, np.where(valid2, 2, 0))
    return resolved_dist, resolved_conf, source, valid1


def _annotate_grid(ax, data):
    for r in range(8):
        for c in range(8):
            val = data[r, c]
            text = "nan" if not np.isfinite(val) else f"{val:.0f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=6, color="white",
                     path_effects=[mpatheffects.withStroke(linewidth=1.5, foreground="black")])
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("column (horizontal)")
    ax.set_ylabel("row (vertical)")


def _annotate_resolved_grid(ax, dist_data, source):
    """Like _annotate_grid, but colors each cell's text by which return it
    came from (white=Object 1, cyan=Object 2, red 'X'=neither trusted) so
    the one-object-per-zone result is legible at a glance."""
    src_color = {0: "red", 1: "white", 2: "#4fc3f7"}
    for r in range(8):
        for c in range(8):
            s = int(source[r, c])
            val = dist_data[r, c]
            text = "X" if s == 0 or not np.isfinite(val) else f"{val:.0f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=6, color=src_color[s],
                     fontweight="bold" if s == 0 else "normal",
                     path_effects=[mpatheffects.withStroke(linewidth=1.5, foreground="black")])
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("column (horizontal)")
    ax.set_ylabel("row (vertical)")


def render_spatial_png(point: PointResult, out_path: str, object_label: str) -> None:
    fig = Figure(figsize=(12, 16))
    (ax_dist, ax_conf), (ax_dist2, ax_conf2), (ax_dist_r, ax_conf_r) = fig.subplots(3, 2)

    n_bands = len(tb.BANDS)
    cmap = colormaps["RdYlBu_r"].resampled(n_bands)

    im_dist = ax_dist.imshow(tb.band_index(point.dist1_median), cmap=cmap,
                              vmin=-0.5, vmax=n_bands - 0.5, origin="upper")
    cbar_dist = fig.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04, ticks=range(n_bands))
    cbar_dist.ax.set_yticklabels([b.name for b in tb.BANDS], fontsize=7)
    cbar_dist.set_label("reporting band")
    ax_dist.set_title("Object 1 (return 1) -- distance (mm)")

    im_conf = ax_conf.imshow(point.conf1_median, cmap="magma", vmin=0, vmax=tb.CONF_MAX, origin="upper")
    fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04).set_label("raw confidence (0-255)")
    ax_conf.set_title("Object 1 (return 1) -- confidence (raw)")

    im_dist2 = ax_dist2.imshow(tb.band_index(point.dist2_median), cmap=cmap,
                                vmin=-0.5, vmax=n_bands - 0.5, origin="upper")
    cbar_dist2 = fig.colorbar(im_dist2, ax=ax_dist2, fraction=0.046, pad=0.04, ticks=range(n_bands))
    cbar_dist2.ax.set_yticklabels([b.name for b in tb.BANDS], fontsize=7)
    cbar_dist2.set_label("reporting band")
    ax_dist2.set_title("Object 2 (return 2, usually the wall) -- distance (mm)")

    im_conf2 = ax_conf2.imshow(point.conf2_median, cmap="magma", vmin=0, vmax=tb.CONF_MAX, origin="upper")
    fig.colorbar(im_conf2, ax=ax_conf2, fraction=0.046, pad=0.04).set_label("raw confidence (0-255)")
    ax_conf2.set_title("Object 2 (return 2, usually the wall) -- confidence (raw)")

    no_data = ~np.isfinite(point.resolved_dist)
    resolved_band_raw = tb.band_index(np.nan_to_num(point.resolved_dist, nan=tb.NO_RETURN_SENTINEL_MM))
    resolved_band = np.ma.masked_where(no_data, resolved_band_raw)
    cmap_r = colormaps["RdYlBu_r"].resampled(n_bands).copy()
    cmap_r.set_bad(color="lightgray")
    im_dist_r = ax_dist_r.imshow(resolved_band, cmap=cmap_r, vmin=-0.5, vmax=n_bands - 0.5, origin="upper")
    cbar_dist_r = fig.colorbar(im_dist_r, ax=ax_dist_r, fraction=0.046, pad=0.04, ticks=range(n_bands))
    cbar_dist_r.ax.set_yticklabels([b.name for b in tb.BANDS], fontsize=7)
    cbar_dist_r.set_label("reporting band (gray = no trusted reading)")
    ax_dist_r.set_title("Resolved (one object per zone) -- distance (mm)")

    resolved_conf_masked = np.ma.masked_where(no_data, point.resolved_conf)
    cmap_conf_r = colormaps["magma"].copy()
    cmap_conf_r.set_bad(color="lightgray")
    im_conf_r = ax_conf_r.imshow(resolved_conf_masked, cmap=cmap_conf_r,
                                  vmin=0, vmax=tb.CONF_MAX, origin="upper")
    fig.colorbar(im_conf_r, ax=ax_conf_r, fraction=0.046, pad=0.04).set_label("raw confidence (0-255)")
    ax_conf_r.set_title("Resolved (one object per zone) -- confidence (raw)")

    fp1 = point.footprints.get(_WALL_PARTITION_MODE)

    for ax, data in ((ax_dist, point.dist1_median), (ax_conf, point.conf1_median)):
        _annotate_grid(ax, data)
        _overlay_analysis_field(ax)
        _overlay_invalid(ax, ~point.obj1_valid)
        _overlay_footprints(ax, point.footprints)

    for ax, data in ((ax_dist2, point.dist2_median), (ax_conf2, point.conf2_median)):
        _annotate_grid(ax, data)
        _overlay_analysis_field(ax)
        _overlay_wall_partition(ax, fp1)
        _overlay_moot_x(ax, point.obj1_valid)

    for ax in (ax_dist_r, ax_conf_r):
        _annotate_resolved_grid(ax, point.resolved_dist, point.resolved_source)
        _overlay_analysis_field(ax)

    handles = [mpatches.Patch(facecolor="none", edgecolor=_MODE_STYLE[m]["edgecolor"],
                               linestyle=_MODE_STYLE[m]["linestyle"], linewidth=2, label=f"obj1 {m}")
               for m in ROUND_MODES]
    handles.append(mpatches.Patch(facecolor="none", edgecolor="orange", label="analysis field"))
    handles.append(mpatches.Patch(facecolor="none", edgecolor="magenta",
                                   label=f"obj1 footprint ({_WALL_PARTITION_MODE}) = occluded for wall"))
    handles.append(mlines.Line2D([], [], color="black", marker="x", linestyle="none",
                                  markersize=8, markeredgewidth=2.4,
                                  label="obj1 valid (current gate) → obj2 moot"))
    handles.append(mpatches.Patch(facecolor="none", edgecolor="none",
                                   label="resolved text: white=obj1, cyan=obj2, red X=neither valid"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)

    z_txt = f"{point.measured_mm:.0f} mm" if point.measured_mm is not None else "n/a"
    z2_txt = f"{point.measured2_mm:.0f} mm" if point.measured2_mm is not None else "n/a"
    fig.suptitle(
        f"{object_label} -- {point.file}\n"
        f"Object 1 nominal {point.nominal_mm:.0f} mm, measured {z_txt} (band: {point.band_name})  |  "
        f"Object 2/wall measured {z2_txt} (band: {point.band2_name})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _savefig(fig, out_path, dpi=140)


_MODE_COLORS = {"floor": "#00b8d4", "nearest": "#2e7d32", "ceil": "#ef6c00"}
_WALL_COLOR = "#6a1b9a"
_FLAG_MARKERS = {"clean_separation": "o", "overlap": "x", "no_phantom_observed": "^"}
_PLOT_X_MM = (80.0, 2200.0)


def _flag_marker(flag: str) -> str:
    return "x" if flag.startswith("overlap") else _FLAG_MARKERS.get(flag, "o")


def _plot_live_curve(ax) -> None:
    """The threshold curve currently loaded in veet_tof_bands."""
    if tb.CONFIDENCE_FIT is not None:
        xs = np.geomspace(max(_PLOT_X_MM[0], tb.Z_FLOOR_MM), _PLOT_X_MM[1], 400)
        ax.plot(xs, tb.min_confidence(xs), color="black", linewidth=1.5,
                label="current threshold curve")
        return
    lo, lut = tb.confidence_curve()  # legacy per-band step gate
    xs = np.append(lo, _PLOT_X_MM[1])
    ax.step(xs, np.append(lut, lut[-1]), where="post", color="black", linewidth=1.5,
            label="current threshold curve")


def _finish_threshold_axes(ax, xlabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*_PLOT_X_MM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Raw confidence (0-255)")
    ax.set_ylim(-5, 260)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    top_axis = ax.secondary_xaxis("top", functions=(lambda z: 1000.0 / np.clip(z, 1e-6, None),
                                                      lambda d: 1000.0 / np.clip(d, 1e-6, None)))
    top_axis.set_xlabel("Vergence [D]")


def render_summary_png(points: list[PointResult], curves: dict, out_path: str, object_label: str) -> None:
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots(1, 1)

    valid_points = [p for p in points if p.measured_mm is not None]
    for mode in ROUND_MODES:
        by_marker: dict = {}
        for p in valid_points:
            rec = p.recs[mode]
            if rec["recommended"] is None:
                continue
            by_marker.setdefault(_flag_marker(rec["flag"]), []).append((p.measured_mm, rec["recommended"]))
        for marker, xy in by_marker.items():
            ax.scatter(*zip(*xy), color=_MODE_COLORS[mode], marker=marker, s=45,
                       label=f"{mode} ({marker})", alpha=0.85)

    wall_xy = [(p.measured2_mm, p.rec2["recommended"]) for p in points
               if p.measured2_mm is not None and p.rec2["recommended"] is not None]
    if wall_xy and any(c["include_wall"] for c in curves.values()):
        ax.scatter(*zip(*wall_xy), color=_WALL_COLOR, marker="s", s=35, alpha=0.7,
                   label="wall (Object 2) anchors")

    _plot_live_curve(ax)
    xs = np.geomspace(tb.Z_FLOOR_MM, _PLOT_X_MM[1], 400)
    for mode in ROUND_MODES:
        fit = curves[mode]["fit"]
        if fit is not None:
            ax.plot(xs, fit.evaluate(xs), color=_MODE_COLORS[mode], linewidth=1.2, linestyle="--",
                    label=f"{mode} fit {fit.a:.3g}·d^-{fit.k:.2f}")

    _finish_threshold_axes(
        ax, f"Measured distance to {object_label} [mm]",
        f"{object_label} bench: per-point recommendations, fitted curves, current curve")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=140)


def render_wall_summary_png(points: list[PointResult], out_path: str, object_label: str) -> None:
    fig = Figure(figsize=(11, 6))
    ax = fig.subplots(1, 1)

    by_marker: dict = {}
    for p in points:
        if p.measured2_mm is None or p.rec2["recommended"] is None:
            continue
        by_marker.setdefault(_flag_marker(p.rec2["flag"]), []).append((p.measured2_mm, p.rec2["recommended"]))
    for marker, xy in by_marker.items():
        ax.scatter(*zip(*xy), color=_WALL_COLOR, marker=marker, s=55, label=f"wall ({marker})", alpha=0.85)

    _plot_live_curve(ax)
    _finish_threshold_axes(
        ax, "Measured distance to Object 2 / wall [mm]",
        f"{object_label} bench: Object 2 (wall) recommended vs. current threshold curve")
    fig.tight_layout()
    _savefig(fig, out_path, dpi=140)


# --- Curve fit -----------------------------------------------------------

def curve_anchors_from_points(points: list[PointResult], mode: str,
                              include_wall: bool = True) -> list[tuple[int, int]]:
    """(distance_mm, min_raw) anchors for the power-law fit: each point's
    Object-1 recommendation for ``mode`` at its measured distance, plus (if
    ``include_wall``) each point's Object-2/wall recommendation at the wall's
    measured distance. Distances round to whole mm; where two anchors land on
    the same mm the higher (stricter) recommendation wins, so the anchors are
    strictly increasing as the v2 thresholds format requires."""
    by_mm: dict = {}

    def add(z, rec):
        if z is None or rec is None:
            return
        d = int(round(z))
        if d >= tb.Z_FLOOR_MM:
            by_mm[d] = max(rec, by_mm.get(d, rec))

    for p in points:
        add(p.measured_mm, p.recs[mode]["recommended"])
        if include_wall:
            add(p.measured2_mm, p.rec2["recommended"])
    return sorted(by_mm.items())


def fit_curves(points: list[PointResult], floor_min_raw: int, include_wall: bool = True) -> dict:
    """mode -> {"anchors", "fit" (tb.PowerLawFit or None), "error", "include_wall"}.

    Uses tb.fit_power_law -- the exact fit veet_tof_bands applies when it
    loads a v2 thresholds file -- so a candidate JSON reproduces this curve.
    """
    curves = {}
    for mode in ROUND_MODES:
        anchors = curve_anchors_from_points(points, mode, include_wall)
        try:
            fit, error = tb.fit_power_law(anchors, floor_min_raw), None
        except ValueError as e:
            fit, error = None, str(e)
        curves[mode] = {"anchors": anchors, "fit": fit, "error": error, "include_wall": include_wall}
    return curves


# --- Reporting -----------------------------------------------------------

def write_points_csv(points: list[PointResult], out_path: str) -> None:
    header = [
        "file", "nominal_mm", "measured_mm", "mad_mm", "n_center_samples",
        "band_idx", "band_name", "n_frames", "current_min_raw",
    ]
    for mode in ROUND_MODES:
        header += [
            f"{mode}_footprint_rows", f"{mode}_footprint_cols", f"{mode}_footprint_zones",
            f"{mode}_n_object_samples", f"{mode}_n_phantom_samples",
            f"{mode}_object_p_lo", f"{mode}_phantom_p_hi",
            f"{mode}_recommended_min_raw", f"{mode}_delta_vs_current", f"{mode}_flag",
        ]
    header += [
        "wall_measured2_mm", "wall_mad2_mm", "wall_band2_idx", "wall_band2_name",
        "wall_current2_min_raw", "wall_n_object_samples", "wall_n_leak_samples",
        "wall_object_p_lo", "wall_leak_p_hi", "wall_recommended_min_raw",
        "wall_delta_vs_current2", "wall_flag",
    ]

    lines = [",".join(header)]
    for p in points:
        row = [
            p.file, f"{p.nominal_mm:.0f}",
            "" if p.measured_mm is None else f"{p.measured_mm:.1f}",
            f"{p.mad_mm:.1f}", str(p.n_center_samples),
            "" if p.band_idx is None else str(p.band_idx), p.band_name,
            str(p.n_frames), "" if p.current_min_raw is None else str(p.current_min_raw),
        ]
        for mode in ROUND_MODES:
            fp = p.footprints[mode]
            rec = p.recs[mode]
            if fp is None:
                row += ["", "", "", "0", "0", "", "", "", "", rec["flag"]]
            else:
                zones = fp["n_rows"] * fp["n_cols"]
                delta = "" if rec["recommended"] is None or p.current_min_raw is None \
                    else str(rec["recommended"] - p.current_min_raw)
                row += [
                    str(fp["n_rows"]), str(fp["n_cols"]), str(zones),
                    str(rec["n_object"]), str(rec["n_phantom"]),
                    "" if rec["p_lo"] is None else f"{rec['p_lo']:.1f}",
                    "" if rec["p_hi"] is None else f"{rec['p_hi']:.1f}",
                    "" if rec["recommended"] is None else str(rec["recommended"]),
                    delta, rec["flag"],
                ]
        rec2 = p.rec2
        delta2 = "" if rec2["recommended"] is None or p.current2_min_raw is None \
            else str(rec2["recommended"] - p.current2_min_raw)
        row += [
            "" if p.measured2_mm is None else f"{p.measured2_mm:.1f}",
            "" if not np.isfinite(p.mad2_mm) else f"{p.mad2_mm:.1f}",
            "" if p.band2_idx is None else str(p.band2_idx), p.band2_name,
            "" if p.current2_min_raw is None else str(p.current2_min_raw),
            str(rec2["n_object"]), str(rec2["n_phantom"]),
            "" if rec2["p_lo"] is None else f"{rec2['p_lo']:.1f}",
            "" if rec2["p_hi"] is None else f"{rec2['p_hi']:.1f}",
            "" if rec2["recommended"] is None else str(rec2["recommended"]),
            delta2, rec2["flag"],
        ]
        lines.append(",".join(row))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_curve_csv(curves: dict, out_path: str) -> None:
    """One row per 10 mm sample of the curve: current threshold and each mode's fit."""
    lo, current = tb.confidence_curve()
    header = ["distance_mm", "current_min_raw"] + [f"{mode}_fit_min_raw" for mode in ROUND_MODES]
    fitted = {mode: (c["fit"].evaluate(lo) if c["fit"] is not None else None) for mode, c in curves.items()}
    lines = [",".join(header)]
    for i, z in enumerate(lo):
        row = [f"{z:.0f}", str(int(current[i]))]
        row += ["" if fitted[mode] is None else str(int(fitted[mode][i])) for mode in ROUND_MODES]
        lines.append(",".join(row))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_candidate_json(curve: dict, out_path: str, provenance: str) -> None:
    """A v2 thresholds file for one mode's fit -- loadable as-is via
    --thresholds-json / reload_confidence_thresholds(). Never touches the live
    veet_tof_confidence_thresholds.json."""
    fit = curve["fit"]
    payload = {
        "_readme": (
            "Candidate confidence-gate curve written by veet_tof_confidence_bench.py. "
            "Same format as veet_tof_confidence_thresholds.json (v2): veet_tof_bands.py "
            "re-fits the anchors at load. Try it with --thresholds-json, or copy the "
            "anchors into the live file after review. This file does NOT replace it."),
        "format_version": 2,
        "provenance": provenance,
        "last_updated": time.strftime("%Y-%m-%d"),
        "rule": "power_law_fit",
        "floor_min_raw": fit.floor_min_raw,
        "fit_preview": {"a": fit.a, "k": fit.k, "n_anchors_fit": fit.n_anchors_fit,
                        "n_anchors_censored": fit.n_anchors_censored},
        "anchors_columns": ["distance_mm", "min_raw"],
        "anchors": [[int(d), int(v)] for d, v in curve["anchors"]],
    }
    text = json.dumps(payload, indent=2)
    text = re.sub(r"\[\n\s+(-?\d+),\n\s+(-?\d+)\n\s+\]", r"[\1, \2]", text)
    text = re.sub(r'\[\n\s+"distance_mm",\n\s+"min_raw"\n\s+\]', '["distance_mm", "min_raw"]', text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def write_report_md(
    points: list[PointResult], curves: dict, candidate_jsons: dict, out_dir: str,
    summary_png: str, object_label: str, wall_summary_png: "str | None" = None,
) -> None:
    lines = []
    lines.append(f"# {object_label} TOF Confidence-Threshold Bench Report\n")
    lines.append(
        f"Derived from the {object_label} bench sweep in `TOF_Conf_Test`. "
        "Recommendations use raw `dist1`/`conf1` only -- never "
        "`veet_tof_bands.resolve_dual_return`/`valid_mask` -- so they don't "
        "circularly depend on the thresholds being re-derived. Three geometric "
        "footprint-rounding conventions (floor/nearest/ceil) are reported side by "
        "side; where they disagree it signals footprint-boundary ambiguity at "
        "that distance, not a script bug. Points with no usable center-zone "
        "return (dropout / beyond-range) are listed with `n/a` rather than "
        "silently dropped.\n\n"
        f"**Confidence thresholds in effect for the X-marks/resolved grid in the spatial "
        f"PNGs and the 'current' columns below:** `{tb.CONFIDENCE_THRESHOLDS_SOURCE}` "
        f"(gate v{tb.CONFIDENCE_GATE_VERSION}: {tb.CONFIDENCE_CURVE_DESCRIPTION})\n"
    )
    lines.append(f"![summary]({os.path.relpath(summary_png, out_dir)})\n")

    include_wall = any(c["include_wall"] for c in curves.values())
    any_fit = next((c["fit"] for c in curves.values() if c["fit"] is not None), None)
    floor_text = f"floored at {any_fit.floor_min_raw}" if any_fit else "floored"
    lines.append("## Fitted threshold curve\n")
    lines.append(
        "Each point's recommendation is an anchor at its measured distance"
        + (" (Object 1 for that mode, plus the Object 2/wall recommendation at the wall's distance)"
           if include_wall else " (Object 1 only)")
        + f". `min_raw = a * d^-k` is fit through the anchors in log-log space, {floor_text}, "
        "clipped at 255; anchors at 0/255 are censored and excluded. This is the same fit "
        "`veet_tof_bands` runs when it loads the candidate JSON.\n")
    lines.append("| Mode | a | k | Anchors fit | Censored | Candidate JSON |")
    lines.append("|---|---|---|---|---|---|")
    for mode in ROUND_MODES:
        c = curves[mode]
        fit = c["fit"]
        if fit is None:
            lines.append(f"| {mode} | | | {len(c['anchors'])} | | no fit: {c['error']} |")
        else:
            rel = os.path.relpath(candidate_jsons[mode], out_dir)
            lines.append(f"| {mode} | {fit.a:.4g} | {fit.k:.3f} | {fit.n_anchors_fit} | "
                          f"{fit.n_anchors_censored} | [{os.path.basename(rel)}]({rel}) |")
    lines.append("")

    lines.append("### Curve at reporting-band near edges (reference only)\n")
    lines.append("The gate no longer has one value per band; these are the curves sampled at "
                  "each band's near edge for comparison. Curves sampled every 10 mm: `threshold_curve.csv`.\n")
    lines.append("| Band | Near edge (mm) | Current | Floor fit | Nearest fit | Ceil fit |")
    lines.append("|---|---|---|---|---|---|")
    for b in tb.BANDS:
        z = max(b.z_min_mm, tb.Z_FLOOR_MM)
        cur = int(tb.min_confidence(np.array([z]))[0])
        cells = ["" if curves[m]["fit"] is None else str(int(curves[m]["fit"].evaluate(z))) for m in ROUND_MODES]
        lines.append(f"| {b.index} `{b.name}` | {z:.0f} | {cur} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines.append("## Per-distance-point detail\n")
    lines.append("| File | Nominal mm | Measured mm | Band | Current | "
                  "Floor rec (flag) | Nearest rec (flag) | Ceil rec (flag) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in points:
        meas = "n/a" if p.measured_mm is None else f"{p.measured_mm:.0f}"
        cur = "" if p.current_min_raw is None else str(p.current_min_raw)
        cells = []
        for mode in ROUND_MODES:
            rec = p.recs[mode]
            val = "n/a" if rec["recommended"] is None else str(rec["recommended"])
            cells.append(f"{val} ({rec['flag']})")
        lines.append(f"| {p.file} | {p.nominal_mm:.0f} | {meas} | {p.band_name} | "
                      f"{cur} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines.append(
        "## Object 2 / wall\n\n"
        "Object 2 (the sensor's secondary return) is usually the wall behind "
        "the target, not the target itself -- the target occludes the wall "
        "wherever it's actually present. Rather than guess a wall \"footprint\" "
        "geometrically (a wall isn't a small target-sized object), this reuses "
        f"the already-derived Object-1 footprint (`{_WALL_PARTITION_MODE}` mode) "
        "as a known partition: its complement is where the wall should read "
        "cleanly (`wall_object_p_lo` in the CSV), and the footprint itself is where "
        "the wall should be occluded, so any confident wall-distance reading "
        "leaking through there (`wall_leak_p_hi`) is genuine crosstalk, not "
        "just background noise."
        + (" These wall recommendations are included as curve anchors above.\n" if include_wall else "\n")
    )
    if wall_summary_png:
        lines.append(f"![wall summary]({os.path.relpath(wall_summary_png, out_dir)})\n")

    lines.append("| File | Obj1 measured mm | Wall measured mm | Wall band | "
                  "Wall current | Wall recommended (flag) |")
    lines.append("|---|---|---|---|---|---|")
    for p in points:
        meas = "n/a" if p.measured_mm is None else f"{p.measured_mm:.0f}"
        meas2 = "n/a" if p.measured2_mm is None else f"{p.measured2_mm:.0f}"
        cur2 = "" if p.current2_min_raw is None else str(p.current2_min_raw)
        rec2 = p.rec2
        val2 = "n/a" if rec2["recommended"] is None else str(rec2["recommended"])
        lines.append(f"| {p.file} | {meas} | {meas2} | {p.band2_name} | {cur2} | "
                      f"{val2} ({rec2['flag']}) |")
    lines.append("")

    lines.append("## Per-point spatial views\n")
    for p in points:
        png_rel = os.path.join("spatial", os.path.splitext(p.file)[0] + ".png")
        lines.append(f"- [{p.file}]({png_rel})")

    with open(os.path.join(out_dir, "threshold_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Console table -----------------------------------------------------------

def print_console_table(points: list[PointResult]) -> None:
    hdr = f"{'file':<24}{'nom mm':>8}{'meas mm':>9}{'band':<18}{'cur':>5}" + \
          "".join(f"{m + ' rec':>12}" for m in ROUND_MODES) + \
          f"{'wall mm':>9}{'wall band':<18}{'wall rec':>10}"
    print(hdr)
    print("-" * len(hdr))
    for p in points:
        meas = "n/a" if p.measured_mm is None else f"{p.measured_mm:.0f}"
        cur = "" if p.current_min_raw is None else str(p.current_min_raw)
        row = f"{p.file:<24}{p.nominal_mm:>8.0f}{meas:>9}{p.band_name:<18}{cur:>5}"
        for mode in ROUND_MODES:
            rec = p.recs[mode]["recommended"]
            row += f"{'n/a' if rec is None else rec:>12}"
        meas2 = "n/a" if p.measured2_mm is None else f"{p.measured2_mm:.0f}"
        rec2 = p.rec2["recommended"]
        row += f"{meas2:>9}{p.band2_name:<18}{'n/a' if rec2 is None else rec2:>10}"
        print(row)


# --- Main --------------------------------------------------------------------

def _slugify(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s or "object"


def run_bench(
    data_dir: str,
    object_label: str = "Object 1",
    file_include: tuple = ("paper",),
    file_exclude: tuple = ("phone", "30deg"),
    object_w_mm: float = 200.0,
    object_h_mm: float = 125.0,
    center_window: int = 2,
    obj_percentile: float = 10.0,
    phantom_percentile: float = 95.0,
    distance_tol_frac: float = 0.15,
    floor_min_raw: int = tb.DEFAULT_FLOOR_MIN_RAW,
    include_wall_anchors: bool = True,
    thresholds_json: "str | None" = None,
    out_dir: "str | None" = None,
) -> dict:
    """Run the full bench analysis end to end. Shared by the CLI (`main`) and
    the GUI (`veet_tof_confidence_gui.py`) -- one code path, so both stay in
    sync automatically.

    Raises ValueError if no files match `file_include`/`file_exclude`.
    Returns a summary dict: out_dir, spatial_dir, n_files, points, curves,
    candidate_jsons, summary_png, wall_summary_png, points_csv, curve_csv,
    report_md.
    """
    if thresholds_json:
        tb.reload_confidence_thresholds(os.path.abspath(thresholds_json))
    print(f"Confidence thresholds in effect: {tb.CONFIDENCE_THRESHOLDS_SOURCE}")
    print(f"  gate v{tb.CONFIDENCE_GATE_VERSION}: {tb.CONFIDENCE_CURVE_DESCRIPTION}")

    data_dir = os.path.abspath(data_dir)
    slug = _slugify(object_label)
    if out_dir:
        out_dir = os.path.abspath(out_dir)
    else:
        # Default folder name incorporates the thresholds file when a
        # non-default one is given, so trying an alternate config can never
        # silently overwrite the default-thresholds run's output.
        suffix = f"_{_slugify(os.path.splitext(os.path.basename(thresholds_json))[0])}" \
            if thresholds_json else ""
        out_dir = os.path.join(data_dir, f"{slug}_threshold_analysis{suffix}")
    spatial_dir = os.path.join(out_dir, "spatial")
    os.makedirs(spatial_dir, exist_ok=True)

    files = discover_files(data_dir, tuple(file_include), tuple(file_exclude))
    if not files:
        raise ValueError(
            f"No files matched in {data_dir} "
            f"(include={list(file_include)}, exclude={list(file_exclude)}) -- nothing to analyze."
        )

    points: list[PointResult] = []
    for path in files:
        try:
            point = analyze_point(
                path, object_w_mm, object_h_mm, center_window,
                obj_percentile, phantom_percentile, distance_tol_frac,
            )
        except Exception as e:
            print(f"WARNING: skipping {os.path.basename(path)} -- {e}", file=sys.stderr)
            continue
        points.append(point)
        png_path = os.path.join(spatial_dir, os.path.splitext(point.file)[0] + ".png")
        render_spatial_png(point, png_path, object_label)

    if not points:
        raise ValueError("Every matched file failed to load -- see warnings above.")

    curves = fit_curves(points, floor_min_raw, include_wall_anchors)
    candidate_jsons = {}
    provenance = (f"veet_tof_confidence_bench.py fit, {object_label}, data {data_dir}, "
                  f"include={list(file_include)} exclude={list(file_exclude)}, "
                  f"wall anchors {'on' if include_wall_anchors else 'off'}")
    for mode in ROUND_MODES:
        c = curves[mode]
        if c["fit"] is None:
            print(f"  {mode}: no fit -- {c['error']}")
            continue
        candidate_jsons[mode] = os.path.join(out_dir, f"candidate_thresholds_{mode}.json")
        write_candidate_json(c, candidate_jsons[mode], f"{provenance}, {mode} footprint rounding")
        print(f"  {mode}: {c['fit'].describe()}")

    summary_png = os.path.join(out_dir, "threshold_summary.png")
    render_summary_png(points, curves, summary_png, object_label)

    wall_summary_png = os.path.join(out_dir, "wall_threshold_summary.png")
    render_wall_summary_png(points, wall_summary_png, object_label)

    points_csv = os.path.join(out_dir, "threshold_points.csv")
    curve_csv = os.path.join(out_dir, "threshold_curve.csv")
    report_md = os.path.join(out_dir, "threshold_report.md")
    write_points_csv(points, points_csv)
    write_curve_csv(curves, curve_csv)
    write_report_md(points, curves, candidate_jsons, out_dir, summary_png, object_label,
                     wall_summary_png)

    return {
        "out_dir": out_dir, "spatial_dir": spatial_dir, "n_files": len(files),
        "points": points, "curves": curves, "candidate_jsons": candidate_jsons,
        "summary_png": summary_png, "wall_summary_png": wall_summary_png,
        "points_csv": points_csv, "curve_csv": curve_csv, "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder containing the per-distance CSVs")
    parser.add_argument("--out-dir", default=None,
                         help="Output folder (default: <data-dir>/<object-label>_threshold_analysis)")
    parser.add_argument("--object-label", default="Object 1",
                         help="Human-readable label for titles/report (e.g. 'Object 2 (paper, 30deg tilt)')")
    parser.add_argument("--file-include", nargs="*", default=["paper"],
                         help="Filename tokens (lowercased) that must ALL be present")
    parser.add_argument("--file-exclude", nargs="*", default=["phone", "30deg"],
                         help="Filename tokens (lowercased) that must NONE be present. "
                              "Pass with no values (--file-exclude) to clear the default.")
    parser.add_argument("--object-w-mm", type=float, default=200.0,
                         help="Object width as presented to the sensor (horizontal/columns)")
    parser.add_argument("--object-h-mm", type=float, default=125.0,
                         help="Object height as presented to the sensor (vertical/rows)")
    parser.add_argument("--center-window", type=int, default=2,
                         help="NxN center block used for the first-pass distance estimate")
    parser.add_argument("--obj-percentile", type=float, default=10.0,
                         help="Low percentile of footprint-zone confidence")
    parser.add_argument("--phantom-percentile", type=float, default=95.0,
                         help="High percentile of background-zone confidence")
    parser.add_argument("--distance-tol-frac", type=float, default=0.15,
                         help="Fractional distance tolerance for sample selection (floor 20mm). "
                              "Widen this for tilted targets -- a tilted planar surface spans a "
                              "real range of physical distances across its footprint.")
    parser.add_argument("--floor-min-raw", type=int, default=tb.DEFAULT_FLOOR_MIN_RAW,
                         help="Lower clamp on the fitted threshold curve (written into the "
                              "candidate JSONs)")
    parser.add_argument("--no-wall-anchors", action="store_true",
                         help="Fit the curve from Object 1 recommendations only (default also "
                              "uses the Object 2 / wall recommendations as anchors)")
    parser.add_argument("--thresholds-json", default=None,
                         help="Alternate veet_tof_confidence_thresholds.json-style file to load "
                              "instead of the default -- affects tb.valid_mask/current_min_raw "
                              "(the X-marks, resolved grid, and 'current' columns) for this run "
                              "only. Does not modify the default file on disk.")
    args = parser.parse_args()

    try:
        result = run_bench(
            data_dir=args.data_dir, object_label=args.object_label,
            file_include=tuple(args.file_include), file_exclude=tuple(args.file_exclude),
            object_w_mm=args.object_w_mm, object_h_mm=args.object_h_mm,
            center_window=args.center_window, obj_percentile=args.obj_percentile,
            phantom_percentile=args.phantom_percentile, distance_tol_frac=args.distance_tol_frac,
            floor_min_raw=args.floor_min_raw,
            include_wall_anchors=not args.no_wall_anchors, thresholds_json=args.thresholds_json, out_dir=args.out_dir,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print()
    print_console_table(result["points"])
    print()
    print(f"Wrote report to {result['out_dir']}")


if __name__ == "__main__":
    main()
