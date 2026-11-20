"""VEET 2 time-of-flight validity gating and dioptric binning.

Two distance-indexed tables, deliberately decoupled:
  1. the per-zone confidence gate (is this return trustworthy?) -- a
     continuous threshold curve evaluated at each zone's own measured
     distance (``min_confidence``)
  2. the dioptric reporting bands (which literature-anchored bin is it?) --
     the 8 ``BANDS``, coupled to the dioptric histogram

The gate used to share the band table (one step threshold per band). That
forced threshold resolution to follow reporting-band width (up to 1000 mm
wide), so it was split out: refining the curve never moves a reporting edge.

Sensor context
--------------
TOF: 8x8 zone grid, 41 deg x 52 deg FOV, usable range ~10-2000+ mm.
Confidence is reported per zone as an unsigned byte, 0-255, where 255 == 100%.

Two-stage validity
------------------
Stage 1 (``valid_mask``) is the per-zone, distance-dependent confidence gate.
Stage 2 (``spatial_corroboration``) exploits the redundancy of the 64-zone
grid: the central 6x6 carries the evaluation and is trusted on its own, while
the 28 perimeter zones sit at the edge of the field where oblique incidence
and partial-target fill make isolated dropouts and range inflation most
likely. A perimeter zone is therefore kept only if at least one adjacent zone
also passed stage 1. Use ``zone_mask`` to apply both.

Corroboration is single-pass, not iterative: adjacency is evaluated against
the stage-1 mask, so pruning one perimeter zone never cascades into pruning
its neighbour. This keeps the operation deterministic and order-independent.

Dual-return resolution (stage 0, mandatory)
--------------------------------------------
The TMF8828 always reports two returns per zone; VEET2 exposes no
object-1-only mode. ``resolve_dual_return`` folds object 1 (primary) and
object 2 (secondary) into a single (z, conf) pair *before* stage 1: object 1
is used whenever it passes the stage-1 gate; otherwise object 2 is consulted
to confirm a confident-far reading, rescue a real target object 1 missed, or
-- if object 2 is itself unusable -- mark the zone ambiguous and excluded.

Confidence semantics
--------------------
TMF8828 confidence is the SNR of the detected histogram peak, NOT a normalised
percentage. Thresholds are raw register values and are only valid for the
encoding they were characterised under -- see CONFIDENCE_ENCODING.

Provenance: thresholds come from bench characterisation and should be
re-derived against a reference target set before being used for any
published analysis. The curve is loaded from ``CONFIDENCE_THRESHOLDS_PATH``
(a JSON file alongside this module: bench anchor points + a power-law fit
rule + a floor), so lab characterisation can update it without touching
code -- see ``reload_confidence_thresholds`` and VEET_TOF_INTEGRATION.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONF_MAX = 255

# Distance in mm at which the sensor's usable range begins. Returns below this
# are physically impossible for a spectacle-mounted device and are dropped
# regardless of reported confidence.
Z_FLOOR_MM = 10.0

# The sensor's minimum reporting distance: it cannot produce a genuine
# time-of-flight return below this. A reading here -- including exactly 0 --
# is a no-return artifact (e.g. nothing in range), not a real near-field
# measurement, and is deterministically reinterpreted as far/optical infinity
# rather than dropped as invalid. This is unconditional (not confidence-gated):
# the confidence byte accompanying a sub-floor reading was calibrated for the
# near-field SNR interpretation, which no longer applies once the distance
# itself is known to be a floor artifact.
NO_RETURN_FLOOR_MM = 10.0

# Finite stand-in for "no return" -- avoids inf breaking isfinite() checks
# elsewhere. Lands exactly on the far band's near edge (BANDS[7], inclusive),
# and to_diopters clips anything >= 2000 mm to the same 0.5 D floor regardless
# of exact magnitude, so the stand-in's precise value doesn't matter beyond
# landing in the far band.
NO_RETURN_SENTINEL_MM = 2000.0


def _resolve_no_return(z_mm: np.ndarray) -> np.ndarray:
    """Reinterpret sub-floor distances (no real return) as far/optical infinity.

    NaN and in-range values pass through unchanged (NaN < floor is False).
    """
    z = np.asarray(z_mm, dtype=float)
    return np.where(z < NO_RETURN_FLOOR_MM, NO_RETURN_SENTINEL_MM, z)

# --- Grid geometry ---------------------------------------------------------
GRID_SHAPE = (8, 8)

# Nominal full-array field of view, degrees.
# CONFIRMED: each row spans 41 deg horizontally, so the 8 columns cover 41 deg
# and the 8 rows cover 52 deg. The field is TALLER THAN IT IS WIDE.
# Named constants are used deliberately -- a bare (41, 52) tuple is ambiguous
# and was previously assigned the wrong way round.
FOV_H_DEG = 41.0  # across columns, array axis -1
FOV_V_DEG = 52.0  # across rows, array axis -2

# Tuples below are ordered (rows, cols) == (vertical, horizontal) to match
# numpy axis order. Do not reorder.
FOV_FULL_DEG = (FOV_V_DEG, FOV_H_DEG)
ZONE_PITCH_DEG = (FOV_V_DEG / GRID_SHAPE[0], FOV_H_DEG / GRID_SHAPE[1])  # 6.5, 5.125

# --- Guard band -----------------------------------------------------------
# Datasheet 7.5.2: VCSEL field of illumination is 41x47 deg at 1/e^2 against a
# sensor FoV of 41x52 deg. Per axis:
#   columns (x): FoI 41 deg == FoV 41 deg -> outer columns fully illuminated
#   rows    (y): FoI 47 deg <  FoV 52 deg -> outer rows overrun illumination by
#                2.5 deg each side, i.e. 38% of the outer row sits beyond 1/e^2
# The weak perimeter is therefore a ROW problem, not a uniform edge problem.
# Default guard band trims the top and bottom rows only and keeps all eight
# columns: 48 zones, full 41 deg horizontal field retained.
ROW_GUARD = np.zeros(GRID_SHAPE, dtype=bool)
ROW_GUARD[1:-1, :] = True  # 6 rows x 8 cols = 48 zones

# Conservative alternative: symmetric 6x6 trim, 36 zones. Use if bench data
# shows column-edge degradation the illumination geometry does not predict.
INTERIOR = np.zeros(GRID_SHAPE, dtype=bool)
INTERIOR[1:-1, 1:-1] = True

ANALYSIS_FIELD = ROW_GUARD
PERIMETER = ~ANALYSIS_FIELD

FOV_ANALYSIS_DEG = (
    FOV_V_DEG * ANALYSIS_FIELD.any(axis=1).sum() / GRID_SHAPE[0],
    FOV_H_DEG * ANALYSIS_FIELD.any(axis=0).sum() / GRID_SHAPE[1],
)  # (vertical, horizontal) = 39.0 V x 41.0 H for the default field

_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OFFSETS_8 = _OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def field_coverage_mm(z_mm: float, field: str = "analysis") -> tuple[float, float]:
    """Linear scene extent (mm) subtended by the field at a given distance.

    ``field`` is "analysis" (central 6x6) or "full" (all 64 zones).

    Returns ``(vertical_mm, horizontal_mm)`` -- ordered to match numpy axes,
    NOT the conventional width-by-height. The VEET field is taller than wide,
    so the two are easy to transpose by accident.
    """
    fov = FOV_ANALYSIS_DEG if field == "analysis" else FOV_FULL_DEG
    return tuple(2.0 * z_mm * math.tan(math.radians(f) / 2.0) for f in fov)


# --- Confidence semantics --------------------------------------------------
# TMF8828 confidence is the SIGNAL-TO-NOISE RATIO of the detected histogram
# peak. It is NOT a normalised 0-100% quality score, and 255 does NOT mean
# "100% confident". Thresholds below are raw register values validated
# empirically against this sensor; the percentages they were originally
# described with were a misreading and have been removed.
#
# Two encodings exist, selected by the logarithmic_confidence register bit:
#   0 (linear) : raw value IS the SNR, clipped at 255
#   1 (log)    : raw 0-40 is SNR directly; above 40,
#                SNR = 40 * 1.053676 ** (raw - 40)
#
# THRESHOLDS ARE ONLY VALID FOR THE ENCODING THEY WERE CHARACTERISED UNDER.
# Under linear encoding the gate spans SNR 52-243 (4.7x). Under logarithmic
# it spans SNR 75-1,627,792 (21,730x). Set CONFIDENCE_ENCODING to the value
# used by the deployed firmware and assert it against session metadata --
# a mismatch breaks every threshold silently, with no error raised.
# CONFIRMED against deployed firmware: logarithmic_confidence = 0.
# Raw value is SNR directly, clipped at 255. Gate therefore spans SNR 52-243.
# Note the ceiling: the band-0 threshold (243) sits 12 counts below saturation,
# so near-field returns cluster against the clip and the top bands cannot be
# discriminated beyond SNR 255. Adequate for gating; inadequate for
# characterising the near-field SNR distribution -- see CHARACTERISATION note.
CONFIDENCE_ENCODING = "linear"  # "linear" | "logarithmic" | "unknown"

# CHARACTERISATION: to measure the true near-field SNR distribution rather than
# a pile-up at 255, re-run bench characterisation with logarithmic_confidence=1
# and convert back via  linear_raw = min(confidence_to_snr(log_raw), 255).
# Do NOT deploy in logarithmic mode without re-deriving every threshold.

_CONF_BREAKPOINT = 40
_CONF_GROWTH_RATE = 1.053676


def confidence_to_snr(raw: np.ndarray, encoding: str | None = None) -> np.ndarray:
    """Decode a raw confidence byte to actual SNR.

    Diagnostic and reporting use only -- the validity gate compares raw values
    directly, so decoding is never required on the hot path.
    """
    enc = encoding or CONFIDENCE_ENCODING
    r = np.asarray(raw, dtype=float)
    if enc == "linear":
        return r
    if enc == "logarithmic":
        return np.where(
            r <= _CONF_BREAKPOINT,
            r,
            _CONF_BREAKPOINT * _CONF_GROWTH_RATE ** (r - _CONF_BREAKPOINT),
        )
    raise ValueError(
        "CONFIDENCE_ENCODING is 'unknown'. Confirm the logarithmic_confidence "
        "register bit in the deployed firmware before decoding SNR."
    )


@dataclass(frozen=True)
class Band:
    """One dioptric reporting band (geometry + literature anchor).

    Bands no longer carry a confidence threshold: the stage-1 gate uses the
    continuous threshold curve (see ``min_confidence``), which is
    deliberately decoupled from the reporting-band edges. Use
    ``band_min_raw_range(band)`` to see what the curve does inside a band.
    """

    index: int
    name: str
    z_min_mm: float          # inclusive
    z_max_mm: float          # exclusive
    anchor: str              # literature anchor for the reporting band

    @property
    def d_max(self) -> float:
        """Vergence in dioptres at the near edge of the band."""
        return math.inf if self.z_min_mm <= 0 else 1000.0 / self.z_min_mm

    @property
    def d_min(self) -> float:
        """Vergence in dioptres at the far edge of the band."""
        return 0.0 if math.isinf(self.z_max_mm) else 1000.0 / self.z_max_mm


# Band geometry (edges, names, literature anchors) is NOT lab-tunable -- it's
# coupled to the dioptric histogram (Invariant: band edges and dioptric bins
# stay coupled). Confidence thresholds are a separate, lab-tunable curve.
_BAND_STRUCTURE = [
    # index, name,                z_min,  z_max,  anchor
    (0, "artifact_occlusion",        0.0,   100.0, "sub-100 mm, treat as occlusion"),
    (1, "extreme_near",            100.0,   200.0, "IMI <20 cm risk threshold"),
    (2, "close_near",              200.0,   300.0, "Sydney Myopia Study; Gajjar 2022"),
    (3, "standard_near",           300.0,   400.0, "clinical near reference 33/40 cm"),
    (4, "near_intermediate",       400.0,   600.0, "upper edge of Ostrin near band"),
    (5, "intermediate",            600.0,  1000.0, "Ostrin intermediate band"),
    (6, "room",                   1000.0,  2000.0, "-"),
    (7, "far",                    2000.0, math.inf, "effective optical infinity"),
]

BANDS: tuple[Band, ...] = tuple(
    Band(index, name, zlo, zhi, anchor) for index, name, zlo, zhi, anchor in _BAND_STRUCTURE
)
# Upper edges used for vectorised lookup. searchsorted(..., side="right")
# places a sample exactly on an edge into the FARTHER band, matching the
# "100-200 mm" convention where 100 mm belongs to band 1.
_EDGES = np.array([b.z_max_mm for b in BANDS[:-1]], dtype=float)


def band_index(z_mm: np.ndarray) -> np.ndarray:
    """Map distances (mm) to band indices 0-7. Shape-preserving."""
    return np.searchsorted(_EDGES, _resolve_no_return(z_mm), side="right")


# --- Confidence gate: continuous threshold curve ------------------------------
#
# The stage-1 threshold is the fitted curve evaluated at each zone's own
# measured distance -- no binning. Distances are clamped to
# [Z_FLOOR_MM, CONF_LUT_MAX_MM] first, so everything at or beyond 2000 mm
# (including the no-return sentinel and NaN) gets the 2000 mm value, which
# is the floor for any realistic fit.
#
# CONF_BIN_LO_MM / the LUT remain for two jobs only: a 10 mm sampling of the
# curve for display and export (``confidence_curve``), and the legacy
# per-band step gate (format v1 / no config), whose band edges are all
# multiples of 10 mm so the step LUT reproduces it exactly.
#
# Bumped whenever gate semantics change, so outputs from different gates are
# never silently compared. 1 = per-band step gate; 2 = 10 mm curve;
# 3 = 10 mm curve + weak-near-unbacked exclusion in resolve_dual_return;
# 4 = continuous curve (no 10 mm binning) + weak-near-unbacked exclusion.
CONFIDENCE_GATE_VERSION = 4

CONF_BIN_MM = 10.0
CONF_LUT_MAX_MM = 2000.0
CONF_BIN_LO_MM = np.arange(Z_FLOOR_MM, CONF_LUT_MAX_MM + CONF_BIN_MM, CONF_BIN_MM)

# Default lower clamp on the fitted curve. A pure power law keeps falling past
# ~1.2 m (5 at 2000 mm for the 2026-09 bench fit), below anything bench data
# supports; the threshold holds at this floor instead.
DEFAULT_FLOOR_MIN_RAW = 10

# Stage-0 rule: an object-1 return that fails the gate, with NO object 2
# behind it, is normally recoded as confident far (a phantom over open
# space). But a genuine near surface also occludes object 2, so a real but
# dim near target (skin, dark material) would be pushed to far -- a bias.
# Below this distance such zones are excluded (ambiguous) instead, turning
# that error into missing data rather than a far bias. Lab-tunable via the
# config's "weak_unbacked_exclude_below_mm"; 0 disables it (legacy v1 files
# load with 0 so they reproduce the old gate exactly).
DEFAULT_WEAK_UNBACKED_EXCLUDE_BELOW_MM = 400.0

# Preliminary bench-characterisation per-band values. Used only as a step
# curve when no config file exists (e.g. fresh clone), and as the base for
# legacy per-band (format v1) config files.
_DEFAULT_MIN_RAW_BY_NAME = {
    "artifact_occlusion": 243,
    "extreme_near": 230,
    "close_near": 205,
    "standard_near": 154,
    "near_intermediate": 103,
    "intermediate": 77,
    "room": 64,
    "far": 52,
}

# Lab validation edits this file in place -- see reload_confidence_thresholds()
# and VEET_TOF_INTEGRATION.md §3.2. Not committed-and-forget: this is the
# expected edit surface as characterisation progresses.
CONFIDENCE_THRESHOLDS_PATH = Path(__file__).with_name("veet_tof_confidence_thresholds.json")

# Set by _load_confidence_lut() below; introspectable by calling tools so a
# GUI can show whether it's showing lab-calibrated or fallback thresholds,
# and which rule produced the curve.
CONFIDENCE_THRESHOLDS_SOURCE: str = ""
CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE: bool = False
CONFIDENCE_RULE: str = ""            # "power_law_fit" | "per_band_step"
CONFIDENCE_CURVE_DESCRIPTION: str = ""
WEAK_UNBACKED_EXCLUDE_BELOW_MM: float = 0.0


@dataclass(frozen=True)
class PowerLawFit:
    """``min_raw(d) = clip(ceil(a * d**-k), floor_min_raw, 255)``, d in mm."""

    a: float
    k: float
    floor_min_raw: int
    n_anchors_fit: int
    n_anchors_censored: int  # anchors at 0 or 255 -- clipped, so excluded from the fit

    def evaluate(self, z_mm: np.ndarray) -> np.ndarray:
        z = np.asarray(z_mm, dtype=float)
        with np.errstate(divide="ignore", over="ignore"):
            raw = np.ceil(self.a * z ** -self.k)
        return np.clip(raw, self.floor_min_raw, CONF_MAX).astype(np.int16)

    def describe(self) -> str:
        return (f"power-law fit min_raw = {self.a:.4g} * d^-{self.k:.3f}, "
                f"floor {self.floor_min_raw}, {self.n_anchors_fit} anchors fit"
                + (f" ({self.n_anchors_censored} censored excluded)" if self.n_anchors_censored else ""))


CONFIDENCE_FIT: "PowerLawFit | None" = None


def fit_power_law(anchors, floor_min_raw: int = DEFAULT_FLOOR_MIN_RAW) -> PowerLawFit:
    """Least-squares fit of log(min_raw) = log(a) - k*log(d) through anchors.

    ``anchors`` is a sequence of (distance_mm, min_raw) pairs. Anchors at 0 or
    CONF_MAX are censored by the byte encoding (the true requirement is
    unknown beyond the clip), so they're excluded from the fit rather than
    flattening its slope. Raises ValueError if fewer than two usable anchors
    remain, or if the fit doesn't fall with distance (k <= 0) -- that means
    the anchors are wrong, not that the sensor got better with range.
    """
    pts = [(float(d), float(v)) for d, v in anchors]
    usable = [(d, v) for d, v in pts if 0 < v < CONF_MAX]
    if len({d for d, _ in usable}) < 2:
        raise ValueError(
            f"Power-law fit needs at least two anchors with 0 < min_raw < {CONF_MAX} "
            f"at distinct distances; got {len(usable)} usable of {len(pts)}.")
    d = np.array([p[0] for p in usable])
    v = np.array([p[1] for p in usable])
    slope, intercept = np.polyfit(np.log(d), np.log(v), 1)
    k = -float(slope)
    if k <= 0:
        raise ValueError(
            f"Power-law fit rises with distance (k={k:.3f}); check the anchors -- "
            f"required confidence should fall as distance grows.")
    return PowerLawFit(float(np.exp(intercept)), k, int(floor_min_raw),
                       len(usable), len(pts) - len(usable))


def _step_lut(min_raw_by_name: dict[str, int]) -> np.ndarray:
    """Per-band step curve on the 10 mm grid. Band edges are multiples of 10 mm
    so every bin falls wholly inside one band -- this reproduces the legacy
    per-band gate exactly."""
    per_band = np.array([min_raw_by_name[b.name] for b in BANDS], dtype=np.int16)
    return per_band[np.searchsorted(_EDGES, CONF_BIN_LO_MM, side="right")]


def _check_byte(value, what: str, target) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not (0 <= value <= CONF_MAX):
        raise ValueError(f"Bad {what} in {target}: {value!r} (must be an integer 0-{CONF_MAX})")
    return value


def _parse_v1(data: dict, target) -> np.ndarray:
    """Legacy per-band map: {"min_raw_by_band": {name: int}} or a bare map."""
    raw_map = data.get("min_raw_by_band", data)
    if not isinstance(raw_map, dict):
        raise ValueError(f"{target}: 'min_raw_by_band' must map band names to integers.")
    unknown = set(raw_map) - set(_DEFAULT_MIN_RAW_BY_NAME)
    if unknown:
        raise ValueError(
            f"Unknown band name(s) in {target}: {sorted(unknown)}. "
            f"Expected a subset of {sorted(_DEFAULT_MIN_RAW_BY_NAME)}.")
    values = dict(_DEFAULT_MIN_RAW_BY_NAME)
    for name, value in raw_map.items():
        values[name] = _check_byte(value, f"threshold for band '{name}'", target)
    return _step_lut(values)


def _parse_v2(data: dict, target) -> "tuple[PowerLawFit, float]":
    """Curve config: anchors + rule, fit at load."""
    version = data.get("format_version", 2)
    if version != 2:
        raise ValueError(f"{target}: unsupported format_version {version!r} (expected 2).")
    rule = data.get("rule")
    if rule != "power_law_fit":
        raise ValueError(f"{target}: unsupported rule {rule!r} (expected 'power_law_fit').")
    floor_min_raw = _check_byte(data.get("floor_min_raw", DEFAULT_FLOOR_MIN_RAW),
                                "floor_min_raw", target)
    cutoff = data.get("weak_unbacked_exclude_below_mm", DEFAULT_WEAK_UNBACKED_EXCLUDE_BELOW_MM)
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) \
            or cutoff < 0:
        raise ValueError(f"{target}: weak_unbacked_exclude_below_mm {cutoff!r} must be a number >= 0.")

    anchors = data.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"{target}: 'anchors' must be a non-empty list of [distance_mm, min_raw].")
    parsed = []
    for i, anchor in enumerate(anchors):
        if not (isinstance(anchor, list) and len(anchor) == 2):
            raise ValueError(f"{target}: anchor #{i} {anchor!r} must be [distance_mm, min_raw].")
        d, v = anchor
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) \
                or d < Z_FLOOR_MM:
            raise ValueError(f"{target}: anchor #{i} distance {d!r} must be a number >= {Z_FLOOR_MM:g} mm.")
        parsed.append((float(d), _check_byte(v, f"min_raw for anchor #{i}", target)))
    distances = [d for d, _ in parsed]
    if any(b <= a for a, b in zip(distances, distances[1:])):
        raise ValueError(f"{target}: anchor distances must be strictly increasing.")
    try:
        return fit_power_law(parsed, floor_min_raw), float(cutoff)
    except ValueError as e:
        raise ValueError(f"{target}: {e}") from e


def _load_confidence_lut(path: "str | Path | None" = None) -> np.ndarray:
    """Load the gate config; returns its 10 mm LUT (display / step gate).

    Format v2 (has "anchors"): power-law fit through the anchors, floored.
    Format v1 (per-band map): legacy step curve, unlisted bands default.
    No file: step curve from the built-in preliminary defaults.
    Raises ValueError on a malformed config -- a bad threshold should fail
    loudly, not gate silently on garbage.
    """
    global CONFIDENCE_THRESHOLDS_SOURCE, CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
    global CONFIDENCE_RULE, CONFIDENCE_CURVE_DESCRIPTION, CONFIDENCE_FIT, WEAK_UNBACKED_EXCLUDE_BELOW_MM
    target = Path(path) if path is not None else CONFIDENCE_THRESHOLDS_PATH

    if not target.exists():
        lut, fit, rule = _step_lut(_DEFAULT_MIN_RAW_BY_NAME), None, "per_band_step"
        cutoff = DEFAULT_WEAK_UNBACKED_EXCLUDE_BELOW_MM
        source, from_file = f"{target} (not found; using built-in defaults)", False
    else:
        with open(target, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse confidence-threshold config {target}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"{target} must be a JSON object.")
        if "anchors" in data or "format_version" in data:
            fit, cutoff = _parse_v2(data, target)
            lut, rule = fit.evaluate(CONF_BIN_LO_MM), "power_law_fit"
        else:
            lut, fit, rule = _parse_v1(data, target), None, "per_band_step"
            cutoff = 0.0  # legacy files reproduce the old gate exactly
        source, from_file = str(target), True

    CONFIDENCE_THRESHOLDS_SOURCE = source
    CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE = from_file
    CONFIDENCE_RULE = rule
    CONFIDENCE_FIT = fit
    WEAK_UNBACKED_EXCLUDE_BELOW_MM = cutoff
    CONFIDENCE_CURVE_DESCRIPTION = (
        (fit.describe() if fit else "per-band step curve (legacy format)")
        + (f"; weak unbacked returns < {cutoff:g} mm excluded" if cutoff > 0 else ""))
    return lut


_CONF_LUT: np.ndarray = _load_confidence_lut()


def reload_confidence_thresholds(path: "str | Path | None" = None) -> np.ndarray:
    """Re-read the confidence-threshold config and re-fit the curve in
    place, without restarting the process -- the expected workflow while lab
    characterisation is ongoing: edit CONFIDENCE_THRESHOLDS_PATH, then call
    this. Returns the new curve sampled on the 10 mm display grid
    (aligned with ``CONF_BIN_LO_MM``).

    All gating functions (``valid_mask``, ``min_confidence``,
    ``resolve_dual_return``, ...) read the module-level ``CONFIDENCE_FIT`` /
    ``_CONF_LUT`` at call time, so they pick up the reload automatically. On a malformed file this
    raises and the previously loaded curve stays in effect.
    """
    global _CONF_LUT
    _CONF_LUT = _load_confidence_lut(path)
    return _CONF_LUT.copy()


def confidence_curve() -> tuple[np.ndarray, np.ndarray]:
    """(distance_mm, min_raw): the curve in effect sampled every 10 mm, for
    display/export. Gating itself evaluates the curve continuously."""
    return CONF_BIN_LO_MM.copy(), _CONF_LUT.copy()


def band_min_raw_range(band: Band) -> tuple[int, int]:
    """(near-edge, far-edge) threshold the curve applies inside a reporting band."""
    lo = max(band.z_min_mm, Z_FLOOR_MM)
    hi = min(band.z_max_mm, CONF_LUT_MAX_MM + CONF_BIN_MM) - 1e-6
    vals = min_confidence(np.array([lo, max(lo, hi)]))
    return int(vals[0]), int(vals[1])


def min_confidence(z_mm: np.ndarray) -> np.ndarray:
    """Minimum raw confidence (0-255) required at each measured distance
    (sub-floor readings resolve to far first). With a fitted curve this is
    ``CONFIDENCE_FIT`` evaluated at the distance itself; ``ceil`` in the fit
    makes ``conf >= min_confidence(z)`` exactly ``conf >= a*z**-k`` for
    integer confidence. Legacy step configs use the per-band LUT."""
    z = _resolve_no_return(z_mm)
    if CONFIDENCE_FIT is not None:
        z = np.where(np.isnan(z), CONF_LUT_MAX_MM, z)
        return CONFIDENCE_FIT.evaluate(np.clip(z, Z_FLOOR_MM, CONF_LUT_MAX_MM))
    idx = np.searchsorted(CONF_BIN_LO_MM, z, side="right") - 1
    return _CONF_LUT[np.clip(idx, 0, len(CONF_BIN_LO_MM) - 1)]


def valid_mask(z_mm: np.ndarray, conf_raw: np.ndarray) -> np.ndarray:
    """Boolean mask of zones passing the distance-dependent confidence gate.

    Sub-floor readings (< NO_RETURN_FLOOR_MM, including 0) are a deterministic
    no-return artifact and are always valid-far, independent of the confidence
    gate -- see NO_RETURN_FLOOR_MM.

    Parameters
    ----------
    z_mm : array of measured distances in mm, any shape (e.g. (n_samples, 8, 8))
    conf_raw : array of per-zone confidence, 0-255, same shape as ``z_mm``

    Returns
    -------
    Boolean array, same shape. True == trust this zone.
    """
    z = np.asarray(z_mm, dtype=float)
    c = np.asarray(conf_raw)
    if z.shape != c.shape:
        raise ValueError(f"shape mismatch: z_mm {z.shape} vs conf_raw {c.shape}")
    no_return = z < NO_RETURN_FLOOR_MM
    z_resolved = _resolve_no_return(z)
    finite = np.isfinite(z_resolved) & (z_resolved >= Z_FLOOR_MM)
    gated = finite & (c >= min_confidence(z_resolved))
    return no_return | gated


def resolve_dual_return(
    z1_mm: np.ndarray,
    conf1_raw: np.ndarray,
    z2_mm: np.ndarray,
    conf2_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the TMF8828's two per-zone returns into one (z, conf) pair.

    Object 1 (the primary/strongest return) is used whenever it passes the
    normal validity gate. When it doesn't -- either a sub-floor/no-return
    reading or a real-but-low-confidence one -- object 2 is consulted:

    - object 2 sub-floor (no second target)  -> confident far (2000mm).
      A low-confidence object-1 reading is typically a phantom return from
      diffuse reflection off a nearby surface, so it should not override a
      clean "nothing there" from object 2. EXCEPT when that weak object-1
      reading is nearer than WEAK_UNBACKED_EXCLUDE_BELOW_MM: a genuine near
      surface also occludes object 2, so a dim real target there would be
      pushed to far. Those zones are ambiguous (excluded) instead -- missing
      data rather than a far bias. Sub-floor/zero object 1 is unaffected
      (it's a no-return, not a weak return) and still resolves far.
    - object 2 passes the validity gate        -> "rescued": object 2's
      (z, conf) becomes the resolved measurement.
    - object 2 is also nonzero-but-low-confidence -> ambiguous: neither
      return can be trusted. Encoded as conf_resolved=0 paired with a
      z_resolved >= NO_RETURN_FLOOR_MM (z2 here; z1 for the weak-near-
      unbacked case above, since z2 is sub-floor there), which
      guarantees the zone fails the existing ``valid_mask`` confidence
      check without needing a separate exclusion mask threaded through
      every downstream function. ``ambiguous_mask`` is returned purely so
      callers can count these zones (see ``zone_yield``'s ambiguous_mask
      parameter) -- it plays no role in gating.

    Object 2 is gated by the same threshold curve as object 1 (same
    sensor, same characterization curve), looked up at object 2's own
    distance.

    Parameters
    ----------
    z1_mm, conf1_raw, z2_mm, conf2_raw : arrays of identical shape.

    Returns
    -------
    (z_resolved, conf_resolved, ambiguous_mask) -- z_resolved/conf_resolved
    feed directly into ``valid_mask``/``analysis_mask`` exactly like a
    single-return (z_mm, conf_raw) pair; ambiguous_mask is diagnostic only.
    """
    z1 = np.asarray(z1_mm, dtype=float)
    c1 = np.asarray(conf1_raw, dtype=float)
    z2 = np.asarray(z2_mm, dtype=float)
    c2 = np.asarray(conf2_raw, dtype=float)
    if not (z1.shape == c1.shape == z2.shape == c2.shape):
        raise ValueError(
            f"shape mismatch: z1 {z1.shape}, conf1 {c1.shape}, "
            f"z2 {z2.shape}, conf2 {c2.shape}"
        )

    obj1_valid = (z1 >= NO_RETURN_FLOOR_MM) & (c1 >= min_confidence(z1))
    obj2_zero = z2 < NO_RETURN_FLOOR_MM
    obj2_valid = (z2 >= NO_RETURN_FLOOR_MM) & (c2 >= min_confidence(z2))
    obj2_low = ~obj2_zero & ~obj2_valid
    weak_near_unbacked = (~obj1_valid & obj2_zero & (z1 >= NO_RETURN_FLOOR_MM)
                          & (z1 < WEAK_UNBACKED_EXCLUDE_BELOW_MM))
    ambiguous = ~obj1_valid & (obj2_low | weak_near_unbacked)

    z_resolved = np.where(
        obj1_valid | weak_near_unbacked, z1,
        np.where(obj2_zero, NO_RETURN_SENTINEL_MM, z2))
    conf_resolved = np.where(
        obj1_valid, c1, np.where(ambiguous, 0.0, np.where(obj2_zero, float(CONF_MAX), c2))
    )
    return z_resolved, conf_resolved, ambiguous


def _neighbor_count(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Count adjacent True zones for every cell of a (..., 8, 8) mask."""
    offsets = _OFFSETS_8 if connectivity == 8 else _OFFSETS_4
    m = mask.astype(np.uint8)
    pad_width = [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)]
    padded = np.pad(m, pad_width, mode="constant", constant_values=0)
    h, w = mask.shape[-2:]
    total = np.zeros_like(m, dtype=np.uint8)
    for dy, dx in offsets:
        total += padded[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return total


def spatial_corroboration(
    gate_mask: np.ndarray,
    connectivity: int = 8,
    require_interior_support: bool = False,
) -> np.ndarray:
    """Prune isolated perimeter zones from a stage-1 confidence mask.

    Interior zones (central 6x6) pass through untouched. A perimeter zone is
    retained only if at least one adjacent zone also passed stage 1.

    Parameters
    ----------
    gate_mask : boolean array of shape (..., 8, 8) from ``valid_mask``
    connectivity : 8 for the Moore neighbourhood (includes diagonals, the
        default and more permissive reading of "adjacent"), or 4 for the
        von Neumann neighbourhood.
    require_interior_support : if True, only interior zones can corroborate a
        perimeter zone. This closes the loophole where two adjacent noisy
        perimeter zones mutually validate each other, at the cost of
        discarding genuine returns along the field edge. Worth testing both
        ways against bench data before fixing the choice.

    Returns
    -------
    Boolean array, same shape as ``gate_mask``.
    """
    gate_mask = np.asarray(gate_mask, dtype=bool)
    if gate_mask.shape[-2:] != GRID_SHAPE:
        raise ValueError(f"expected trailing shape {GRID_SHAPE}, got {gate_mask.shape[-2:]}")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    support = gate_mask & INTERIOR if require_interior_support else gate_mask
    has_neighbor = _neighbor_count(support, connectivity) > 0
    return gate_mask & (INTERIOR | has_neighbor)


def analysis_mask(z_mm: np.ndarray, conf_raw: np.ndarray) -> np.ndarray:
    """PRIMARY entry point. Confidence gate restricted to the 6x6 analysis field.

    The 28 perimeter zones are excluded unconditionally. They sit at the edge
    of the field where oblique incidence and partial target fill make range
    inflation most likely, and no free-living ground truth exists against which
    a recovery heuristic could be validated. Excluding them is a declared,
    lab-verifiable specification; recovering them is not.

    All reported measures derive from this mask.
    """
    return valid_mask(z_mm, conf_raw) & ANALYSIS_FIELD


def zone_mask(
    z_mm: np.ndarray,
    conf_raw: np.ndarray,
    connectivity: int = 8,
    require_interior_support: bool = False,
) -> np.ndarray:
    """DIAGNOSTIC ONLY. Full 64-zone mask with perimeter corroboration.

    Retained to exploit the guard band as an internal consistency channel and
    to preserve Phase 2 optionality. Do NOT use for reported measures --
    ``analysis_mask`` is the reporting path.
    """
    gate = valid_mask(z_mm, conf_raw)
    return spatial_corroboration(gate, connectivity, require_interior_support)


def zone_yield(
    z_mm: np.ndarray,
    conf_raw: np.ndarray,
    ambiguous_mask: np.ndarray | None = None,
    **kwargs,
) -> dict[str, float]:
    """Diagnostic: what each validity stage costs, as fractions of all zones.

    ``perimeter_yield`` is the guard-band health indicator. A persistent drop
    relative to ``interior_yield`` is expected; a sharp change over time or
    across devices is a flag worth chasing.

    ``ambiguous_mask``, if given (from ``resolve_dual_return``), adds
    ``ambiguous_dual_return_fraction`` -- the share of analysis-field zones
    excluded because neither TOF return could be trusted. Omitted entirely
    for single-return callers, so this stat only appears for sessions with
    dual-return data.
    """
    gate = valid_mask(z_mm, conf_raw)
    final = spatial_corroboration(gate, **kwargs)
    n = float(gate.size)
    n_perim = float(gate[..., PERIMETER].size)
    result = {
        "passed_confidence_gate": float(gate.sum()) / n,
        "analysis_field_yield": float((gate & ANALYSIS_FIELD).sum())
        / float(gate[..., ANALYSIS_FIELD].size),
        "passed_both_stages": float(final.sum()) / n,
        "pruned_by_corroboration": float((gate & ~final).sum()) / n,
        "interior_yield": float(final[..., INTERIOR].sum()) / float(gate[..., INTERIOR].size),
        "perimeter_yield": float(final[..., PERIMETER].sum()) / n_perim if n_perim else 0.0,
    }
    if ambiguous_mask is not None:
        am = np.asarray(ambiguous_mask, dtype=bool)
        result["ambiguous_dual_return_fraction"] = float(am[..., ANALYSIS_FIELD].sum()) / float(
            am[..., ANALYSIS_FIELD].size
        )
    return result


def to_diopters(z_mm: np.ndarray, z_cap_mm: float = 2000.0) -> np.ndarray:
    """Convert distance (mm) to vergence (D), clamping the far tail.

    Returns beyond ``z_cap_mm`` are censored to the cap rather than allowed to
    approach zero asymptotically, since the sensor cannot distinguish 3 m from
    10 m and both are optically equivalent for accommodative demand. Sub-floor
    (no-return) readings are resolved the same way -- see NO_RETURN_FLOOR_MM.
    """
    z = _resolve_no_return(z_mm)
    return 1000.0 / np.clip(z, Z_FLOOR_MM, z_cap_mm)


# ---------------------------------------------------------------------------
# Dioptric histogram: uniform 0.25 D bins, chosen to match the eye's depth of
# focus (~+/-0.25 D). Binning finer resolves differences the retina cannot.
# ---------------------------------------------------------------------------

D_BIN_WIDTH = 0.25
D_BIN_MAX = 10.0
D_BIN_EDGES = np.arange(0.0, D_BIN_MAX + D_BIN_WIDTH, D_BIN_WIDTH)  # 41 edges


def dioptric_histogram(
    z_mm: np.ndarray,
    conf_raw: np.ndarray,
    sample_period_s: float = 5.0,
) -> np.ndarray:
    """Time-weighted dioptric histogram over valid returns.

    Returns an array of length ``len(D_BIN_EDGES)`` where the final element is
    the overflow bin (> D_BIN_MAX, i.e. closer than 100 mm). Units are seconds.

    ``sample_period_s`` defaults to 5.0 s (0.2 Hz nominal sampling).
    """
    mask = analysis_mask(z_mm, conf_raw)
    d = to_diopters(np.asarray(z_mm, dtype=float))[mask]
    counts, _ = np.histogram(d, bins=np.append(D_BIN_EDGES, np.inf))
    return counts.astype(float) * sample_period_s


def dioptric_dose(
    z_mm: np.ndarray,
    conf_raw: np.ndarray,
    sample_period_s: float = 5.0,
) -> float:
    """Time-integrated accommodative demand, in dioptre-hours.

    The distance analogue of lux-hours on the ALS channel.
    """
    mask = analysis_mask(z_mm, conf_raw)
    if not mask.any():
        return 0.0
    d = to_diopters(np.asarray(z_mm, dtype=float))[mask]
    return float(d.sum() * sample_period_s / 3600.0)


if __name__ == "__main__":
    print(f"Confidence thresholds: {CONFIDENCE_THRESHOLDS_SOURCE}")
    print(f"Gate v{CONFIDENCE_GATE_VERSION}: {CONFIDENCE_CURVE_DESCRIPTION}")
    hdr = f"{'idx':>3}  {'band':<20} {'mm range':>15} {'D range':>15} {'min raw':>9}"
    print(hdr)
    print("-" * len(hdr))
    for b in BANDS:
        zr = f"{b.z_min_mm:.0f}-{'inf' if math.isinf(b.z_max_mm) else f'{b.z_max_mm:.0f}'}"
        dr = f"{'inf' if math.isinf(b.d_max) else f'{b.d_max:.2f}'}-{b.d_min:.2f}"
        near, far = band_min_raw_range(b)
        print(f"{b.index:>3}  {b.name:<20} {zr:>15} {dr:>15} {f'{near}-{far}':>9}")
