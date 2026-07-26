"""VEET 2 time-of-flight validity gating and dioptric binning.

Single source of truth for the distance-band definitions used by both:
  1. the per-zone confidence gate (is this return trustworthy?)
  2. the dioptric reporting bands (which literature-anchored bin is it?)

Keeping these coupled prevents the gate and the histogram from drifting apart
as thresholds are refined.

Sensor context
--------------
TOF: 8x8 zone grid, 41 deg x 52 deg FOV, usable range ~20-2000+ mm.
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

Confidence semantics
--------------------
TMF8828 confidence is the SNR of the detected histogram peak, NOT a normalised
percentage. Thresholds are raw register values and are only valid for the
encoding they were characterised under -- see CONFIDENCE_ENCODING.

Provenance: thresholds below are from a preliminary bench characterisation
and should be re-derived against a reference target set before being used
for any published analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CONF_MAX = 255

# Distance in mm at which the sensor's usable range begins. Returns below this
# are physically impossible for a spectacle-mounted device and are dropped
# regardless of reported confidence.
Z_FLOOR_MM = 20.0

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
    """One distance band: validity gate plus dioptric reporting metadata."""

    index: int
    name: str
    z_min_mm: float          # inclusive
    z_max_mm: float          # exclusive
    conf_raw: int            # minimum raw confidence (SNR domain) for validity
    anchor: str              # literature anchor for the reporting band

    @property
    def d_max(self) -> float:
        """Vergence in dioptres at the near edge of the band."""
        return math.inf if self.z_min_mm <= 0 else 1000.0 / self.z_min_mm

    @property
    def d_min(self) -> float:
        """Vergence in dioptres at the far edge of the band."""
        return 0.0 if math.isinf(self.z_max_mm) else 1000.0 / self.z_max_mm


_SPEC = [
    # index, name,                z_min,  z_max,  min_raw, anchor
    (0, "artifact_occlusion",        0.0,   100.0, 243, "sub-100 mm, treat as occlusion"),
    (1, "extreme_near",            100.0,   200.0, 230, "IMI <20 cm risk threshold"),
    (2, "close_near",              200.0,   300.0, 205, "Sydney Myopia Study; Gajjar 2022"),
    (3, "standard_near",           300.0,   400.0, 154, "clinical near reference 33/40 cm"),
    (4, "near_intermediate",       400.0,   600.0, 103, "upper edge of Ostrin near band"),
    (5, "intermediate",            600.0,  1000.0,  77, "Ostrin intermediate band"),
    (6, "room",                   1000.0,  2000.0,  64, "-"),
    (7, "far",                    2000.0, math.inf,  52, "effective optical infinity"),
]

BANDS: tuple[Band, ...] = tuple(
    Band(i, name, zlo, zhi, raw, anchor)
    for i, name, zlo, zhi, raw, anchor in _SPEC
)

# Upper edges used for vectorised lookup. searchsorted(..., side="right")
# places a sample exactly on an edge into the FARTHER band, matching the
# "100-200 mm" convention where 100 mm belongs to band 1.
_EDGES = np.array([b.z_max_mm for b in BANDS[:-1]], dtype=float)
_MIN_RAW = np.array([b.conf_raw for b in BANDS], dtype=np.int16)


def band_index(z_mm: np.ndarray) -> np.ndarray:
    """Map distances (mm) to band indices 0-7. Shape-preserving."""
    return np.searchsorted(_EDGES, np.asarray(z_mm, dtype=float), side="right")


def min_confidence(z_mm: np.ndarray) -> np.ndarray:
    """Minimum raw confidence (0-255) required at each measured distance."""
    return _MIN_RAW[band_index(z_mm)]


def valid_mask(z_mm: np.ndarray, conf_raw: np.ndarray) -> np.ndarray:
    """Boolean mask of zones passing the distance-dependent confidence gate.

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
    finite = np.isfinite(z) & (z >= Z_FLOOR_MM)
    return finite & (c >= min_confidence(z))


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


def zone_yield(z_mm: np.ndarray, conf_raw: np.ndarray, **kwargs) -> dict[str, float]:
    """Diagnostic: what each validity stage costs, as fractions of all zones.

    ``perimeter_yield`` is the guard-band health indicator. A persistent drop
    relative to ``interior_yield`` is expected; a sharp change over time or
    across devices is a flag worth chasing.
    """
    gate = valid_mask(z_mm, conf_raw)
    final = spatial_corroboration(gate, **kwargs)
    n = float(gate.size)
    n_perim = float(gate[..., PERIMETER].size)
    return {
        "passed_confidence_gate": float(gate.sum()) / n,
        "analysis_field_yield": float((gate & ANALYSIS_FIELD).sum())
        / float(gate[..., ANALYSIS_FIELD].size),
        "passed_both_stages": float(final.sum()) / n,
        "pruned_by_corroboration": float((gate & ~final).sum()) / n,
        "interior_yield": float(final[..., INTERIOR].sum()) / float(gate[..., INTERIOR].size),
        "perimeter_yield": float(final[..., PERIMETER].sum()) / n_perim if n_perim else 0.0,
    }


def to_diopters(z_mm: np.ndarray, z_cap_mm: float = 2000.0) -> np.ndarray:
    """Convert distance (mm) to vergence (D), clamping the far tail.

    Returns beyond ``z_cap_mm`` are censored to the cap rather than allowed to
    approach zero asymptotically, since the sensor cannot distinguish 3 m from
    10 m and both are optically equivalent for accommodative demand.
    """
    z = np.asarray(z_mm, dtype=float)
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
    hdr = f"{'idx':>3}  {'band':<20} {'mm range':>15} {'D range':>15} {'min raw':>8}"
    print(hdr)
    print("-" * len(hdr))
    for b in BANDS:
        zr = f"{b.z_min_mm:.0f}-{'inf' if math.isinf(b.z_max_mm) else f'{b.z_max_mm:.0f}'}"
        dr = f"{'inf' if math.isinf(b.d_max) else f'{b.d_max:.2f}'}-{b.d_min:.2f}"
        print(f"{b.index:>3}  {b.name:<20} {zr:>15} {dr:>15} {b.conf_raw:>8}")
