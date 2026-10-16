# veet_tof_bands.py

Single source of truth for VEET 2 time-of-flight (TOF) validity gating and dioptric binning. It answers two questions for every zone of every frame: **is this return trustworthy?** and **if so, which reporting bin does it belong to?**

For the full technical spec — provenance, invariants, integration checklist, open questions — see [VEET_TOF_INTEGRATION.md](VEET_TOF_INTEGRATION.md). This document is the practical, task-oriented companion: what to import, what to call, what the numbers mean, and what breaks if you get it wrong.

> **Status: pre-validation.** The confidence gate is a power-law curve fit to bench anchors (paper/phone targets, ~200–2000 mm). It has not yet been validated against a reference target set — see [Confidence encoding](#confidence-encoding-read-this-first).

---

## Table of contents

- [What this module does](#what-this-module-does)
- [What this module does not do](#what-this-module-does-not-do)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The sensor](#the-sensor)
- [Data contract](#data-contract)
- [Confidence encoding — read this first](#confidence-encoding-read-this-first)
- [Confidence thresholds are lab-editable](#confidence-thresholds-are-lab-editable)
- [The two-stage (three-stage) validity pipeline](#the-two-stage-three-stage-validity-pipeline)
  - [Stage 0 — dual-return resolution (mandatory)](#stage-0--dual-return-resolution-mandatory)
  - [Stage 1 — confidence gate](#stage-1--confidence-gate)
  - [Stage 2 — spatial corroboration / analysis field](#stage-2--spatial-corroboration--analysis-field)
- [Distance bands](#distance-bands)
- [Dioptric conversion and binning](#dioptric-conversion-and-binning)
- [API reference](#api-reference)
- [Common recipes](#common-recipes)
- [Diagnostics](#diagnostics)
- [Gotchas](#gotchas)
- [Running the module directly](#running-the-module-directly)
- [Testing](#testing)
- [Provenance](#provenance)

---

## What this module does

- Decides whether a raw TOF zone reading is trustworthy (`valid_mask`, `analysis_mask`)
- Folds the sensor's two per-zone returns (object 1 / object 2), always present, into one resolved reading before gating (`resolve_dual_return`)
- Excludes structurally weak perimeter zones from reported measures, while keeping them available as a diagnostic channel (`ANALYSIS_FIELD`, `spatial_corroboration`, `zone_mask`)
- Converts trusted distances to dioptric vergence and bins them into a time-weighted histogram and a time-integrated dose (`to_diopters`, `dioptric_histogram`, `dioptric_dose`)
- Classifies distances into eight literature-anchored reporting bands (`band_index`, `BANDS`)
- Reports field-of-view geometry and per-distance scene coverage (`field_coverage_mm`)
- Reports pipeline health/yield stats for QA and cross-device comparison (`zone_yield`)

The validity gate and the reporting bands are deliberately **separate**. The gate is a continuous threshold curve on a 10 mm grid, looked up from each zone's own measured distance. The 8 reporting bands only classify and bin. Refining the curve therefore never moves a reporting edge. The curve is lab-editable via a JSON config file (see [Confidence thresholds are lab-editable](#confidence-thresholds-are-lab-editable)); band geometry is not.

## What this module does not do

Out of scope, by design — build these in the calling tool and call into this module, don't extend it:

- File I/O / CSV parsing
- Session segmentation, wear detection
- ALS / PHO / IMU fusion
- Temporal filtering, epoch/bout analysis

## Requirements

- Python 3.9+ (uses `from __future__ import annotations` and `X | None` syntax)
- `numpy`

No other dependencies. The module is self-contained and has no import-time side effects beyond building lookup arrays.

## Quick start

```python
import numpy as np
import veet_tof_bands as tb

# One frame: 8x8 zone distances (mm) and confidences (raw byte, 0-255)
z_mm = np.random.uniform(50, 2500, size=(8, 8))
conf_raw = np.random.randint(0, 256, size=(8, 8))

# Which zones can be trusted, restricted to the reporting field?
mask = tb.analysis_mask(z_mm, conf_raw)

# Vergence in dioptres for the trusted zones
d = tb.to_diopters(z_mm)[mask]

# Time-weighted histogram over a session (n_samples, 8, 8), 5 s per sample
z_session = np.random.uniform(50, 2500, size=(2000, 8, 8))
c_session = np.random.randint(0, 256, size=(2000, 8, 8))
hist = tb.dioptric_histogram(z_session, c_session, sample_period_s=5.0)
dose = tb.dioptric_dose(z_session, c_session, sample_period_s=5.0)

print(f"Accommodative dose: {dose:.2f} D-hours")
```

VEET2 sessions always carry a second TOF return (Dist2/Conf2); resolve it first:

```python
z_resolved, conf_resolved, ambiguous = tb.resolve_dual_return(z1, c1, z2, c2)
mask = tb.analysis_mask(z_resolved, conf_resolved)
```

## The sensor

VEET 2 uses a TMF8828: an 8×8 zone depth array, nominal field of view **41° horizontal × 52° vertical** (taller than wide — easy to get backwards, see [Gotchas](#gotchas)), usable range roughly 10–2000+ mm. Confidence is reported per zone as an unsigned byte, 0–255. Sampling rate is configuration-dependent and is recorded in the INF line of `Sensor_Data.csv` — read it from there rather than assuming a fixed rate. Nominal sampling for VEET2 is 0.5 Hz (2 s/sample).

The 8×8 frame is assembled from four interleaved sub-captures over a 66–258 ms ranging period, so zones within one logical frame are not strictly simultaneous — a consideration for any cross-zone metric, not something this module corrects for.

## Data contract

Every function takes parallel arrays:

| Arg | Meaning | Dtype | Shape | Units |
|---|---|---|---|---|
| `z_mm` | measured distance per zone | float | `(..., 8, 8)` | mm |
| `conf_raw` | reported confidence per zone | int | `(..., 8, 8)` | 0–255 |

- Leading dimensions are free: pass `(8, 8)` for one frame, `(n_samples, 8, 8)` for a session.
- `z_mm` and `conf_raw` shapes must match exactly, or a `ValueError` is raised.
- **Do not pre-convert confidence to a percentage.** Thresholds are raw integers, compared directly.
- **Do not pre-filter to NaN/0 before calling.** The gate already handles zeros, sub-floor values, and non-finite values.

## Confidence encoding — read this first

TMF8828 confidence is the **signal-to-noise ratio** of the detected histogram peak, not a normalized quality percentage. 255 does not mean "100% confident."

VEET2 firmware is fixed to linear confidence encoding (`logarithmic_confidence = 0`, confirmed against deployed firmware). The TMF8828's logarithmic mode exists in silicon but is not exposed on this hardware, so there is no per-session register bit to track. Under linear encoding the raw byte **is** the SNR, clipped at 255 — `CONFIDENCE_ENCODING` (module constant) is `"linear"`.

`confidence_to_snr(raw, encoding=None)` decodes a raw byte to true SNR for diagnostics/reporting. It's never needed on the gating hot path — the gate compares raw values directly.

## Confidence thresholds are lab-editable

The gate threshold (`min_confidence(z)`) is a curve, sampled on a 10 mm grid from 10 mm to 2000 mm plus one open-ended bin ≥ 2000 mm. Each bin uses the curve's value at its near edge. The curve is

```
min_raw(d) = clip(ceil(a · d^-k), floor_min_raw, 255)        d in mm
```

with `a` and `k` fit at load time (least squares, log-log) through bench anchor points in `veet_tof_confidence_thresholds.json`, next to `veet_tof_bands.py`:

```json
{
  "format_version": 2,
  "rule": "power_law_fit",
  "floor_min_raw": 10,
  "anchors": [[200, 245], [250, 200], [300, 150], [400, 100], [600, 50], [1000, 15], [2000, 10]]
}
```

- `anchors` are `[distance_mm, min_raw]`, strictly increasing in distance, distance ≥ 10 mm, value an integer 0–255. Anchors at exactly 0 or 255 are clipped by the encoding and are excluded from the fit.
- `floor_min_raw` clamps the far tail (default 10). Near range saturates at 255.
- `veet_tof_confidence_bench.py` writes ready-to-load `candidate_thresholds_<mode>.json` files from its per-point recommendations, using the same `fit_power_law` the module uses at load.
- Legacy per-band files (`{"min_raw_by_band": {"far": 52, ...}}`) still load, as a step curve identical to the old per-band gate. They're useful for before/after comparisons. No file at all falls back to a step curve of the preliminary built-in defaults.
- `veet_tof_bands.reload_confidence_thresholds(path=None)` re-reads the file, re-fits, and rebuilds the curve in place, so a running session doesn't need restarting after an edit. Both `VEET_TOF_Viewer.py` and `VEET_Data_Charting.py`'s TOF Analysis window expose this as a "Reload Confidence Thresholds" button that re-analyzes the loaded session and confirms explicitly: a status bar message, plus a "Confidence Thresholds" tab showing the fit (`a`, `k`, floor) and each band's near→far threshold.
- A malformed file (bad value, unsorted anchors, fewer than two usable anchors, a fit that rises with distance, unknown band name in a legacy file) raises `ValueError` immediately and leaves the previous curve in effect. It does not gate silently on garbage.
- `CONFIDENCE_THRESHOLDS_SOURCE`, `CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE`, `CONFIDENCE_RULE`, `CONFIDENCE_FIT` and `CONFIDENCE_CURVE_DESCRIPTION` report what's currently in effect. `confidence_curve()` returns the full 10 mm table.
- `CONFIDENCE_GATE_VERSION` is 3 (10 mm curve + weak-near-unbacked exclusion); 2 was the curve alone, 1 the per-band step gate. Don't compare yields or histograms across gate versions as if nothing changed.

## The two-stage (three-stage) validity pipeline

```
z1,conf1,z2,conf2 --[resolve_dual_return]--> z,conf   (stage 0, mandatory)
z,conf            --[valid_mask]-----------> gate     (stage 1: confidence gate)
gate              --[& ANALYSIS_FIELD]------> analysis_mask  (stage 2: reporting field)
gate              --[spatial_corroboration]-> zone_mask       (stage 2': diagnostic only)
```

### Stage 0 — dual-return resolution (mandatory)

The TMF8828 always reports two returns per zone: object 1 (primary/strongest) and object 2 (secondary). VEET2 exposes no configuration that collects object 1 only. `resolve_dual_return(z1, c1, z2, c2)` folds them into the single `(z, conf)` pair stage 1 expects:

1. **Object 1 passes the stage-1 gate** → used as-is; object 2 is never inspected.
2. Otherwise:
   - **Object 2 is sub-floor (no second target)** → resolved as confident far (2000 mm, `CONF_MAX`), regardless of why object 1 failed. A phantom low-confidence object-1 return must not override a clean "nothing there" from object 2.
     **Exception — weak near return with nothing behind it.** If object 1 is a real (≥ 10 mm) reading that fails the gate *and* is nearer than `weak_unbacked_exclude_below_mm` (config, default 400 mm), the zone is **ambiguous (excluded)** instead of confident far. A genuine near surface also blocks object 2, so a real but dim near target (skin, dark material) is indistinguishable here from a phantom; recoding it as far would bias near-work time downward, while excluding it only costs data. A zero/sub-floor object 1 is a no-return, not a weak return, and still resolves far. Legacy per-band (v1) configs load with the rule disabled so they reproduce the old gate exactly. In session 25060544-4 this moved 5.7 zone-hours (3.4% of analysis-field zone-time) from far to excluded and left every other band unchanged.
   - **Object 2 passes the stage-1 gate** → "rescued": object 2's `(z, conf)` becomes the resolved measurement.
   - **Object 2 is also nonzero-but-low-confidence** → ambiguous. Encoded as `conf_resolved = 0` (paired with `z_resolved = z2`), which guarantees the zone fails the ordinary stage-1 check with no separate exclusion mask needed anywhere downstream. Returned as the third tuple element, `ambiguous_mask`, purely for counting (feed it to `zone_yield`).

There is no VEET2 configuration without a second return, so this stage always runs — pass its output, never `(z_mm, conf_raw)` directly, to `valid_mask`/`analysis_mask`.

### Stage 1 — confidence gate

`valid_mask(z_mm, conf_raw)` — the per-zone, distance-dependent confidence gate. Required confidence scales with distance (see [Distance bands](#distance-bands)): weak far returns are cheap and common, weak near returns are suspicious.

Two floors, both unconditional (not confidence-gated):

- `Z_FLOOR_MM = 10.0` — physically impossible for a spectacle-mounted device below this; dropped regardless of confidence.
- `NO_RETURN_FLOOR_MM = 10.0` — the sensor cannot produce a genuine return below this. A sub-floor reading (including exactly `0`) is deterministically reinterpreted as far/optical infinity (`NO_RETURN_SENTINEL_MM = 2000.0`), not dropped — see `_resolve_no_return`.

**Known, accepted limitation — the gate is self-referential.** The confidence threshold is looked up using the very distance being evaluated. A spurious 1500 mm return only needs raw ≥ 64; a genuine 150 mm return needs raw ≥ 230. Errors that *inflate* distance are policed far more loosely than errors that shorten it — and inflation is the more common TOF failure mode. This biases the dioptric distribution toward the far tail (i.e. erodes the near-work signal). It's a documented, accepted trade-off, not a bug to "fix" locally.

### Stage 2 — spatial corroboration / analysis field

The 64-zone grid is split into an **analysis field** (central 6 rows × all 8 columns = 48 zones) and a **perimeter guard band** (top + bottom rows = 16 zones), based on the VCSEL illumination pattern being narrower in the vertical axis than the sensor FOV (datasheet §7.5.2). The perimeter is where oblique incidence and partial-target fill make isolated dropouts and range inflation most likely.

- `analysis_mask(z_mm, conf_raw)` — **the primary entry point.** Stage-1 gate restricted to `ANALYSIS_FIELD`. Perimeter zones are excluded unconditionally — no free-living ground truth exists to validate a recovery heuristic for them. **All reported measures should derive from this.**
- `zone_mask(z_mm, conf_raw, connectivity=8, require_interior_support=False)` — **diagnostic only.** Full 64-zone mask where a perimeter zone is kept if it has at least one passing neighbour (`spatial_corroboration`). Exists to use the guard band as an internal consistency channel, not to feed reported measures.
- `INTERIOR` (symmetric 6×6, 36 zones) is available as a more conservative fallback field if bench data ever shows column-edge degradation the illumination geometry doesn't predict.

Corroboration is single-pass: adjacency is checked against the *stage-1* mask, so pruning one perimeter zone never cascades into pruning its neighbour. Deterministic and order-independent by design — do not turn this into an iterative erosion loop.

## Distance bands

Eight literature-anchored reporting bands. Edges are inclusive-low, exclusive-high (100 mm belongs to band 1, not band 0). Band geometry is fixed in source (`_BAND_STRUCTURE`). The bands no longer carry their own confidence threshold; the last column below just shows what the shipped 10 mm gate curve does across each band (`band_min_raw_range(band)`).

| # | Name | Range (mm) | Vergence (D) | Gate min raw, near → far edge | Anchor |
|---|---|---|---|---|---|
| 0 | `artifact_occlusion` | 0–100 | > 10.0 | 255 → 255 | sub-100 mm, treat as occlusion |
| 1 | `extreme_near` | 100–200 | 10.0–5.0 | 255 → 255 | IMI <20 cm risk threshold |
| 2 | `close_near` | 200–300 | 5.0–3.33 | 255 → 142 | Sydney Myopia Study; Gajjar 2022 |
| 3 | `standard_near` | 300–400 | 3.33–2.5 | 134 → 86 | clinical near reference 33/40 cm |
| 4 | `near_intermediate` | 400–600 | 2.5–1.67 | 82 → 42 | upper edge of Ostrin near band |
| 5 | `intermediate` | 600–1000 | 1.67–1.0 | 41 → 18 | Ostrin intermediate band |
| 6 | `room` | 1000–2000 | 1.0–0.5 | 17 → 10 | — |
| 7 | `far` | 2000–∞ | < 0.5 | 10 | effective optical infinity |

Run the module directly (`python veet_tof_bands.py`) to print the table **as currently loaded**, including the fitted `a`/`k`. It reflects any edits to `veet_tof_confidence_thresholds.json`.

Each `Band` is a frozen dataclass with `.d_max`/`.d_min` properties (vergence at the near/far edge). `band_index(z_mm)` and `min_confidence(z_mm)` each do their own vectorised `np.searchsorted` lookup, one on the band edges and the other on the 10 mm gate grid.

## Dioptric conversion and binning

The eye responds to reciprocal distance (vergence), not distance itself — accommodative demand and retinal defocus both scale as `1/z`. `to_diopters(z_mm, z_cap_mm=2000.0)` converts and clamps the far tail: beyond `z_cap_mm` the sensor can't distinguish 3 m from 10 m, and both are optically equivalent, so the tail is censored rather than left to approach zero asymptotically.

`D = 1000 / z_mm`

Histogram bins are **0.25 D wide, 0–10 D, plus one overflow bin** (`D_BIN_EDGES`, 41 edges total) — sized to the eye's depth of focus (~±0.25 D). Finer bins would resolve differences the retina can't. Don't change this to a round number of millimetres; the whole point is that it's uniform in dioptres, not mm.

- `dioptric_histogram(z_mm, conf_raw, sample_period_s=5.0)` — time-weighted histogram (seconds per bin) over `analysis_mask`-passing zones only.
- `dioptric_dose(z_mm, conf_raw, sample_period_s=5.0)` — time-integrated accommodative demand in dioptre-hours. The distance analogue of lux-hours on the ALS channel. Returns `0.0` if no zone passes — **the caller must distinguish "no valid data" from "genuinely zero exposure"** (`dioptric_dose` cannot tell you which).

## API reference

Import surface, in dependency order:

```python
from veet_tof_bands import (
    BANDS, GRID_SHAPE, ANALYSIS_FIELD, INTERIOR, PERIMETER,
    FOV_FULL_DEG, FOV_ANALYSIS_DEG, D_BIN_EDGES, D_BIN_WIDTH,
    band_index, min_confidence, valid_mask,
    analysis_mask, to_diopters,
    dioptric_histogram, dioptric_dose,
    zone_mask, spatial_corroboration, zone_yield, field_coverage_mm,
    resolve_dual_return, confidence_to_snr, reload_confidence_thresholds,
)
```

| Function / constant | Returns | Notes |
|---|---|---|
| `analysis_mask(z, c)` | bool `(...,8,8)` | **Primary reporting path.** Gate ∩ analysis field. |
| `valid_mask(z, c)` | bool `(...,8,8)` | Stage 1 only, all 64 zones. |
| `reload_confidence_thresholds(path=None)` | int `(200,)` | Re-read `veet_tof_confidence_thresholds.json` (or `path`), re-fit, and rebuild the 10 mm gate curve in place. See [Confidence thresholds are lab-editable](#confidence-thresholds-are-lab-editable). |
| `resolve_dual_return(z1, c1, z2, c2)` | `(z, conf, ambiguous)` | Stage 0, mandatory. Output feeds straight into `valid_mask`/`analysis_mask`. |
| `to_diopters(z, z_cap_mm=2000.0)` | float, same shape | Vergence, far tail clamped. No masking applied — mask separately. |
| `dioptric_histogram(z, c, sample_period_s=5.0)` | float `(41,)` | Seconds per bin; last bin is overflow (> 10 D / < 100 mm). |
| `dioptric_dose(z, c, sample_period_s=5.0)` | float | Dioptre-hours. `0.0` if no zone valid. |
| `band_index(z)` | int `(...,8,8)` | Reporting band 0–7 for every zone, gate not applied. |
| `min_confidence(z)` | int `(...,8,8)` | Required raw confidence at each distance. |
| `field_coverage_mm(z, field="analysis")` | `(vertical_mm, horizontal_mm)` | Scene extent at distance `z`. `field` is `"analysis"` or `"full"`. Order is (V, H), not (W, H) — see [Gotchas](#gotchas). |
| `zone_yield(z, c, ambiguous_mask=None, **kwargs)` | dict | Diagnostic. Per-stage pass rates; `ambiguous_dual_return_fraction` appears only if `ambiguous_mask` is passed. |
| `confidence_to_snr(raw, encoding=None)` | float, same shape | Diagnostic. Decodes raw byte to true SNR; raises if encoding unknown. |
| `zone_mask(z, c, connectivity=8, require_interior_support=False)` | bool `(...,8,8)` | Diagnostic only — never for reported measures. |
| `spatial_corroboration(gate_mask, connectivity=8, require_interior_support=False)` | bool, same shape | Diagnostic only. Operates on an existing stage-1 mask. |
| `BANDS` | `tuple[Band, ...]` | The 8 reporting bands. Geometry only; the gate is the separate 10 mm curve. |
| `confidence_curve()` | `(bin_lo_mm, min_raw)` | The 10 mm gate curve currently in effect. |
| `band_min_raw_range(band)` | `(int, int)` | Gate threshold at a band's near/far edge. Display only. |
| `fit_power_law(anchors, floor_min_raw=10)` | `PowerLawFit` | The load-time fit; `.evaluate(z)`, `.a`, `.k`, `.describe()`. |
| `GRID_SHAPE` | `(8, 8)` | |
| `ANALYSIS_FIELD` / `PERIMETER` / `INTERIOR` | bool `(8, 8)` | Field masks. `ANALYSIS_FIELD` = `~PERIMETER`. |
| `FOV_FULL_DEG` / `FOV_ANALYSIS_DEG` | `(vertical_deg, horizontal_deg)` | |
| `D_BIN_EDGES` / `D_BIN_WIDTH` | array / float | Histogram bin edges (41) and width (0.25 D). |
| `Z_FLOOR_MM`, `NO_RETURN_FLOOR_MM`, `NO_RETURN_SENTINEL_MM`, `CONF_MAX`, `CONFIDENCE_ENCODING` | float / int / str | See relevant sections above. |

`sample_period_s` defaults to `5.0`. Pass the session's actual sample period (from the INF line; 0.5 Hz nominal for VEET2) — do not rely on the default for real data.

## Common recipes

**Single frame, already resolved (e.g. output of `resolve_dual_return`, or a legacy single-return log):**

```python
mask = tb.analysis_mask(z_mm, conf_raw)          # (8, 8) bool
trusted_d = tb.to_diopters(z_mm)[mask]
```

**Full session with dual-return data:**

```python
z, conf, ambiguous = tb.resolve_dual_return(z1_mm, conf1_raw, z2_mm, conf2_raw)
hist = tb.dioptric_histogram(z, conf, sample_period_s=actual_period)
dose = tb.dioptric_dose(z, conf, sample_period_s=actual_period)
yields = tb.zone_yield(z, conf, ambiguous_mask=ambiguous)
```

**Per-band time breakdown for a session:**

```python
mask = tb.analysis_mask(z_mm, conf_raw)
idx = tb.band_index(z_mm)[mask]
for b in tb.BANDS:
    seconds = (idx == b.index).sum() * sample_period_s
    print(f"{b.name:20s} {seconds:8.0f} s")
```

**Scene coverage at working distance (e.g. for a peripheral-defocus proxy):**

```python
v_mm, h_mm = tb.field_coverage_mm(z_mm=300.0, field="analysis")  # (vertical, horizontal)
```

## Diagnostics

`zone_yield(z_mm, conf_raw, ambiguous_mask=None, **kwargs)` reports, as fractions of applicable zones:

- `passed_confidence_gate` — stage 1 only, all 64 zones
- `analysis_field_yield` — stage 1 restricted to the analysis field
- `passed_both_stages` — after `spatial_corroboration` (diagnostic path)
- `pruned_by_corroboration` — zones stage 1 passed but corroboration removed
- `interior_yield` / `perimeter_yield` — split by field region; `perimeter_yield` is the guard-band health indicator. A persistent gap vs. `interior_yield` is expected; a *sharp change over time or across devices* is worth investigating.
- `ambiguous_dual_return_fraction` — only present if `ambiguous_mask` (from `resolve_dual_return`) is passed. Fraction of analysis-field zones where neither TOF return was trustworthy.

Log this per session. It's the cheapest early signal for a mounting problem or hardware degradation.

## Gotchas

- **FOV axes are (vertical, horizontal), not (width, height).** The VEET field is taller than wide (52° V × 41° H) — the opposite of most camera intuition. `FOV_FULL_DEG`, `FOV_ANALYSIS_DEG`, and `field_coverage_mm`'s return value are all `(vertical, horizontal)`. Indexing them positionally as if they were `(width, height)` produces a plausible-looking but wrong number.
- **`analysis_mask` is the only path that should feed reported numbers.** `valid_mask`, `zone_mask`, and `spatial_corroboration` all exist for legitimate reasons (raw gating, diagnostics, guard-band QA) but none of them apply the analysis-field restriction — using them directly for reporting will silently include perimeter zones.
- **`dioptric_dose` returning `0.0` is ambiguous.** It means either "genuinely zero accommodative demand" or "no zone in this frame/session passed the gate." Check `mask.any()` (or `zone_yield`) separately if that distinction matters — don't let a no-data frame silently dilute a daily mean.
- **Sub-floor / zero distance is not missing data.** It's deterministically treated as a confident far/no-return reading (2000 mm sentinel), independent of whatever confidence byte came with it. Do not special-case `0` upstream of this module.
- **The confidence gate is self-referential** (see [Stage 1](#stage-1--confidence-gate)) — it's an accepted bias toward under-flagging inflated (far) returns, not a defect to patch locally.
- **Thresholds are pre-validation.** Every confidence threshold needs re-derivation against a reference target set (varying reflectance, incidence angle) before being used in published analysis — see [Confidence thresholds are lab-editable](#confidence-thresholds-are-lab-editable) for how to apply updated values.

## Running the module directly

```bash
python veet_tof_bands.py
```

Prints which confidence-threshold config is loaded, then the band table (index, name, mm range, dioptre range, minimum raw confidence) — a quick sanity check that `veet_tof_confidence_thresholds.json` parses as expected after an edit.

## Testing

```bash
pytest tests/test_tof_bands_dual_return.py
```

Covers `resolve_dual_return`'s branch logic (pass-through, confident-far, rescue, ambiguous) including confirming a rescued zone actually clears `analysis_mask` downstream. See also `tests/test_tof_array_export.py` and [verify_tof_export.py](verify_tof_export.py) for the CSV export path that produces the arrays this module consumes.

## Provenance

Sensor geometry, illumination figures, and the confidence encoding definition are from the TMF882X datasheet DS000693 v8-00 (2023-Aug-02), §§7.4.7 and 7.5.2 and Figures 30–32. Reporting-band edges are anchored to published myopia literature: the <20 cm threshold from the IMI Clinical Management Guidelines, the <30 cm threshold from the Sydney Myopia Study and the Gajjar 2022 systematic review, and the near/intermediate split from Ostrin's Clouclip and Actiwatch work. Anchors are recorded per-band in `_SPEC` in the source so provenance travels with the code. Full detail, including the real-session dual-return outcome statistics and the integration checklist, is in [VEET_TOF_INTEGRATION.md](VEET_TOF_INTEGRATION.md).
