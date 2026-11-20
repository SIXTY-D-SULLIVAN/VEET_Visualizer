# VEET 2 TOF — integration spec

Companion document to `veet_tof_bands.py`. Read this before modifying that module or building analysis on top of it.

This module is the **single source of truth** for two coupled things: which time-of-flight returns are trustworthy, and how trusted returns are binned for reporting. They are deliberately defined in one place so the validity gate and the reporting histogram cannot drift apart.

Status: pre-validation. Thresholds are provisional (see [Provenance](#provenance)).

---

## 1. Scope

**In scope:** per-zone validity gating, distance-to-vergence conversion, dioptric binning, time-integrated exposure metrics, field-of-view geometry.

**Out of scope:** file I/O, session segmentation, wear-detection, ALS/PHO/IMU fusion, temporal filtering, epoch/bout analysis. Build those in the larger tool and call into this module — do not add them here.

---

## 2. Sensor and data contract

VEET 2 TOF is an 8×8 zone depth array, nominal FOV 41° × 52°, usable range roughly 10–2000+ mm. Sampling rate is configuration-dependent and is recorded in the INF line of `sensor_data.csv` — read it from there rather than assuming a fixed rate. Nominal sampling for VEET2 is 0.5 Hz (one sample per 2 s).

Every function takes two parallel arrays:

| Arg | Meaning | Dtype | Shape | Units |
|---|---|---|---|---|
| `z_mm` | measured distance per zone | float | `(..., 8, 8)` | mm |
| `conf_raw` | reported confidence per zone | integer | `(..., 8, 8)` | 0–255, where 255 = 100% |

Leading dimensions are free — pass `(8, 8)` for one frame or `(n_samples, 8, 8)` for a session. Shapes must match exactly; a mismatch raises `ValueError`. Trailing shape must be `(8, 8)`.

**Do not pre-convert confidence to a percentage.** Thresholds are stored as raw integers and compared directly.

**Do not pre-filter invalid returns to NaN or 0 before calling.** The gate handles them. Zeros and sub-10 mm values are dropped by `Z_FLOOR_MM`; non-finite values are dropped by `valid_mask`.

---

## 3. Validity model

### 3.1 Dual-return resolution (stage 0, mandatory)

The TMF8828 always reports two returns per zone: object 1 (primary/strongest) and object 2 (secondary). VEET2 exposes no configuration that collects object 1 only. `resolve_dual_return(z1, c1, z2, c2)` folds both into the single `(z, conf)` pair that stage 1 expects, and must run upstream of `valid_mask`/`analysis_mask` for every session.

Resolution, per zone per frame:

1. **Object 1 passes the stage-1 gate** → use it. Object 2 is never inspected.
2. Otherwise (object 1 is sub-floor/0, or nonzero-but-low-confidence):
   - **Object 2 is sub-floor/0** → confident far (2000 mm), regardless of *why* object 1 failed. A low-confidence object-1 reading is typically a phantom return from diffuse reflection off a nearby surface, so it must not override a clean "nothing there" from object 2.
     **Exception — weak near return with nothing behind it.** If object 1 is a real (≥ 10 mm) reading that fails the gate *and* is nearer than `weak_unbacked_exclude_below_mm` (config, default 400 mm), the zone is **ambiguous (excluded)** instead of confident far. A genuine near surface also blocks object 2, so a real but dim near target (skin, dark material) is indistinguishable here from a phantom; recoding it as far would bias near-work time downward, while excluding it only costs data. A zero/sub-floor object 1 is a no-return, not a weak return, and still resolves far. Legacy per-band (v1) configs load with the rule disabled so they reproduce the old gate exactly. In session 25060544-4 this moved 5.7 zone-hours (3.4% of analysis-field zone-time) from far to excluded and left every other band unchanged.
   - **Object 2 passes the stage-1 gate** → "rescued": object 2's `(z, conf)` becomes the resolved measurement.
   - **Object 2 is also nonzero-but-low-confidence** → ambiguous. Neither return can be trusted; the zone is excluded and counted (see `zone_yield`'s `ambiguous_dual_return_fraction`), not silently mis-binned.

Object 2 is gated by the same threshold curve as object 1 (§3.2), looked up at object 2's own distance — same sensor, same characterisation curve, no separate threshold.

**Encoding, not a second mask.** The ambiguous branch resolves to `conf_resolved = 0` paired with `z_resolved = z2` (always ≥ the floor in that branch). This guarantees the zone fails the ordinary stage-1 confidence check on its own, so `valid_mask`/`analysis_mask`/`dioptric_histogram`/`dioptric_dose`/`band_index`/`to_diopters` all keep working unchanged on the resolved array — no dual-return-aware branch needed anywhere downstream, and `analysis_mask` remains the sole reporting path (Invariant §6.2).

**"No second object" is a literal `0`.** Confirmed against a real dual-return session (`25060544-4_tof_arrays.csv`) — `z2_mm`/`conf2_raw` are never `NaN`.

**Real-data outcome shares** (same session, 402,368 zone-frames): object-1-valid 16.0%, confident-far 72.3%, rescued-by-object-2 1.9%, ambiguous 9.8%. The ambiguous share is a real, non-trivial signal worth trending via `ambiguous_dual_return_fraction`, not a bug. Rescues come almost entirely from the "object 1 nonzero-but-low-confidence" bucket rather than "object 1 exactly 0" (0% of sub-floor object-1 zones were rescued in this session) — consistent with the sensor reporting its strongest return as object 1.

**Yield comparability caveat.** `zone_yield()`'s fractions (`analysis_field_yield`, `perimeter_yield`, etc.) reflect the *resolved* array, which includes object-2 rescues. They are not directly comparable to yields from an older primary-only log predating dual-return export — a shift across that cutover is a metric-definition change, not a hardware change.

### 3.2 Confidence gate (stage 1)

**Confidence is SNR, not a percentage.** Per TMF882X datasheet §7.4.7, the reported byte is the signal-to-noise ratio of the detected histogram peak. A value of 255 does not mean "100% confident." Earlier revisions of this module described the thresholds as percentages; that was a misreading and has been removed.

VEET2 firmware is fixed to linear confidence encoding (`logarithmic_confidence = 0`, confirmed against deployed firmware); the TMF8828's logarithmic mode exists in silicon but is not exposed on this hardware. Under linear encoding the raw byte *is* the SNR, clipped at 255.

Confidence required scales with measured distance — weak far returns are cheap and common, weak near returns are suspicious.

**The threshold is a continuous curve, not one value per reporting band** (`CONFIDENCE_GATE_VERSION = 4`: v2 introduced the curve on a 10 mm grid, v3 added the weak-near-unbacked exclusion in §3.1, v4 dropped the 10 mm grid). The curve is evaluated at each zone's own measured distance, clamped to `Z_FLOOR_MM`–2000 mm (anything ≥ 2000 mm gets the 2000 mm value). Because confidence is an integer, `conf ≥ ceil(a·d^-k)` is exactly `conf ≥ a·d^-k`. Dropping the grid (which used each bin's near-edge value) loosened the gate by at most 12 counts, at 250–260 mm; on session 25060544-4 it changed 0.28% of zone decisions and every band's hours by under 2%. The curve is:

```
min_raw(d) = clip(ceil(a · d^-k), floor_min_raw, 255)        d in mm
```

`a` and `k` are not stored. They are fit at load time by least squares in log-log space through the bench **anchor points** in the config file. Anchors at exactly 0 or 255 are clipped by the byte encoding (the true requirement is unknown), so they are excluded from the fit rather than flattening its slope. `floor_min_raw` (currently 10) stops the far tail from falling below anything bench data supports; a pure power law would reach ~5 at 2000 mm. A fit that rises with distance (`k ≤ 0`) raises rather than loading — it means the anchors are wrong.

Shipped curve (2026-10 paper/phone bench anchors plus 2026-11 near-work anchors from iPad, Switch, laptop and picture book; `a ≈ 1.58e6`, `k ≈ 1.67`, floor 10), summarised at the reporting bands for reference only:

| Band | Name | Distance (mm) | Vergence (D) | Min raw (SNR), near → far edge |
|---|---|---|---|---|
| 0 | `artifact_occlusion` | < 100 | > 10.0 | 255 → 255 |
| 1 | `extreme_near` | 100–200 | 10.00–5.00 | 255 → 234 |
| 2 | `close_near` | 200–300 | 5.00–3.33 | 234 → 119 |
| 3 | `standard_near` | 300–400 | 3.33–2.50 | 119 → 74 |
| 4 | `near_intermediate` | 400–600 | 2.50–1.67 | 74 → 38 |
| 5 | `intermediate` | 600–1000 | 1.67–1.00 | 38 → 17 |
| 6 | `room` | 1000–2000 | 1.00–0.50 | 17 → 10 |
| 7 | `far` | > 2000 | < 0.50 | 10 |

`python veet_tof_bands.py` prints this table for whatever is currently loaded.

**Lab-editable, not hardcoded.** The curve loads from `veet_tof_confidence_thresholds.json` (alongside `veet_tof_bands.py`) at import time:

```json
{
  "format_version": 2,
  "rule": "power_law_fit",
  "floor_min_raw": 10,
  "anchors": [[200, 245], [250, 200], [300, 150], ...]
}
```

`anchors` are `[distance_mm, min_raw]` pairs, strictly increasing in distance, each distance ≥ 10 mm and each value an integer 0–255. Edit the anchors (or the floor) after each characterisation pass. `veet_tof_confidence_bench.py` writes ready-to-load candidate files (`candidate_thresholds_<mode>.json`) from its per-point recommendations, so anchors don't have to be transcribed by hand. Reporting-band geometry is not in this file and is not lab-tunable, since it's coupled to the dioptric histogram (Invariant §6.1). The gate and the bands are now deliberately separate: refining the curve never moves a reporting edge.

Legacy per-band files (`{"min_raw_by_band": {...}}`, format v1) still load, as a step curve that reproduces the old per-band gate exactly. Use them for before/after regression comparisons. With no file at all, the gate falls back to a step curve built from the preliminary built-in per-band defaults.

Call `veet_tof_bands.reload_confidence_thresholds()` to pick up an edit without restarting a running session. Both `VEET_TOF_Viewer.py` and `VEET_Data_Charting.py`'s TOF Analysis window expose this as a "Reload Confidence Thresholds" button, re-analyze the loaded session, and confirm explicitly: a status bar message, plus a "Confidence Thresholds" legend tab showing the source file, the fitted `a`/`k`/floor and each band's near→far threshold. A reload is never silent. A malformed file (bad value, unknown band name, unsorted anchors, too few usable anchors, a rising fit) raises immediately and leaves the previous curve in effect, rather than gating silently on garbage. `CONFIDENCE_THRESHOLDS_SOURCE`, `CONFIDENCE_RULE`, `CONFIDENCE_FIT` and `CONFIDENCE_CURVE_DESCRIPTION` report what's currently in effect.

The gate compares raw values directly, so decoding is never needed on the hot path. `confidence_to_snr()` exists for diagnostics.

Band edges are inclusive-low, exclusive-high: a reading of exactly 100 mm belongs to band 1.

**Known limitation — the gate is self-referential.** The threshold is looked up using the very reading being evaluated. With the shipped curve, a spurious 1500 mm return needs only raw ≥ 10; a genuine 150 mm return needs raw ≥ 255. (Under the old per-band step gate these were 64 and 230. The bench data says far returns really are that weak, but the asymmetry is now steeper.) Errors that *inflate* distance are therefore policed far more loosely than errors that shorten it, and inflation is the more common TOF failure mode. The residual bias pushes the dioptric distribution toward the far tail — precisely the direction that erodes the near-work signal. This is accepted, not solved. Do not remove the comment documenting it.

### 3.3 Analysis field (stage 2)

**The declared analysis field is the central 6 rows × all 8 columns (48 zones). The top and bottom rows are a guard band: logged in full, excluded from every reported measure.**

The guard band is asymmetric because the underlying defect is. Datasheet §7.5.2 gives VCSEL field of illumination as 47×57° at 5% of max, **41×47° at 1/e²**, and 30×32° FWHM, against a sensor FoV of 41×52°:

- **Columns (x):** FoI 1/e² = 41° equals FoV = 41°. Outer columns are fully illuminated. No guard needed.
- **Rows (y):** FoI 1/e² = 47° is 5° narrower than FoV = 52°. The outer rows span 19.5–26° off-axis while illumination falls to 1/e² at 23.5°, so 38% of the outer row's angular extent is lit at between 5% and 13.5% of peak.

Empirically observed perimeter weakness therefore has a physical cause, and it is a row phenomenon. Trimming rows only preserves 48 zones and the full 41° horizontal field.

Beyond that, free-living data has **no ground truth** — there is no reference against which a perimeter-recovery heuristic could be scored. This is not a budget problem; the measurement does not exist. A narrower declared field is lab-verifiable against reference targets, cheaply and definitively.

`INTERIOR` (symmetric 6×6, 36 zones) remains available as the conservative fallback if bench data shows column-edge degradation the illumination geometry does not predict.

`spatial_corroboration()` and `zone_mask()` recover guard-band zones having a valid neighbour. **These are diagnostic only.** They exist to use the guard band as an internal consistency channel and to preserve Phase 2 optionality. They must not feed reported measures.

### 3.4 Field geometry and coverage

FoV is **41° horizontal (x, columns) × 52° vertical (y, rows)** — taller than wide. Confirmed against datasheet Figures 30–32, which annotate 41° on the x axis, and against SPAD pitch (16.8 µm in x, 38.8 µm in y). Zone pitch is 5.125° H, 6.5° V. The 6×8 analysis field subtends **41° H × 39° V**.

Scene coverage, mm, as width × height:

| Distance | Analysis field (6×8) | Full array (8×8) | 6×6 fallback |
|---|---|---|---|
| 200 mm | 150 × 142 | 150 × 195 | 110 × 142 |
| 300 mm | 224 × 212 | 224 × 293 | 165 × 212 |
| 400 mm | 299 × 283 | 299 × 390 | 220 × 283 |
| 600 mm | 449 × 425 | 449 × 585 | 330 × 425 |

Retaining all eight columns keeps a letter-width page (216 mm) inside the analysis field from 300 mm outward. The symmetric 6×6 fallback does not reach that until ~400 mm, which matters for `scene_dioptric_range` as a peripheral-defocus proxy — below that the field reads page interior rather than page-plus-surround. Central vergence, dioptric dose, and the histogram are unaffected either way; they are carried by the central zones.

## 4. Dioptric conversion

Vergence `D = 1000 / z_mm`. The eye responds to reciprocal distance, not distance — accommodative demand and retinal defocus both scale as 1/z. Linear binning oversamples the far field where nothing happens and undersamples the near field where the mechanism lives: 200→300 mm costs 1.67 D, while 1000→1100 mm costs 0.09 D.

`to_diopters()` clamps at `z_cap_mm = 2000.0`. Beyond that the sensor cannot distinguish 3 m from 10 m and both are optically equivalent, so the far tail is censored rather than allowed to approach zero asymptotically.

**Histogram bins are 0.25 D wide, 0 to 10 D, plus an overflow bin (41 total).** The width is set by the eye's depth of focus (~±0.25 D) — binning finer resolves differences the retina cannot detect. Do not change this to a round number of millimetres.

---

## 5. Public API

Import surface, in dependency order:

```python
from veet_tof_bands import (
    BANDS, GRID_SHAPE, ANALYSIS_FIELD, INTERIOR, PERIMETER,
    FOV_FULL_DEG, FOV_ANALYSIS_DEG, D_BIN_EDGES, D_BIN_WIDTH,
    band_index, min_confidence, valid_mask,
    analysis_mask, to_diopters,
    dioptric_histogram, dioptric_dose,
    zone_mask, spatial_corroboration, zone_yield, field_coverage_mm,
    resolve_dual_return, reload_confidence_thresholds,
    confidence_curve, band_min_raw_range, fit_power_law,
)
```

| Function | Returns | Use |
|---|---|---|
| `analysis_mask(z, c)` | bool `(...,8,8)` | **Primary.** Gate ∩ analysis field. All reporting derives from this. |
| `valid_mask(z, c)` | bool `(...,8,8)` | Stage 1 only. Use when you need the gate independent of the field. |
| `reload_confidence_thresholds(path=None)` | int `(200,)` | Re-read `CONFIDENCE_THRESHOLDS_PATH` (or `path`), re-fit, and replace the gate curve in place — the lab-characterisation edit loop. Returns the new curve. See §3.2. |
| `confidence_curve()` | `(distance_mm, min_raw)` | The curve currently in effect, sampled every 10 mm, for display/export. |
| `band_min_raw_range(band)` | `(int, int)` | Threshold at a reporting band's near and far edge — display only. |
| `fit_power_law(anchors, floor)` | `PowerLawFit` | The load-time fit, exposed so the bench produces exactly the curve a candidate JSON will load as. |
| `resolve_dual_return(z1, c1, z2, c2)` | `(z, conf, ambiguous)` | **Stage 0, mandatory.** Folds object 1/object 2 into one `(z, conf)` pair; feed the result into `valid_mask`/`analysis_mask` exactly like a single-return pair. See §3.1. |
| `to_diopters(z)` | float, same shape | Vergence, far tail clamped. |
| `dioptric_histogram(z, c, dt)` | float `(41,)` | Time-weighted, seconds per bin. Last element is overflow. |
| `dioptric_dose(z, c, dt)` | float | Dioptre-hours. The distance analogue of lux-hours. |
| `band_index(z)` | int `(...,8,8)` | Reporting band 0–7. |
| `min_confidence(z)` | int `(...,8,8)` | Required raw confidence at each distance (continuous curve). |
| `field_coverage_mm(z, field)` | `(float, float)` | Scene extent in mm at a distance. |
| `zone_yield(z, c, ambiguous_mask=None)` | dict | **Diagnostic.** Per-stage yields including guard-band health; adds `ambiguous_dual_return_fraction` when `ambiguous_mask` (from `resolve_dual_return`) is passed. |
| `confidence_to_snr(raw, enc)` | float, same shape | **Diagnostic.** Decode raw byte to SNR. Raises if encoding unknown. |
| `zone_mask`, `spatial_corroboration` | bool `(...,8,8)` | **Diagnostic only.** Never for reported measures. |

`sample_period_s` defaults to 5.0 throughout. Pass the actual period if the session used a different rate — do not assume.

---

## 6. Invariants

Changing any of these silently invalidates comparisons against prior runs. Bump a version constant and re-run the full pipeline if you touch them.

1. Band edges (mm/vergence geometry) and dioptric bin edges must stay coupled. Band geometry is not exposed for lab editing. The confidence gate is deliberately *not* tied to the bands: it is a separate continuous curve (§3.2), so tuning it never moves a reporting edge. `CONFIDENCE_GATE_VERSION` (currently 4: the continuous curve plus the weak-near-unbacked exclusion; 3 was the same on a 10 mm grid, 2 the 10 mm curve alone, 1 the per-band step gate) must be bumped if gate semantics change, and outputs from different gate versions must not be compared directly.
2. `CONFIDENCE_ENCODING` is fixed at `"linear"` — VEET2 firmware does not expose the logarithmic option, so no per-session assertion is needed.
2. `analysis_mask` is the only path to reported measures. `resolve_dual_return` runs strictly upstream of stage 1 for every session; ambiguous zones are excluded via `conf_resolved = 0`, which stage 1 already rejects, not via a separate mask threaded through the pipeline.
3. `spatial_corroboration` is single-pass. Adjacency is evaluated against the stage-1 mask, so pruning a zone never cascades into pruning its neighbour. Iterative erosion would be order-dependent — do not "improve" it into a loop.
4. `D_BIN_WIDTH = 0.25` is justified by depth of focus, not convenience.
5. `Z_FLOOR_MM = 10.0` is the sensor's physical near limit.
6. Retain all 64 zones in storage regardless of what is analyzed. The guard band costs nothing to log and cannot be recovered retroactively.

---

## 7. Integration checklist

- [ ] Confirm `sample_period_s` from session metadata (the INF line) rather than defaulting.
- [ ] Account for 8×8 sub-capture timing: the frame is assembled from four interleaved sub-captures over a 66 ms (125 k iterations) to 258 ms (550 k) ranging period, so zones within one frame are not simultaneous. Head motion during the burst is a spatial artefact source for any cross-zone metric.
- [ ] Never index coverage tuples positionally without checking the order — they are `(vertical, horizontal)`, and the field being taller than wide makes a transposition look plausible.
- [ ] Condition any `scene_dioptric_range` reporting on working distance; below ~350 mm the analysis field is narrower than a page (§3.4).
- [ ] Apply wear-detection upstream. Non-wear periods will otherwise register as valid near returns from whatever surface the device is resting on — band 0 exists partly to catch this, but it is not a wear detector.
- [ ] Log `zone_yield()` per session. Track `analysis_field_yield` and `perimeter_yield` over time and across devices; a sharp change is a hardware or mounting flag.
- [ ] Also track `ambiguous_dual_return_fraction` (expect ~10% based on one real session, §3.1); a sharp change flags the same kind of hardware/mounting issue as the other yields. Remember it isn't comparable to yields from older primary-only logs predating dual-return export (§3.1).
- [ ] Handle all-invalid frames explicitly. `dioptric_dose` returns 0.0, but a frame with no valid zones is missing data, not zero exposure — do not let it dilute a daily mean.
- [ ] Code far-field non-returns as band 7, **not** as missing data. Dropping them systematically deletes outdoor time and biases the day toward near.
- [ ] Cross-check outdoor classification against ALS lux. Genuine far-field and sensor dropout look identical to the TOF alone.
- [ ] Before adopting a new thresholds config, run real sessions through it with `python veet_tof_gate_compare.py <device folder> [--candidate new.json]`. By default this compares against the frozen pre-curve gate (`veet_tof_confidence_thresholds_v1_2026-08-12.json`, do not edit). It reports outcome shares, yields, dose and band hours for both gates, plus which zone-frames were gained, lost or moved, and how coherent they are against unchanged ones. Output goes to `<device folder>/gate_comparison/`.
- [ ] After any lab characterisation pass, edit the anchors in `veet_tof_confidence_thresholds.json` (or start from a bench `candidate_thresholds_<mode>.json`), not the code, and call `reload_confidence_thresholds()`; record what changed and why outside the file itself (git history), since the file's own `last_updated` field is not authoritative.

---

## 8. Open questions

**~~FOV axis assignment.~~ RESOLVED.** Each row spans 41° horizontally, so the 8 columns cover 41° and the 8 rows cover 52°. The field is taller than wide. Encoded as `FOV_H_DEG` / `FOV_V_DEG` rather than a bare tuple, because the original `(41, 52)` was assigned the wrong way round and read as plausible either way. Tuples in the module are `(rows, cols)` = `(vertical, horizontal)`; `field_coverage_mm` returns in that order, **not** the conventional width-by-height.

**~~Corroboration connectivity.~~ RESOLVED.** VEET2 always collects the full 8×8 zone grid — there is no reduced-grid configuration where a corner zone would be structurally undercorroborated. Connectivity is fixed at 8-connected (Moore); 4-connectivity is not exposed as an option.

**Mutual validation loophole.** By default two adjacent noisy perimeter zones can corroborate each other. `require_interior_support=True` closes it. On synthetic data the two modes differ substantially in perimeter yield — resolve empirically against bench data if the diagnostic path ever becomes load-bearing.

**Threshold re-derivation.** The gate is now a power-law curve fit to bench anchors (§3.2). The current anchors come from flat paper/phone targets at 0° and 30°, with and without a wall behind, and object data covers only ~200–700 mm; below 200 mm the curve is extrapolated (it saturates at 255), and above ~1250 mm it sits on the floor. The anchors still need validating against a reference target set spanning the full range, at varying surface reflectance and incidence angle, before any published analysis. Also still open: whether the floor of 10 is right, and how much worse the self-referential bias (§3.2) gets with the lower far-field thresholds.

**Row-guard depth.** One trimmed row per edge brings the field inside the 1/e² illumination cone. Whether the second row also needs trimming is an empirical question; the geometry does not require it.

---

## Provenance

Confidence thresholds originate from a preliminary bench characterisation and are explicitly provisional — see `veet_tof_confidence_thresholds.json`'s own `provenance`/`last_updated` fields for the currently-loaded values' status, and §3.2 for how to update them. They were originally described as percentages; per TMF882X datasheet §7.4.7 the underlying quantity is SNR, and that framing has been corrected throughout. Sensor geometry, illumination figures, and the confidence encoding definition are from TMF882X datasheet DS000693 v8-00 (2023-Aug-02), §§7.4.7 and 7.5.2 and Figures 30–32. Reporting-band edges are anchored to published myopia literature: the < 20 cm threshold from the IMI Clinical Management Guidelines, the < 30 cm threshold from the Sydney Myopia Study and the Gajjar 2022 systematic review, and the near/intermediate split from Ostrin's Clouclip and Actiwatch work. Anchors are recorded per-band in `_BAND_STRUCTURE` in the source so the provenance travels with the code.
