"""veet_tof_gate_compare.py -- run one real session through two confidence gates.

Standalone, headless CLI. Loads a VEET session (a device folder containing
*Sensor_Data.csv, or a *_tof_arrays.npz/.csv export) and runs it through two
veet_tof_confidence_thresholds.json-style configs -- by default the frozen
pre-curve per-band gate (veet_tof_confidence_thresholds_v1_2026-08-12.json)
as the baseline and the live config as the candidate. Reports, per gate:

  1. Dual-return outcome shares over the analysis field (object 1 valid /
     confident far / rescued by object 2 / ambiguous).
  2. zone_yield() stats, dioptric dose, median vergence, and valid
     zone-hours per reporting band.
  3. Which zone-frames change between the gates, and at what distance:
     newly accepted, newly rejected, or accepted by both but at a different
     resolved distance ("moved" -- e.g. baseline "confident far" that the
     candidate accepts as a real object-1 return).
  4. Plausibility of the newly accepted and moved zone-frames versus
     zone-frames unchanged between the gates, in the same band: median confidence, spatial coherence
     (>= 2 of 8 neighbours valid at a distance within 15%), and temporal
     stability (same zone valid at a distance within 15% in the previous frame).
     Changed returns that are much less coherent than the unchanged ones are
     probably phantoms getting through.

Writes gate_compare.png and gate_compare_report.md, and prints the tables.
Uses veet_tof_bands for all gating; borrows VEET_TOF_Viewer.py's session
loaders rather than duplicating its Sensor_Data.csv parser (that module
imports PySide6, but no window is created).

No wear gating is applied (same as VEET_TOF_Viewer.py), so desk/non-wear
time is included in both runs equally.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import veet_tof_bands as tb

BASELINE_V1_PATH = os.path.join(SCRIPT_DIR, "veet_tof_confidence_thresholds_v1_2026-08-12.json")

COHERENCE_TOL_FRAC = 0.15
COHERENCE_MIN_NEIGHBOURS = 2
_COLORS = ("#7f7f7f", "#2e7d32")


def load_session(path: str):
    """Device folder -> Sensor_Data.csv parser; file -> array-export loader."""
    import VEET_TOF_Viewer as viewer  # deferred: pulls in PySide6
    if os.path.isdir(path):
        return viewer.load_from_device_folder(path)
    return viewer.load_array_file(path)


def gate_session(z1, c1, z2, c2, dt: float, thresholds_path: str) -> dict:
    """Run one session through one thresholds config; returns per-gate results.

    Reloads tb's thresholds -- callers should reload the default afterwards.
    """
    tb.reload_confidence_thresholds(thresholds_path)
    field = tb.ANALYSIS_FIELD
    if z2 is not None:
        z, c, amb = tb.resolve_dual_return(z1, c1, z2, c2)
        obj1_valid = (z1 >= tb.NO_RETURN_FLOOR_MM) & (c1 >= tb.min_confidence(z1))
        obj2_zero = z2 < tb.NO_RETURN_FLOOR_MM
        outcomes = {
            "object 1 valid": obj1_valid,
            "confident far": ~obj1_valid & obj2_zero & ~amb,
            "rescued by object 2": ~obj1_valid & ~obj2_zero & ~amb,
            "ambiguous": amb,
        }
    else:
        z, c, amb = np.asarray(z1, float), np.asarray(c1, float), None
        outcomes = {"object 1 valid": tb.valid_mask(z, c)}

    mask = tb.analysis_mask(z, c)
    d = tb.to_diopters(z)[mask]
    bidx = tb.band_index(z)[mask]
    return {
        "source": tb.CONFIDENCE_THRESHOLDS_SOURCE,
        "description": tb.CONFIDENCE_CURVE_DESCRIPTION,
        "z": z, "conf": c, "mask": mask,
        "outcomes": {k: float(v[..., field].mean()) for k, v in outcomes.items()},
        "yields": tb.zone_yield(z, c, ambiguous_mask=amb),
        "dose_dh": tb.dioptric_dose(z, c, dt),
        "hist_s": tb.dioptric_histogram(z, c, dt),
        "median_vergence": float(np.median(d)) if d.size else float("nan"),
        "valid_frame_frac": float(mask.any(axis=(-2, -1)).mean()),
        "band_hours": [float((bidx == b.index).sum() * dt / 3600.0) for b in tb.BANDS],
    }


def _neighbour_agreement(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per zone: how many of its 8 neighbours are valid at a distance within
    COHERENCE_TOL_FRAC of the zone's own distance."""
    zz = np.where(mask, z, np.nan)
    padded = np.pad(zz, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
    count = np.zeros(z.shape, dtype=int)
    with np.errstate(invalid="ignore"):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nb = padded[:, 1 + dr:9 + dr, 1 + dc:9 + dc]
                count += np.abs(nb - z) <= COHERENCE_TOL_FRAC * z
    return count


def _temporal_stable(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prev = np.full(z.shape, np.nan)
    prev[1:] = np.where(mask, z, np.nan)[:-1]
    with np.errstate(invalid="ignore"):
        return np.abs(prev - z) <= COHERENCE_TOL_FRAC * z


def compare(base: dict, cand: dict) -> dict:
    """How the candidate gate changes each analysis-field zone-frame, and how
    plausible the changed ones look, per reporting band (candidate distance).

    - gained:    rejected by baseline, accepted by candidate
    - lost:      accepted by baseline, rejected by candidate
    - moved:     accepted by both, but at a different resolved distance --
                 typically baseline "confident far" (2000 mm sentinel) that
                 the candidate now accepts as a real object-1 return
    - unchanged: accepted by both at the same distance; the reference group
    """
    field = tb.ANALYSIS_FIELD
    bm, cm = base["mask"] & field, cand["mask"] & field
    gained = ~bm & cm
    lost = bm & ~cm
    both = bm & cm
    moved = both & (base["z"] != cand["z"])
    unchanged = both & ~moved
    z, c = cand["z"], cand["conf"]
    coherent = _neighbour_agreement(z, cand["mask"]) >= COHERENCE_MIN_NEIGHBOURS
    stable = _temporal_stable(z, cand["mask"])

    def stats(sel):
        n = int(sel.sum())
        if n == 0:
            return {"n": 0}
        return {"n": n, "conf_median": float(np.median(c[sel])),
                "conf_p10": float(np.percentile(c[sel], 10)), "conf_p90": float(np.percentile(c[sel], 90)),
                "coherent": float(coherent[sel].mean()), "stable": float(stable[sel].mean())}

    cand_bi = tb.band_index(z)
    base_bi = tb.band_index(base["z"])
    per_band = []
    for b in tb.BANDS:
        in_band = cand_bi == b.index
        per_band.append({
            "band": b,
            "lost_n": int((lost & (base_bi == b.index)).sum()),
            "gained": stats(gained & in_band), "moved": stats(moved & in_band),
            "unchanged": stats(unchanged & in_band),
        })
    return {"gained_n": int(gained.sum()), "lost_n": int(lost.sum()), "moved_n": int(moved.sum()),
            "n_field_zone_frames": int(bm[..., field].size),
            "per_band": per_band}


# --- Output ------------------------------------------------------------------

def _pct(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def _summary_rows(runs: dict) -> list[tuple[str, list[str]]]:
    names = list(runs)
    rows = []
    outcome_keys = list(dict.fromkeys(k for r in runs.values() for k in r["outcomes"]))
    for k in outcome_keys:
        rows.append((f"outcome: {k}", [_pct(runs[n]["outcomes"].get(k)) for n in names]))
    for k in ("analysis_field_yield", "interior_yield", "perimeter_yield", "ambiguous_dual_return_fraction"):
        if all(k in runs[n]["yields"] for n in names):
            rows.append((k, [_pct(runs[n]["yields"][k]) for n in names]))
    rows.append(("frames with >= 1 valid zone", [_pct(runs[n]["valid_frame_frac"]) for n in names]))
    rows.append(("median vergence (D)", [f"{runs[n]['median_vergence']:.3f}" for n in names]))
    rows.append(("dioptric dose (D-h)", [f"{runs[n]['dose_dh']:.1f}" for n in names]))
    for b in tb.BANDS:
        rows.append((f"zone-hours: {b.index} {b.name}", [f"{runs[n]['band_hours'][b.index]:.2f}" for n in names]))
    return rows


_GROUPS = (("gained", "newly accepted"), ("moved", "moved distance"), ("unchanged", "unchanged (reference)"))
_PLAUS_HEADER = ["Band", "Group", "n", "Conf median (p10-p90)", "Coherent", "Stable"]


def _plausibility_rows(cmp: dict) -> list[list[str]]:
    """Per band with any gained/moved zones: one row per group, unchanged last."""
    rows = []
    for pb in cmp["per_band"]:
        if not (pb["gained"]["n"] or pb["moved"]["n"]):
            continue
        for key, label in _GROUPS:
            st = pb[key]
            if not st["n"]:
                continue
            rows.append([
                f"{pb['band'].index} `{pb['band'].name}`", label, str(st["n"]),
                f"{st['conf_median']:.0f} ({st['conf_p10']:.0f}-{st['conf_p90']:.0f})",
                _pct(st["coherent"]), _pct(st["stable"]),
            ])
    return rows


def _change_rows(cmp: dict) -> list[list[str]]:
    return [[f"{pb['band'].index} `{pb['band'].name}`", str(pb["gained"]["n"]), str(pb["moved"]["n"]),
             str(pb["lost_n"])]
            for pb in cmp["per_band"] if pb["gained"]["n"] or pb["moved"]["n"] or pb["lost_n"]]


def print_report(session_label: str, runs: dict, cmp: dict) -> None:
    names = list(runs)
    print(f"\nSession: {session_label}")
    for n in names:
        print(f"  {n}: {runs[n]['description']}  [{runs[n]['source']}]")
    print()
    print(f"{'':<38}" + "".join(f"{n:>22}" for n in names))
    for label, vals in _summary_rows(runs):
        print(f"{label:<38}" + "".join(f"{v:>22}" for v in vals))
    print(f"\nchanges ({names[0]} -> {names[1]}), of {cmp['n_field_zone_frames']} analysis-field zone-frames: "
          f"gained {cmp['gained_n']}, moved {cmp['moved_n']}, lost {cmp['lost_n']}")
    print(f"  {'band':<22}{'gained':>9}{'moved':>9}{'lost':>7}   (gained/moved by new band, lost by old band)")
    for band, g, m, l in _change_rows(cmp):
        print(f"  {band.replace('`', ''):<22}{g:>9}{m:>9}{l:>7}")
    print("\nplausibility (candidate gate), by band:")
    print("  " + " | ".join(_PLAUS_HEADER))
    for row in _plausibility_rows(cmp):
        print("  " + " | ".join(c.replace("`", "") for c in row))


def write_report_md(path: str, session_label: str, runs: dict, cmp: dict, png_name: str) -> None:
    names = list(runs)
    lines = [f"# Confidence-gate comparison: {session_label}\n"]
    for n in names:
        lines.append(f"- **{n}**: {runs[n]['description']} (`{runs[n]['source']}`)")
    lines.append(f"\n![comparison]({png_name})\n")
    lines.append("No wear gating is applied; both gates see the same frames.\n")
    lines.append("## Summary\n")
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for label, vals in _summary_rows(runs):
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append(
        f"\n## Changes\n\nOf {cmp['n_field_zone_frames']} analysis-field zone-frames, under {names[1]}: "
        f"**{cmp['gained_n']}** newly accepted, **{cmp['moved_n']}** accepted by both but at a different "
        f"distance (typically baseline \"confident far\" now accepted as a real return), "
        f"**{cmp['lost_n']}** newly rejected. Gained/moved are binned by the new distance, lost by the old.\n")
    lines.append("| Band | Gained | Moved | Lost |")
    lines.append("|---|---|---|---|")
    for row in _change_rows(cmp):
        lines.append("| " + " | ".join(row) + " |")
    lines.append(
        "\n## Plausibility\n\n"
        f"*Coherent*: at least {COHERENCE_MIN_NEIGHBOURS} of 8 neighbours valid at a distance within "
        f"{COHERENCE_TOL_FRAC:.0%}. *Stable*: same zone valid within {COHERENCE_TOL_FRAC:.0%} in the "
        "previous frame. Compare the newly accepted and moved groups against the unchanged group in "
        "the same band. Much lower coherence means the change is mostly admitting phantoms.\n")
    lines.append("| " + " | ".join(_PLAUS_HEADER) + " |")
    lines.append("|" + "---|" * len(_PLAUS_HEADER))
    for row in _plausibility_rows(cmp):
        lines.append("| " + " | ".join(row) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render_png(path: str, session_label: str, runs: dict) -> None:
    names = list(runs)
    fig = Figure(figsize=(12, 4.8))
    ax1, ax2 = fig.subplots(1, 2)
    edges = tb.D_BIN_EDGES
    width = 0.25 / len(names)
    for i, (n, col) in enumerate(zip(names, _COLORS)):
        ax1.bar(edges[:-1] + i * width, runs[n]["hist_s"][:-1] / 3600.0, width=width, align="edge",
                color=col, label=n)
    ax1.set_yscale("log")
    ax1.set_xlabel("Vergence [D] (0.25 D bins)")
    ax1.set_ylabel("Zone-hours (log)")
    ax1.set_title("Dioptric histogram, analysis field")
    ax1.legend()

    x = np.arange(len(tb.BANDS))
    bw = 0.8 / len(names)
    for i, (n, col) in enumerate(zip(names, _COLORS)):
        ax2.bar(x - 0.4 + bw * (i + 0.5), runs[n]["band_hours"], width=bw, color=col, label=n)
    ax2.set_xticks(x, [b.name for b in tb.BANDS], rotation=35, ha="right", fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel("Zone-hours (log)")
    ax2.set_title("Valid zone-hours per reporting band")
    ax2.legend()
    fig.suptitle(f"{session_label}: {names[0]} vs {names[1]}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# --- Main --------------------------------------------------------------------

def run_compare(session_path: str, baseline_json: str = BASELINE_V1_PATH,
                candidate_json: "str | None" = None, baseline_label: str = "baseline",
                candidate_label: str = "candidate", out_dir: "str | None" = None) -> dict:
    session_path = os.path.abspath(session_path)
    candidate_json = candidate_json or str(tb.CONFIDENCE_THRESHOLDS_PATH)
    session = load_session(session_path)
    label = os.path.basename(session_path.rstrip("\\/"))
    print(f"Loaded {label}: {session.z1_mm.shape[0]} frames, {session.sample_period_s:g} s period, "
          f"{'dual' if session.z2_mm is not None else 'single'} return")

    args = (session.z1_mm, session.conf1_raw, session.z2_mm, session.conf2_raw, session.sample_period_s)
    try:
        runs = {baseline_label: gate_session(*args, os.path.abspath(baseline_json)),
                candidate_label: gate_session(*args, os.path.abspath(candidate_json))}
    finally:
        tb.reload_confidence_thresholds()
    cmp = compare(runs[baseline_label], runs[candidate_label])

    if out_dir is None:
        base_dir = session_path if os.path.isdir(session_path) else os.path.dirname(session_path)
        out_dir = os.path.join(base_dir, "gate_comparison")
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "gate_compare.png")
    md = os.path.join(out_dir, "gate_compare_report.md")
    render_png(png, label, runs)
    write_report_md(md, label, runs, cmp, os.path.basename(png))
    print_report(label, runs, cmp)
    print(f"\nWrote {md}\n      {png}")
    return {"runs": runs, "compare": cmp, "out_dir": out_dir, "png": png, "report_md": md}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="Device folder (with *Sensor_Data.csv) or *_tof_arrays.npz/.csv")
    parser.add_argument("--baseline", default=BASELINE_V1_PATH,
                        help="Baseline thresholds JSON (default: frozen 2026-08-12 per-band gate)")
    parser.add_argument("--candidate", default=None,
                        help="Candidate thresholds JSON (default: live veet_tof_confidence_thresholds.json)")
    parser.add_argument("--baseline-label", default="v1 step (2026-08-12)")
    parser.add_argument("--candidate-label", default="live config")
    parser.add_argument("--out-dir", default=None, help="Output folder (default: <session>/gate_comparison)")
    args = parser.parse_args()
    try:
        run_compare(args.session, args.baseline, args.candidate, args.baseline_label,
                    args.candidate_label, args.out_dir)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
