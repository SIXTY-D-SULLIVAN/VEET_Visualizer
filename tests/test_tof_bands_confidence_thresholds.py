import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import veet_tof_bands as tb


@pytest.fixture(autouse=True)
def _restore_default_thresholds():
    """Every test in this module reloads thresholds; always leave the module
    back on the repo's default config so other test files aren't affected."""
    yield
    tb.reload_confidence_thresholds()


def _write(tmp_path, payload):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload))
    return path


def _v2(anchors, floor=10, **extra):
    return {"format_version": 2, "rule": "power_law_fit", "floor_min_raw": floor,
            "anchors": [list(a) for a in anchors], **extra}


# Exact power law min_raw = 1e6 * d^-1.6 sampled at a few distances.
_EXACT_ANCHORS = [(d, round(1e6 * d ** -1.6)) for d in (300, 500, 800, 1200)]


# --- Shipped config ---------------------------------------------------------

def test_shipped_config_is_loaded_and_matches_its_own_file():
    # Compare against whatever the file contains right now -- not a frozen
    # snapshot of numbers, since lab characterisation edits it in place.
    assert tb.CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
    with open(tb.CONFIDENCE_THRESHOLDS_PATH) as f:
        on_disk = json.load(f)
    assert tb.CONFIDENCE_RULE == on_disk["rule"] == "power_law_fit"
    expected = tb.fit_power_law(on_disk["anchors"], on_disk["floor_min_raw"])
    _, lut = tb.confidence_curve()
    np.testing.assert_array_equal(lut, expected.evaluate(tb.CONF_BIN_LO_MM))


def test_loaded_curve_is_valid_bytes_and_non_increasing():
    _, lut = tb.confidence_curve()
    assert lut.min() >= 0 and lut.max() <= tb.CONF_MAX
    assert np.all(np.diff(lut) <= 0)


# --- 10 mm grid -------------------------------------------------------------

def test_grid_spans_floor_to_2000mm_in_10mm_bins():
    lo = tb.CONF_BIN_LO_MM
    assert lo[0] == tb.Z_FLOOR_MM and lo[-1] == tb.CONF_LUT_MAX_MM
    assert np.all(np.diff(lo) == 10.0)


def test_lookup_uses_bin_near_edge(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS, floor=0)))
    fit = tb.CONFIDENCE_FIT
    # Anywhere inside [300, 310) gets the value at 300; 310 starts the next bin.
    at_300 = int(fit.evaluate(300.0))
    assert tb.min_confidence([300.0, 305.0, 309.99]).tolist() == [at_300] * 3
    assert tb.min_confidence([310.0])[0] == int(fit.evaluate(310.0))
    assert at_300 > int(fit.evaluate(310.0))


def test_beyond_2000mm_uses_final_bin_and_no_return_resolves_far(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS, floor=0)))
    far = tb.min_confidence([2000.0])[0]
    assert tb.min_confidence([2500.0, 10_000.0, np.inf, np.nan]).tolist() == [far] * 4
    # Sub-floor readings are no-return -> far, never the near-field threshold.
    assert tb.min_confidence([0.0, 5.0]).tolist() == [far, far]


def test_lookup_is_shape_preserving():
    z = np.full((3, 8, 8), 450.0)
    assert tb.min_confidence(z).shape == (3, 8, 8)


# --- Power-law fit ----------------------------------------------------------

def test_fit_recovers_known_power_law():
    fit = tb.fit_power_law(_EXACT_ANCHORS, floor_min_raw=0)
    # Anchors are rounded to integers, so the recovered fit is close, not exact.
    assert fit.k == pytest.approx(1.6, abs=0.02)
    assert fit.a == pytest.approx(1e6, rel=0.15)


def test_floor_clamps_the_far_tail(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS, floor=10)))
    # Unfloored, 1e6 * 2000^-1.6 ~= 5.2 -> would round up to 6.
    assert tb.min_confidence([2000.0, 5000.0]).tolist() == [10, 10]


def test_near_field_saturates_at_255(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS)))
    assert tb.min_confidence([10.0, 50.0, 100.0]).tolist() == [255, 255, 255]


def test_censored_anchors_excluded_from_fit():
    with_censored = [(50, 255), (100, 255)] + _EXACT_ANCHORS
    fit = tb.fit_power_law(with_censored, floor_min_raw=0)
    assert fit.n_anchors_censored == 2
    assert fit.n_anchors_fit == len(_EXACT_ANCHORS)
    assert fit.k == pytest.approx(tb.fit_power_law(_EXACT_ANCHORS, 0).k)


def test_rising_fit_raises(tmp_path):
    path = _write(tmp_path, _v2([(300, 20), (600, 80), (900, 150)]))
    with pytest.raises(ValueError, match="rises with distance"):
        tb.reload_confidence_thresholds(path)


@pytest.mark.parametrize("anchors", [[(300, 100)], [(300, 255), (600, 255), (900, 50)]])
def test_too_few_usable_anchors_raises(tmp_path, anchors):
    with pytest.raises(ValueError, match="at least two anchors"):
        tb.reload_confidence_thresholds(_write(tmp_path, _v2(anchors)))


# --- v2 validation ----------------------------------------------------------

@pytest.mark.parametrize("payload, match", [
    (_v2(_EXACT_ANCHORS, rule="interpolate"), "unsupported rule"),
    ({**_v2(_EXACT_ANCHORS), "format_version": 3}, "unsupported format_version"),
    (_v2(_EXACT_ANCHORS, floor=256), "floor_min_raw"),
    (_v2(_EXACT_ANCHORS, floor=True), "floor_min_raw"),
    ({**_v2(_EXACT_ANCHORS), "anchors": []}, "non-empty list"),
    (_v2([(300, 100), (300, 90), (600, 50)]), "strictly increasing"),
    (_v2([(600, 50), (300, 100)]), "strictly increasing"),
    (_v2([(5, 255), (300, 100), (600, 50)]), "distance"),
    (_v2([(300, 100), (600, 50.5)]), "min_raw for anchor"),
    (_v2([(300, 100), (600, 256)]), "min_raw for anchor"),
    ({**_v2(_EXACT_ANCHORS), "anchors": [[300, 100, 1]]}, r"\[distance_mm, min_raw\]"),
])
def test_bad_v2_config_raises(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        tb.reload_confidence_thresholds(_write(tmp_path, payload))


def test_failed_reload_keeps_previous_curve(tmp_path):
    before = tb.confidence_curve()[1]
    with pytest.raises(ValueError):
        tb.reload_confidence_thresholds(_write(tmp_path, _v2([(300, 100)])))
    np.testing.assert_array_equal(tb.confidence_curve()[1], before)


# --- Reload propagation -----------------------------------------------------

def test_reload_propagates_into_gating_functions(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS, floor=0)))
    need = int(tb.min_confidence([500.0])[0])
    z = np.array([500.0, 500.0])
    c = np.array([need, need - 1])
    assert tb.valid_mask(z, c).tolist() == [True, False]
    # Object 2 is gated by the same curve at its own distance.
    z1 = np.array([300.0, 300.0])
    c1 = np.zeros(2)
    _, conf_res, ambiguous = tb.resolve_dual_return(z1, c1, z, c)
    assert conf_res.tolist() == [need, 0.0]
    assert ambiguous.tolist() == [False, True]


# --- Legacy per-band (v1) files ----------------------------------------------

def test_v1_step_curve_reproduces_legacy_band_gate(tmp_path):
    values = {"artifact_occlusion": 243, "extreme_near": 230, "close_near": 205,
              "standard_near": 154, "near_intermediate": 103, "intermediate": 77,
              "room": 64, "far": 20}
    tb.reload_confidence_thresholds(_write(tmp_path, {"min_raw_by_band": values}))
    assert tb.CONFIDENCE_RULE == "per_band_step" and tb.CONFIDENCE_FIT is None
    z = np.arange(10.0, 3000.0, 0.5)
    legacy = np.array([values[b.name] for b in tb.BANDS])[tb.band_index(z)]
    np.testing.assert_array_equal(tb.min_confidence(z), legacy)


def test_v1_partial_override_and_bare_mapping(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, {"far": 30, "room": 200}))
    assert tb.min_confidence([2500.0, 1500.0, 150.0]).tolist() == [
        30, 200, tb._DEFAULT_MIN_RAW_BY_NAME["extreme_near"]]


def test_v1_unknown_band_name_raises(tmp_path):
    path = _write(tmp_path, {"min_raw_by_band": {"not_a_real_band": 100}})
    with pytest.raises(ValueError, match="Unknown band name"):
        tb.reload_confidence_thresholds(path)


@pytest.mark.parametrize("bad_value", [-1, 256, 100.5, "200", True])
def test_v1_out_of_range_or_wrong_type_raises(tmp_path, bad_value):
    path = _write(tmp_path, {"min_raw_by_band": {"far": bad_value}})
    with pytest.raises(ValueError):
        tb.reload_confidence_thresholds(path)


# --- File-level -------------------------------------------------------------

def test_malformed_json_raises(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="Failed to parse"):
        tb.reload_confidence_thresholds(path)


def test_missing_file_falls_back_to_default_step_curve(tmp_path):
    tb.reload_confidence_thresholds(tmp_path / "does_not_exist.json")
    assert not tb.CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
    assert tb.CONFIDENCE_RULE == "per_band_step"
    assert tb.min_confidence([2500.0])[0] == tb._DEFAULT_MIN_RAW_BY_NAME["far"]


def test_band_geometry_unaffected_by_threshold_reload(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS)))
    # Reporting bands stay coupled to the dioptric bins, independent of the gate.
    assert [b.z_min_mm for b in tb.BANDS] == [0, 100, 200, 300, 400, 600, 1000, 2000]
    assert tb.band_index([299.9, 300.0]).tolist() == [2, 3]
    assert tb.D_BIN_WIDTH == 0.25


def test_band_min_raw_range_reports_curve_inside_band(tmp_path):
    tb.reload_confidence_thresholds(_write(tmp_path, _v2(_EXACT_ANCHORS, floor=0)))
    near, far = tb.band_min_raw_range(tb.BANDS[4])  # 400-600 mm
    assert near == tb.min_confidence([400.0])[0]
    assert far == tb.min_confidence([599.0])[0]
    assert near > far


@pytest.mark.parametrize("bad", [-1, "400", True, float("nan")])
def test_bad_weak_unbacked_cutoff_raises(tmp_path, bad):
    payload = _v2(_EXACT_ANCHORS)
    payload["weak_unbacked_exclude_below_mm"] = bad
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload, allow_nan=True))
    with pytest.raises(ValueError, match="weak_unbacked_exclude_below_mm"):
        tb.reload_confidence_thresholds(path)
