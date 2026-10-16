import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import veet_tof_bands as tb

# Pass/fail confidences derived from whatever threshold curve is loaded, so
# these tests exercise dual-return logic, not today's calibration numbers.
Z_INTERMEDIATE = 800.0
CONF_INTERMEDIATE_PASS = float(tb.min_confidence([Z_INTERMEDIATE])[0])
CONF_INTERMEDIATE_FAIL = CONF_INTERMEDIATE_PASS - 1

Z_ROOM = 1500.0
CONF_ROOM_FAIL = float(tb.min_confidence([Z_ROOM])[0]) - 1
assert CONF_INTERMEDIATE_FAIL > 0 and CONF_ROOM_FAIL > 0  # must be nonzero-but-low


def test_object1_valid_passes_through_unchanged():
    z, conf, ambiguous = tb.resolve_dual_return(
        z1_mm=np.array([Z_INTERMEDIATE]),
        conf1_raw=np.array([CONF_INTERMEDIATE_PASS]),
        z2_mm=np.array([0.0]),
        conf2_raw=np.array([0.0]),
    )
    assert z[0] == Z_INTERMEDIATE
    assert conf[0] == CONF_INTERMEDIATE_PASS
    assert not ambiguous[0]


def test_confident_far_when_object1_is_zero_and_object2_is_zero():
    z, conf, ambiguous = tb.resolve_dual_return(
        z1_mm=np.array([0.0]),
        conf1_raw=np.array([0.0]),
        z2_mm=np.array([0.0]),
        conf2_raw=np.array([0.0]),
    )
    assert z[0] == tb.NO_RETURN_SENTINEL_MM
    assert conf[0] == tb.CONF_MAX
    assert not ambiguous[0]


def test_confident_far_when_object1_is_low_confidence_and_object2_is_zero():
    """A phantom (low-confidence, nonzero) object-1 return must not override
    a clean "nothing there" from object 2 -- see resolve_dual_return docstring."""
    z, conf, ambiguous = tb.resolve_dual_return(
        z1_mm=np.array([Z_INTERMEDIATE]),
        conf1_raw=np.array([CONF_INTERMEDIATE_FAIL]),
        z2_mm=np.array([0.0]),
        conf2_raw=np.array([0.0]),
    )
    assert z[0] == tb.NO_RETURN_SENTINEL_MM
    assert conf[0] == tb.CONF_MAX
    assert not ambiguous[0]


def test_rescued_by_object2_when_object1_fails():
    z, conf, ambiguous = tb.resolve_dual_return(
        z1_mm=np.array([0.0]),
        conf1_raw=np.array([0.0]),
        z2_mm=np.array([Z_INTERMEDIATE]),
        conf2_raw=np.array([CONF_INTERMEDIATE_PASS]),
    )
    assert z[0] == Z_INTERMEDIATE
    assert conf[0] == CONF_INTERMEDIATE_PASS
    assert not ambiguous[0]

    # A zone rescued this way must actually clear the ordinary gate downstream.
    grid_z1 = np.zeros((8, 8))
    grid_c1 = np.zeros((8, 8))
    grid_z2 = np.zeros((8, 8))
    grid_c2 = np.zeros((8, 8))
    grid_z2[3, 3] = Z_INTERMEDIATE
    grid_c2[3, 3] = CONF_INTERMEDIATE_PASS

    z_resolved, conf_resolved, _ = tb.resolve_dual_return(grid_z1, grid_c1, grid_z2, grid_c2)
    assert tb.analysis_mask(z_resolved, conf_resolved)[3, 3]


def test_ambiguous_when_both_returns_are_low_confidence():
    grid_z1 = np.zeros((8, 8))
    grid_c1 = np.zeros((8, 8))
    grid_z2 = np.zeros((8, 8))
    grid_c2 = np.zeros((8, 8))
    grid_z1[3, 3] = Z_ROOM
    grid_c1[3, 3] = CONF_ROOM_FAIL
    grid_z2[3, 3] = Z_ROOM
    grid_c2[3, 3] = CONF_ROOM_FAIL

    z_resolved, conf_resolved, ambiguous = tb.resolve_dual_return(grid_z1, grid_c1, grid_z2, grid_c2)
    assert ambiguous[3, 3]
    assert not tb.analysis_mask(z_resolved, conf_resolved)[3, 3]

    yields = tb.zone_yield(z_resolved, conf_resolved, ambiguous_mask=ambiguous)
    assert yields["ambiguous_dual_return_fraction"] == 1.0 / tb.ANALYSIS_FIELD.sum()


def test_ambiguous_flag_does_not_leak_into_object1_valid_zones():
    """A coincidentally low-confidence object 2 must not mark a zone ambiguous
    when object 1 already passed the gate on its own -- object 2 is never
    even inspected in that case."""
    z, conf, ambiguous = tb.resolve_dual_return(
        z1_mm=np.array([Z_INTERMEDIATE]),
        conf1_raw=np.array([CONF_INTERMEDIATE_PASS]),
        z2_mm=np.array([Z_ROOM]),
        conf2_raw=np.array([CONF_ROOM_FAIL]),
    )
    assert z[0] == Z_INTERMEDIATE
    assert conf[0] == CONF_INTERMEDIATE_PASS
    assert not ambiguous[0]


def test_shape_mismatch_raises():
    try:
        tb.resolve_dual_return(
            z1_mm=np.zeros((8, 8)),
            conf1_raw=np.zeros((8, 8)),
            z2_mm=np.zeros((4, 4)),
            conf2_raw=np.zeros((4, 4)),
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


# --- weak near return with no object 2 behind it ------------------------------

Z_NEAR = 300.0
CONF_NEAR_FAIL = float(tb.min_confidence([Z_NEAR])[0]) - 1


def _resolve_one(z1, c1, z2=0.0, c2=0.0):
    z, conf, amb = tb.resolve_dual_return(
        np.array([z1]), np.array([c1]), np.array([z2]), np.array([c2]))
    return z[0], conf[0], amb[0]


def test_weak_near_unbacked_is_excluded_not_recoded_far():
    assert tb.WEAK_UNBACKED_EXCLUDE_BELOW_MM > Z_NEAR  # shipped config enables the rule here
    z, conf, amb = _resolve_one(Z_NEAR, CONF_NEAR_FAIL)
    assert amb
    assert z == Z_NEAR and conf == 0.0
    assert not tb.valid_mask(np.array([z]), np.array([conf]))[0]


def test_weak_unbacked_beyond_cutoff_still_resolves_far():
    z_far_side = tb.WEAK_UNBACKED_EXCLUDE_BELOW_MM + 100.0
    weak = float(tb.min_confidence([z_far_side])[0]) - 1
    z, conf, amb = _resolve_one(z_far_side, weak)
    assert not amb
    assert z == tb.NO_RETURN_SENTINEL_MM and conf == tb.CONF_MAX


def test_zero_object1_still_resolves_far_even_below_cutoff():
    # A sub-floor object 1 is a no-return, not a weak return.
    z, conf, amb = _resolve_one(0.0, 0.0)
    assert not amb and z == tb.NO_RETURN_SENTINEL_MM


def test_near_object1_passing_is_unaffected():
    z, conf, amb = _resolve_one(Z_NEAR, 255.0)
    assert not amb and z == Z_NEAR and conf == 255.0


def test_cutoff_is_configurable_and_zero_disables(tmp_path):
    import json
    cfg = {"format_version": 2, "rule": "power_law_fit", "floor_min_raw": 10,
           "weak_unbacked_exclude_below_mm": 0,
           "anchors": [[200, 245], [300, 150], [600, 50], [1000, 15]]}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(cfg))
    try:
        tb.reload_confidence_thresholds(path)
        assert tb.WEAK_UNBACKED_EXCLUDE_BELOW_MM == 0
        weak = float(tb.min_confidence([Z_NEAR])[0]) - 1
        z, _, amb = _resolve_one(Z_NEAR, weak)
        assert not amb and z == tb.NO_RETURN_SENTINEL_MM
    finally:
        tb.reload_confidence_thresholds()


def test_legacy_v1_config_disables_the_rule():
    import veet_tof_gate_compare as gc
    try:
        tb.reload_confidence_thresholds(gc.BASELINE_V1_PATH)
        assert tb.WEAK_UNBACKED_EXCLUDE_BELOW_MM == 0
    finally:
        tb.reload_confidence_thresholds()
