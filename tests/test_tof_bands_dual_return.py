import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import veet_tof_bands as tb

# "intermediate" band: 600-1000mm, min raw confidence 77
Z_INTERMEDIATE = 800.0
CONF_INTERMEDIATE_PASS = 100.0
CONF_INTERMEDIATE_FAIL = 50.0

# "room" band: 1000-2000mm, min raw confidence 64
Z_ROOM = 1500.0
CONF_ROOM_FAIL = 40.0


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
