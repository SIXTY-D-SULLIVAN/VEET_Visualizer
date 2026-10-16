import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import veet_tof_bands as tb
import veet_tof_gate_compare as gc


@pytest.fixture(autouse=True)
def _restore_default_thresholds():
    yield
    tb.reload_confidence_thresholds()


def _session(n=20):
    """A flat 1500 mm wall at confidence 30 everywhere, object 2 empty:
    fails a per-band room threshold of 64, passes a curve floored near 10."""
    z1 = np.full((n, 8, 8), 1500.0)
    c1 = np.full((n, 8, 8), 30.0)
    return z1, c1, np.zeros_like(z1), np.zeros_like(c1), 2.0


def test_frozen_v1_baseline_loads_as_step_curve():
    tb.reload_confidence_thresholds(gc.BASELINE_V1_PATH)
    assert tb.CONFIDENCE_RULE == "per_band_step"
    with open(gc.BASELINE_V1_PATH) as f:
        assert tb.min_confidence([1500.0])[0] == json.load(f)["min_raw_by_band"]["room"]


def test_weak_wall_moves_from_far_to_room(tmp_path):
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"format_version": 2, "rule": "power_law_fit", "floor_min_raw": 10,
                                 "anchors": [[300, 150], [600, 50], [1000, 15]]}))
    base = gc.gate_session(*_session(), gc.BASELINE_V1_PATH)
    cand = gc.gate_session(*_session(), str(curve))

    room = [b.name for b in tb.BANDS].index("room")
    far = [b.name for b in tb.BANDS].index("far")
    assert base["outcomes"]["confident far"] == 1.0
    assert cand["outcomes"]["object 1 valid"] == 1.0
    assert base["band_hours"][room] == 0.0 and cand["band_hours"][room] > 0.0
    assert cand["band_hours"][far] == 0.0

    cmp = gc.compare(base, cand)
    # Valid under both gates (base as confident far), so nothing is gained or
    # lost -- every analysis-field zone-frame "moved" from 2000 mm to 1500 mm.
    n_field = 20 * int(tb.ANALYSIS_FIELD.sum())
    assert (cmp["gained_n"], cmp["lost_n"], cmp["moved_n"]) == (0, 0, n_field)
    room_row = cmp["per_band"][room]
    assert room_row["moved"]["n"] == n_field and room_row["unchanged"]["n"] == 0
    # A uniform wall is fully coherent; stable in every frame after the first.
    assert room_row["moved"]["coherent"] == 1.0
    assert room_row["moved"]["stable"] == pytest.approx(19 / 20)


def test_gained_and_unchanged_are_separated(tmp_path):
    z1, c1, z2, c2, dt = _session()
    c1[:10] = 255.0  # first half passes both gates at the same distance
    z2[10:] = 1500.0  # second half: object 2 present but weak -> v1 ambiguous
    c2[10:] = 30.0
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"format_version": 2, "rule": "power_law_fit", "floor_min_raw": 10,
                                 "anchors": [[300, 150], [600, 50], [1000, 15]]}))
    cmp = gc.compare(gc.gate_session(z1, c1, z2, c2, dt, gc.BASELINE_V1_PATH),
                     gc.gate_session(z1, c1, z2, c2, dt, str(curve)))
    half = 10 * int(tb.ANALYSIS_FIELD.sum())
    assert (cmp["gained_n"], cmp["moved_n"], cmp["lost_n"]) == (half, 0, 0)
    room_row = cmp["per_band"][[b.name for b in tb.BANDS].index("room")]
    assert room_row["gained"]["n"] == half and room_row["unchanged"]["n"] == half


def test_coherence_flags_isolated_zone():
    z = np.full((3, 8, 8), 1000.0)
    mask = np.zeros(z.shape, dtype=bool)
    mask[:, 3, 3] = True          # isolated
    mask[:, 0:3, 5:8] = True      # 3x3 patch
    agree = gc._neighbour_agreement(z, mask)
    assert agree[0, 3, 3] == 0
    assert agree[0, 1, 6] == 8
