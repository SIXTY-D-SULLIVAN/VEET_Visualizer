"""Footprint auto-location and boundary guard in veet_tof_confidence_bench."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import veet_tof_confidence_bench as bench  # noqa: E402


def test_locate_block_finds_off_centre_object():
    score = np.full((8, 8), 20.0)          # weak ghost everywhere
    score[2:6, 1:8] = 200.0                # object high and to the right
    assert bench.locate_block(score, 4, 7) == (2, 1)


def test_locate_block_flat_map_falls_back_to_centre():
    score = np.zeros((8, 8))
    assert bench.locate_block(score, 4, 4) == (2, 2)
    assert bench.locate_block(score, 8, 8) == (0, 0)


def test_shift_mask_grow_and_shrink():
    m = bench._block_mask(3, 2, 3, 2)
    grown = bench.shift_mask(m, 1, grow=True)
    assert grown.sum() == 16 and grown[2:6, 2:6].all()
    assert not bench.shift_mask(m, 1, grow=False).any()
    assert (bench.shift_mask(m, 0, grow=True) == m).all()


def test_shrink_keeps_zones_on_grid_edge():
    m = bench._block_mask(2, 4, 0, 8)      # object spans the full width
    inner = bench.shift_mask(m, 1, grow=False)
    assert inner[3:5, :].all() and inner.sum() == 16


def test_compute_footprint_uses_evidence_for_placement():
    centred = bench.compute_footprint(1000.0, 150.0, 150.0, "nearest")
    nr, nc = centred["n_rows"], centred["n_cols"]
    ev = np.zeros((8, 8))
    ev[0:nr, 8 - nc:8] = 255.0             # object in the top-right corner
    located = bench.compute_footprint(1000.0, 150.0, 150.0, "nearest", ev)
    assert (nr, nc) == (located["n_rows"], located["n_cols"])
    assert (located["row_lo"], located["col_lo"]) == (0, 8 - nc)
    assert not centred["located"] and located["located"]


def test_gate_validation_on_synthetic_frames():
    n, z = 5, 500.0
    dist = np.full((n, 8, 8), z)
    conf = np.full((n, 8, 8), 2.0)          # background ghost at object distance
    conf[:, 3:5, 3:5] = 250.0               # object
    data = bench.TofFrameData(times=np.arange(n, dtype=float), conf1=conf, conf2=np.zeros_like(conf),
                              dist1=dist, dist2=np.zeros_like(dist), sample_period_s=1.0, n_frames=n)
    fp = {m: dict(mask=bench._block_mask(3, 2, 3, 2)) for m in bench.ROUND_MODES}
    v = bench.gate_validation(data, z, fp, 0.15, guard_zones=1)
    assert v["object_pass"] == 1.0 and v["zones_kept"] == 1.0
    assert v["background_reject"] == 1.0 and v["n_background"] > 0
