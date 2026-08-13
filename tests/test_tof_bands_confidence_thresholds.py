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


def test_shipped_config_is_loaded_and_matches_its_own_file():
    # veet_tof_bands.py is imported fresh at collection time, so BANDS
    # already reflects CONFIDENCE_THRESHOLDS_PATH. Compare against whatever
    # the file actually contains right now -- not a frozen snapshot of
    # specific numbers, since lab characterisation is expected to edit this
    # file in place (that's the whole point of it being externalized).
    assert tb.CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
    with open(tb.CONFIDENCE_THRESHOLDS_PATH) as f:
        on_disk = json.load(f)["min_raw_by_band"]
    for band in tb.BANDS:
        assert band.conf_raw == on_disk.get(band.name, tb._DEFAULT_MIN_RAW_BY_NAME[band.name])


def test_all_loaded_thresholds_are_valid_snr_bytes():
    for band in tb.BANDS:
        assert isinstance(band.conf_raw, (int, np.integer))
        assert 0 <= band.conf_raw <= tb.CONF_MAX


def test_partial_override_only_changes_listed_bands(tmp_path):
    path = _write(tmp_path, {"min_raw_by_band": {"far": 30, "room": 200}})
    tb.reload_confidence_thresholds(path)
    assert tb.BANDS[7].conf_raw == 30
    assert tb.BANDS[6].conf_raw == 200
    # Untouched bands fall back to the built-in defaults, not to zero/stale.
    assert tb.BANDS[1].conf_raw == tb._DEFAULT_MIN_RAW_BY_NAME["extreme_near"]


def test_reload_propagates_into_gating_functions(tmp_path):
    path = _write(tmp_path, {"min_raw_by_band": {"room": 200}})
    tb.reload_confidence_thresholds(path)
    # 1500mm is in the "room" band -- min_confidence must reflect the override.
    assert tb.min_confidence([1500.0])[0] == 200


def test_bare_mapping_without_wrapper_key_also_works(tmp_path):
    path = _write(tmp_path, {"far": 30})
    tb.reload_confidence_thresholds(path)
    assert tb.BANDS[7].conf_raw == 30


def test_unknown_band_name_raises(tmp_path):
    path = _write(tmp_path, {"min_raw_by_band": {"not_a_real_band": 100}})
    with pytest.raises(ValueError, match="Unknown band name"):
        tb.reload_confidence_thresholds(path)


@pytest.mark.parametrize("bad_value", [-1, 256, 100.5, "200", True])
def test_out_of_range_or_wrong_type_raises(tmp_path, bad_value):
    path = _write(tmp_path, {"min_raw_by_band": {"far": bad_value}})
    with pytest.raises(ValueError):
        tb.reload_confidence_thresholds(path)


def test_malformed_json_raises(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="Failed to parse"):
        tb.reload_confidence_thresholds(path)


def test_missing_file_falls_back_to_defaults(tmp_path):
    tb.reload_confidence_thresholds(tmp_path / "does_not_exist.json")
    assert not tb.CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
    assert tb.BANDS[7].conf_raw == tb._DEFAULT_MIN_RAW_BY_NAME["far"]


def test_band_geometry_unaffected_by_threshold_reload(tmp_path):
    path = _write(tmp_path, {"min_raw_by_band": {"far": 30}})
    tb.reload_confidence_thresholds(path)
    # Only conf_raw is lab-tunable; mm ranges stay coupled to the dioptric bins.
    assert tb.BANDS[7].z_min_mm == 2000.0
    assert tb.D_BIN_WIDTH == 0.25
