import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "veet_charting",
    Path(__file__).resolve().parents[1] / "VEET_Data_Charting_1.2.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_tof_arrays_to_dataframe_flattens_returns():
    arrays = {
        "times": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"]),
        "z1_mm": np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8),
        "conf1_raw": np.ones((2, 8, 8), dtype=np.float32),
        "z2_mm": (np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8) + 100),
        "conf2_raw": np.ones((2, 8, 8), dtype=np.float32) * 2,
    }

    df = module.tof_arrays_to_dataframe(arrays)

    assert df.shape[0] == 2
    assert "time_stamp" in df.columns
    assert "z1_mm_0_0" in df.columns
    assert "conf1_raw_0_0" in df.columns
    assert "z2_mm_0_0" in df.columns
    assert df.iloc[0]["z1_mm_0_0"] == 0.0
    assert df.iloc[0]["z2_mm_0_0"] == 100.0
    assert df.iloc[1]["conf2_raw_7_7"] == 2.0
