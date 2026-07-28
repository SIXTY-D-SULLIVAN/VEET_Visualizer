import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('veet_charting', Path('VEET_Data_Charting_1.2.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

arrays = {
    'times': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:00:01']),
    'z1_mm': np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8),
    'conf1_raw': np.ones((2, 8, 8), dtype=np.float32),
    'z2_mm': (np.arange(2 * 8 * 8, dtype=np.float32).reshape(2, 8, 8) + 100),
    'conf2_raw': np.ones((2, 8, 8), dtype=np.float32) * 2,
}
df = module.tof_arrays_to_dataframe(arrays)
print(df.shape)
print(df.columns[:6].tolist())
print(df.iloc[0]['z1_mm_0_0'], df.iloc[0]['z2_mm_0_0'], df.iloc[1]['conf2_raw_7_7'])
