"""VEET_TOF_Viewer.py -- standalone TOF-only viewer.

Independent of VEET_Data_Charting.py (no import of that module). Loads TOF
data either from a raw device folder (Sensor_Data.csv, parsing only the INF
and TOF rows) or from a previously exported *_tof_arrays.npz / .csv (see
VEET_Data_Charting.py's "Export TOF Arrays" buttons).

All validity gating and dioptric binning is delegated to veet_tof_bands.py --
see VEET_TOF_INTEGRATION.md for the spec. This file must not reimplement any
of that math.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSlider, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import colormaps
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import veet_tof_bands as tb


def find_file_with_suffix(directory, suffix):
    """Find a file in ``directory`` ending with ``suffix`` (case-insensitive)."""
    if not directory or not os.path.isdir(directory):
        return None
    for filename in os.listdir(directory):
        if filename.lower().endswith(suffix.lower()):
            return os.path.join(directory, filename)
    return None


# --- Loading -----------------------------------------------------------

TOF_RETURN_COLS = 64  # zones per return
# time_stamp, TOF, Conf1 x64, Conf2 x64, Dist1 x64, Dist2 x64 -- same column
# order as VEET_Data_Charting.py's self.headers['TOF'].
TOF_FIELDS_PER_ROW = 2 + 4 * TOF_RETURN_COLS


def _parse_timestamps(raw_values):
    """Epoch seconds or milliseconds (auto-detected) -> UTC datetimes."""
    ts = pd.to_numeric(pd.Series(raw_values), errors='coerce')
    sample = ts.dropna().iloc[0] if ts.notna().any() else 0
    unit = 'ms' if len(str(int(sample))) >= 13 else 's'
    return pd.to_datetime(ts, unit=unit, utc=True)


def _sample_period_from_inf(inf_line):
    """Sample period in seconds from the INF line's four per-sensor interval
    values (ms) -- see VEET_TOF_INTEGRATION.md §2. None if unavailable."""
    if not inf_line:
        return None
    parts = [p.strip() for p in inf_line.split(',')]
    if len(parts) < 7:
        return None
    try:
        values_ms = [float(v) for v in parts[-5:-1]]
    except ValueError:
        return None
    if len(set(values_ms)) == 1 and values_ms[0] > 0:
        return values_ms[0] / 1000.0
    return None


@dataclass
class TofSession:
    times: "pd.Series"           # UTC datetimes, one per frame
    z1_mm: np.ndarray            # (n, 8, 8)
    conf1_raw: np.ndarray
    z2_mm: "np.ndarray | None"   # None if the session has no second return
    conf2_raw: "np.ndarray | None"
    sample_period_s: float
    serial_number: str = ''
    firmware: str = ''
    source: str = ''


def _pad_or_trim(fields, n):
    if len(fields) < n:
        return fields + [''] * (n - len(fields))
    return fields[:n]


def load_from_device_folder(folder):
    """Parse only the INF and TOF rows of Sensor_Data.csv.

    Deliberately not shared with VEET_Data_Charting.py's parser, which is
    entangled with the other five sensor types and the multi-window GUI --
    this stays small and standalone.
    """
    sensor_path = find_file_with_suffix(folder, "Sensor_Data.csv")
    if not sensor_path:
        raise FileNotFoundError(f"No *Sensor_Data.csv found in {folder}")

    inf_line = None
    tof_rows = []
    with open(sensor_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if inf_line is None and ',INF,' in line:
                inf_line = line.strip()
            elif ',TOF,' in line:
                tof_rows.append(line.strip())

    if not tof_rows:
        raise ValueError(f"No TOF rows found in {sensor_path}")

    parts = [_pad_or_trim(row.split(','), TOF_FIELDS_PER_ROW) for row in tof_rows]
    arr = np.array(parts, dtype=object)

    times = _parse_timestamps(arr[:, 0])
    values = pd.DataFrame(arr[:, 2:]).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    n = values.shape[0]

    conf1 = values[:, 0 * TOF_RETURN_COLS: 1 * TOF_RETURN_COLS].reshape(n, 8, 8)
    conf2 = values[:, 1 * TOF_RETURN_COLS: 2 * TOF_RETURN_COLS].reshape(n, 8, 8)
    dist1 = values[:, 2 * TOF_RETURN_COLS: 3 * TOF_RETURN_COLS].reshape(n, 8, 8)
    dist2 = values[:, 3 * TOF_RETURN_COLS: 4 * TOF_RETURN_COLS].reshape(n, 8, 8)

    order = np.argsort(times.to_numpy())
    times = times.iloc[order].reset_index(drop=True)
    dist1, conf1, dist2, conf2 = dist1[order], conf1[order], dist2[order], conf2[order]

    z2_present = bool(np.isfinite(dist2).any() or np.isfinite(conf2).any())

    period = _sample_period_from_inf(inf_line)
    if not period:
        diffs = times.diff().dt.total_seconds().dropna()
        period = float(diffs.median()) if not diffs.empty else 5.0

    firmware = serial = ''
    if inf_line:
        ip = [p.strip() for p in inf_line.split(',')]
        if len(ip) > 3:
            serial = ip[3]
        if len(ip) > 4:
            firmware = ip[4]

    return TofSession(
        times=times, z1_mm=dist1, conf1_raw=conf1,
        z2_mm=dist2 if z2_present else None,
        conf2_raw=conf2 if z2_present else None,
        sample_period_s=period, serial_number=serial, firmware=firmware,
        source=os.path.basename(sensor_path),
    )


def load_from_npz(path):
    """Load the standardized *_tof_arrays.npz export (sample period and
    encoding already resolved in its meta_json -- see export_tof_arrays)."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data['meta_json']))
    times = pd.Series(pd.to_datetime(data['timestamps_utc_s'], unit='s', utc=True))
    z1, c1 = data['z1_mm'].astype(float), data['conf1_raw'].astype(float)
    z2 = data['z2_mm'].astype(float) if 'z2_mm' in data else None
    c2 = data['conf2_raw'].astype(float) if 'conf2_raw' in data else None
    return TofSession(
        times=times, z1_mm=z1, conf1_raw=c1, z2_mm=z2, conf2_raw=c2,
        sample_period_s=float(meta.get('sample_period_s') or 5.0),
        serial_number=meta.get('serial_number', ''),
        firmware=meta.get('firmware', ''),
        source=os.path.basename(path),
    )


def load_from_csv_export(path):
    """Load the flattened *_tof_arrays.csv export (inverse of
    tof_arrays_to_dataframe). No metadata block, so sample period is
    inferred from the timestamp column."""
    df = pd.read_csv(path)
    n = len(df)

    def stack(prefix):
        cols = [f'{prefix}_{i}_{j}' for i in range(8) for j in range(8)]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].to_numpy(dtype=float).reshape(n, 8, 8)

    z1, c1 = stack('z1_mm'), stack('conf1_raw')
    if z1 is None or c1 is None:
        raise ValueError(f"{path} is missing required z1_mm_*/conf1_raw_* columns")
    z2, c2 = stack('z2_mm'), stack('conf2_raw')

    if 'time_stamp' in df.columns:
        times = pd.to_datetime(df['time_stamp'], utc=True, errors='coerce')
    else:
        times = pd.Series(pd.NaT, index=df.index)
    diffs = times.diff().dt.total_seconds().dropna()
    period = float(diffs.median()) if not diffs.empty else 5.0

    return TofSession(
        times=times, z1_mm=z1, conf1_raw=c1, z2_mm=z2, conf2_raw=c2,
        sample_period_s=period, source=os.path.basename(path),
    )


def load_array_file(path):
    if path.lower().endswith('.npz'):
        return load_from_npz(path)
    if path.lower().endswith('.csv'):
        return load_from_csv_export(path)
    raise ValueError(f"Unsupported array file type: {path}")


# --- Analysis ------------------------------------------------------------

def compute_stats(session: TofSession):
    """Mirrors VEET_Data_Charting.py's MainWindow.compute_tof_stats, minus
    the wear-gating split (no IMU data available standalone -- wear
    detection is explicitly out of scope for veet_tof_bands, see
    VEET_TOF_INTEGRATION.md §1)."""
    if session.z2_mm is not None:
        z_mm, conf_raw, ambiguous_mask = tb.resolve_dual_return(
            session.z1_mm, session.conf1_raw, session.z2_mm, session.conf2_raw)
    else:
        z_mm, conf_raw, ambiguous_mask = session.z1_mm, session.conf1_raw, None

    interval = session.sample_period_s
    mask = tb.analysis_mask(z_mm, conf_raw)
    n_frames = z_mm.shape[0]
    valid_frames = int(mask.any(axis=(1, 2)).sum())

    hist = tb.dioptric_histogram(z_mm, conf_raw, interval)
    dose = tb.dioptric_dose(z_mm, conf_raw, interval)

    diopters = tb.to_diopters(z_mm)[mask]
    median_vergence = float(np.median(diopters)) if diopters.size else None

    band_indices = tb.band_index(z_mm)[mask]
    band_share = []
    for band in tb.BANDS:
        count = int((band_indices == band.index).sum())
        z_hi = band.z_max_mm
        range_text = f"{band.z_min_mm:.0f}-{'inf' if z_hi == float('inf') else f'{z_hi:.0f}'} mm"
        pct = count / diopters.size * 100.0 if diopters.size else 0.0
        band_share.append((band.name, range_text, pct, count * interval / 3600.0))

    yields = tb.zone_yield(z_mm, conf_raw, ambiguous_mask=ambiguous_mask)

    return {
        'z_resolved': z_mm, 'conf_resolved': conf_raw, 'mask': mask,
        'n_frames': n_frames, 'valid_frames': valid_frames, 'sample_period': interval,
        'dose_dh': dose, 'hist_seconds': hist, 'bin_edges': tb.D_BIN_EDGES,
        'median_vergence': median_vergence, 'band_share': band_share, 'yields': yields,
    }


# --- GUI -------------------------------------------------------------------

class SummaryTab(QWidget):
    """Time-weighted dioptric histogram + band-occupancy chart -- same
    layout as VEET_Data_Charting.py's TOFAnalysisWindow, standalone."""

    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(11, 5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def set_data(self, session: TofSession, stats: dict):
        fig = self.fig
        fig.clear()
        ax_hist, ax_bands = fig.subplots(1, 2)

        edges = np.asarray(stats['bin_edges'])
        hist = np.asarray(stats['hist_seconds']) / 3600.0
        regular = len(edges) - 1
        ax_hist.bar(edges[:regular], hist[:regular], width=0.25, align='edge',
                    color='steelblue', alpha=0.9)
        ax_hist.set_xlabel("Vergence [D]  (D = 1000 / distance mm)")
        ax_hist.set_ylabel("Exposure [zone·hours]")
        ax_hist.set_title("Time-weighted dioptric histogram (0.25 D bins)")
        ax_hist.grid(True, axis='y', linestyle=':', linewidth=0.5)

        top_axis = ax_hist.secondary_xaxis('top')
        ref_diopters = [0.5, 1.0, 2.0, 3.33, 5.0, 10.0]
        top_axis.set_xticks(ref_diopters)
        top_axis.set_xticklabels(['2000', '1000', '500', '300', '200', '100'])
        top_axis.set_xlabel("Distance [mm]")

        overflow_h = hist[regular]
        if overflow_h > 0:
            ax_hist.annotate(f"overflow (<100 mm): {overflow_h:.2f} zone·h",
                              xy=(0.98, 0.98), xycoords='axes fraction',
                              ha='right', va='top', fontsize=8, color='gray')

        names = [f"{name}\n{rng}" for name, rng, pct, hours in stats['band_share']]
        shares = [pct for name, rng, pct, hours in stats['band_share']]
        hours_list = [hours for name, rng, pct, hours in stats['band_share']]
        y_pos = np.arange(len(names))[::-1]
        colors = ['firebrick' if 1 <= i <= 3 else 'steelblue' for i in range(len(names))]
        ax_bands.barh(y_pos, shares, color=colors, alpha=0.85)
        ax_bands.set_yticks(y_pos)
        ax_bands.set_yticklabels(names, fontsize=8)
        ax_bands.set_xlabel("Share of valid zone-time [%]")
        ax_bands.set_title("Reporting-band occupancy")
        ax_bands.grid(True, axis='x', linestyle=':', linewidth=0.5)
        for y, pct, hours in zip(y_pos, shares, hours_list):
            if pct > 0:
                ax_bands.annotate(f" {pct:.1f}%  ({hours:.1f} zone·h)",
                                  xy=(pct, y), va='center', fontsize=7)

        yields = stats['yields']
        summary = (f"dose {stats['dose_dh']:.1f} D·h (zone-weighted)"
                   + (f"  |  median vergence {stats['median_vergence']:.2f} D"
                      if stats['median_vergence'] is not None else "")
                   + f"  |  {stats['valid_frames']}/{stats['n_frames']} frames with valid zones"
                   + f"  |  sample period {stats['sample_period']:.2f}s"
                   + f"  |  analysis-field yield {yields['analysis_field_yield']:.2f}"
                   + f"  |  perimeter yield {yields['perimeter_yield']:.2f}")
        title = session.source or session.serial_number or "session"
        fig.suptitle(f"{title} — TOF dioptric analysis\n{summary}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        self.canvas.draw()


class SpatialTab(QWidget):
    """8x8 zone-grid heatmap with a frame scrubber -- the view this tool
    adds that VEET_Data_Charting.py's TOF window doesn't have.

    Renders distance and confidence side by side for one source at a time
    (Resolved / Object 1 / Object 2), rather than one grid you flip between
    quantities on -- a zone's distance and its confidence are read together,
    which is the actual question ("is this near reading trustworthy?") most
    of the time.
    """

    def __init__(self):
        super().__init__()
        self.session = None
        self.stats = None
        self.sources = {}  # source name -> (distance_array, confidence_array)

        self.fig = Figure(figsize=(11, 5.5))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._draw)

        self.chk_guard = QCheckBox("Show guard band")
        self.chk_guard.setChecked(True)
        self.chk_guard.stateChanged.connect(self._draw)

        self.chk_invalid = QCheckBox("Mark invalid zones")
        self.chk_invalid.setChecked(True)
        self.chk_invalid.stateChanged.connect(self._draw)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)

        self.spin = QSpinBox()
        self.spin.setMinimum(0)
        self.spin.valueChanged.connect(self._on_spin)

        self.time_label = QLabel("")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Source:"))
        controls.addWidget(self.combo)
        controls.addWidget(self.chk_guard)
        controls.addWidget(self.chk_invalid)
        controls.addStretch(1)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        frame_row.addWidget(self.slider, 1)
        frame_row.addWidget(self.spin)
        frame_row.addWidget(self.time_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(controls)
        layout.addLayout(frame_row)

    def _on_slider(self, value):
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._draw()

    def _on_spin(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._draw()

    def set_data(self, session: TofSession, stats: dict):
        self.session = session
        self.stats = stats
        n = session.z1_mm.shape[0]

        self.sources = {
            'Resolved': (stats['z_resolved'], stats['conf_resolved']),
            'Object 1': (session.z1_mm, session.conf1_raw),
        }
        if session.z2_mm is not None:
            self.sources['Object 2'] = (session.z2_mm, session.conf2_raw)

        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(self.sources.keys()))
        self.combo.blockSignals(False)

        for widget in (self.slider, self.spin):
            widget.blockSignals(True)
            widget.setMaximum(max(n - 1, 0))
            widget.setValue(0)
            widget.blockSignals(False)

        self._draw()

    def _overlay(self, ax, idx):
        if self.chk_guard.isChecked():
            # ANALYSIS_FIELD trims row 0 and row 7 (see VEET_TOF_INTEGRATION.md §3.3)
            ax.axhline(0.5, color='orange', linewidth=2)
            ax.axhline(6.5, color='orange', linewidth=2)

        if self.chk_invalid.isChecked():
            frame_mask = self.stats['mask'][idx]
            for r in range(8):
                for c in range(8):
                    if not frame_mask[r, c]:
                        ax.add_patch(mpatches.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1, fill=False,
                            hatch='xx', edgecolor='red', linewidth=1.2))

    def _draw(self):
        if self.stats is None or not self.sources:
            return
        idx = self.slider.value()
        source = self.combo.currentText()
        if source not in self.sources:
            return
        dist_data, conf_data = (arr[idx] for arr in self.sources[source])

        # Clearing the whole figure (not just the axes) avoids a stale
        # colorbar axes surviving across redraws -- ax.clear() alone leaves
        # the colorbar's own axes orphaned and its later .remove() fails.
        self.fig.clear()
        ax_dist, ax_conf = self.fig.subplots(1, 2)

        # Distance panel: color = reporting-band membership (tb.band_index),
        # not this frame's local min/max -- so the scale is both static
        # (same 8 colors every frame) and grouped by the module's own
        # binning scheme (VEET_TOF_INTEGRATION.md distance-band table),
        # rather than a continuous scale that reshuffles per slice.
        band_idx = tb.band_index(dist_data)
        n_bands = len(tb.BANDS)
        cmap = colormaps['RdYlBu_r'].resampled(n_bands)  # near=warm, far=cool
        im_dist = ax_dist.imshow(band_idx, cmap=cmap, vmin=-0.5, vmax=n_bands - 0.5, origin='upper')
        cbar_dist = self.fig.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04, ticks=range(n_bands))
        cbar_dist.ax.set_yticklabels([b.name for b in tb.BANDS], fontsize=7)
        cbar_dist.set_label('reporting band (§ Distance bands)')
        ax_dist.set_title(f"{source} — distance (mm)")

        # Confidence panel: fixed 0-255 scale, static across frames for the
        # same reason as the distance panel above.
        im_conf = ax_conf.imshow(conf_data, cmap='magma', vmin=0, vmax=tb.CONF_MAX, origin='upper')
        cbar_conf = self.fig.colorbar(im_conf, ax=ax_conf, fraction=0.046, pad=0.04)
        cbar_conf.set_label('raw confidence (SNR domain, 0-255)')
        ax_conf.set_title(f"{source} — confidence (raw)")

        for ax, data in ((ax_dist, dist_data), (ax_conf, conf_data)):
            for r in range(8):
                for c in range(8):
                    ax.text(c, r, f"{data[r, c]:.0f}", ha='center', va='center',
                             fontsize=6, color='white',
                             path_effects=[mpatheffects.withStroke(linewidth=1.5, foreground='black')])
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.set_xlabel("column (horizontal)")
            ax.set_ylabel("row (vertical)")
            self._overlay(ax, idx)

        ts = self.session.times.iloc[idx] if self.session.times is not None else None
        self.time_label.setText(str(ts) if ts is not None and pd.notna(ts) else f"frame {idx}")

        self.fig.tight_layout()
        self.canvas.draw_idle()


class ThresholdsTab(QWidget):
    """Legend: the confidence-threshold gate and band geometry actually in
    effect right now, so a reload's result is visible, not just asserted."""

    def __init__(self):
        super().__init__()
        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.addWidget(self.source_label)
        layout.addWidget(self.table)

        self.refresh()

    def refresh(self, reloaded=False):
        import datetime
        loaded = tb.CONFIDENCE_THRESHOLDS_LOADED_FROM_FILE
        origin = "file" if loaded else "built-in defaults (file not found)"
        prefix = "Reloaded" if reloaded else "Loaded"
        when = datetime.datetime.now().strftime('%H:%M:%S')
        self.source_label.setText(
            f"{prefix} {when} — confidence thresholds: {origin} — {tb.CONFIDENCE_THRESHOLDS_SOURCE}")

        headers = ["Band", "Name", "Distance (mm)", "Vergence (D)", "Min raw (SNR)"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        bands = tb.BANDS
        self.table.setRowCount(len(bands))
        for row, b in enumerate(bands):
            z_hi = 'inf' if b.z_max_mm == float('inf') else f"{b.z_max_mm:.0f}"
            d_hi = 'inf' if b.d_max == float('inf') else f"{b.d_max:.2f}"
            values = [
                str(b.index), b.name, f"{b.z_min_mm:.0f}-{z_hi}",
                f"{d_hi}-{b.d_min:.2f}", str(b.conf_raw),
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if 1 <= b.index <= 3:  # near-work bands, matches SummaryTab's highlight
                    item.setBackground(QColor(255, 228, 225))
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VEET TOF Viewer")
        self.resize(1200, 800)
        self.session = None

        central = QWidget()
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        btn_folder = QPushButton("Open Device Folder...")
        btn_folder.clicked.connect(self.open_folder)
        btn_file = QPushButton("Open Array File (.npz/.csv)...")
        btn_file.clicked.connect(self.open_array_file)
        btn_reload_thresholds = QPushButton("Reload Confidence Thresholds")
        btn_reload_thresholds.setToolTip(
            "Re-read veet_tof_confidence_thresholds.json after a lab-characterisation "
            "edit and re-analyze the loaded session -- no restart needed.")
        btn_reload_thresholds.clicked.connect(self.reload_thresholds)
        self.status_label = QLabel("No data loaded.")
        controls.addWidget(btn_folder)
        controls.addWidget(btn_file)
        controls.addWidget(btn_reload_thresholds)
        controls.addWidget(self.status_label, 1)
        layout.addLayout(controls)

        self.tabs = QTabWidget()
        self.summary_tab = SummaryTab()
        self.spatial_tab = SpatialTab()
        self.thresholds_tab = ThresholdsTab()
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.spatial_tab, "Spatial (8×8)")
        self.tabs.addTab(self.thresholds_tab, "Confidence Thresholds")
        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready.")

    def reload_thresholds(self):
        try:
            tb.reload_confidence_thresholds()
        except ValueError as e:
            QMessageBox.critical(self, "Reload Failed", str(e))
            self.statusBar().showMessage("Reload failed -- see error dialog.", 8000)
            return

        self.thresholds_tab.refresh(reloaded=True)
        self.tabs.setCurrentWidget(self.thresholds_tab)  # jump to the legend so the change is visible immediately

        if self.session is not None:
            # Preserve where the user was looking -- reload is meant for
            # comparing before/after on the same frame, not jumping back to 0.
            prev_frame = self.spatial_tab.slider.value()
            prev_source = self.spatial_tab.combo.currentText()
            self._apply_session(self.session)
            if 0 <= prev_frame <= self.spatial_tab.slider.maximum():
                self.spatial_tab.slider.setValue(prev_frame)
            if prev_source in self.spatial_tab.sources:
                self.spatial_tab.combo.setCurrentText(prev_source)
            reanalyzed = " Session re-analyzed."
        else:
            reanalyzed = ""

        QMessageBox.information(
            self, "Confidence Thresholds Reloaded",
            f"Read {tb.CONFIDENCE_THRESHOLDS_SOURCE}.{reanalyzed}\n\n"
            f"See the \"Confidence Thresholds\" tab for the values now in effect.")
        self.statusBar().showMessage(
            f"Confidence thresholds reloaded from {tb.CONFIDENCE_THRESHOLDS_SOURCE}.{reanalyzed}", 8000)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select VEET device folder")
        if not folder:
            return
        try:
            session = load_from_device_folder(folder)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return
        self._apply_session(session)

    def open_array_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TOF Array File", "", "TOF arrays (*.npz *.csv)")
        if not path:
            return
        try:
            session = load_array_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return
        self._apply_session(session)

    def _apply_session(self, session: TofSession):
        try:
            stats = compute_stats(session)
        except Exception as e:
            QMessageBox.critical(self, "Analysis Failed", str(e))
            return
        self.session = session
        self.summary_tab.set_data(session, stats)
        self.spatial_tab.set_data(session, stats)
        self.status_label.setText(
            f"{session.source}  |  serial {session.serial_number or '?'}  |  "
            f"firmware {session.firmware or '?'}  |  {session.z1_mm.shape[0]} frames  |  "
            f"{session.sample_period_s:.2f}s/sample"
        )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
