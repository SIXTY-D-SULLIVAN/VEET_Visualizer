"""veet_tof_confidence_gui.py -- simple GUI front-end for veet_tof_confidence_bench.py.

For a technician running many bench scenarios (different objects, tilts,
distance sweeps, candidate threshold configs) without needing to remember or
type CLI flags. Pick a data folder, pick an object preset (or fill in custom
values), optionally pick an alternate thresholds JSON, click Run.

This file contains no analysis logic of its own -- it only collects inputs
and calls veet_tof_confidence_bench.run_bench(), the same function the CLI
(`veet_tof_confidence_bench.py --data-dir ...`) uses. Both stay in sync
automatically since there is only one analysis code path.

Run with:  python veet_tof_confidence_gui.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import traceback

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import veet_tof_confidence_bench as bench
import veet_tof_bands as tb


# --- Object presets ----------------------------------------------------------
# Edit this dict to add new object/tilt combinations -- no other code changes
# needed. "Custom" (always last) leaves every field as the technician set it.
PRESETS = {
    "Paper -- flat / normal (0 deg)": dict(
        include="paper", exclude="phone,30deg", w_mm=200.0, h_mm=125.0, tol_frac=0.15),
    "Paper -- 30 deg back tilt": dict(
        include="paper,30deg", exclude="phone", w_mm=200.0, h_mm=125.0, tol_frac=0.25),
    "Phone -- flat / normal (0 deg)": dict(
        include="phone", exclude="30deg", w_mm=75.0, h_mm=160.0, tol_frac=0.15),
    "Phone -- 30 deg back tilt": dict(
        include="phone,30deg", exclude="", w_mm=75.0, h_mm=160.0, tol_frac=0.25),
    "Custom...": dict(include="", exclude="", w_mm=200.0, h_mm=125.0, tol_frac=0.15),
}


def _split_tokens(text: str) -> tuple:
    return tuple(t.strip() for t in text.split(",") if t.strip())


class _QtLogStream(io.TextIOBase):
    """Minimal writable stream that forwards text to a Qt signal, so
    run_bench()'s existing print() calls show up live in the GUI log panel
    without any change to the analysis code."""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit = emit_fn

    def write(self, s):
        if s:
            self._emit(s)
        return len(s)

    def flush(self):
        pass


class BenchWorker(QThread):
    log = Signal(str)
    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    def run(self):
        stream = _QtLogStream(self.log.emit)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                result = bench.run_bench(**self._kwargs)
        except Exception as e:
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(e))
            return
        self.succeeded.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VEET TOF Confidence Threshold Bench")
        self.resize(880, 900)
        self.worker: "BenchWorker | None" = None
        self.last_result: "dict | None" = None

        central = QWidget()
        outer = QVBoxLayout(central)

        intro = QLabel(
            "Run the per-distance confidence-threshold bench analysis on a folder of TOF "
            "CSVs (see veet_tof_confidence_bench.py). Pick a data folder and an object "
            "preset, then click Run."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()

        # Data folder
        self.data_dir_edit = QLineEdit()
        btn_data_dir = QPushButton("Browse...")
        btn_data_dir.clicked.connect(self._pick_data_dir)
        form.addRow("Data folder:", self._row(self.data_dir_edit, btn_data_dir))

        # Preset
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        form.addRow("Object preset:", self.preset_combo)

        # Object label
        self.label_edit = QLineEdit("Object 1 (paper, flat)")
        form.addRow("Report label:", self.label_edit)

        # Include / exclude
        self.include_edit = QLineEdit()
        form.addRow("Filename must include (comma-separated):", self.include_edit)
        self.exclude_edit = QLineEdit()
        form.addRow("Filename must exclude (comma-separated):", self.exclude_edit)

        # Object size
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(1.0, 2000.0)
        self.w_spin.setSuffix(" mm")
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(1.0, 2000.0)
        self.h_spin.setSuffix(" mm")
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("width (horizontal):"))
        size_row.addWidget(self.w_spin)
        size_row.addWidget(QLabel("height (vertical):"))
        size_row.addWidget(self.h_spin)
        size_widget = QWidget()
        size_widget.setLayout(size_row)
        form.addRow("Object size as presented to sensor:", size_widget)

        outer.addLayout(form)

        # Thresholds
        thresh_box = QGroupBox("Confidence thresholds")
        thresh_layout = QHBoxLayout(thresh_box)
        self.thresholds_edit = QLineEdit()
        self.thresholds_edit.setReadOnly(True)
        self.thresholds_edit.setPlaceholderText("(built-in default -- veet_tof_confidence_thresholds.json)")
        btn_thresholds = QPushButton("Browse...")
        btn_thresholds.clicked.connect(self._pick_thresholds)
        btn_thresholds_clear = QPushButton("Use Default")
        btn_thresholds_clear.clicked.connect(lambda: self.thresholds_edit.setText(""))
        thresh_layout.addWidget(self.thresholds_edit, 1)
        thresh_layout.addWidget(btn_thresholds)
        thresh_layout.addWidget(btn_thresholds_clear)
        outer.addWidget(thresh_box)

        # Output folder (optional override)
        out_box = QGroupBox("Output folder")
        out_layout = QHBoxLayout(out_box)
        self.custom_out_check = QCheckBox("Use custom output folder")
        self.custom_out_check.toggled.connect(self._toggle_custom_out)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setEnabled(False)
        self.out_dir_edit.setPlaceholderText("(default: auto-named under the data folder)")
        btn_out_dir = QPushButton("Browse...")
        btn_out_dir.clicked.connect(self._pick_out_dir)
        btn_out_dir.setEnabled(False)
        self._btn_out_dir = btn_out_dir
        out_layout.addWidget(self.custom_out_check)
        out_layout.addWidget(self.out_dir_edit, 1)
        out_layout.addWidget(btn_out_dir)
        outer.addWidget(out_box)

        # Advanced
        adv_box = QGroupBox("Advanced (usually fine at defaults)")
        adv_form = QFormLayout(adv_box)

        self.center_window_spin = QSpinBox()
        self.center_window_spin.setRange(1, 8)
        self.center_window_spin.setValue(2)
        adv_form.addRow("Center window (zones):", self.center_window_spin)

        self.obj_pct_spin = QDoubleSpinBox()
        self.obj_pct_spin.setRange(0.0, 100.0)
        self.obj_pct_spin.setValue(10.0)
        adv_form.addRow("Object confidence percentile:", self.obj_pct_spin)

        self.phantom_pct_spin = QDoubleSpinBox()
        self.phantom_pct_spin.setRange(0.0, 100.0)
        self.phantom_pct_spin.setValue(95.0)
        adv_form.addRow("Phantom confidence percentile:", self.phantom_pct_spin)

        self.tol_frac_spin = QDoubleSpinBox()
        self.tol_frac_spin.setRange(0.01, 1.0)
        self.tol_frac_spin.setSingleStep(0.01)
        self.tol_frac_spin.setValue(0.15)
        adv_form.addRow("Distance tolerance fraction:", self.tol_frac_spin)

        self.floor_spin = QSpinBox()
        self.floor_spin.setRange(0, tb.CONF_MAX)
        self.floor_spin.setValue(tb.DEFAULT_FLOOR_MIN_RAW)
        self.floor_spin.setToolTip("Lower clamp on the fitted threshold curve")
        adv_form.addRow("Curve floor (min raw):", self.floor_spin)

        self.wall_anchors_check = QCheckBox("Use Object 2 / wall recommendations as curve anchors")
        self.wall_anchors_check.setChecked(True)
        adv_form.addRow("", self.wall_anchors_check)

        outer.addWidget(adv_box)

        # Run controls
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.hide()
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.progress, 1)
        outer.addLayout(run_row)

        # Log panel
        outer.addWidget(QLabel("Log:"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        font = self.log_edit.font()
        font.setFamily("Consolas")
        self.log_edit.setFont(font)
        self.log_edit.setMinimumHeight(160)
        outer.addWidget(self.log_edit)

        # Results panel
        self.results_box = QGroupBox("Result")
        self.results_box.hide()
        results_layout = QVBoxLayout(self.results_box)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        results_layout.addWidget(self.result_label)
        results_buttons = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        self.btn_open_report = QPushButton("Open Report")
        self.btn_open_report.clicked.connect(self._open_report)
        results_buttons.addWidget(self.btn_open_folder)
        results_buttons.addWidget(self.btn_open_report)
        results_layout.addLayout(results_buttons)
        self.summary_image = QLabel()
        self.summary_image.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.summary_image)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        results_layout.addWidget(scroll)
        outer.addWidget(self.results_box)

        scroll_outer = QScrollArea()
        scroll_outer.setWidget(central)
        scroll_outer.setWidgetResizable(True)
        self.setCentralWidget(scroll_outer)

        self.statusBar().showMessage("Ready.")
        self.preset_combo.setCurrentIndex(0)
        self._apply_preset(self.preset_combo.currentText())

    # -- small layout helper --------------------------------------------
    def _row(self, *widgets):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            lay.addWidget(widget, 1 if isinstance(widget, QLineEdit) else 0)
        return w

    # -- preset handling ---------------------------------------------------
    def _apply_preset(self, name: str):
        preset = PRESETS.get(name)
        if not preset:
            return
        self.include_edit.setText(preset["include"])
        self.exclude_edit.setText(preset["exclude"])
        self.w_spin.setValue(preset["w_mm"])
        self.h_spin.setValue(preset["h_mm"])
        self.tol_frac_spin.setValue(preset["tol_frac"])
        if name != "Custom...":
            self.label_edit.setText(name.replace(" --", ","))

    # -- file/folder pickers -------------------------------------------
    def _pick_data_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        if path:
            self.data_dir_edit.setText(path)

    def _pick_thresholds(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select thresholds JSON", "", "JSON files (*.json)")
        if path:
            self.thresholds_edit.setText(path)

    def _toggle_custom_out(self, checked: bool):
        self.out_dir_edit.setEnabled(checked)
        self._btn_out_dir.setEnabled(checked)

    def _pick_out_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self.out_dir_edit.setText(path)

    # -- run ----------------------------------------------------------------
    def _on_run_clicked(self):
        data_dir = self.data_dir_edit.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, "Missing data folder", "Choose a valid data folder first.")
            return
        include = _split_tokens(self.include_edit.text())
        if not include:
            QMessageBox.warning(self, "Missing filter",
                                 "\"Filename must include\" needs at least one token "
                                 "(e.g. \"paper\") so the tool knows which files to load.")
            return
        exclude = _split_tokens(self.exclude_edit.text())

        kwargs = dict(
            data_dir=data_dir,
            object_label=self.label_edit.text().strip() or "Object",
            file_include=include,
            file_exclude=exclude,
            object_w_mm=self.w_spin.value(),
            object_h_mm=self.h_spin.value(),
            center_window=self.center_window_spin.value(),
            obj_percentile=self.obj_pct_spin.value(),
            phantom_percentile=self.phantom_pct_spin.value(),
            distance_tol_frac=self.tol_frac_spin.value(),
            floor_min_raw=self.floor_spin.value(),
            include_wall_anchors=self.wall_anchors_check.isChecked(),
            thresholds_json=self.thresholds_edit.text().strip() or None,
            out_dir=self.out_dir_edit.text().strip() if self.custom_out_check.isChecked() else None,
        )

        self.log_edit.clear()
        self.results_box.hide()
        self.run_button.setEnabled(False)
        self.progress.show()
        self.statusBar().showMessage("Running...")

        self.worker = BenchWorker(kwargs)
        self.worker.log.connect(self._append_log)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_failure)
        self.worker.start()

    def _append_log(self, text: str):
        self.log_edit.moveCursor(self.log_edit.textCursor().MoveOperation.End)
        self.log_edit.insertPlainText(text)
        self.log_edit.moveCursor(self.log_edit.textCursor().MoveOperation.End)

    def _run_finished_ui(self):
        self.run_button.setEnabled(True)
        self.progress.hide()

    def _on_success(self, result: dict):
        self._run_finished_ui()
        self.statusBar().showMessage("Done.")
        self.last_result = result
        n_points = len(result["points"])
        fits = "\n".join(
            f"  {mode}: " + (c["fit"].describe() if c["fit"] else f"no fit -- {c['error']}")
            for mode, c in result["curves"].items())
        self.result_label.setText(
            f"Analyzed {n_points} of {result['n_files']} distance points.\n"
            f"Fitted curves (candidate JSONs in output folder):\n{fits}\n"
            f"Output folder: {result['out_dir']}"
        )
        pix = QPixmap(result["summary_png"])
        if not pix.isNull():
            self.summary_image.setPixmap(pix.scaledToWidth(800, Qt.SmoothTransformation))
        self.results_box.show()

    def _on_failure(self, message: str):
        self._run_finished_ui()
        self.statusBar().showMessage("Failed -- see log.")
        QMessageBox.critical(self, "Run failed", message)

    # -- results panel actions -------------------------------------------
    def _open_output_folder(self):
        if self.last_result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_result["out_dir"]))

    def _open_report(self):
        if self.last_result:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_result["report_md"]))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
