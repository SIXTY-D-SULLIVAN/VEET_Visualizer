"""
VEET Chart - Direct VEET Log Visualization Tool (v1.14)

This script processes VEET CSV files (log.csv and Sensor_Data.csv) to generate interactive 
visualizations of device data. It works with files that have various naming conventions as 
long as they end with "log.csv" and "Sensor_Data.csv".

Key Features:
- Enhanced GUI interface with progress dialogs for data loading and plot generation
- Improved user experience with visual feedback during long operations
- Flexible file handling for logs with date prefixes or other naming variations
- Nine selectable plot types for comprehensive device data analysis:
  • Battery Voltage - with comprehensive device state markers
  • Battery Voltage Rate of Change Over Time - discharge rate analysis
  • TOF (Time-of-Flight) Average Distance measurements
  • ALS Lux - calculated luminance values with optional log scaling
  • ALS Basic Counts - gain & integration time normalized wavelength counts (<400nm, 400-700nm, >700nm)
  • PHO Basic Counts - gain & integration time normalized photosensor data (415-910nm)
  • IMU Translational Acceleration - 3-axis accelerometer data
  • System Temperature from IMU sensor
  • Sensor Sample Interval - timing analysis showing acquisition rates
- Interactive plots with zoom, pan, and coordinate inspection
- Ctrl+Click on plots to copy coordinates to clipboard
- Timezone adjustment with dropdown selection (GMT-12 to GMT+14)
- Automatic detection of device information from data files
- Advanced time range controls:
  • Hours to display - limit X-axis to specific time window
  • Hours offset from end - start display window before latest data point
- Battery analysis with comprehensive device state visualization:
  • Deep Sleep events, CPU speed changes (300MHz/12MHz modes)
  • Charge rate indicators (Fast/Slow/No charge)
  • VBUS connection status, USB connection state
  • Disk mount status, App Disk Ready state
  • Reboot event markers
- Gain and integration time normalization for ALS and PHO data with accurate calibration factors
- Logarithmic or linear scale options for ALS Lux, ALS Basic Counts, and PHO Basic Counts
- Export plots as PNG images for reports (individual or batch processing)
- Batch processing capability for multiple device folders
- Optimized legends with proper placement to avoid overlap with data
- Interactive legends for PHO Basic Counts plots - click to show/hide data series
- Draggable legends for better plot viewing
- Clear labeling and organization of chart elements

Plot Title and Information Sources:
- Plot titles are automatically derived from the selected folder name (via parseDeviceFolder function)
- If folder name cannot be determined, uses "Unknown_Device" with serial number if available
- System information box displays:
  • Data is extracted automatically from the first INF line found in Sensor_Data.csv
  • Firmware version (from 5th column of INF line in Sensor_Data.csv)
  • Device configuration data including sensor intervals (from last 6 values of INF line)

When running the script a GUI window will open to:
1. Select the device input folder location
2. Load data automatically from log.csv and Sensor_Data.csv files (with flexible naming)
   *Note: Data loading shows a progress dialog and can take several seconds for large datasets*
3. Choose which data types to plot using checkboxes
4. Configure logarithmic scaling for applicable plot types
5. Set visualization options (timezone, time range, display window)
6. Generate interactive plot windows with the selected data
   *Note: Plot generation shows a progress dialog and can take several seconds for large datasets or multiple plots*
7. Export plots individually or in batch mode

The script expects the following input files in the selected folder:
- Any file ending with "log.csv" (e.g., "2025-08-03 23-46 log.csv")
- Any file ending with "Sensor_Data.csv" (e.g., "2025-08-03 23-46 Sensor_Data.csv")

All user-configurable settings are in the settings dictionary below.
- Check filters settings if data does not plot
- Check applyALSgain flag if ALS data does not look gain normalized
- Adjust plot_options for changing line width, marker style, etc.
- Configure useLogScale settings for default scaling behavior

Interactive Plot Tips:
- Hover over matplotlib toolbar to see available hot keys
- Zoom while holding y or x key for axis-specific zooming
- Ctrl+Click anywhere on a plot to copy the coordinates to clipboard
- Legends are optimally positioned and draggable for better plot viewing
- Click on PHO Basic Counts legend items to toggle data series visibility
- Battery Voltage plot legend includes organized markers for various device states
- The window title is automatically copied to clipboard for easy file naming when saving
"""
import sys
import os
import io
import re
import csv
import json
import threading
from PySide6.QtCore import Qt, QCoreApplication, QThread, Signal, QTimer, QSettings, QDateTime
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QCheckBox, QFileDialog, QMainWindow, QComboBox, QMessageBox, QLineEdit, QProgressDialog, QDateTimeEdit, QSplashScreen, QDockWidget, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QScrollArea
from PySide6.QtGui import QClipboard, QPixmap, QColor

# numpy/pandas/pytz and matplotlib are imported lazily so the GUI can show a
# splash screen immediately at startup instead of blocking on library imports.
# matplotlib is deferred all the way to the first plot window.
np = pd = pytz = None
FigureCanvas = NavigationToolbar = Figure = mdates = MaxNLocator = None


def _ensure_core_modules():
    """Import numpy/pandas/pytz on first use (needed for data loading)."""
    global np, pd, pytz
    if pd is None:
        import numpy as _np
        import pandas as _pd
        import pytz as _pytz
        np = _np
        pd = _pd
        pytz = _pytz


def _ensure_matplotlib():
    """Import matplotlib pieces on first plot (not needed for data loading)."""
    global FigureCanvas, NavigationToolbar, Figure, mdates, MaxNLocator
    if FigureCanvas is None:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar
        from matplotlib.figure import Figure as _Figure
        import matplotlib.dates as _mdates
        from matplotlib.ticker import MaxNLocator as _MaxNLocator
        FigureCanvas = _FigureCanvas
        NavigationToolbar = _NavigationToolbar
        Figure = _Figure
        mdates = _mdates
        MaxNLocator = _MaxNLocator

def _import_tof_bands():
    """Import the veet_tof_bands module (validity gating + dioptric binning).

    Looked up next to this script; returns None if unavailable so TOF dioptric
    statistics degrade gracefully instead of breaking the load path.
    """
    try:
        import veet_tof_bands
        return veet_tof_bands
    except ImportError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        try:
            import veet_tof_bands
            return veet_tof_bands
        except ImportError:
            return None


def find_file_with_suffix(directory, suffix):
    """Find a file in the specified directory that ends with the given suffix (case-insensitive)."""
    if not os.path.exists(directory):
        return None
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(suffix.lower()):
            return os.path.join(directory, filename)
    
    # If no matching file is found, return None
    return None

class PlotWindow(QMainWindow):
    def __init__(self, selected_plots, plot_labels, settings, tz_offset, data, device_folder,
                 ALS_wave_col_names, PHO_wave_col_names, ALSgainRatios, PHOgainRatios, phoToScale, phoScales,
                 firmware_info=None, additional_info=None, inf_line=None, serial_number=None,
                 log_events=None, wear_windows=None):
        _ensure_core_modules()
        _ensure_matplotlib()
        super().__init__()
        self.selected_plots = selected_plots
        self.plot_labels = plot_labels
        self.settings = settings
        self.tz_offset = tz_offset
        self.data = data
        self.device_folder = device_folder
        self.ALS_wave_col_names = ALS_wave_col_names
        self.PHO_wave_col_names = PHO_wave_col_names
        self.ALSgainRatios = ALSgainRatios
        self.PHOgainRatios = PHOgainRatios
        self.phoToScale = phoToScale
        self.phoScales = phoScales
        self.firmware_info = firmware_info
        self.additional_info = additional_info
        self.inf_line = inf_line
        self.serial_number = serial_number
        self.log_events = log_events or {}
        # Precomputed per-window wear classification (see MainWindow.compute_wear_windows);
        # None when unavailable (e.g. insufficient IMU data)
        self.wear_windows = wear_windows
        # Full-resolution data behind display-decimated lines: ax -> [(line, xdat, xnum, y)]
        self._dyn_lines = {}
        self._dyn_axes = set()

        self.canvas = FigureCanvas(Figure(figsize=(10, 15)))
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Setup click-to-copy functionality
        self.setup_click_events()

        self.plot_data()
    
    def setup_click_events(self):
        """Set up event handling for mouse clicks on the plot"""
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Create a status label to show when values are copied
        self.statusBar().showMessage("Ctrl+Click on plot points to copy values to clipboard")
    
    def on_plot_click(self, event):
        """Handle mouse clicks on the plot by copying the x,y values to clipboard"""
        # Only trigger when Ctrl key is pressed during click (event.key contains modifier keys)
        if event.xdata is not None and event.ydata is not None and event.key == 'control':
            # Format the datetime for the x-value (convert from matplotlib date to datetime)
            x_datetime = mdates.num2date(event.xdata)
            x_datetime = x_datetime.astimezone(self.tz_offset)
            x_str = x_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format the y-value with 3 decimal places
            y_str = f"{event.ydata:.3f}"
            
            # Create the text to copy
            copy_text = f"Time: {x_str}, Value: {y_str}"
            
            # Copy to clipboard
            QApplication.clipboard().setText(copy_text)
            
            # Update status bar to provide feedback
            self.statusBar().showMessage(f"Copied to clipboard: {copy_text}", 3000)  # Show for 3 seconds
            
            print(f"Copied to clipboard: {copy_text}")
        elif event.key is not None and event.xdata is not None and event.ydata is not None:
            # If another key was pressed during click, show a hint message
            self.statusBar().showMessage(f"Use Ctrl+Click to copy coordinates (you pressed {event.key})", 2000)

    def get_time_span_hours(self, ax):
        """Get the current visible time span in hours"""
        xlim = ax.get_xlim()
        span_days = xlim[1] - xlim[0]  # matplotlib dates are in days
        return span_days * 24
    
    def setup_smart_time_axis(self, ax):
        """Setup intelligent time axis formatting with boundary-aware positioning"""
        # Use matplotlib's smart auto locator for good density
        auto_locator = mdates.AutoDateLocator(tz=self.tz_offset)
        
        # But customize the intervals to snap to meaningful boundaries
        # This tells AutoDateLocator what intervals to prefer at each scale
        auto_locator.intervald = {
            mdates.YEARLY: [1, 2, 5, 10, 20, 50, 100],
            mdates.MONTHLY: [1, 2, 3, 4, 6, 12],
            mdates.DAILY: [1, 2, 3, 7, 14, 21],
            mdates.HOURLY: [1, 2, 3, 4, 6, 8, 12],
            mdates.MINUTELY: [1, 5, 10, 15, 20, 30],
            mdates.SECONDLY: [1, 5, 10, 15, 20, 30],
            mdates.MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500]
        }
        
        # Use the concise formatter which automatically adapts format based on zoom level
        formatter = mdates.ConciseDateFormatter(auto_locator, tz=self.tz_offset)
        
        # Customize the format strings for better boundary indication
        formatter.formats = [
            '%Y',           # Year
            '%b',           # Month
            '%d',           # Day  
            '%H:%M',        # Hour:Minute
            '%H:%M',        # Minute
            '%S.%f',        # Second.Microsecond
        ]
        
        # Customize zero formats (what shows when a unit is at its boundary)
        formatter.zero_formats = [
            '',             # Year (always show)
            '%Y',           # Month (show year when at Jan)
            '%b',           # Day (show month when at 1st)
            '%b %d',        # Hour (show month-day when at midnight)
            '%H:%M',        # Minute (show hour:min when at top of hour)
            '%H:%M',        # Second (keep showing minutes)
        ]
        
        # Apply the locator and formatter
        ax.xaxis.set_major_locator(auto_locator)
        ax.xaxis.set_major_formatter(formatter)
        
        # Add minor locator for better granularity indication
        minor_locator = mdates.AutoDateLocator(tz=self.tz_offset)
        # Make minor ticks more frequent than major
        minor_locator.intervald = {
            mdates.YEARLY: [1, 2, 4],
            mdates.MONTHLY: [1, 2, 3],
            mdates.DAILY: [1],
            mdates.HOURLY: [1, 2, 3, 4, 6],
            mdates.MINUTELY: [1, 5, 10, 15],
            mdates.SECONDLY: [1, 5, 10, 15],
            mdates.MICROSECONDLY: [1, 2, 5, 10, 20, 50]
        }
        ax.xaxis.set_minor_locator(minor_locator)
        
        # Enable minor ticks
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=6)

    def format_coord(self, x, y, tz_offset):
        x_datetime = mdates.num2date(x)
        x_datetime = x_datetime.astimezone(tz_offset)
        return f"x={x_datetime:%Y-%m-%d %H:%M:%S}, y={y:.3f}\n"
    
    def make_legend_interactive(self, ax, legend):
        """Make legend interactive - click to toggle line visibility"""
        # Map legend lines and text to actual plot lines
        line_map = {}  # Maps plot lines to their legend line and text
        
        for legend_line, legend_text, orig_line in zip(legend.get_lines(), legend.get_texts(), ax.get_lines()):
            line_map[orig_line] = {'leg_line': legend_line, 'leg_text': legend_text}
        
        def on_click(event):
            # Only process if click is within the legend
            if legend.get_window_extent().contains(event.x, event.y):
                # Check each legend element
                for orig_line, leg_items in line_map.items():
                    leg_line = leg_items['leg_line']
                    leg_text = leg_items['leg_text']
                    
                    # Check if click is on this legend line or text
                    if (leg_line.contains(event)[0] or leg_text.contains(event)[0]):
                        # Toggle visibility
                        visible = not orig_line.get_visible()
                        orig_line.set_visible(visible)
                        
                        # Update legend appearance
                        alpha = 1.0 if visible else 0.2
                        leg_line.set_alpha(alpha)
                        leg_text.set_alpha(alpha)
                        
                        # Redraw
                        self.canvas.draw_idle()
                        break
        
        # Connect the button press event (works even when toolbar is active)
        self.canvas.mpl_connect('button_press_event', on_click)

    # Maximum points drawn per line for the current view; zooming re-densifies
    DISPLAY_MAX_POINTS = 5000

    def _decimate_indices(self, y, max_points):
        """Indices of ~max_points samples of y, keeping each bucket's min and max
        so peaks survive decimation. All-NaN buckets keep one point so gaps stay
        visible; the first and last points are always kept."""
        n = len(y)
        if n <= max_points:
            return np.arange(n)
        n_buckets = max(1, max_points // 2)
        bucket = int(np.ceil(n / n_buckets))
        n_b = int(np.ceil(n / bucket))
        pad = n_b * bucket - n
        ypad = np.concatenate([y, np.full(pad, np.nan)]).reshape(n_b, bucket)
        finite = np.isfinite(ypad)
        has_finite = finite.any(axis=1)
        base = np.arange(n_b) * bucket
        imin = np.where(finite, ypad, np.inf).argmin(axis=1)
        imax = np.where(finite, ypad, -np.inf).argmax(axis=1)
        keep = np.concatenate([
            (base + imin)[has_finite],
            (base + imax)[has_finite],
            base[~has_finite],
            [0, n - 1],
        ])
        return np.unique(keep)

    def plot_downsampled(self, ax, x, y, **kwargs):
        """ax.plot wrapper that keeps the full-resolution series but only draws
        ~DISPLAY_MAX_POINTS for the visible x-range, re-densifying whenever the
        axis limits change (zoom/pan)."""
        x = pd.to_datetime(pd.Series(x))
        if x.dt.tz is not None:
            # Store as UTC-naive; the axis formatter applies the display timezone
            x = x.dt.tz_convert('UTC').dt.tz_localize(None)
        y = pd.to_numeric(pd.Series(y), errors='coerce').to_numpy(dtype=float)

        valid_x = x.notna().to_numpy()
        xdat = x.to_numpy()[valid_x]
        y = y[valid_x]
        order = np.argsort(xdat, kind='stable')
        xdat = xdat[order]
        y = y[order]
        xnum = mdates.date2num(xdat)

        keep = self._decimate_indices(y, self.DISPLAY_MAX_POINTS)
        line, = ax.plot(xdat[keep], y[keep], **kwargs)

        self._dyn_lines.setdefault(ax, []).append((line, xdat, xnum, y))
        if ax not in self._dyn_axes:
            self._dyn_axes.add(ax)
            ax.callbacks.connect('xlim_changed', self._refresh_dynamic_lines)
        return line

    # Local nighttime window for day/night shading (hours in display timezone)
    NIGHT_START_HOUR = 20
    NIGHT_END_HOUR = 6

    def add_night_shading(self, axs):
        """Draw light gray bands over local nighttime (20:00-06:00) on every subplot,
        spanning the full extent of the loaded data."""
        try:
            tmin = tmax = None
            for df in self.data.values():
                if df.empty or 'time_stamp' not in df.columns:
                    continue
                if not pd.api.types.is_datetime64_any_dtype(df['time_stamp']):
                    continue
                col = df['time_stamp'].dropna()
                if col.empty:
                    continue
                lo, hi = col.min(), col.max()
                if tmin is None or lo < tmin:
                    tmin = lo
                if tmax is None or hi > tmax:
                    tmax = hi
            if tmin is None:
                return

            tmin_local = tmin.tz_convert(self.tz_offset)
            tmax_local = tmax.tz_convert(self.tz_offset)
            n_days = (tmax_local - tmin_local).days + 2
            if n_days > 370:
                message = f"Night shading skipped: loaded time range spans {n_days} days (> 1 year)"
                print(message)
                self.statusBar().showMessage(message, 8000)
                return

            first_day = tmin_local.normalize() - pd.Timedelta(days=1)
            bands = []
            for day_index in range(n_days + 1):
                day = first_day + pd.Timedelta(days=day_index)
                night_start = day + pd.Timedelta(hours=self.NIGHT_START_HOUR)
                night_end = day + pd.Timedelta(hours=24 + self.NIGHT_END_HOUR)
                if night_end < tmin_local or night_start > tmax_local:
                    continue
                bands.append((night_start, night_end))

            for ax in axs:
                for band_start, band_end in bands:
                    ax.axvspan(band_start, band_end, facecolor='gray', alpha=0.12,
                               zorder=0, linewidth=0)
        except Exception as e:
            print(f"Error adding night shading: {e}")

    def _refresh_dynamic_lines(self, ax):
        """xlim_changed callback: re-decimate every dynamic line for the new view."""
        lo, hi = ax.get_xlim()
        for line, xdat, xnum, y in self._dyn_lines.get(ax, []):
            i0 = max(int(np.searchsorted(xnum, lo, side='left')) - 1, 0)
            i1 = min(int(np.searchsorted(xnum, hi, side='right')) + 1, len(xnum))
            keep = i0 + self._decimate_indices(y[i0:i1], self.DISPLAY_MAX_POINTS)
            line.set_data(xdat[keep], y[keep])
        self.canvas.draw_idle()

    def plot_data(self):
        try:
            num_plots = len(self.selected_plots)
            if num_plots == 0:
                return

            fig = self.canvas.figure
            fig.clear()  # Clear any previous plots
            axs = fig.subplots(num_plots, 1, sharex=True)

            if num_plots == 1:
                axs = [axs]
            else:
                axs = axs.flatten()

            for i, plot_name in enumerate(self.selected_plots):
                try:
                    ax = axs[i]
                    ax.grid(True, axis='y', linewidth=0.5, linestyle=':')
                    ax.grid(True, axis='x', linewidth=0.5, linestyle=':')
                    ax.set_ylabel(self.plot_labels.get(plot_name, ""))
                    ax.set_xlabel("Time Stamp")
                    
                    # Use smart time axis formatting
                    self.setup_smart_time_axis(ax)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto'))

                    # Handle each plot type with proper error trapping
                    if plot_name == 'Battery Voltage':
                        if 'BAT' in self.data and not self.data['BAT'].empty:
                            self.plot_bat(self.data['BAT'], self.data['REBOOT'], self.data['LOG'], ax, self.settings['plot_options']['BAT'])
                            leg1 = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
                            leg1.set_draggable(state=True)
                        else:
                            ax.text(0.5, 0.5, "No BAT data available", 
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=ax.transAxes)
                    if plot_name == 'Battery Voltage Change Over Time':
                        self.plot_bat_rate(self.data['BAT'], ax, self.settings['plot_options']['BAT Rate'])
                    if plot_name == 'TOF Average Distance':
                        self.plot_tof(self.data['TOF'], ax, self.settings['plot_options']['TOF'])
                    if plot_name == 'ALS Lux':
                        self.plot_lux(self.data['ALS'], ax, self.settings['plot_options']['ALS_lux'])
                    if plot_name == 'ALS Basic Counts':
                        # Pass the actual ALSgainRatios dictionary instead of empty list
                        self.plot_alsW(self.data['ALS'], ax, self.settings['plot_options']['ALS_wave'], self.ALSgainRatios)
                    if plot_name == 'PHO Basic Counts':
                        # Pass the actual PHOgainRatios dictionary instead of empty list
                        self.plot_pho(self.data['PHO'], ax, self.settings['plot_options']['PHO'], self.PHOgainRatios)
                        leg1 = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
                        leg1.set_draggable(state=True)
                        self.make_legend_interactive(ax, leg1)
                    if plot_name == 'IMU Translational Acceleration':
                        self.plot_imu(self.data['IMU'], ax, self.settings['plot_options']['IMU'], [])
                        leg1 = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
                        leg1.set_draggable(state=True)
                    if plot_name == 'System Temperature (IMU)':
                        self.plot_imu_temp(self.data['IMU'], ax, self.settings['plot_options']['IMU_temp'])
                        leg1 = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
                        leg1.set_draggable(state=True)
                    if plot_name == 'Sensor Sample Interval':
                        self.plot_sampling_intervals(self.data, ax, self.settings['plot_options']['Sample_intervals'])
                        leg1 = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
                        leg1.set_draggable(state=True)
            
                    # Set the format_coord for each ax
                    ax.format_coord = lambda x, y, fmt=self.format_coord, offset=self.tz_offset: fmt(x, y, offset)
                
                except Exception as e:
                    print(f"Error plotting {plot_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Add text to the plot indicating error
                    ax.text(0.5, 0.5, f"Error plotting {plot_name}",
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, color='red')
            
            # Shade local nighttime bands behind the data on every subplot
            self.add_night_shading(axs)

            # Set X-axis limits: an explicit date range overrides hours/offset
            if self.settings.get('use_date_range') and self.settings.get('range_start') and self.settings.get('range_end'):
                try:
                    start_time = pd.Timestamp(self.settings['range_start'], tz=self.tz_offset)
                    end_time = pd.Timestamp(self.settings['range_end'], tz=self.tz_offset)
                    if end_time > start_time:
                        for ax in axs:
                            ax.set_xlim(start_time, end_time)
                    else:
                        print("Warning: date range end is not after start; ignoring")
                except Exception as e:
                    print(f"Warning: invalid date range: {e}")
            elif self.settings.get('hours_to_display') and float(self.settings['hours_to_display']) > 0:
                # Find the latest timestamp across all datasets
                max_time = None
                for key in self.data:
                    if not self.data[key].empty and 'time_stamp' in self.data[key].columns:
                        dataset_max = self.data[key]['time_stamp'].max()
                        if max_time is None or dataset_max > max_time:
                            max_time = dataset_max
                
                if max_time is not None:
                    # Apply offset from the latest data point if specified
                    hours_offset = 0
                    if self.settings.get('hours_offset') and self.settings['hours_offset'].strip():
                        try:
                            hours_offset = float(self.settings['hours_offset'])
                        except ValueError:
                            print(f"Warning: Invalid offset value: {self.settings['hours_offset']}")
                    
                    # Calculate the end time as the latest timestamp minus offset
                    end_time = max_time - pd.Timedelta(hours=hours_offset)
                    
                    # Calculate the start time based on hours to display
                    hours_to_display = float(self.settings['hours_to_display'])
                    start_time = end_time - pd.Timedelta(hours=hours_to_display)
                    
                    # Set the x-axis limits for all subplots
                    for ax in axs:
                        ax.set_xlim(start_time, end_time)
            
            # Set the title and draw the plot - Fix potential issue with title
            try:
                # Check if device_folder is a valid string
                if self.device_folder is None or self.device_folder == '':
                    plot_title = "Unknown Device"
                else:
                    plot_title = str(self.device_folder)
                
                # Add the title at the center with enough space for info on the right
                fig.suptitle(plot_title, fontsize=16, x=0.4)
                
                # Add firmware and additional info to the right of the title if available
                if self.firmware_info or self.additional_info or self.inf_line or self.serial_number:
                    # Handle INFO line label with custom styling
                    info_y_start = 0.98
                    current_y = info_y_start
                    
                    # Add "INFO line" label in small grey text (left justified)
                    if self.inf_line:
                        fig.text(0.70, current_y, "INFO line", 
                                fontsize=8, color='gray', ha='left', va='top',
                                alpha=0.7, style='italic')
                        current_y -= 0.035  # Better spacing from the box
                    
                    # Collect all info for display inside the box
                    info_text = []
                    
                    # Combine serial number and firmware info on the same line
                    first_line_parts = []
                    if self.serial_number:
                        first_line_parts.append(self.serial_number)
                    if self.firmware_info:
                        first_line_parts.append(self.firmware_info)
                    
                    if first_line_parts:
                        info_text.append(" - ".join(first_line_parts))
                    
                    if self.additional_info:
                        info_text.append(self.additional_info)
                    
                    # Add info text in box if any exists
                    if info_text:
                        combined_info = '\n'.join(info_text)
                        
                        # Add text in box below the INFO line label
                        fig.text(0.70, current_y, combined_info, 
                                fontsize=10, ha='left', va='top',
                                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
                
                # Add instructional text at the top left corner
                instruction_text = [
                    "Mouse over plot region to view coordinates (window top right)",
                    "Ctrl+Click to copy coordinates to clipboard",
                    "Click legend items to hide/show data sets (PHO & ALS count plots only)",
                    "Hold 'x' key while Zoom/Panning to only adjust time axis", 
                    "Hold 'y' key while Zoom/Panning to only adjust magnitude axis"
                ]
                
                # Add each line of instruction text
                for i, line in enumerate(instruction_text):
                    fig.text(0.01, 0.98 - (i * 0.018), line,
                             fontsize=8, color='gray', ha='left', va='top',
                             alpha=0.7, style='italic')
                
                self.canvas.draw()
                
                # Copy just the title to clipboard for easy pasting when saving
                # Clean up plot_title by removing characters that might cause issues in filenames
                clipboard_text = re.sub(r'[\\/*?:"<>|]', '_', plot_title)
                
                # Set to clipboard
                QApplication.clipboard().setText(clipboard_text)
                print(f"Title copied to clipboard: {clipboard_text}")
                
            except Exception as e:
                print(f"Error setting plot title: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try a simpler title if there was an error
                try:
                    fig.suptitle("Data Plot", fontsize=16)
                    self.canvas.draw()
                except Exception as e2:
                    print(f"Critical error with plot title: {str(e2)}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"Fatal error in plot_data: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_bat(self, bat_data, reboot_data, log_data, ax, plot_options):
        try:
            if not bat_data.empty and 'time_stamp' in bat_data.columns and 'Voltage' in bat_data.columns:
                # Ensure Voltage is numeric type
                bat_data['Voltage'] = pd.to_numeric(bat_data['Voltage'], errors='coerce')
                
                # Plot the battery voltage
                self.plot_downsampled(ax, bat_data['time_stamp'], bat_data['Voltage'],
                        lw=plot_options['lw'], marker=plot_options['marker'],
                        markersize=plot_options['marker_size'], label='Battery Voltage')
                
                # Add a line to the legend for reboot events if there are any reboot events
                if not reboot_data.empty and 'time_stamp' in reboot_data.columns:
                    # Plot vertical lines for reboot events
                    for idx, row in reboot_data.iterrows():
                        ax.axvline(x=row['time_stamp'], color='r', linestyle='--', alpha=0.2, linewidth=0.5)
                    
                    # Add entry to legend for reboot events
                    ax.plot([], [], color='r', linestyle='--', alpha=0.6, linewidth=1.0, label='Reboot Event')
                
                # Set Y-axis limits to 3000-4250 mV range
                ax.set_ylim(3000, 4350)
                
                # Add markers for Deep Sleep events and STATE events pre-parsed
                # from log.csv during load_data (no file re-read at plot time)
                try:
                    deep_sleep_events = self.log_events.get('deep_sleep', [])
                    cpu_speed_events = self.log_events.get('cpu_speed', {})
                    charge_rate_events = self.log_events.get('charge_rate', {})
                    vbus_events = self.log_events.get('vbus', {})
                    disk_mounted_events = self.log_events.get('disk_mounted', {})
                    usb_connected_events = self.log_events.get('usb_connected', {})
                    app_disk_ready_events = self.log_events.get('app_disk_ready', {})

                    # Plot Deep Sleep events
                    if deep_sleep_events:
                        # Convert to pandas datetime with timezone
                        deep_sleep_ts = pd.to_datetime(deep_sleep_events, unit='s')
                        deep_sleep_ts = deep_sleep_ts.tz_localize('UTC').tz_convert(self.tz_offset)
                        
                        print(f"Adding {len(deep_sleep_events)} Deep Sleep markers to plot")
                        
                        # Position the Deep Sleep markers at Y=3500
                        ax.scatter(deep_sleep_ts, 
                                [3500] * len(deep_sleep_ts),
                                marker='v',        # Inverted triangle
                                s=40,              # Marker size
                                color='purple',    # Purple color
                                alpha=1.0,         # Fully opaque
                                edgecolor='black',  # Black border
                                linewidth=0.3,
                                label='Deep Sleep')  # Ensure label is added for legend

                    # Plot CPU Speed events
                    for speed, timestamps in cpu_speed_events.items():
                        if timestamps:
                            # Convert to pandas datetime
                            event_ts = pd.to_datetime(timestamps, unit='s')
                            event_ts = event_ts.tz_localize('UTC').tz_convert(self.tz_offset)
                            
                            # Different markers/colors for different speeds
                            if speed == '300MHz':
                                marker = 'D'  # Diamond
                                color = 'deepskyblue'
                                display_label = '300Mhz'
                            elif speed == '12MHz':
                                marker = 'o'  # Circle down centered at base
                                color = 'orchid'
                                display_label = '12Mhz'
                            else:  # 12MHz_ExternPower
                                marker = 'o'  # Circle down centered at base
                                color = 'deepskyblue'
                                display_label = '12Mhz XPwr'
                            
                            print(f"Adding {len(timestamps)} CPU {speed} markers to plot")
                            
                            # Position at Y=4050
                            ax.scatter(event_ts,
                                    [4200] * len(event_ts),
                                    marker=marker,
                                    s=40,
                                    color=color,
                                    alpha=0.8,
                                    edgecolor='black',
                                    linewidth=0.3,
                                    label=f'{display_label}')  # Removed "CPU " prefix

                    # Plot Charge Rate events
                    for rate, timestamps in charge_rate_events.items():
                        if timestamps:
                            event_ts = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert(self.tz_offset)
                            if rate == 'Fast':
                                marker, color = '^', 'green'
                                label = 'Fast Chg'
                            elif rate == 'Slow':
                                marker, color = 'v', 'lightgreen'
                                label = 'Slow Chg'
                            else:  # No Charge
                                marker, color = 's', 'gray'
                                label = 'No Chg'
                            ax.scatter(event_ts, [3300] * len(event_ts), marker=marker, s=40, color=color,
                                       alpha=0.8, edgecolor='black', linewidth=0.3, label=label)

                    # Plot Disk Mounted events
                    for status, timestamps in disk_mounted_events.items():
                        if timestamps:
                            event_ts = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert(self.tz_offset)
                            if status == '1':
                                marker, color = 'o', 'orange'
                                ax.scatter(event_ts, [4000] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='Disk Mnt')
                            else:  # status == '0'
                                # Use filled marker with cross pattern instead of 'x'
                                marker, color = 'X', 'orange'  # Capital 'X' is a filled marker
                                ax.scatter(event_ts, [4000] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='Disk Unmnt')

                    # Plot USB Connected events
                    for status, timestamps in usb_connected_events.items():
                        if timestamps:
                            event_ts = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert(self.tz_offset)
                            if status == '1':
                                marker, color = 'o', 'blue'
                                ax.scatter(event_ts, [3600] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='USB Conn')
                            else:  # status == '0'
                                # Use filled marker with cross pattern instead of 'x'
                                marker, color = 'X', 'blue'  # Capital 'X' is a filled marker
                                ax.scatter(event_ts, [3600] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='USB Disconn')

                    # Plot App Disk Ready events
                    for status, timestamps in app_disk_ready_events.items():
                        if timestamps:
                            event_ts = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert(self.tz_offset)
                            if status == '1':
                                marker, color = 'o', 'gold'
                                ax.scatter(event_ts, [4100] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='App Disk+')
                            else:  # status == '0'
                                # Use filled marker with cross pattern instead of 'x'
                                marker, color = 'X', 'gold'  # Capital 'X' is a filled marker
                                ax.scatter(event_ts, [4100] * len(event_ts), marker=marker, s=40, color=color,
                                           alpha=0.8, edgecolor='black', linewidth=0.3, label='App Disk-')
                    
                    # Plot VBUS events
                    for status, timestamps in vbus_events.items():
                        if timestamps:
                            event_ts = pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert(self.tz_offset)
                            marker, color = ('o', 'red') if status == '1' else ('X', 'red')
                            label = 'VBUS+' if status == '1' else 'VBUS-'
                            ax.scatter(event_ts, [3500] * len(event_ts), marker=marker, s=40, color=color,
                                      alpha=0.8, edgecolor='black', linewidth=0.3, label=label)

                    # Ensure the legend is displayed
                    # Instead of letting matplotlib automatically order items, collect and order them manually
                    handles, labels = ax.get_legend_handles_labels()
                    
                    # Define the desired order categories
                    order_dict = {
                        'Battery Voltage': 0,
                        'Reboot Event': 1,
                        'Deep Sleep': 2,
                        'App Disk+': 3, 'App Disk-': 4,
                        'Disk Mnt': 5, 'Disk Unmnt': 6,
                        'USB Conn': 7, 'USB Disconn': 8,
                        'VBUS+': 9, 'VBUS-': 10,
                        'Fast Chg': 11, 'Slow Chg': 12, 'No Chg': 13,
                        '300Mhz': 14, '12Mhz': 15, '12Mhz XPwr': 16
                    }
                    
                    # Sort handles and labels by the order dictionary
                    if handles:
                        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: order_dict.get(pair[1], 999))
                        handles, labels = zip(*sorted_pairs)
                    
                    ax.legend(handles, labels, bbox_to_anchor=(1.35, 1.05), 
                             loc='upper right', fontsize='x-small').set_draggable(True)
                
                except Exception as e:
                    print(f"Error adding event markers: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Continue with the plot even if deep sleep markers fail
        
        except Exception as e:
            print(f"Error in plot_bat method: {str(e)}")
            import traceback
            traceback.print_exc()

    def plot_bat_rate(self, bat_data, ax, plot_options):
        if not bat_data.empty and 'time_stamp' in bat_data.columns and 'Voltage' in bat_data.columns:
            # Create a copy of the dataframe to avoid modifying the original
            bat_data = bat_data.copy()
            
            # Ensure Voltage is numeric type and sort by timestamp
            bat_data['Voltage'] = pd.to_numeric(bat_data['Voltage'], errors='coerce')
            bat_data = bat_data.sort_values('time_stamp')
            
            # Calculate voltage difference and time difference
            bat_data['Voltage_diff'] = bat_data['Voltage'].diff()
            bat_data['time_diff'] = bat_data['time_stamp'].diff().dt.total_seconds() / 60
            # Calculate the rate of change in mV per minute
            bat_data['rate'] = bat_data['Voltage_diff'] / bat_data['time_diff']
            
            # Filter out extreme values (infinities and NaNs)
            bat_data = bat_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['rate'])
            
            # Further filter to reasonable range based on expected battery behavior
            bat_data = bat_data[bat_data['rate'].between(-100, 100)]
            
            # Apply a rolling average of 11 points to smooth the data
            window_size = 11
            bat_data['rate_rolling_avg'] = bat_data['rate'].rolling(
                window=window_size, min_periods=1, center=True).mean()
            
            # Plot the rolling average rate of change
            self.plot_downsampled(ax, bat_data['time_stamp'], bat_data['rate_rolling_avg'],
                    lw=plot_options['lw'], marker=plot_options['marker'],
                    markersize=plot_options['marker_size'],
                    label=f'{window_size}-point Rolling Average (Centered)')

            # Plot the original rate with improved prominence while maintaining style
            self.plot_downsampled(ax, bat_data['time_stamp'], bat_data['rate'],
                    linestyle='solid', linewidth=0.7, alpha=0.7,
                    color='grey', label='Raw Rate')
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Set y-axis limits to show important variations
            y_min = max(bat_data['rate_rolling_avg'].min() * 1.1 if not bat_data.empty else -50, -50)
            y_max = min(bat_data['rate_rolling_avg'].max() * 1.1 if not bat_data.empty else 50, 50)
            ax.set_ylim(y_min, y_max)
            
            # Add a legend
            leg = ax.legend(loc='best')
            leg.set_draggable(state=True)
    
    def plot_tof(self, tof_data, ax, plot_options):
        if not tof_data.empty and 'time_stamp' in tof_data.columns:
            # Calculate average distance across all sensors
            dist_cols = [col for col in tof_data.columns if 'Dist' in col]
            conf_cols = [col for col in tof_data.columns if 'Conf' in col]
            
            if dist_cols and conf_cols:
                # Convert distance and confidence columns to numeric
                for col in dist_cols + conf_cols:
                    tof_data[col] = pd.to_numeric(tof_data[col], errors='coerce')
                
                # Filter distances based on confidence threshold - Create a copy to avoid fragmentation
                tof_data_filtered = tof_data.copy()
                conf_threshold = self.settings.get('conf_thresh', 0)

                for i, conf_col in enumerate(conf_cols):
                    if i < len(dist_cols):
                        dist_col = dist_cols[i]
                        # Set distance to NaN where confidence is below threshold
                        mask = tof_data[conf_col] < conf_threshold
                        tof_data_filtered.loc[mask, dist_col] = np.nan

                # Sub-floor readings (including 0) are a no-return artifact, not
                # a real near-field measurement -- reinterpret as far/optical
                # infinity (matching veet_tof_bands.NO_RETURN_FLOOR_MM) instead
                # of letting them drag the average toward "very close". Checked
                # against the raw distance (unconditional, like valid_mask's
                # no-return handling) so it isn't masked out by the confidence
                # filter above -- a sub-floor reading is a deterministic sensor
                # artifact regardless of its accompanying confidence byte.
                tof_bands = _import_tof_bands()
                no_return_floor = tof_bands.NO_RETURN_FLOOR_MM if tof_bands else 10.0
                no_return_sentinel = tof_bands.NO_RETURN_SENTINEL_MM if tof_bands else 2000.0
                for dist_col in dist_cols:
                    no_return = tof_data[dist_col] < no_return_floor
                    tof_data_filtered.loc[no_return, dist_col] = no_return_sentinel

                # Calculate average distance (ignoring NaNs) - Pre-calculate this before adding to dataframe
                avg_dist = tof_data_filtered[dist_cols].mean(axis=1)
                
                # Create a new dataframe with just timestamp and avg_dist to avoid fragmentation
                plot_df = pd.DataFrame({
                    'time_stamp': tof_data_filtered['time_stamp'],
                    'avg_dist': avg_dist
                })
                
                # Plot average distance
                self.plot_downsampled(ax, plot_df['time_stamp'], plot_df['avg_dist'],
                        lw=plot_options['lw'], marker=plot_options['marker'],
                        markersize=plot_options['marker_size'], label='Avg Distance')

    def plot_lux(self, als_data, ax, plot_options):
        if not als_data.empty and 'time_stamp' in als_data.columns and 'lux' in als_data.columns:
            als_data['lux'] = pd.to_numeric(als_data['lux'], errors='coerce')
            
            self.plot_downsampled(ax, als_data['time_stamp'], als_data['lux'],
                    lw=plot_options['lw'], marker=plot_options['marker'],
                    markersize=plot_options['marker_size'], label='Lux')
            
            # Set y-axis scale based on user preference
            if self.settings['useLogScale']['ALS_lux'] and not als_data[als_data['lux'] > 0].empty:
                # Set y-axis to logarithmic scale
                ax.set_yscale('log')
                # For log scale, we need a small positive number instead of 0
                min_positive = als_data[als_data['lux'] > 0]['lux'].min()
                ax.set_ylim(min_positive * 0.8, ax.get_ylim()[1])
            else:
                # Set y-axis to linear scale
                ax.set_yscale('linear')
                # For linear scale, include 0 in the range
                if not als_data.empty:
                    y_min = max(0, als_data['lux'].min() * 0.9 if als_data['lux'].min() < 0 else 0)
                    ax.set_ylim(y_min, ax.get_ylim()[1])
            # No min/max points or annotations
    def compute_basic_counts(self, values, gains, int_times, gain_ratios):
        """Vectorized basic-counts conversion: value / gain_ratio / int_time.

        Rows with a gain not present in gain_ratios use the closest available
        gain (an exact match is distance 0, so one lookup covers both cases).
        Rows with missing gain/value/int_time or int_time <= 0 become NaN.
        """
        values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
        gains = pd.to_numeric(gains, errors='coerce').to_numpy(dtype=float)
        int_times = pd.to_numeric(int_times, errors='coerce').to_numpy(dtype=float)

        gain_keys = np.array(list(gain_ratios.keys()), dtype=float)
        gain_vals = np.array(list(gain_ratios.values()), dtype=float)

        valid = ~np.isnan(gains) & ~np.isnan(values) & ~np.isnan(int_times) & (int_times > 0)
        # Substitute a real gain for invalid rows so the lookup stays NaN-free
        safe_gains = np.where(valid, gains, gain_keys[0])
        closest_idx = np.abs(safe_gains[:, None] - gain_keys[None, :]).argmin(axis=1)
        ratios = gain_vals[closest_idx]

        with np.errstate(divide='ignore', invalid='ignore'):
            counts = values / ratios / int_times
        return np.where(valid, counts, np.nan)

    def plot_alsW(self, als_data, ax, plot_options, ALSgainRatios):
        if not als_data.empty and 'time_stamp' in als_data.columns:
            # Create a copy of the dataframe to avoid modifying the original
            als_data_plot = als_data.copy()
            
            # Ensure gain columns and int_time are numeric
            for gain_col in ['uvGain', 'visGain', 'irGain']:
                if gain_col in als_data_plot.columns:
                    als_data_plot[gain_col] = pd.to_numeric(als_data_plot[gain_col], errors='coerce')
            if 'int_time' in als_data_plot.columns:
                als_data_plot['int_time'] = pd.to_numeric(als_data_plot['int_time'], errors='coerce')
            
            # Process each wavelength channel
            for wavelength in self.ALS_wave_col_names:
                if wavelength in als_data_plot.columns:
                    # Convert wavelength data to numeric
                    als_data_plot[wavelength] = pd.to_numeric(als_data_plot[wavelength], errors='coerce')
                    
                    # Apply gain compensation if enabled
                    if self.settings.get('applyALSgain', True) and ALSgainRatios:
                        # Determine which gain column applies to this wavelength
                        gain_col = None
                        if wavelength == '<400nm' and 'uvGain' in als_data_plot.columns:
                            gain_col = 'uvGain'
                        elif wavelength == '400-700nm' and 'visGain' in als_data_plot.columns:
                            gain_col = 'visGain'
                        elif wavelength == '>700nm' and 'irGain' in als_data_plot.columns:
                            gain_col = 'irGain'
                        
                        if gain_col:
                            if 'int_time' in als_data_plot.columns:
                                int_times = als_data_plot['int_time']
                            else:
                                int_times = pd.Series(1.0, index=als_data_plot.index)
                            # To convert to basic counts we divide by the gain ratio AND int_time
                            # Integration time normalization ensures counts are comparable across different int_time settings
                            basic_counts = self.compute_basic_counts(
                                als_data_plot[wavelength], als_data_plot[gain_col],
                                int_times, ALSgainRatios)

                            # Plot the basic counts
                            self.plot_downsampled(ax, als_data_plot['time_stamp'], basic_counts,
                                    lw=plot_options['lw'], marker=plot_options['marker'],
                                    markersize=plot_options['marker_size'],
                                    label=f"{wavelength}")
                    else:
                        # If gain compensation is disabled, just plot the raw values
                        self.plot_downsampled(ax, als_data_plot['time_stamp'], als_data_plot[wavelength],
                                lw=plot_options['lw'], marker=plot_options['marker'],
                                markersize=plot_options['marker_size'], label=f"{wavelength}")
              # Add an interactive legend for ALS Wavelength chart
            leg = ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
            leg.set_draggable(state=True)
            self.make_legend_interactive(ax, leg)
            
            # Set y-axis scale based on user preference
            has_positive_data = False
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if np.any(ydata > 0):
                    has_positive_data = True
                    break
                        
            if self.settings['useLogScale']['ALS_wave'] and has_positive_data:
                # Set y-axis to logarithmic scale
                ax.set_yscale('log')
                # Find minimum positive value across all plotted lines to set y-axis minimum
                min_positive = float('inf')
                for line in ax.get_lines():
                    ydata = line.get_ydata()
                    positive_vals = ydata[ydata > 0]
                    if len(positive_vals) > 0:
                        min_positive = min(min_positive, np.min(positive_vals))
                
                if min_positive != float('inf'):
                    ax.set_ylim(min_positive * 0.8, ax.get_ylim()[1])
            else:
                # Set y-axis to linear scale
                ax.set_yscale('linear')
                # For linear scale, use a sensible y-axis range
                y_min = 0
                ax.set_ylim(y_min, ax.get_ylim()[1])
    
    def plot_alsF(self, als_data, ax, plot_options):
        if not als_data.empty and 'time_stamp' in als_data.columns and 'flicker' in als_data.columns:
            als_data['flicker'] = pd.to_numeric(als_data['flicker'], errors='coerce')
            
            ax.plot(als_data['time_stamp'], als_data['flicker'], 
                    lw=plot_options['lw'], marker=plot_options['marker'], 
                    markersize=plot_options['marker_size'], label='Flicker')
            
            # No min/max points or annotations
    
    def plot_pho(self, pho_data, ax, plot_options, PHOgainRatios):
        if not pho_data.empty and 'time_stamp' in pho_data.columns:
            # Create a copy of the dataframe to avoid modifying the original
            pho_data_plot = pho_data.copy()
            
            # Ensure Gain and int_time columns are numeric
            if 'Gain' in pho_data_plot.columns:
                pho_data_plot['Gain'] = pd.to_numeric(pho_data_plot['Gain'], errors='coerce')
            if 'int_time' in pho_data_plot.columns:
                pho_data_plot['int_time'] = pd.to_numeric(pho_data_plot['int_time'], errors='coerce')
            
            # Process each wavelength channel
            for wavelength in self.PHO_wave_col_names:
                if wavelength in pho_data_plot.columns:
                    # Convert wavelength data to numeric
                    pho_data_plot[wavelength] = pd.to_numeric(pho_data_plot[wavelength], errors='coerce')
                    
                    # Apply gain compensation if enabled
                    if self.settings.get('applyPHOgain', True) and PHOgainRatios and 'Gain' in pho_data_plot.columns:
                        if 'int_time' in pho_data_plot.columns:
                            int_times = pho_data_plot['int_time']
                        else:
                            int_times = pd.Series(1.0, index=pho_data_plot.index)
                        # For PHO sensors we DIVIDE by ratio AND int_time to get basic counts
                        # Integration time normalization ensures counts are comparable across different int_time settings
                        basic_counts = self.compute_basic_counts(
                            pho_data_plot[wavelength], pho_data_plot['Gain'],
                            int_times, PHOgainRatios)

                        # Apply additional wavelength scaling if enabled (NaN stays NaN)
                        if self.settings.get('scalePHO', False) and wavelength in self.phoToScale:
                            basic_counts = basic_counts * self.phoScales.get(wavelength, 1.0)

                        # Plot the basic counts
                        self.plot_downsampled(ax, pho_data_plot['time_stamp'], basic_counts,
                                lw=plot_options['lw'], marker=plot_options['marker'],
                                markersize=plot_options['marker_size'],
                                label=f"{wavelength}")
                    else:
                        # If gain compensation is disabled, just plot the raw values
                        self.plot_downsampled(ax, pho_data_plot['time_stamp'], pho_data_plot[wavelength],
                                lw=plot_options['lw'], marker=plot_options['marker'],
                                markersize=plot_options['marker_size'], label=f"{wavelength} (Raw)")

              # Set y-axis scale based on user preference
            has_positive_data = False
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if np.any(ydata > 0):
                    has_positive_data = True
                    break
            
            if self.settings['useLogScale']['PHO'] and has_positive_data:
                # Set y-axis to logarithmic scale
                ax.set_yscale('log')
                # Find minimum positive value across all plotted lines to set y-axis minimum
                min_positive = float('inf')
                for line in ax.get_lines():
                    ydata = line.get_ydata()
                    positive_vals = ydata[ydata > 0]
                    if len(positive_vals) > 0:
                        min_positive = min(min_positive, np.min(positive_vals))
                
                if min_positive != float('inf'):
                    ax.set_ylim(min_positive * 0.8, ax.get_ylim()[1])
            else:
                # Set y-axis to linear scale
                ax.set_yscale('linear')
                # For linear scale, use a sensible y-axis range
                y_min = 0
                ax.set_ylim(y_min, ax.get_ylim()[1])
    
    def plot_imu(self, imu_data, ax, plot_options, ALSgainRatios):
        if not imu_data.empty and 'time_stamp' in imu_data.columns:
            for axis in ['Acc_x', 'Acc_y', 'Acc_z']:
                if axis in imu_data.columns:
                    imu_data[axis] = pd.to_numeric(imu_data[axis], errors='coerce')

                    self.plot_downsampled(ax, imu_data['time_stamp'], imu_data[axis],
                            lw=plot_options['lw'], marker=plot_options['marker'],
                            markersize=plot_options['marker_size'], label=axis)

            self.add_wear_shading(ax)

    def add_wear_shading(self, ax):
        """Shade windows classified "not worn" (stationary, or excluded because
        the device was charging -- see MainWindow.compute_wear_windows) directly
        on the IMU plot, the same way add_night_shading marks time of day."""
        windows = self.wear_windows
        if not windows:
            return
        window_s = windows.get('window_s', 60)
        not_worn = windows['monitored'] & ~windows['worn']
        if not not_worn.any():
            return
        first = True
        for start, flag in zip(windows['window_starts'], not_worn):
            if not flag:
                continue
            end = start + pd.Timedelta(seconds=window_s)
            ax.axvspan(start, end, facecolor='red', alpha=0.10, zorder=0, linewidth=0,
                       label='Not Worn' if first else None)
            first = False

    def plot_imu_temp(self, imu_data, ax, plot_options):
        # Flag to track if we displayed any temperature data
        any_temp_data_plotted = False
        
        # Part 1: Plot IMU temperature from Sensor_Data.csv
        if not imu_data.empty and 'time_stamp' in imu_data.columns:
            # Look for temperature column - specifically the 'Temp' column that we added to headers
            if 'Temp' in imu_data.columns:
                # Convert to numeric
                imu_data['Temp'] = pd.to_numeric(imu_data['Temp'], errors='coerce')
                
                # Plot temperature from Sensor_Data.csv
                self.plot_downsampled(ax, imu_data['time_stamp'], imu_data['Temp'],
                        lw=plot_options['lw'], marker=plot_options['marker'],
                        markersize=plot_options['marker_size'],
                        label='System Temp', color='blue')
                
                any_temp_data_plotted = True
        
        # Part 2: Plot IMU temperature pre-parsed from log.csv CHARGESTRING
        # entries during load_data (no file re-read at plot time)
        try:
            log_temp_times = self.log_events.get('imu_temp_times', [])
            log_temp_values = self.log_events.get('imu_temp_values', [])
            print(f"Using {len(log_temp_times)} pre-parsed IMU temperature values")

            # Plot the temperature data if found
            if log_temp_times:
                try:
                    # Convert to pandas datetime with timezone
                    log_temp_ts = pd.to_datetime(log_temp_times, unit='s')
                    log_temp_ts = log_temp_ts.tz_localize('UTC').tz_convert(self.tz_offset)
                    
                    # Show first few values for debugging
                    sample_size = min(5, len(log_temp_times))
                    if sample_size > 0:
                        print(f"Sample of {sample_size} temperature values:")
                        for i in range(sample_size):
                            print(f"  Time: {log_temp_ts[i]}, Temp: {log_temp_values[i]}")
                    
                    # Check for valid data before plotting
                    if len(log_temp_ts) > 0 and len(log_temp_values) > 0:
                        # Create an explicit Series with proper datetime index
                        temp_df = pd.DataFrame({
                            'timestamp': log_temp_ts,
                            'temperature': log_temp_values
                        })
                        

                        # Plot with explicit settings to ensure visibility
                        self.plot_downsampled(ax, temp_df['timestamp'], temp_df['temperature'],
                               color='red',
                               linestyle='-',
                               linewidth=2,
                               marker='o',
                               markersize=4,
                               label='IMU Temp (Log)')
                        

                        # Add a legend entry for the log temperature data
                        leg = ax.legend(loc='best')
                        leg.set_draggable(state=True)
                        
                        any_temp_data_plotted = True
                        print("Successfully plotted temperature data")
                except Exception as e:
                    print(f"Error during plotting: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No temperature values found to plot")
        except Exception as e:
            print(f"Error adding IMU temperature from log.csv: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Common elements for the plot if we have any temperature data
        if any_temp_data_plotted:
            # Add horizontal line at 25°C as reference temperature
            ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='25°C Reference')
            
            # Set reasonable y-axis limits with some padding based on all data
            y_values = []
            for line in ax.get_lines():
                if line.get_label() != '25°C Reference':  # Don't include the reference line in y-range calculation
                    y_values.extend(line.get_ydata())
            
            if y_values:
                temp_min = min(y_values)
                temp_max = max(y_values)
                
                # Add 10% padding
                padding = max(1, (temp_max - temp_min) * 0.1)
                ax.set_ylim(temp_min - padding, temp_max + padding)
        else:
            # If no temperature data found from either source
            ax.text(0.5, 0.5, "No temperature data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, color='red')


    def plot_sampling_intervals(self, all_data, ax, plot_options):
        """Plot the time intervals between consecutive samples for different sensors"""
        colors = {'ALS': 'blue', 'PHO': 'green', 'TOF': 'red', 'IMU': 'purple'}
        markers = {'ALS': 'o', 'PHO': 's', 'TOF': '^', 'IMU': 'x'}
        
        for sensor_type in ['ALS', 'PHO', 'TOF', 'IMU']:
            if sensor_type in all_data and not all_data[sensor_type].empty and 'time_stamp' in all_data[sensor_type].columns:
                # Make a copy of the dataframe to avoid modifying original
                sensor_df = all_data[sensor_type].copy()
                
                # Sort by timestamp (in case they're not already sorted)
                sensor_df = sensor_df.sort_values('time_stamp')
                
                # Calculate time differences in seconds
                sensor_df['time_diff'] = sensor_df['time_stamp'].diff().dt.total_seconds()
                
                # Filter out the first row (which has NaN time_diff) and any extreme outliers
                sensor_df = sensor_df.dropna(subset=['time_diff'])
                # Filter out unreasonable values (e.g., negative or extremely large intervals)
                sensor_df = sensor_df[sensor_df['time_diff'].between(0, 600)]  # Max 10 minutes interval
                
                if not sensor_df.empty:
                    # Plot the intervals
                    self.plot_downsampled(ax, sensor_df['time_stamp'], sensor_df['time_diff'],
                            lw=plot_options['lw'],
                            marker=markers.get(sensor_type, plot_options['marker']),
                            markersize=plot_options['marker_size'],
                            color=colors.get(sensor_type),
                            alpha=0.7,
                            label=f"{sensor_type} Interval")
                    
                    # Calculate and print summary statistics
                    avg_interval = sensor_df['time_diff'].mean()
                    std_interval = sensor_df['time_diff'].std()
                    median_interval = sensor_df['time_diff'].median()
                    print(f"{sensor_type} sampling stats - Mean: {avg_interval:.2f}s, Median: {median_interval:.2f}s, StdDev: {std_interval:.2f}s")
        
        # Add a grid for better readability
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Always use linear scale as preferred by the user
        ax.set_yscale('linear')
        
        # Find reasonable y-axis limits without considering logarithmic scale
        all_intervals = []
        for sensor_type in ['ALS', 'PHO', 'TOF', 'IMU']:
            if sensor_type in all_data and not all_data[sensor_type].empty and 'time_stamp' in all_data[sensor_type].columns:
                # Get time differences as we did above
                diffs = all_data[sensor_type]['time_stamp'].diff().dt.total_seconds()
                valid_diffs = [d for d in diffs.dropna().tolist() if 0 < d < 600]  # Filter out extreme values
                all_intervals.extend(valid_diffs)
        
        if all_intervals:
            # Find min and max for setting y-axis limits
            min_interval = min(all_intervals)
            max_interval = max(all_intervals)
            
            # Add padding to the y-axis limits
            padding = (max_interval - min_interval) * 0.1
            y_min = max(0, min_interval - padding)  # Don't go below zero
            y_max = max_interval + padding
            
            # Set y-axis limits
            ax.set_ylim(y_min, y_max)
        
        # Set y-axis label
        ax.set_ylabel("Interval [seconds]")
        
        # Add a reference line for some common sampling rates (e.g., 1 sec, 10 sec, 1 min)
        for rate, style in [(1, '--'), (10, ':'), (60, '-.')]:
            ax.axhline(y=rate, color='gray', linestyle=style, alpha=0.5, 
                       label=f"{rate}s Reference" if rate == 1 else None)


class TOFAnalysisWindow(QMainWindow):
    """Dioptric TOF analysis: time-weighted vergence histogram + band occupancy.

    Renders from a precomputed tof-stats dict (see MainWindow.compute_tof_stats)
    so the window itself carries no gating logic — veet_tof_bands stays the
    single source of truth.
    """

    def __init__(self, tof_stats, device_folder):
        _ensure_core_modules()
        _ensure_matplotlib()
        super().__init__()
        self.setWindowTitle(f"TOF Dioptric Analysis - {device_folder}")
        self.canvas = FigureCanvas(Figure(figsize=(13, 6)))
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.plot(tof_stats, device_folder)

    def plot(self, s, device_folder):
        fig = self.canvas.figure
        fig.clear()
        ax_hist, ax_bands = fig.subplots(1, 2)

        # --- Left: time-weighted dioptric histogram (0.25 D bins + overflow) ---
        edges = np.asarray(s['bin_edges'])          # 41 edges, 0..10 D
        hist_all = np.asarray(s['hist_seconds_all']) / 3600.0   # 41 values, last = overflow
        regular = len(edges) - 1                     # 40 regular bins
        ax_hist.bar(edges[:regular], hist_all[:regular], width=0.25, align='edge',
                    color='steelblue', alpha=0.45 if s['hist_seconds_worn'] is not None else 0.9,
                    label='All frames')
        if s['hist_seconds_worn'] is not None:
            hist_worn = np.asarray(s['hist_seconds_worn']) / 3600.0
            ax_hist.bar(edges[:regular], hist_worn[:regular], width=0.25, align='edge',
                        color='darkorange', alpha=0.9, label='Worn periods only')
            ax_hist.legend()
        ax_hist.set_xlabel("Vergence [D]  (D = 1000 / distance mm)")
        ax_hist.set_ylabel("Exposure [zone·hours]")
        ax_hist.set_title("Time-weighted dioptric histogram (0.25 D bins)")
        ax_hist.grid(True, axis='y', linestyle=':', linewidth=0.5)

        # Distance reference on the top axis
        top_axis = ax_hist.secondary_xaxis('top')
        ref_diopters = [0.5, 1.0, 2.0, 3.33, 5.0, 10.0]
        top_axis.set_xticks(ref_diopters)
        top_axis.set_xticklabels(['2000', '1000', '500', '300', '200', '100'])
        top_axis.set_xlabel("Distance [mm]")

        overflow_h = hist_all[regular]
        if overflow_h > 0:
            ax_hist.annotate(f"overflow (<100 mm): {overflow_h:.2f} zone·h",
                             xy=(0.98, 0.98), xycoords='axes fraction',
                             ha='right', va='top', fontsize=8, color='gray')

        # --- Right: reporting-band occupancy (% of valid zone-time) ---
        names = [f"{name}\n{rng}" for name, rng, pct, hours in s['band_share']]
        shares = [pct for name, rng, pct, hours in s['band_share']]
        hours_list = [hours for name, rng, pct, hours in s['band_share']]
        y_pos = np.arange(len(names))[::-1]  # nearest band at the top
        # Near-work bands (extreme/close/standard near) highlighted
        colors = ['firebrick' if 1 <= i <= 3 else 'steelblue' for i in range(len(names))]
        ax_bands.barh(y_pos, shares, color=colors, alpha=0.85)
        ax_bands.set_yticks(y_pos)
        ax_bands.set_yticklabels(names, fontsize=8)
        ax_bands.set_xlabel("Share of valid zone-time [%]")
        ax_bands.set_title("Reporting-band occupancy (all frames)")
        ax_bands.grid(True, axis='x', linestyle=':', linewidth=0.5)
        for y, pct, hours in zip(y_pos, shares, hours_list):
            if pct > 0:
                ax_bands.annotate(f" {pct:.1f}%  ({hours:.1f} zone·h)",
                                  xy=(pct, y), va='center', fontsize=7)

        summary = (f"dose {s['dose_dh_all']:.1f} D·h (zone-weighted)"
                   + (f", worn-only {s['dose_dh_worn']:.1f} D·h" if s['dose_dh_worn'] is not None else "")
                   + (f"  |  median vergence {s['median_vergence']:.2f} D" if s['median_vergence'] is not None else "")
                   + f"  |  {s['valid_frames']}/{s['n_frames']} frames with valid zones"
                   + f"  |  sample period {s['sample_period']:.1f}s")
        fig.suptitle(f"{device_folder} — TOF dioptric analysis\n{summary}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        self.canvas.draw()


class DataLoadWorker(QThread):
    """Runs parse_folder_data off the GUI thread with progress and cancel."""
    progress = Signal(int, str)
    finished_ok = Signal(object)   # result dict, or None if canceled
    failed = Signal(str)

    def __init__(self, main_window, device_path):
        super().__init__(main_window)
        self.main_window = main_window
        self.device_path = device_path
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = self.main_window.parse_folder_data(
                self.device_path,
                progress_cb=self.progress.emit,
                cancel_event=self.cancel_event)
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        _ensure_core_modules()
        super().__init__()

        # Define all attributes BEFORE calling initUI()
        self.plot_labels = {
            'Battery Voltage': "Voltage [mV]",
            'Battery Voltage Change Over Time': "Voltage Rate [mV/min]",
            'TOF Average Distance': "TOF Avg. Dist. [mm]",
            'ALS Lux': "ALS [lux]",
            'ALS Basic Counts': "ALS [counts]",
            'PHO Basic Counts': "PHO [counts]",
            'IMU Translational Acceleration': "Accel [m/s^2]",
            'System Temperature (IMU)': "Temperature [°C]",
            'Sensor Sample Interval': "Sample Interval [s]"
        }
        self.headers = {
            'BAT': ['time_stamp', 'BAT', 'Voltage'],
            'TOF': ['time_stamp', 'TOF'] + 
                    [f"S{i}_Conf1" for i in range(0, 64)] + 
                    [f"S{i}_Conf2" for i in range(0, 64)] +                            
                    [f"S{i}_Dist1" for i in range(0, 64)] +                          
                    [f"S{i}_Dist2" for i in range(0, 64)],                              
            'ALS': ['time_stamp', 'ALS', 'int_time', 'uvGain', 'visGain', 'irGain', '<400nm', 
                    '400-700nm', '>700nm', 'flicker','lux'],
            'PHO': ['time_stamp', 'PHO', 'int_time', 'Gain', '415', '445', '480', '515', '555', 
                    '590', '630', '680', '910', 'Dark', 'Clear'],
            'REBOOT': ['time_stamp', 'REBOOT', 'State'],
            'IMU': ['time_stamp', 'IMU', 'Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z', 'Temp'],
            'LOG': ['time_stamp', 'LOG', 'State', 'Value']
        }
        self.ALS_wave_col_names = ['<400nm', '400-700nm', '>700nm']
        self.ALSgainsIndx = ['uvGain', 'visGain', 'irGain']
        self.PHO_wave_col_names = ['415', '445', '480', '515', '555', '590', '630', '680','910','Dark', 'Clear']
        self.IMUaccel = ['Acc_x', 'Acc_y', 'Acc_z']
        self.IMUgyro = ['Gyr_x', 'Gyr_y', 'Gyr_z']
        
        # Get system timezone offset for initialization
        import time
        system_tz_offset = self.get_system_timezone_offset()
        
        # Initialize settings BEFORE calling initUI
        self.settings = {
            'time_zone': {'hour_offset': system_tz_offset},
            'filters': {'pre2023': False, 'time_range': False},                          
            'time_range': {'A': '2023-05-24 13:50:00', 'B': '2023-05-24 14:10:00'},
            'conf_thresh': 0,
            'hours_to_display': '',  # New setting for controlling X-axis display hours
            'hours_offset': '0',     # New setting for offset from the end of the data
            'plot_options': {
                'BAT': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'BAT Rate': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'TOF': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'ALS_lux': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'ALS_wave': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'ALS_flick': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'PHO': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'IMU': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'IMU_temp': {'lw': 1, 'marker': 'o', 'marker_size': 2},
                'Sample_intervals': {'lw': 1, 'marker': 'o', 'marker_size': 3}
            },
            'applyALSgain': True,
            'scalePHO': False,
            'applyPHOgain': True,
            'useLogScale': {
                'ALS_lux': True,  # Default to log scale for ALS Lux
                'ALS_wave': True,  # Default to log scale for ALS Wavelength
                'PHO': True        # Default to log scale for PHO
            },
            'ALSgainRatios': {0.5: 1/249.13, 1: 1/123.85, 2: 1/62.97, 4: 1/31.72, 8: 1/15.53, 16: 1/7.97, 32: 1/3.99, 64: 1/2.01, 128: 1, 256: 1.93, 512: 3.80, 1024: 7.42, 2048: 14.06, 4096: 25.35},
            'PHOgainRatios': {0.5: 0.008, 1: 0.016, 2: 0.032, 4: 0.065, 8: 0.125, 16: 0.25, 32: 0.50, 64: 1.00, 128: 2.00, 256: 3.95, 512: 7.75}
        }
        
        self.tz = pytz.timezone('GMT')
        self.tz_offset = pytz.FixedOffset(self.settings['time_zone']['hour_offset'] * 60)
        self.ALSgainRatios = {0.5: 1/249.13, 1: 1/123.85, 2: 1/62.97, 4: 1/31.72, 8: 1/15.53, 16: 1/7.97, 32: 1/3.99, 64: 1/2.01, 128: 1, 256: 1.93, 512: 3.80, 1024: 7.42, 2048: 14.06, 4096: 25.35}
        self.PHOgainRatios = {0.5: 0.008, 1: 0.016, 2: 0.032, 4: 0.065, 8: 0.125, 16: 0.25, 32: 0.50, 64: 1.00, 128: 2.00, 256: 3.95, 512: 7.75}
        self.phoToScale = ['415', '445', '480', '515', '555', '590', '630', '680', '910', 'Dark', 'Clear']
        self.phoScales = {'415': 1.425, '445': 1.031, '480': 1.342, '515': 0.550, '555': 0.872, '590': 0.373, '630': 0.648, '680': 0.425, '910': 0.173, 'Dark': 1.0, 'Clear': 0.76}
        self.data = {}
        self.log_events = {}
        self.device_folder = ''
        self.selected_plots = []
        self.firmware_info = None
        self.additional_info = None
        self.inf_line = None
        self.serial_number = None
        self.parse_warnings = []
        self._load_worker = None
        self._load_progress = None

        # Custom date-range settings (override hours/offset when enabled)
        self.settings['use_date_range'] = False
        self.settings['range_start'] = ''
        self.settings['range_end'] = ''

        # Session statistics: gap threshold (multiples of the expected sample
        # interval) and whether exports get a stats stamp
        self.settings['gap_factor'] = 5
        self.settings['stamp_stats'] = False
        self.session_stats = None

        # Debounce timer so hours/offset typing doesn't rebuild the plot per keystroke
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(600)
        self._rebuild_timer.timeout.connect(self._rebuild_plot_if_open)

        # NOW call initUI after all attributes are initialized
        self.initUI()

    def initUI(self):
        content = QWidget()
        layout = QVBoxLayout(content)

        # --- Data ---
        data_group = QGroupBox("Data", self)
        data_layout = QVBoxLayout(data_group)

        self.btn_select_folder = QPushButton("Select Folder && Load Data", self)
        self.btn_select_folder.clicked.connect(self.select_folder)
        data_layout.addWidget(self.btn_select_folder)

        self.label_folder = QLabel("Folder not selected", self)
        data_layout.addWidget(self.label_folder)

        # Data summary panel: SN, firmware, extents, record counts, warnings
        self.summary_label = QLabel("No data loaded — select or drop a device folder", self)
        self.summary_label.setStyleSheet(
            "background: #f5f5f5; border: 1px solid #d0d0d0; border-radius: 3px; "
            "padding: 6px; font-family: Consolas, monospace; font-size: 11px;")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        data_layout.addWidget(self.summary_label)

        layout.addWidget(data_group)

        # --- Plot selection: each option's checkbox, with its log-scale
        # companion (if any) on the same row instead of a row below it ---
        plot_group = QGroupBox("Plot Selection", self)
        plot_layout = QVBoxLayout(plot_group)

        scale_option_for = {
            'ALS Lux': ('ALS_lux', "Use Log Scale for ALS Lux"),
            'ALS Basic Counts': ('ALS_wave', "Use Log Scale for ALS Basic Counts"),
            'PHO Basic Counts': ('PHO', "Use Log Scale for PHO Basic Counts"),
        }
        self.checkboxes = []
        self.scale_checkboxes = {}
        for plot_option in ['Battery Voltage', 'Battery Voltage Change Over Time', 'Sensor Sample Interval', 'System Temperature (IMU)', 'TOF Average Distance', 'ALS Lux', 'ALS Basic Counts', 'PHO Basic Counts', 'IMU Translational Acceleration']:
            row = QHBoxLayout()
            cb = QCheckBox(plot_option, self)
            # Set Battery Voltage to be checked by default
            if plot_option == 'Battery Voltage':
                cb.setChecked(True)
            cb.stateChanged.connect(self.update_selected_plots)
            self.checkboxes.append(cb)
            row.addWidget(cb)

            if plot_option in scale_option_for:
                settings_key, label = scale_option_for[plot_option]
                scale_cb = QCheckBox(label, self)
                scale_cb.setChecked(self.settings['useLogScale'][settings_key])
                scale_cb.stateChanged.connect(lambda state, pt=settings_key: self.update_scale_setting(pt, state))
                row.addWidget(scale_cb)
                self.scale_checkboxes[settings_key] = scale_cb

            row.addStretch(1)
            plot_layout.addLayout(row)

        layout.addWidget(plot_group)

        # --- X-axis time range control (including timezone) ---
        time_group = QGroupBox("X-Axis Time Range Control", self)
        x_axis_layout = QVBoxLayout(time_group)

        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Timezone offset (GMT):", self))
        self.timezone_combo = QComboBox(self)        # Populate with timezone options from -12 to +14
        for i in range(-12, 15):
            self.timezone_combo.addItem(f"GMT{'+' if i >= 0 else ''}{i}", i)
        # Set default to system timezone
        system_tz_offset = self.get_system_timezone_offset()
        index = self.timezone_combo.findData(system_tz_offset)
        if index >= 0:
            self.timezone_combo.setCurrentIndex(index)
        self.timezone_combo.currentIndexChanged.connect(self.update_timezone)
        tz_layout.addWidget(self.timezone_combo, 1)
        x_axis_layout.addLayout(tz_layout)

        hours_layout = QHBoxLayout()
        hours_label = QLabel("Hours to display:", self)
        self.hours_input = QLineEdit(self)
        self.hours_input.setPlaceholderText("Leave empty for full range")
        self.hours_input.setText(self.settings['hours_to_display'])
        self.hours_input.textChanged.connect(self.update_hours_setting)
        hours_layout.addWidget(hours_label)
        hours_layout.addWidget(self.hours_input)
        x_axis_layout.addLayout(hours_layout)

        # Add the new hours offset control
        offset_layout = QHBoxLayout()
        offset_label = QLabel("Hours offset from end:", self)
        self.offset_input = QLineEdit(self)
        self.offset_input.setPlaceholderText("0")
        self.offset_input.setText(self.settings.get('hours_offset', '0'))
        self.offset_input.textChanged.connect(self.update_offset_setting)
        offset_layout.addWidget(offset_label)
        offset_layout.addWidget(self.offset_input)
        x_axis_layout.addLayout(offset_layout)

        # Date-range picker: pre-filled with data extents after load; overrides
        # the hours/offset controls (kept above as a power-user option)
        self.use_date_range_cb = QCheckBox("Use custom date range (overrides hours settings)", self)
        self.use_date_range_cb.stateChanged.connect(self.update_date_range)
        x_axis_layout.addWidget(self.use_date_range_cb)

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("From:", self))
        self.range_start_edit = QDateTimeEdit(self)
        self.range_start_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.range_start_edit.setCalendarPopup(True)
        self.range_start_edit.dateTimeChanged.connect(self.update_date_range)
        range_layout.addWidget(self.range_start_edit)
        range_layout.addWidget(QLabel("To:", self))
        self.range_end_edit = QDateTimeEdit(self)
        self.range_end_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.range_end_edit.setCalendarPopup(True)
        self.range_end_edit.dateTimeChanged.connect(self.update_date_range)
        range_layout.addWidget(self.range_end_edit)
        x_axis_layout.addLayout(range_layout)

        layout.addWidget(time_group)

        # --- Actions ---
        action_group = QGroupBox("Actions", self)
        action_layout = QGridLayout(action_group)

        self.btn_show_plot = QPushButton("Show Plot", self)
        self.btn_show_plot.clicked.connect(self.show_plot_window)
        action_layout.addWidget(self.btn_show_plot, 0, 0)

        self.btn_tof_analysis = QPushButton("TOF Dioptric Analysis", self)
        self.btn_tof_analysis.clicked.connect(self.show_tof_analysis)
        action_layout.addWidget(self.btn_tof_analysis, 0, 1)

        self.btn_export_tof_arrays = QPushButton("Export TOF Arrays (.npz)", self)
        self.btn_export_tof_arrays.clicked.connect(self.export_tof_arrays)
        action_layout.addWidget(self.btn_export_tof_arrays, 1, 0)

        self.btn_export_plot = QPushButton("Export Plot", self)
        self.btn_export_plot.clicked.connect(self.export_plot)
        action_layout.addWidget(self.btn_export_plot, 1, 1)

        self.stamp_stats_cb = QCheckBox("Stamp session statistics on exported PNGs", self)
        self.stamp_stats_cb.setChecked(self.settings.get('stamp_stats', False))
        self.stamp_stats_cb.stateChanged.connect(
            lambda state: self.settings.__setitem__('stamp_stats', state == 2))
        action_layout.addWidget(self.stamp_stats_cb, 2, 0, 1, 2)

        # Renamed for clarity; spans both columns since it's a lower-priority/deprecated action
        self.btn_export_all_plots = QPushButton("Export All Subdirectory Plots (deprecated function)", self)
        self.btn_export_all_plots.clicked.connect(self.export_all_plots)
        action_layout.addWidget(self.btn_export_all_plots, 3, 0, 1, 2)

        layout.addWidget(action_group)
        layout.addStretch(1)

        # Scroll area so the control panel can be shrunk below its natural
        # content height on small screens instead of forcing the window to
        # grow to fit every control
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)

        # Dockable session-statistics table (hidden until data loads)
        self._init_stats_dock()

        # Accept a device folder dropped anywhere on the window
        self.setAcceptDrops(True)

        # Standard window behavior: fit on-screen and stay resizable, and
        # remember size/position/maximized state across launches like any
        # other Windows application (restore_preferences below handles the
        # rest of the persisted settings)
        self.setMinimumSize(360, 320)
        self._apply_initial_window_geometry()

        # Restore persisted preferences (may change checkbox states)
        self.restore_preferences()

        # Initialize selected plots to capture any checkboxes that are checked by default
        self.update_selected_plots()

    def _apply_initial_window_geometry(self):
        """Restore the last saved window size/position/maximized state, or
        fall back to a default sized to fit on the available screen (rather
        than growing to fit the full control panel content)."""
        qsettings = QSettings("VEET", "VEET_Data_Charting")
        geometry = qsettings.value("window_geometry")
        if geometry is not None and self.restoreGeometry(geometry):
            return
        screen = self.screen()
        avail = screen.availableGeometry() if screen else None
        if avail is not None:
            width = min(480, avail.width() - 40)
            height = min(880, avail.height() - 80)
        else:
            width, height = 480, 800
        self.resize(width, height)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if any(u.isLocalFile() and os.path.isdir(u.toLocalFile()) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isdir(path):
                self.start_threaded_load(path)
                break

    def restore_preferences(self):
        """Restore persisted preferences: last folder, timezone, checkbox states."""
        qsettings = QSettings("VEET", "VEET_Data_Charting")

        last_folder = qsettings.value("last_folder", "")
        if last_folder and os.path.isdir(last_folder):
            self.label_folder.setText(last_folder)

        tz_value = qsettings.value("timezone_offset", None)
        if tz_value is not None:
            try:
                index = self.timezone_combo.findData(int(tz_value))
                if index >= 0:
                    self.timezone_combo.setCurrentIndex(index)
            except (TypeError, ValueError):
                pass

        def as_bool(value):
            return value in (True, 'true', '1', 1)

        for cb in self.checkboxes:
            value = qsettings.value(f"plots/{cb.text()}", None)
            if value is not None:
                cb.setChecked(as_bool(value))

        for key, cb in self.scale_checkboxes.items():
            value = qsettings.value(f"logscale/{key}", None)
            if value is not None:
                cb.setChecked(as_bool(value))

        self.hours_input.setText(str(qsettings.value("hours_to_display", self.settings['hours_to_display'])))
        self.offset_input.setText(str(qsettings.value("hours_offset", self.settings['hours_offset'])))

        stamp_value = qsettings.value("stamp_stats", None)
        if stamp_value is not None:
            self.stamp_stats_cb.setChecked(as_bool(stamp_value))

    def save_preferences(self):
        qsettings = QSettings("VEET", "VEET_Data_Charting")
        folder = self.label_folder.text()
        if os.path.isdir(folder):
            qsettings.setValue("last_folder", folder)
        qsettings.setValue("timezone_offset", self.settings['time_zone']['hour_offset'])
        for cb in self.checkboxes:
            qsettings.setValue(f"plots/{cb.text()}", cb.isChecked())
        for key, cb in self.scale_checkboxes.items():
            qsettings.setValue(f"logscale/{key}", cb.isChecked())
        qsettings.setValue("hours_to_display", self.settings['hours_to_display'])
        qsettings.setValue("hours_offset", self.settings['hours_offset'])
        qsettings.setValue("stamp_stats", self.stamp_stats_cb.isChecked())
        qsettings.setValue("window_geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_preferences()
        super().closeEvent(event)
    
    def export_all_plots(self):
        """Batch export charts for all subdirectories of a given directory."""
        # Capture the CURRENT plot selections from the GUI (not just startup defaults)
        self.update_selected_plots()
        
        # Check if any plots are selected before proceeding
        if not self.selected_plots:
            QMessageBox.warning(self, "No Plots Selected", "Please select at least one plot type before exporting.")
            return
            
        print(f"Batch exporting with CURRENT selected plots: {self.selected_plots}")
        print(f"(Based on what you have checked in the GUI right now)")
        
        root_directory = QFileDialog.getExistingDirectory(self, "Select Root Directory")
        if not root_directory:
            return

        subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
        if not subdirectories:
            QMessageBox.information(self, "No Subdirectories", "No subdirectories found in the selected directory.")
            return

        for subdir in subdirectories:
            self.label_folder.setText(subdir)
            try:
                print(f"Loading data from: {subdir}")
                self.load_data()
                
                # Check if we have selected plots and data
                if not self.selected_plots:
                    print(f"Warning: No plots selected for {subdir}")
                    continue
                    
                # Check if we have data
                if not self.data:
                    print(f"Warning: No data loaded for {subdir}")
                    continue
                
                print(f"Data loaded successfully. Selected plots: {self.selected_plots}")
                
                # Get subdirectory name for unique filename
                subdir_name = os.path.basename(subdir)
                # Call export with the root directory and subdirectory name
                self.export_plot_batch(root_directory, subdir_name)
            except Exception as e:
                print(f"Error processing directory {subdir}: {e}")
                continue

        QMessageBox.information(self, "Batch Export Complete", "All plots have been exported.")

    def _stamp_stats_on_figure(self, fig):
        """Stamp the session statistics in the bottom-left of an exported figure."""
        if not self.settings.get('stamp_stats') or not getattr(self, 'session_stats', None):
            return
        stamp_text = self.format_session_stats(self.session_stats, max_gaps=3)
        if stamp_text:
            fig.text(0.01, 0.005, stamp_text, fontsize=6.5, family='monospace',
                     ha='left', va='bottom', color='#303030',
                     bbox=dict(facecolor='white', alpha=0.75, boxstyle='round,pad=0.3'))

    def export_plot_batch(self, save_directory, subdir_name):
        """Export plot for batch processing with custom directory and filename."""
        if self.device_folder == "Unknown_Device":
            serial_number = self.get_serial_number_from_inf(self.label_folder.text())
            if serial_number:
                self.device_folder = f"Unknown_Device (SN: {serial_number})"

        # Create the plot window
        self.plot_window = PlotWindow(
            self.selected_plots,
            self.plot_labels,
            self.settings,
            self.tz_offset,
            self.data,
            self.device_folder,
            self.ALS_wave_col_names,
            self.PHO_wave_col_names,
            self.ALSgainRatios,
            self.PHOgainRatios,
            self.phoToScale,
            self.phoScales,
            self.firmware_info,
            self.additional_info,
            log_events=self.log_events,
            wear_windows=self.compute_wear_windows(),
        )

        # Show and maximize the plot window properly
        self.plot_window.show()
        QApplication.processEvents()  # Ensure GUI updates
        self.plot_window.showMaximized()
        QApplication.processEvents()  # Ensure maximization is processed
        
        # Wait a moment for the plot to fully render
        import time
        time.sleep(0.5)
        QApplication.processEvents()  # Final processing to ensure plot is drawn

        # Create unique filename using just the subdirectory name
        clean_subdir_name = re.sub(r'[\\/*?:"<>|]', '_', subdir_name)
        filename = f"{clean_subdir_name}.png"
        
        # Save to the specified root directory
        save_path = os.path.join(save_directory, filename)
        print(f"Attempting to save to: {save_path}")

        try:
            # Ensure the canvas is fully drawn
            self.plot_window.canvas.draw()
            self.plot_window.canvas.flush_events()
            QApplication.processEvents()
            
            # Check if figure has any content
            if len(self.plot_window.canvas.figure.axes) == 0:
                print(f"Warning: No axes found in figure for {subdir_name}")
                return
            
            # Set figure size to 2032x1002 pixels at 100 DPI
            self.plot_window.canvas.figure.set_size_inches(2032 / 100, 1002 / 100)
            self._stamp_stats_on_figure(self.plot_window.canvas.figure)

            # Ensure the save directory exists
            os.makedirs(save_directory, exist_ok=True)
            
            # Save the figure
            self.plot_window.canvas.figure.savefig(
                save_path, dpi=100, bbox_inches="tight", facecolor="white"
            )
            
            # Verify the file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"Successfully exported: {save_path} (Size: {file_size} bytes)")
            else:
                print(f"Error: File was not created at {save_path}")
                
        except Exception as e:
            print(f"Failed to save plot for {subdir_name}: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")

        # Close the plot window
        self.plot_window.close()

    def export_plot(self):
        """Show the plot window, maximize it, save the plot as a .png file, and close the window."""
        if self.device_folder == "Unknown_Device":
            serial_number = self.get_serial_number_from_inf(self.label_folder.text())
            if serial_number:
                self.device_folder = f"Unknown_Device (SN: {serial_number})"

        # Create the plot window
        self.plot_window = PlotWindow(
            self.selected_plots,
            self.plot_labels,
            self.settings,
            self.tz_offset,
            self.data,
            self.device_folder,
            self.ALS_wave_col_names,
            self.PHO_wave_col_names,
            self.ALSgainRatios,
            self.PHOgainRatios,
            self.phoToScale,
            self.phoScales,
            self.firmware_info,
            self.additional_info,
            log_events=self.log_events,
            wear_windows=self.compute_wear_windows(),
        )

        # Maximize the plot window
        self.plot_window.showMaximized()

        # Save the plot
        clipboard_text = QApplication.clipboard().text()
        if not clipboard_text:
            clipboard_text = "plot_export"
        filename = re.sub(r'[\\/*?:"<>|]', '_', clipboard_text) + ".png"

        # Save to the parent directory of the data folder
        data_folder = self.label_folder.text()
        parent_directory = os.path.dirname(data_folder)
        save_path = os.path.join(parent_directory, filename)

        try:
            # Set figure size to 2032x1002 pixels at 100 DPI
            self.plot_window.canvas.figure.set_size_inches(2032 / 100, 1002 / 100)
            self._stamp_stats_on_figure(self.plot_window.canvas.figure)
            self.plot_window.canvas.figure.savefig(
                save_path, dpi=100, bbox_inches="tight", facecolor="white"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save plot: {str(e)}")

        # Close the plot window
        self.plot_window.close()

    def get_system_timezone_offset(self):
        """Get the system's timezone offset from GMT in hours"""
        import time
        import datetime
        
        # Get local timezone offset in seconds
        # The negative sign is needed because time.timezone gives the offset in opposite direction
        if time.localtime().tm_isdst and time.daylight:
            offset_seconds = -time.altzone
        else:
            offset_seconds = -time.timezone
            
        # Convert to hours
        offset_hours = offset_seconds / 3600
        return int(offset_hours)  # Return as an integer for the combobox
        
    def update_scale_setting(self, plot_type, state):
        """Update the scale setting when a checkbox state changes"""
        # Convert Qt.Checked (2) to True and Qt.Unchecked (0) to False
        is_checked = (state == 2)
        self.settings['useLogScale'][plot_type] = is_checked
        
        # If a plot window is already open, close it and reopen to apply changes
        if hasattr(self, 'plot_window') and self.plot_window is not None and self.plot_window.isVisible():
            self.plot_window.close()
            self.show_plot_window()
    
    def _schedule_plot_rebuild(self):
        """Debounce plot rebuilds so typing doesn't rebuild on every keystroke."""
        self._rebuild_timer.start()

    def _rebuild_plot_if_open(self):
        if hasattr(self, 'plot_window') and self.plot_window is not None and self.plot_window.isVisible():
            self.plot_window.close()
            self.show_plot_window()

    def update_hours_setting(self, text):
        """Update the hours_to_display setting when the input changes (debounced)"""
        self.settings['hours_to_display'] = text
        self._schedule_plot_rebuild()

    def update_offset_setting(self, text):
        """Update the hours_offset setting when the input changes (debounced)"""
        self.settings['hours_offset'] = text
        self._schedule_plot_rebuild()

    def update_date_range(self, *args):
        """Update the custom date-range settings from the picker widgets (debounced)"""
        self.settings['use_date_range'] = self.use_date_range_cb.isChecked()
        self.settings['range_start'] = self.range_start_edit.dateTime().toString('yyyy-MM-dd HH:mm:ss')
        self.settings['range_end'] = self.range_end_edit.dateTime().toString('yyyy-MM-dd HH:mm:ss')
        self._schedule_plot_rebuild()


    def update_timezone(self, index):
        """Update the timezone when the user selects a different one from the dropdown"""
        offset_hours = self.timezone_combo.itemData(index)
        self.settings['time_zone']['hour_offset'] = offset_hours
        self.tz_offset = pytz.FixedOffset(offset_hours * 60)
        
        # If data is already loaded, update timestamps
        if self.data:
            for key in self.data:
                if not self.data[key].empty and 'time_stamp' in self.data[key].columns:
                    # First convert all timestamps to UTC, then to the new timezone
                    if pd.api.types.is_datetime64_dtype(self.data[key]['time_stamp']):
                        # Get the current timezone info from the timestamps
                        orig_tz = self.data[key]['time_stamp'].dt.tz
                        
                        # Convert to UTC first (if they already have timezone info)
                        if orig_tz is not None:
                            self.data[key]['time_stamp'] = self.data[key]['time_stamp'].dt.tz_convert('UTC')
                            
                        # Now convert from UTC to the new timezone
                        self.data[key]['time_stamp'] = self.data[key]['time_stamp'].dt.tz_convert(self.tz_offset)
            
            # Refresh displayed statistics in the new timezone (without forcing
            # the dock open if the user closed it)
            self.update_summary_panel()
            self.update_stats_table(show=False)

            # Update the plot ONLY if it already exists and is open
            if hasattr(self, 'plot_window') and self.plot_window is not None and self.plot_window.isVisible():
                self.plot_window.close()
                self.show_plot_window()

    def select_folder(self):
        start_dir = self.label_folder.text()
        if not os.path.isdir(start_dir):
            start_dir = QSettings("VEET", "VEET_Data_Charting").value("last_folder", "")
            if not os.path.isdir(start_dir):
                start_dir = ""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if folder_path:
            self.start_threaded_load(folder_path)

    def start_threaded_load(self, folder_path):
        """Load a device folder on a worker thread with real progress and Cancel."""
        # Check for required files up front, on the GUI thread
        sensor_data_path = find_file_with_suffix(folder_path, "Sensor_Data.csv")
        log_data_path = find_file_with_suffix(folder_path, "log.csv")
        if not sensor_data_path or not log_data_path:
            missing_files = []
            if not sensor_data_path:
                missing_files.append("Sensor_Data.csv")
            if not log_data_path:
                missing_files.append("log.csv")
            QMessageBox.critical(self, "Error", f"Required files ({', '.join(missing_files)}) are missing.\n\nMake sure files ending with these names exist in the selected folder.")
            return

        # Ignore a second request while a load is already running
        if getattr(self, '_load_worker', None) is not None and self._load_worker.isRunning():
            print("A data load is already in progress")
            return

        self.label_folder.setText(folder_path)
        print(f"Starting data load from: {folder_path}")

        progress = QProgressDialog("Loading VEET data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Loading VEET Data")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        self._load_progress = progress

        worker = DataLoadWorker(self, folder_path)
        self._load_worker = worker
        progress.canceled.connect(worker.cancel_event.set)
        worker.progress.connect(self._on_load_progress)
        worker.finished_ok.connect(lambda result: self._on_load_finished(result, folder_path))
        worker.failed.connect(self._on_load_failed)
        worker.start()

    def _on_load_progress(self, percent, message):
        if self._load_progress is not None:
            self._load_progress.setLabelText(message)
            self._load_progress.setValue(percent)

    def _on_load_finished(self, result, folder_path):
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        if result is None:
            print("Data load canceled")
            self.summary_label.setText("Load canceled")
            return
        self.apply_parse_result(result, folder_path)

    def _on_load_failed(self, message):
        if self._load_progress is not None:
            self._load_progress.close()
            self._load_progress = None
        print(f"Error loading data: {message}")
        QMessageBox.critical(self, "Error", f"Error during data loading: {message}")


    def is_convertible(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False    
             
    def unload_data(self):
        self.data = {}
        self.device_folder = ''
        self.label_folder.setText("Folder not selected")
        print('Data Unloaded')
        QMessageBox.information(self, "Information", "Data Unloaded")
        
        if hasattr(self, 'plot_window') and self.plot_window is not None:
            self.plot_window.close()
            self.plot_window = None
    
    def process_line(self, line):
        """Process and clean a line, removing bad characters."""
        try:
            decoded_line = line.strip()
            # Use regex to check for bad characters
            if re.search(r'[^ -~]', decoded_line):
                decoded_line = ''  # Discard the line
        except Exception as e:
            print(f"An error occurred: {e}")
            decoded_line = ''
        return decoded_line

    def safe_to_numeric(self, series):
        """Safely convert a series to numeric values without using deprecated errors='ignore'"""
        try:
            return pd.to_numeric(series)
        except (ValueError, TypeError):
            # For non-numeric values, maintain original values
            return series

    def load_data_clicked(self):
        try:
            self.load_data()
        except Exception as e:
            print(e)
            print('Error loading data')
            return
        print('Data Loaded')
        QMessageBox.information(self, "Information", "Data Loaded")
        
    def parseDeviceFolder(self, path): 
        # If the path is a drive root (e.g., "D:\"), use the drive letter as the device folder
        if re.match(r'^[A-Za-z]:\\$', path):



            device_folder = path.split(':')[0] + "_Drive"
            print(device_folder)
            return device_folder
            
        # Otherwise, use the original regex pattern to extract the folder name
        pattern = r'[^/\\]+$'
        match = re.search(pattern, path)
        if match:
            device_folder = match.group()
            print(device_folder)
            return device_folder
        
        # If both methods fail, use a default name
        return "Unknown_Device"
    
    def get_serial_number_from_inf(self, device_path):
        """Extract serial number from the first INF line in sensor_data.csv"""
        # Use the helper function to find Sensor_Data.csv file with flexible naming
        sensor_data_path = find_file_with_suffix(device_path, "Sensor_Data.csv")
        
        if sensor_data_path and os.path.exists(sensor_data_path):
            try:
                with open(sensor_data_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        if ',INF,' in line:
                            parts = line.strip().split(',')
                            # A typical INF line: 1747772806,INF,VEET,25010306,FW2.1.19RC May 13 2025 07:19:05 ,abc123,abc123,-8,2000,2000,2000,2000,R
                            # We want the 4th column (index 3)
                            if len(parts) > 3:
                                serial_number = parts[3].strip()
                                print(f"Found serial number: {serial_number}")

                                return serial_number
            except Exception as e:
                print(f"Error reading serial number: {e}")
        return None
    
    def extract_inf_info(self, device_path):
        """Extract firmware revision and additional info from INF line in Sensor_Data.csv"""
        firmware_info = None
        additional_info = None
        inf_line = None
        serial_number = None

        # Use the helper function to find Sensor_Data.csv file with flexible naming
        sensor_data_path = find_file_with_suffix(device_path, "Sensor_Data.csv")
        
        if sensor_data_path and os.path.exists(sensor_data_path):
            try:
                with open(sensor_data_path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        if ',INF,' in line:
                            # Store the raw INF line
                            inf_line = line.strip()
                            
                            parts = line.strip().split(',')
                            # Extract serial number (4th column, index 3)
                            if len(parts) > 3:
                                serial_number = parts[3].strip()
                                print(f"Found serial number: {serial_number}")
                            
                            # Extract firmware info (5th column, index 4)
                            if len(parts) > 4:
                                firmware_info = parts[4].strip()
                                print(f"Found firmware info: {firmware_info}")
                            
                            # Extract last 6 values if they exist
                            if len(parts) >= 7:  # At least 7 parts needed for last 6 values
                                last_six_values = parts[-6:]
                                additional_info = ','.join(last_six_values)
                                print(f"Found additional info: {additional_info}")
                            
                            # Break after processing the first INF line
                            break
            except Exception as e:
                print(f"Error extracting INF info: {e}")
                import traceback
                traceback.print_exc()
        
        return firmware_info, additional_info, inf_line, serial_number
    
    def build_dataframe(self, filtered_data, keyword, tz_offset, warnings=None):
        """Build a typed DataFrame from processed CSV line strings for the keyword.

        Uses pandas' C parser for splitting and numeric type inference in one
        pass — roughly 10x faster than per-column to_numeric on large files.
        Rows are truncated to the header count to avoid field-count errors."""
        headers = self.headers.get(keyword, [])
        if not filtered_data:
            # Create empty DataFrame with proper headers
            return pd.DataFrame(columns=headers)

        header_count = len(headers)
        if header_count == 0:
            # No headers defined for this keyword (e.g. INF): keep row count only,
            # matching the old zero-column truncation behavior
            return pd.DataFrame(index=range(len(filtered_data)))

        rows = []
        max_fields = 0
        for line in filtered_data:
            parts = line.split(',')
            if len(parts) > header_count:
                parts = parts[:header_count]
            if len(parts) > max_fields:
                max_fields = len(parts)
            rows.append(','.join(parts))

        df = pd.read_csv(io.StringIO('\n'.join(rows)), header=None,
                         names=headers[:max_fields], quoting=csv.QUOTE_NONE,
                         engine='c')

        # Convert timestamps to datetime - FIXED to properly handle epoch time
        if 'time_stamp' in df.columns:
            # First try parsing as seconds since epoch (standard Unix timestamp)
            try:
                # Convert to numeric first to handle any formatting
                df['time_stamp'] = pd.to_numeric(df['time_stamp'], errors='coerce')

                # Check if timestamps are in milliseconds (13 digits) or seconds (10 digits),
                # using the first VALID sample rather than row 0 -- a corrupted/unparseable
                # first row would otherwise raise on int(nan) and take the whole column down
                valid_ts = df['time_stamp'].dropna()
                sample_ts = valid_ts.iloc[0] if not valid_ts.empty else 0

                if len(str(int(sample_ts))) >= 13:  # milliseconds
                    df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='ms')
                else:  # seconds
                    df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='s')

                df['time_stamp'] = df['time_stamp'].dt.tz_localize('UTC').dt.tz_convert(tz_offset)

                # Reject implausible dates -- a single corrupted/garbled numeric
                # timestamp elsewhere in the file (not just row 0) can otherwise
                # land far in the future or past (e.g. year 318857 if a garbage
                # value gets parsed as epoch seconds). One such row poisons every
                # min/max-based calculation downstream (night shading's date
                # range, data extents, session stats) and can crash any code
                # that later calls strftime on it, since pandas' wide-range
                # non-nanosecond datetimes exceed what Python's datetime/strftime
                # supports.
                min_valid = pd.Timestamp('2000-01-01', tz='UTC')
                max_valid = pd.Timestamp('2100-01-01', tz='UTC')
                implausible = (df['time_stamp'] < min_valid) | (df['time_stamp'] > max_valid)
                if implausible.any():
                    n_bad = int(implausible.sum())
                    df.loc[implausible, 'time_stamp'] = pd.NaT
                    print(f"{keyword}: dropped {n_bad} row(s) with an implausible timestamp")
                    if warnings is not None:
                        warnings.append(f"{keyword}: {n_bad} row(s) had an implausible timestamp and were dropped")
            except Exception as e:
                print(f"Error converting timestamps: {e}")
                if warnings is not None:
                    warnings.append(f"{keyword}: timestamp conversion failed ({e})")
                # If conversion fails, try alternate format or leave as is

        # Numeric columns were already typed by the C parser; mixed columns stay object
        return df

    def parse_log_state_events(self, line, log_events):
        """Parse Deep Sleep / STATE / CHARGESTRING event info from one log.csv line."""
        # Process Deep Sleep events
        if 'Deep Sleep' in line:
            parts = line.strip().split(',')
            if len(parts) >= 1 and parts[0].strip().isdigit():
                log_events['deep_sleep'].append(int(parts[0].strip()))

        # Process STATE events
        # Example: "1750979974,STATE,CPUSpeed_12MHz,No Charge,VBUS=0,isDiskMounted=0,USBIsConnected=0,APP_DISK_Ready=1"
        elif ',STATE,' in line:
            parts = line.strip().split(',')
            if len(parts) >= 5 and parts[0].strip().isdigit():
                timestamp = int(parts[0].strip())

                # Parse CPU Speed
                if 'CPUSpeed_12MHz' in parts[2]:
                    if 'CPUSpeed_12MHz_ExternPower' in parts[2]:
                        log_events['cpu_speed']['12MHz_ExternPower'].append(timestamp)
                    else:
                        log_events['cpu_speed']['12MHz'].append(timestamp)
                elif 'CPUSpeed_300MHz' in parts[2]:
                    log_events['cpu_speed']['300MHz'].append(timestamp)

                # Parse Charge Rate
                if 'Fast Charge' in parts[3]:
                    log_events['charge_rate']['Fast'].append(timestamp)
                elif 'Slow Charge' in parts[3]:
                    log_events['charge_rate']['Slow'].append(timestamp)
                elif 'No Charge' in parts[3]:
                    log_events['charge_rate']['No'].append(timestamp)

                # Parse VBUS Status
                if 'VBUS=0' in parts[4]:
                    log_events['vbus']['0'].append(timestamp)
                elif 'VBUS=1' in parts[4]:
                    log_events['vbus']['1'].append(timestamp)

                # Parse additional information if available
                if len(parts) >= 6 and 'isDiskMounted=' in parts[5]:
                    if 'isDiskMounted=1' in parts[5]:
                        log_events['disk_mounted']['1'].append(timestamp)
                    elif 'isDiskMounted=0' in parts[5]:
                        log_events['disk_mounted']['0'].append(timestamp)

                if len(parts) >= 7 and 'USBIsConnected=' in parts[6]:
                    if 'USBIsConnected=1' in parts[6]:
                        log_events['usb_connected']['1'].append(timestamp)
                    elif 'USBIsConnected=0' in parts[6]:
                        log_events['usb_connected']['0'].append(timestamp)

                if len(parts) >= 8 and 'APP_DISK_Ready=' in parts[7]:
                    if 'APP_DISK_Ready=1' in parts[7]:
                        log_events['app_disk_ready']['1'].append(timestamp)
                    elif 'APP_DISK_Ready=0' in parts[7]:
                        log_events['app_disk_ready']['0'].append(timestamp)

        # Process CHARGESTRING IMU temperature entries
        if 'CHARGESTRING' in line and 'imu_temperature=' in line:
            try:
                parts = line.strip().split(',')
                if len(parts) >= 1 and parts[0].strip().isdigit():
                    timestamp = int(parts[0].strip())

                    temp_part = line.split('imu_temperature=')
                    if len(temp_part) > 1:
                        # Take only numeric characters and decimal point
                        temp_str = temp_part[1]
                        temp_val = ""
                        for char in temp_str:
                            if char.isdigit() or char == '.' or (char == '-' and not temp_val):
                                temp_val += char
                            else:
                                break

                        # Only proceed if we have a valid number
                        if temp_val and self.is_convertible(temp_val):
                            log_events['imu_temp_times'].append(timestamp)
                            log_events['imu_temp_values'].append(float(temp_val))
            except Exception as e:
                print(f"Error parsing temperature: {e}")

    def load_data(self):
        """Synchronous load (used by batch export and scripts). The GUI paths use
        start_threaded_load, which runs the same parser on a worker thread."""
        print('Loading data...')
        device_path = self.label_folder.text()

        # Use the helper function to find files with flexible naming
        sensor_data_path = find_file_with_suffix(device_path, "Sensor_Data.csv")
        log_data_path = find_file_with_suffix(device_path, "log.csv")
        
        if not sensor_data_path or not log_data_path:
            missing_files = []
            if not sensor_data_path:
                missing_files.append("Sensor_Data.csv")
            if not log_data_path:
                missing_files.append("log.csv")
                
            QMessageBox.critical(self, "Error", f"Required files ({', '.join(missing_files)}) are missing.\n\nMake sure files ending with these names exist in the selected folder.")
            return
        
        try:
            result = self.parse_folder_data(device_path)
            self.apply_parse_result(result, device_path)
        except Exception as e:
            print(f"Error during data loading: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error during data loading: {str(e)}")
            return

    def parse_folder_data(self, device_path, progress_cb=None, cancel_event=None):
        """Parse Sensor_Data.csv and log.csv, each in a single pass.

        GUI-free so it can run on a worker thread. progress_cb(percent, message)
        reports progress; cancel_event (threading.Event) aborts the parse, in
        which case None is returned. Returns a result dict for apply_parse_result.
        """
        settings = self.settings
        tz_offset = self.tz_offset
        warnings = []

        sensor_data_path = find_file_with_suffix(device_path, "Sensor_Data.csv")
        log_data_path = find_file_with_suffix(device_path, "log.csv")
        if not sensor_data_path or not log_data_path:
            raise FileNotFoundError("Required Sensor_Data.csv / log.csv files are missing")

        total_bytes = os.path.getsize(sensor_data_path) + os.path.getsize(log_data_path)
        processed_bytes = 0
        discarded = 0

        def report(message):
            if progress_cb:
                # File passes cover 0-90%; DataFrame building is the last 10%
                progress_cb(min(int(processed_bytes / max(total_bytes, 1) * 90), 90), message)

        # --- Pass 1: Sensor_Data.csv (was 5 keyword passes + 2 INF reads) ---
        sensor_keywords = ['IMU', 'PHO', 'TOF', 'ALS', 'INF']
        buckets = {kw: [] for kw in sensor_keywords}
        inf_line = None
        sensor_name = os.path.basename(sensor_data_path)
        with open(sensor_data_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line_no, line in enumerate(file):
                processed_bytes += len(line)
                if line_no % 5000 == 0:
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                    report(f"Reading {sensor_name}...")
                row = line.split(',', 2)
                if len(row) > 1:
                    field = row[1]
                    for keyword in sensor_keywords:
                        if keyword in field:
                            processed_line = self.process_line(line)
                            if processed_line:
                                buckets[keyword].append(processed_line)
                            else:
                                discarded += 1
                    if inf_line is None and ',INF,' in line:
                        inf_line = line.strip()

        # Extract device info from the first INF line (no extra file reads)
        firmware_info = additional_info = serial_number = None
        if inf_line:
            parts = inf_line.split(',')
            if len(parts) > 3:
                serial_number = parts[3].strip()
                print(f"Found serial number: {serial_number}")
            if len(parts) > 4:
                firmware_info = parts[4].strip()
                print(f"Found firmware info: {firmware_info}")
            if len(parts) >= 7:
                additional_info = ','.join(parts[-6:])
                print(f"Found additional info: {additional_info}")

        # --- Pass 2: log.csv (BAT/REBOOT/LOG rows + device state events) ---
        log_rows = {'BAT': [], 'LOG': []}
        reboot_entries = []
        log_events = {
            'deep_sleep': [],
            'cpu_speed': {'300MHz': [], '12MHz': [], '12MHz_ExternPower': []},
            'charge_rate': {'Fast': [], 'Slow': [], 'No': []},
            'vbus': {'1': [], '0': []},
            'disk_mounted': {'1': [], '0': []},
            'usb_connected': {'1': [], '0': []},
            'app_disk_ready': {'1': [], '0': []},
            'imu_temp_times': [],
            'imu_temp_values': [],
        }

        log_name = os.path.basename(log_data_path)
        with open(log_data_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line_no, line in enumerate(file):
                processed_bytes += len(line)
                if line_no % 5000 == 0:
                    if cancel_event is not None and cancel_event.is_set():
                        return None
                    report(f"Reading {log_name}...")

                for keyword in ('BAT', 'LOG'):
                    if keyword in line:
                        processed_line = self.process_line(line)
                        if processed_line:
                            log_rows[keyword].append(processed_line)
                        else:
                            discarded += 1

                # Special handling for REBOOT entries. Deliberately excludes the
                # config.json/calib.json lines the firmware dumps right after
                # each reboot (same timestamp, same burst) -- those aren't
                # reboot events themselves and previously inflated every real
                # reboot into 9 counted rows (1 REBOOT + 4 config.json + 4
                # calib.json), badly overcounting reboot stats and cluttering
                # the Battery Voltage plot with duplicate reboot markers.
                if ',REBOOT,' in line:
                    processed_line = self.process_line(line)
                    if processed_line:
                        items = processed_line.split(',')
                        if len(items) > 0 and self.is_convertible(items[0]):
                            reboot_entries.append((int(items[0]), items))

                # Device state / temperature events (previously re-parsed on every plot)
                self.parse_log_state_events(line, log_events)

        # Sort REBOOT entries by timestamp
        reboot_entries.sort(key=lambda x: x[0])

        # --- Build typed DataFrames ---
        if progress_cb:
            progress_cb(92, "Building data tables...")
        if cancel_event is not None and cancel_event.is_set():
            return None

        data = {}
        for keyword in sensor_keywords:
            data[keyword] = self.build_dataframe(buckets[keyword], keyword, tz_offset, warnings)
        data['BAT'] = self.build_dataframe(log_rows['BAT'], 'BAT', tz_offset, warnings)
        data['LOG'] = self.build_dataframe(log_rows['LOG'], 'LOG', tz_offset, warnings)
        data['REBOOT'] = self.build_dataframe([','.join(entry[1]) for entry in reboot_entries], 'REBOOT', tz_offset, warnings)

        # Apply filters if needed
        if settings['filters']['pre2023']:
            time_threshold = pd.Timestamp('2023-01-01', tz=tz_offset)

            for key in data:
                if not data[key].empty and 'time_stamp' in data[key].columns:
                    data[key] = data[key][data[key]['time_stamp'] >= time_threshold]

        if settings['filters']['time_range']:
            time_A = pd.Timestamp(settings['time_range']['A'], tz=tz_offset)
            time_B = pd.Timestamp(settings['time_range']['B'], tz=tz_offset)
            for key in data:
                if not data[key].empty and 'time_stamp' in data[key].columns:
                    data[key] = data[key][(data[key]['time_stamp'] >= time_A) & (data[key]['time_stamp'] <= time_B)]

        if discarded:
            warnings.append(f"{discarded} corrupted lines discarded (non-printable characters)")

        if progress_cb:
            progress_cb(100, "Done")

        return {
            'data': data,
            'log_events': log_events,
            'firmware_info': firmware_info,
            'additional_info': additional_info,
            'inf_line': inf_line,
            'serial_number': serial_number,
            'warnings': warnings,
        }

    def apply_parse_result(self, result, device_path):
        """Adopt a parse_folder_data result on the GUI thread and refresh the UI."""
        self.data = result['data']
        self.log_events = result['log_events']
        self.firmware_info = result['firmware_info']
        self.additional_info = result['additional_info']
        self.inf_line = result['inf_line']
        self.serial_number = result['serial_number']
        self.parse_warnings = result['warnings']

        self.device_folder = self.parseDeviceFolder(device_path)
        if self.device_folder == "Unknown_Device" and self.serial_number:
            self.device_folder = f"Unknown_Device (SN: {self.serial_number})"

        self.session_stats = self.compute_session_stats()
        self.update_summary_panel()
        self.update_stats_table()
        self.prefill_date_range()

        # Full gap list to the console (the panel shows only the longest few)
        if self.session_stats and self.session_stats['gaps']:
            gaps = self.session_stats['gaps']
            print(f"Detected {len(gaps)} data gaps (> {self.session_stats['gap_factor']:.0f}x expected interval):")
            for gap_start, gap_end in gaps:
                duration = (gap_end - gap_start).total_seconds()
                print(f"  {gap_start.tz_convert(self.tz_offset):%Y-%m-%d %H:%M:%S} → "
                      f"{gap_end.tz_convert(self.tz_offset):%Y-%m-%d %H:%M:%S}  ({self.format_duration(duration)})")

        qsettings = QSettings("VEET", "VEET_Data_Charting")
        qsettings.setValue("last_folder", device_path)
        print('Data loaded successfully')

    def get_configured_intervals(self):
        """Sensor sample intervals in seconds from the INF line, or None per sensor.

        The INF line ends with: tz_offset, four per-sensor interval values (ms),
        mode flag. The order of the four values isn't documented, so they are
        only used when all four agree (the normal case, e.g. 2000ms everywhere).
        Otherwise coverage falls back to each sensor's median observed interval.
        """
        sensors = ['ALS', 'PHO', 'TOF', 'IMU']
        result = dict.fromkeys(sensors)
        if self.inf_line:
            parts = [p.strip() for p in self.inf_line.split(',')]
            if len(parts) >= 7:
                try:
                    values_ms = [float(v) for v in parts[-5:-1]]
                    if len(set(values_ms)) == 1 and values_ms[0] > 0:
                        for sensor in sensors:
                            result[sensor] = values_ms[0] / 1000.0
                except ValueError:
                    pass
        return result

    def compute_session_stats(self):
        """Coverage, gap, reboot and deep-sleep statistics for the loaded session."""
        if not self.data:
            return None
        gap_factor = float(self.settings.get('gap_factor', 5))
        configured = self.get_configured_intervals()
        stats = {'sensors': {}, 'gaps': [], 'gap_factor': gap_factor}

        raw_gaps = []
        for sensor in ['ALS', 'PHO', 'TOF', 'IMU']:
            df = self.data.get(sensor)
            if df is None or df.empty or 'time_stamp' not in df.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df['time_stamp']):
                continue
            ts = df['time_stamp'].dropna().sort_values().reset_index(drop=True)
            if len(ts) < 2:
                continue
            diffs = ts.diff().dt.total_seconds().iloc[1:].reset_index(drop=True)

            interval = configured.get(sensor)
            interval_source = 'INF'
            if not interval:
                interval = float(diffs.median())
                interval_source = 'median'
            if not interval or interval <= 0:
                continue

            span_seconds = (ts.iloc[len(ts) - 1] - ts.iloc[0]).total_seconds()
            expected_samples = span_seconds / interval + 1
            coverage = min(100.0, len(ts) / expected_samples * 100.0)

            gap_mask = diffs > gap_factor * interval
            sensor_gaps = 0
            for i in gap_mask[gap_mask].index:
                raw_gaps.append((ts.iloc[i], ts.iloc[i + 1]))
                sensor_gaps += 1

            stats['sensors'][sensor] = {
                'count': len(ts),
                'interval': interval,
                'interval_source': interval_source,
                'coverage': coverage,
                'gap_count': sensor_gaps,
                'span_seconds': span_seconds,
            }

        # Merge overlapping per-sensor gaps into distinct device-level gaps
        raw_gaps.sort(key=lambda g: g[0])
        merged = []
        for gap_start, gap_end in raw_gaps:
            if merged and gap_start <= merged[-1][1] + pd.Timedelta(seconds=5):
                if gap_end > merged[-1][1]:
                    merged[-1][1] = gap_end
            else:
                merged.append([gap_start, gap_end])
        stats['gaps'] = [(s, e) for s, e in merged]

        # Reboots and mean uptime between them
        reboot_count = 0
        mean_uptime_seconds = None
        reboot_df = self.data.get('REBOOT')
        if (reboot_df is not None and not reboot_df.empty and 'time_stamp' in reboot_df.columns
                and pd.api.types.is_datetime64_any_dtype(reboot_df['time_stamp'])):
            reboot_ts = reboot_df['time_stamp'].dropna().sort_values()
            reboot_count = len(reboot_ts)
            if reboot_count >= 2:
                mean_uptime_seconds = float(reboot_ts.diff().dt.total_seconds().iloc[1:].mean())
        stats['reboot_count'] = reboot_count
        stats['mean_uptime_seconds'] = mean_uptime_seconds
        stats['deep_sleep_count'] = len(self.log_events.get('deep_sleep', []))

        stats['battery'] = self.compute_battery_stats()
        stats['light'] = self.compute_light_stats()
        stats['wear'] = self.compute_wear_stats()
        stats['tof'] = self.compute_tof_stats()
        return stats

    def get_tof_arrays(self):
        """TOF primary-return arrays reshaped for veet_tof_bands: (z_mm, conf_raw, times).

        z_mm and conf_raw have shape (n_frames, 8, 8). Zones S0..S63 are mapped
        row-major onto the 8x8 grid. Only the primary return (Dist1/Conf1) is
        used. Returns None when the TOF table or its columns are unavailable.
        """
        tof = self.data.get('TOF')
        if tof is None or tof.empty or 'time_stamp' not in tof.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(tof['time_stamp']):
            return None
        dist_cols = [f"S{i}_Dist1" for i in range(64)]
        conf_cols = [f"S{i}_Conf1" for i in range(64)]
        if not all(col in tof.columns for col in dist_cols + conf_cols):
            return None
        df = tof[['time_stamp'] + dist_cols + conf_cols].dropna(subset=['time_stamp'])
        df = df.sort_values('time_stamp').reset_index(drop=True)
        if df.empty:
            return None
        try:
            z_mm = df[dist_cols].to_numpy(dtype=float).reshape(-1, 8, 8)
            conf_raw = df[conf_cols].to_numpy(dtype=float).reshape(-1, 8, 8)
        except (TypeError, ValueError):
            return None
        return z_mm, conf_raw, df['time_stamp']

    def get_tof_arrays_full(self):
        """Both TOF returns reshaped to (n, 8, 8), for the array-file export.

        Returns a dict with 'times' plus 'z1_mm'/'conf1_raw' (primary return,
        required) and 'z2_mm'/'conf2_raw' (secondary return, None if the
        columns are absent). Zones S0..S63 map row-major onto the 8x8 grid.
        """
        tof = self.data.get('TOF')
        if tof is None or tof.empty or 'time_stamp' not in tof.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(tof['time_stamp']):
            return None

        def reshape(return_no):
            dist_cols = [f"S{i}_Dist{return_no}" for i in range(64)]
            conf_cols = [f"S{i}_Conf{return_no}" for i in range(64)]
            if not all(col in tof.columns for col in dist_cols + conf_cols):
                return None, None
            try:
                z = tof[dist_cols].to_numpy(dtype=float).reshape(-1, 8, 8)
                c = tof[conf_cols].to_numpy(dtype=float).reshape(-1, 8, 8)
            except (TypeError, ValueError):
                return None, None
            return z, c

        order = tof['time_stamp'].dropna().sort_values().index
        tof = tof.loc[order].reset_index(drop=True)
        if tof.empty:
            return None
        z1, c1 = reshape(1)
        if z1 is None:
            return None
        z2, c2 = reshape(2)
        return {
            'times': tof['time_stamp'],
            'z1_mm': z1, 'conf1_raw': c1,
            'z2_mm': z2, 'conf2_raw': c2,
        }

    def export_tof_arrays(self):
        """Create the standardized TOF array file (.npz) from the loaded session.

        The file carries everything veet_tof_bands' data contract needs —
        (n, 8, 8) distance and raw-confidence arrays for both returns, epoch
        timestamps, and a JSON metadata block (confidence encoding, sample
        period, zone-order convention, device identity) so downstream analysis
        never has to re-parse Sensor_Data.csv or guess session parameters.
        """
        arrays = self.get_tof_arrays_full()
        if arrays is None:
            QMessageBox.information(
                self, "Export TOF Arrays",
                "No TOF data loaded.\n\nLoad a device folder with TOF data first.")
            return

        clean_name = re.sub(r'[\\/*?:"<>|]', '_', self.device_folder or "session")
        default_path = os.path.join(self.label_folder.text() or "", f"{clean_name}_tof_arrays.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Export TOF Arrays", default_path,
                                              "NumPy archives (*.npz)")
        if not path:
            return

        try:
            times = arrays['times']
            timestamps_utc_s = np.array([ts.timestamp() for ts in times], dtype=np.float64)

            interval = self.get_configured_intervals().get('TOF')
            if not interval:
                interval = float(times.diff().dt.total_seconds().median())

            tof_bands = _import_tof_bands()
            encoding = getattr(tof_bands, 'CONFIDENCE_ENCODING', 'unknown') if tof_bands else 'unknown'

            sensor_path = find_file_with_suffix(self.label_folder.text(), "Sensor_Data.csv")
            meta = {
                'format_version': 1,
                'source_file': os.path.basename(sensor_path) if sensor_path else '',
                'serial_number': self.serial_number or '',
                'firmware': self.firmware_info or '',
                'sample_period_s': interval,
                'confidence_encoding': encoding,
                'zone_order': ('S0..S63 row-major onto (8, 8); axis order (row, col) '
                               '= (vertical, horizontal); FOV 41 deg H x 52 deg V'),
                'returns_included': [1] + ([2] if arrays['z2_mm'] is not None else []),
                'n_frames': int(arrays['z1_mm'].shape[0]),
                'units': {'z': 'mm', 'conf': 'raw 0-255', 'timestamps': 'seconds since epoch, UTC'},
                'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
            }

            payload = {
                'timestamps_utc_s': timestamps_utc_s,
                'z1_mm': arrays['z1_mm'].astype(np.float32),
                'conf1_raw': arrays['conf1_raw'].astype(np.float32),
                'meta_json': np.array(json.dumps(meta)),
            }
            if arrays['z2_mm'] is not None:
                payload['z2_mm'] = arrays['z2_mm'].astype(np.float32)
                payload['conf2_raw'] = arrays['conf2_raw'].astype(np.float32)

            np.savez_compressed(path, **payload)
            size_mb = os.path.getsize(path) / 1e6
            print(f"TOF arrays exported to {path} ({meta['n_frames']:,} frames, {size_mb:.1f} MB)")
            self.statusBar().showMessage(
                f"TOF arrays exported: {meta['n_frames']:,} frames, {size_mb:.1f} MB -> {path}", 8000)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Failed", f"Failed to write TOF array file: {e}")

    def compute_tof_stats(self):
        """Dioptric TOF statistics via veet_tof_bands (validity gate + binning).

        Follows the integration spec: the actual configured sample period is
        passed (never the module's 5s default), wear detection is applied
        upstream (frames in confirmed not-worn windows are excluded from the
        worn-only series), and zone_yield diagnostics are logged per session.
        """
        tof_bands = _import_tof_bands()
        if tof_bands is None:
            return None
        arrays = self.get_tof_arrays()
        if arrays is None:
            return None
        z_mm, conf_raw, times = arrays

        interval = self.get_configured_intervals().get('TOF')
        if not interval:
            interval = float(times.diff().dt.total_seconds().median())
        if not interval or interval <= 0:
            return None

        mask = tof_bands.analysis_mask(z_mm, conf_raw)
        n_frames = z_mm.shape[0]
        valid_frames = int(mask.any(axis=(1, 2)).sum())

        # Wear gating upstream of dioptric reporting: a not-worn device returns
        # valid near readings from whatever it is resting on
        excluded_not_worn = 0
        keep = np.ones(n_frames, dtype=bool)
        windows = self.compute_wear_windows()
        if windows is not None:
            not_worn_windows = set(windows['window_starts'][windows['monitored'] & ~windows['worn']])
            frame_windows = times.dt.floor(f'{self.WEAR_WINDOW_S}s')
            keep = ~frame_windows.isin(not_worn_windows).to_numpy()
            excluded_not_worn = int((~keep).sum())

        hist_all = tof_bands.dioptric_histogram(z_mm, conf_raw, interval)
        dose_all = tof_bands.dioptric_dose(z_mm, conf_raw, interval)
        if excluded_not_worn > 0:
            if keep.any():
                hist_worn = tof_bands.dioptric_histogram(z_mm[keep], conf_raw[keep], interval)
                dose_worn = tof_bands.dioptric_dose(z_mm[keep], conf_raw[keep], interval)
            else:
                hist_worn = np.zeros_like(hist_all)
                dose_worn = 0.0
        else:
            hist_worn = None
            dose_worn = None

        diopters = tof_bands.to_diopters(z_mm)[mask]
        median_vergence = float(np.median(diopters)) if diopters.size else None

        band_indices = tof_bands.band_index(z_mm)[mask]
        band_share = []
        for band in tof_bands.BANDS:
            count = int((band_indices == band.index).sum())
            z_hi = band.z_max_mm
            range_text = f"{band.z_min_mm:.0f}-{'inf' if z_hi == float('inf') else f'{z_hi:.0f}'} mm"
            pct = count / diopters.size * 100.0 if diopters.size else 0.0
            band_share.append((band.name, range_text, pct, count * interval / 3600.0))

        yields = tof_bands.zone_yield(z_mm, conf_raw)
        print(f"TOF zone yield: analysis_field {yields['analysis_field_yield']:.3f}, "
              f"perimeter {yields['perimeter_yield']:.3f}")

        return {
            'n_frames': n_frames,
            'valid_frames': valid_frames,
            'sample_period': interval,
            'dose_dh_all': dose_all,
            'dose_dh_worn': dose_worn,
            'hist_seconds_all': hist_all,
            'hist_seconds_worn': hist_worn,
            'bin_edges': tof_bands.D_BIN_EDGES,
            'median_vergence': median_vergence,
            'band_share': band_share,
            'yields': yields,
            'excluded_not_worn': excluded_not_worn,
        }

    # Wear-time detection parameters
    WEAR_WINDOW_S = 60                 # classification window length
    WEAR_STD_THRESHOLD = 0.05          # combined accel std below this = not worn (m/s^2)
    WEAR_MIN_SAMPLES_PER_WINDOW = 5    # windows with fewer samples are ignored

    def compute_wear_windows(self):
        """Per-window wear classification from IMU accelerometer variance,
        screened by charger state.

        The session is cut into WEAR_WINDOW_S windows; a window whose combined
        3-axis acceleration standard deviation is below WEAR_STD_THRESHOLD is
        classified "not worn" (device stationary), anything above it "worn".
        Windows inside data gaps are unmonitored and counted as neither.

        The device charges on wall power and cannot be worn while doing so, so
        any window where get_charging_state() reports the charger connected is
        forced out of "worn" regardless of accelerometer variance (e.g. a
        charging dock getting bumped shouldn't count as wear).

        Returns a dict of aligned per-window arrays (window_starts, worn,
        monitored, credit_s, charging, excluded_charging) for aggregation and
        for upstream gating of other sensors (e.g. excluding not-worn periods
        from TOF dioptric stats).
        """
        imu = self.data.get('IMU')
        accel_cols = ['Acc_x', 'Acc_y', 'Acc_z']
        if imu is None or imu.empty or 'time_stamp' not in imu.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(imu['time_stamp']):
            return None
        if not all(col in imu.columns for col in accel_cols):
            return None

        df = imu[['time_stamp'] + accel_cols].copy()
        for col in accel_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().sort_values('time_stamp').reset_index(drop=True)
        if len(df) < 2 * self.WEAR_MIN_SAMPLES_PER_WINDOW:
            return None

        interval = self.get_configured_intervals().get('IMU')
        if not interval:
            interval = float(df['time_stamp'].diff().dt.total_seconds().median())

        window_start = df['time_stamp'].dt.floor(f'{self.WEAR_WINDOW_S}s')
        grouped = df.groupby(window_start)
        stds = grouped[accel_cols].std()
        counts = grouped.size()

        monitored = counts >= self.WEAR_MIN_SAMPLES_PER_WINDOW
        combined_std = np.sqrt((stds ** 2).sum(axis=1))
        worn_by_motion = (monitored & (combined_std > self.WEAR_STD_THRESHOLD)).to_numpy()

        window_starts = pd.DatetimeIndex(monitored.index)
        charging, _ = self.get_charging_state(window_starts)
        if charging is None:
            charging = np.zeros(len(window_starts), dtype=bool)

        worn = worn_by_motion & ~charging
        excluded_charging = worn_by_motion & charging

        # Credit each window with its sampled time, capped at the window length
        credit_s = (counts * interval).clip(upper=self.WEAR_WINDOW_S)

        return {
            'window_starts': window_starts,
            'worn': worn,
            'monitored': monitored.to_numpy(),
            'credit_s': credit_s.to_numpy(dtype=float),
            'interval': interval,
            'window_s': self.WEAR_WINDOW_S,
            'charging': charging,
            'excluded_charging': excluded_charging,
        }

    def compute_wear_stats(self):
        """Aggregate per-window wear classification into session/daily totals."""
        windows = self.compute_wear_windows()
        if windows is None:
            return None
        worn = windows['worn']
        monitored = windows['monitored']
        credit_s = windows['credit_s']
        excluded_charging = windows['excluded_charging']

        worn_s = float(credit_s[worn].sum())
        not_worn_s = float(credit_s[monitored & ~worn].sum())
        monitored_s = worn_s + not_worn_s
        wear_pct = worn_s / monitored_s * 100.0 if monitored_s > 0 else None
        excluded_charging_s = float(credit_s[excluded_charging].sum())

        daily = []
        window_dates = windows['window_starts'].tz_convert(self.tz_offset).date
        for day in sorted(set(window_dates)):
            day_mask = window_dates == day
            day_worn = float(credit_s[day_mask & worn].sum()) / 3600.0
            day_monitored = float(credit_s[day_mask & monitored].sum()) / 3600.0
            daily.append((str(day), day_worn, day_monitored))

        return {
            'worn_hours': worn_s / 3600.0,
            'not_worn_hours': not_worn_s / 3600.0,
            'wear_pct': wear_pct,
            'excluded_charging_hours': excluded_charging_s / 3600.0,
            'daily': daily,
            'window_s': self.WEAR_WINDOW_S,
            'threshold': self.WEAR_STD_THRESHOLD,
        }

    # Battery analytics parameters
    BAT_GAP_PAIR_LIMIT_S = 600         # ignore sample pairs spanning > 10 min
    CHARGE_SLOPE_MV_PER_H = 20.0       # smoothed rise above this = on charger (heuristic)

    def get_charging_state(self, query_times):
        """Boolean array: is the device on charger power at each of query_times?

        VBUS events from log.csv are authoritative when present (firmware-
        reported USB/wall power connection). Falls back to the same smoothed
        BAT voltage-rise heuristic used in compute_battery_stats (rise faster
        than CHARGE_SLOPE_MV_PER_H) when no VBUS events exist. Returns
        (None, None) if neither source is available (e.g. no BAT data loaded).
        """
        if len(query_times) == 0:
            return np.zeros(0, dtype=bool), None
        query_epoch = np.array([ts.timestamp() for ts in query_times])

        vbus = self.log_events.get('vbus', {})
        vbus_events = sorted([(ts, 1) for ts in vbus.get('1', [])] +
                             [(ts, 0) for ts in vbus.get('0', [])])
        if vbus_events:
            event_times = np.array([e[0] for e in vbus_events], dtype=float)
            event_states = np.array([e[1] for e in vbus_events])
            idx = np.searchsorted(event_times, query_epoch, side='right') - 1
            # Before the first event, assume the opposite of that event's state
            state_before_first = 1 - vbus_events[0][1]
            states = np.where(idx >= 0, event_states[np.clip(idx, 0, None)], state_before_first)
            return states.astype(bool), "VBUS events"

        bat = self.data.get('BAT')
        if bat is None or bat.empty or 'time_stamp' not in bat.columns or 'Voltage' not in bat.columns:
            return None, None
        if not pd.api.types.is_datetime64_any_dtype(bat['time_stamp']):
            return None, None
        df = bat[['time_stamp', 'Voltage']].copy()
        df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
        df = df.dropna().sort_values('time_stamp').reset_index(drop=True)
        if len(df) < 3:
            return None, None

        t = df['time_stamp']
        v = df['Voltage']
        t_prev = t.iloc[:-1].reset_index(drop=True)
        t_next = t.iloc[1:].reset_index(drop=True)
        dt_hours = (t_next - t_prev).dt.total_seconds() / 3600.0
        v_smooth = v.rolling(9, center=True, min_periods=1).median()
        dv_smooth = (v_smooth.iloc[1:].reset_index(drop=True)
                     - v_smooth.iloc[:-1].reset_index(drop=True))
        with np.errstate(divide='ignore', invalid='ignore'):
            slope = dv_smooth / dt_hours
        pair_charging = (slope > self.CHARGE_SLOPE_MV_PER_H).to_numpy()
        pair_mid_epoch = np.array([ts.timestamp() for ts in (t_prev + (t_next - t_prev) / 2)])

        idx = np.searchsorted(pair_mid_epoch, query_epoch, side='right') - 1
        idx = np.clip(idx, 0, len(pair_charging) - 1)
        return pair_charging[idx], "voltage-rise heuristic"

    def compute_battery_stats(self):
        """Discharge rate, runtime estimate, charger vs battery time, charge cycles.

        Charger periods come from VBUS events in log.csv when present; otherwise
        a voltage-rise heuristic (smoothed slope > +20 mV/hr) classifies each
        sample pair. The mean discharge rate averages raw voltage deltas over
        on-battery pairs only, and the 3.4V estimate is a naive linear
        extrapolation from the latest voltage at that rate.
        """
        bat = self.data.get('BAT')
        if bat is None or bat.empty or 'time_stamp' not in bat.columns or 'Voltage' not in bat.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(bat['time_stamp']):
            return None
        df = bat[['time_stamp', 'Voltage']].copy()
        df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
        df = df.dropna().sort_values('time_stamp').reset_index(drop=True)
        if len(df) < 3:
            return None

        t = df['time_stamp']
        v = df['Voltage']
        t_prev = t.iloc[:-1].reset_index(drop=True)
        t_next = t.iloc[1:].reset_index(drop=True)
        dt_hours = (t_next - t_prev).dt.total_seconds() / 3600.0
        dv = v.iloc[1:].reset_index(drop=True) - v.iloc[:-1].reset_index(drop=True)
        valid = (dt_hours > 0) & (dt_hours <= self.BAT_GAP_PAIR_LIMIT_S / 3600.0)

        midpoints = t_prev + (t_next - t_prev) / 2
        charging_arr, charger_source = self.get_charging_state(midpoints)
        charging = pd.Series(charging_arr)

        if charger_source == "VBUS events":
            vbus = self.log_events.get('vbus', {})
            vbus_events = sorted([(ts, 1) for ts in vbus.get('1', [])] +
                                 [(ts, 0) for ts in vbus.get('0', [])])
            charge_cycles = sum(1 for i, (ts, state) in enumerate(vbus_events)
                                if state == 1 and (i == 0 or vbus_events[i - 1][1] == 0))
        else:
            # Charge cycles = distinct runs of >= 3 consecutive charging pairs
            charge_cycles = 0
            run_length = 0
            for flag in (charging & valid):
                if flag:
                    run_length += 1
                    if run_length == 3:
                        charge_cycles += 1
                else:
                    run_length = 0

        discharge_mask = valid & ~charging
        charge_mask = valid & charging
        time_on_battery_h = float(dt_hours[discharge_mask].sum())
        time_on_charger_h = float(dt_hours[charge_mask].sum())

        discharge_rate = None
        if time_on_battery_h > 0.1:
            discharge_rate = float(dv[discharge_mask].sum() / time_on_battery_h)

        v_last = float(v.tail(5).median())
        est_hours = None
        if discharge_rate is not None and discharge_rate < -1.0:
            est_hours = max(0.0, (v_last - 3400.0) / -discharge_rate)

        return {
            'discharge_rate_mv_per_h': discharge_rate,
            'est_hours_to_3v4': est_hours,
            'v_last': v_last,
            'time_on_battery_h': time_on_battery_h,
            'time_on_charger_h': time_on_charger_h,
            'charge_cycles': charge_cycles,
            'charger_source': charger_source,
        }

    def compute_light_stats(self, thresholds=(1000, 10000)):
        """Time above lux thresholds, cumulative lux-hours, and daily min/mean/max.

        Each ALS sample is credited with the time until the next sample, capped
        at 5x the expected interval so data gaps don't count as exposure time.
        """
        als = self.data.get('ALS')
        if als is None or als.empty or 'time_stamp' not in als.columns or 'lux' not in als.columns:
            return None
        if not pd.api.types.is_datetime64_any_dtype(als['time_stamp']):
            return None
        df = als[['time_stamp', 'lux']].copy()
        df['lux'] = pd.to_numeric(df['lux'], errors='coerce')
        df = df.dropna().sort_values('time_stamp').reset_index(drop=True)
        if len(df) < 2:
            return None

        interval = self.get_configured_intervals().get('ALS')
        observed = df['time_stamp'].diff().dt.total_seconds()
        if not interval:
            interval = float(observed.median())
        dt_seconds = observed.shift(-1).clip(upper=5 * interval).fillna(interval)
        lux = df['lux']

        minutes_above = {thr: float(dt_seconds[lux > thr].sum() / 60.0) for thr in thresholds}
        lux_hours = float((lux * dt_seconds).sum() / 3600.0)

        local_dates = df['time_stamp'].dt.tz_convert(self.tz_offset).dt.date
        daily = []
        for day, day_lux in lux.groupby(local_dates):
            daily.append((str(day), float(day_lux.min()), float(day_lux.mean()), float(day_lux.max())))

        return {
            'minutes_above': minutes_above,
            'lux_hours': lux_hours,
            'daily': daily,
            'thresholds': thresholds,
        }

    def format_duration(self, seconds):
        if seconds is None:
            return "n/a"
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f} min"
        if seconds < 86400:
            return f"{seconds / 3600:.1f} h"
        return f"{seconds / 86400:.1f} d"

    def format_session_stats(self, stats, max_gaps=5):
        """Multi-line stats text for the summary panel and the PNG stamp."""
        if not stats or not stats['sensors']:
            return ""
        lines = []

        first = next(iter(stats['sensors'].values()))
        coverage_parts = [f"{sensor} {s['coverage']:.1f}%" for sensor, s in stats['sensors'].items()]
        if first['interval_source'] == 'INF':
            header = f"Coverage (vs {first['interval'] * 1000:.0f}ms):"
        else:
            header = "Coverage (vs median interval):"
        lines.append(header + " " + "  ".join(coverage_parts))

        gaps = stats['gaps']
        if gaps:
            longest = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:max_gaps]
            gap_parts = [
                f"{self.format_duration((e - s).total_seconds())} @ {s.tz_convert(self.tz_offset):%m-%d %H:%M:%S}"
                for s, e in longest]
            more = f"  (+{len(gaps) - max_gaps} more)" if len(gaps) > max_gaps else ""
            lines.append(f"Gaps >{stats['gap_factor']:.0f}x interval: {len(gaps)}  |  "
                         + ";  ".join(gap_parts) + more)
        else:
            lines.append(f"Gaps >{stats['gap_factor']:.0f}x interval: none")

        battery = stats.get('battery')
        if battery and battery['discharge_rate_mv_per_h'] is not None:
            battery_line = f"Battery: {-battery['discharge_rate_mv_per_h']:.1f} mV/hr discharge"
            if battery['est_hours_to_3v4'] is not None:
                battery_line += f", ~{self.format_duration(battery['est_hours_to_3v4'] * 3600)} to 3.4V"
            battery_line += (f"; charger {self.format_duration(battery['time_on_charger_h'] * 3600)}"
                             f" / battery {self.format_duration(battery['time_on_battery_h'] * 3600)}"
                             f" ({battery['charge_cycles']} charge cycles)")
            lines.append(battery_line)

        light = stats.get('light')
        if light:
            threshold_parts = [
                f"{light['minutes_above'][thr]:.0f} min >{thr // 1000}k lux"
                for thr in light['thresholds']]
            lines.append("Light: " + ", ".join(threshold_parts)
                         + f", {light['lux_hours']:,.0f} lux·h")

        wear = stats.get('wear')
        if wear and wear['wear_pct'] is not None:
            monitored_hours = wear['worn_hours'] + wear['not_worn_hours']
            wear_line = (f"Wear: {wear['worn_hours']:.1f} h worn of "
                         f"{monitored_hours:.1f} h monitored ({wear['wear_pct']:.0f}%)")
            if wear['excluded_charging_hours'] > 0:
                wear_line += f"; {wear['excluded_charging_hours']:.1f} h excluded (charging)"
            lines.append(wear_line)

        tof = stats.get('tof')
        if tof:
            tof_line = f"TOF: {tof['dose_dh_all']:,.0f} D·h dose"
            if tof['median_vergence'] is not None:
                tof_line += f", median {tof['median_vergence']:.2f} D"
            tof_line += (f", {tof['valid_frames'] / tof['n_frames'] * 100:.0f}% frames valid"
                         f" (see TOF Dioptric Analysis)")
            lines.append(tof_line)

        reboot_line = f"Reboots: {stats['reboot_count']}"
        if stats['mean_uptime_seconds'] is not None:
            reboot_line += f" (mean uptime {self.format_duration(stats['mean_uptime_seconds'])})"
        reboot_line += f"   Deep sleep events: {stats['deep_sleep_count']}"
        lines.append(reboot_line)
        return "\n".join(lines)

    def _init_stats_dock(self):
        """Dockable session-statistics table with CSV export."""
        self.stats_table = QTableWidget(0, 3, self)
        self.stats_table.setHorizontalHeaderLabels(["Category", "Metric", "Value"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.btn_export_stats = QPushButton("Export stats to CSV", self)
        self.btn_export_stats.clicked.connect(self.export_stats_csv)

        dock_body = QWidget()
        dock_layout = QVBoxLayout(dock_body)
        dock_layout.setContentsMargins(4, 4, 4, 4)
        dock_layout.addWidget(self.stats_table)
        dock_layout.addWidget(self.btn_export_stats)

        self.stats_dock = QDockWidget("Session Statistics", self)
        self.stats_dock.setObjectName("session_stats_dock")
        self.stats_dock.setWidget(dock_body)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        self.stats_dock.hide()

        # View menu so the dock can be reopened after closing it
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.stats_dock.toggleViewAction())

    def build_stats_rows(self):
        """(category, metric, value) rows for the stats table and CSV export.

        This is the single source for session statistics output: extend this
        method to add new statistics and both the dock table and the CSV
        export pick them up automatically."""
        rows = []
        if self.serial_number:
            rows.append(("Device", "Serial number", str(self.serial_number)))
        if self.firmware_info:
            rows.append(("Device", "Firmware", self.firmware_info))

        extents = self.get_data_extents()
        if extents:
            t0 = extents[0].tz_convert(self.tz_offset)
            t1 = extents[1].tz_convert(self.tz_offset)
            rows.append(("Session", "Start", f"{t0:%Y-%m-%d %H:%M:%S}"))
            rows.append(("Session", "End", f"{t1:%Y-%m-%d %H:%M:%S}"))
            rows.append(("Session", "Span", self.format_duration((t1 - t0).total_seconds())))

        for key, df in self.data.items():
            if len(df):
                rows.append(("Records", key, f"{len(df):,}"))

        stats = self.session_stats
        if stats:
            for sensor, s in stats['sensors'].items():
                source = "INF" if s['interval_source'] == 'INF' else "median observed"
                rows.append((sensor, "Expected interval", f"{s['interval'] * 1000:.0f} ms ({source})"))
                rows.append((sensor, "Coverage", f"{s['coverage']:.2f} %"))
                rows.append((sensor, "Gaps", str(s['gap_count'])))

            gaps = stats['gaps']
            rows.append(("Gaps", f"Distinct gaps (>{stats['gap_factor']:.0f}x interval)", str(len(gaps))))
            for i, (gap_start, gap_end) in enumerate(gaps, 1):
                duration = (gap_end - gap_start).total_seconds()
                rows.append(("Gaps", f"Gap {i}",
                             f"{gap_start.tz_convert(self.tz_offset):%Y-%m-%d %H:%M:%S} → "
                             f"{gap_end.tz_convert(self.tz_offset):%H:%M:%S}  ({self.format_duration(duration)})"))

            rows.append(("Power", "Reboots", str(stats['reboot_count'])))
            rows.append(("Power", "Mean uptime between reboots",
                         self.format_duration(stats['mean_uptime_seconds'])))
            rows.append(("Power", "Deep sleep events", str(stats['deep_sleep_count'])))

            battery = stats.get('battery')
            if battery:
                if battery['discharge_rate_mv_per_h'] is not None:
                    rows.append(("Battery", "Mean discharge rate",
                                 f"{-battery['discharge_rate_mv_per_h']:.1f} mV/hr (on-battery segments)"))
                else:
                    rows.append(("Battery", "Mean discharge rate", "n/a"))
                if battery['est_hours_to_3v4'] is not None:
                    rows.append(("Battery", "Est. time to 3.4V",
                                 f"{self.format_duration(battery['est_hours_to_3v4'] * 3600)}"
                                 f" (from {battery['v_last']:.0f} mV, linear)"))
                else:
                    rows.append(("Battery", "Est. time to 3.4V", "n/a"))
                total_hours = battery['time_on_battery_h'] + battery['time_on_charger_h']

                def with_pct(hours):
                    share = f" ({hours / total_hours * 100:.0f}%)" if total_hours > 0 else ""
                    return self.format_duration(hours * 3600) + share

                rows.append(("Battery", "Time on battery", with_pct(battery['time_on_battery_h'])))
                rows.append(("Battery", "Time on charger", with_pct(battery['time_on_charger_h'])))
                rows.append(("Battery", "Charge cycles", str(battery['charge_cycles'])))
                rows.append(("Battery", "Charger detection", battery['charger_source']))

            light = stats.get('light')
            if light:
                for threshold in light['thresholds']:
                    rows.append(("Light (ALS)", f"Time >{threshold:,} lux",
                                 f"{light['minutes_above'][threshold]:.1f} min"))
                rows.append(("Light (ALS)", "Cumulative exposure", f"{light['lux_hours']:,.1f} lux·h"))
                for day, day_min, day_mean, day_max in light['daily']:
                    rows.append(("Light (ALS)", f"Daily {day}",
                                 f"min {day_min:,.0f} / mean {day_mean:,.0f} / max {day_max:,.0f} lux"))

            wear = stats.get('wear')
            if wear:
                pct_text = f" ({wear['wear_pct']:.0f}% of monitored)" if wear['wear_pct'] is not None else ""
                rows.append(("Wear (IMU)", "Worn time",
                             self.format_duration(wear['worn_hours'] * 3600) + pct_text))
                rows.append(("Wear (IMU)", "Not worn",
                             self.format_duration(wear['not_worn_hours'] * 3600)))
                if wear['excluded_charging_hours'] > 0:
                    rows.append(("Wear (IMU)", "Excluded (charging)",
                                 self.format_duration(wear['excluded_charging_hours'] * 3600)))
                for day, day_worn, day_monitored in wear['daily']:
                    rows.append(("Wear (IMU)", f"Daily {day}",
                                 f"{day_worn:.1f} h worn of {day_monitored:.1f} h monitored"))
                rows.append(("Wear (IMU)", "Detection",
                             f"{wear['window_s']}s windows, accel std < {wear['threshold']} m/s² = not worn; "
                             f"charging windows always forced not-worn"))

            tof = stats.get('tof')
            if tof:
                rows.append(("TOF (dioptric)", "Valid frames",
                             f"{tof['valid_frames']:,} / {tof['n_frames']:,}"
                             f" ({tof['valid_frames'] / tof['n_frames'] * 100:.1f}%)"))
                rows.append(("TOF (dioptric)", "Dioptric dose",
                             f"{tof['dose_dh_all']:,.1f} D·h (zone-weighted, all frames)"))
                if tof['dose_dh_worn'] is not None:
                    rows.append(("TOF (dioptric)", "Dioptric dose (worn only)",
                                 f"{tof['dose_dh_worn']:,.1f} D·h"
                                 f" ({tof['excluded_not_worn']:,} not-worn frames excluded)"))
                if tof['median_vergence'] is not None:
                    vergence_text = f"{tof['median_vergence']:.2f} D"
                    if tof['median_vergence'] > 0:
                        vergence_text += f" (≈{1000.0 / tof['median_vergence']:.0f} mm)"
                    rows.append(("TOF (dioptric)", "Median vergence", vergence_text))
                rows.append(("TOF (dioptric)", "Analysis-field yield",
                             f"{tof['yields']['analysis_field_yield'] * 100:.1f}%"))
                rows.append(("TOF (dioptric)", "Guard-band yield",
                             f"{tof['yields']['perimeter_yield'] * 100:.1f}%"))
                rows.append(("TOF (dioptric)", "Sample period", f"{tof['sample_period']:.1f} s"))
                for name, range_text, pct, hours in tof['band_share']:
                    if pct >= 0.05:
                        rows.append(("TOF (dioptric)", f"Band: {name}",
                                     f"{pct:.1f}%  ({hours:,.1f} zone·h, {range_text})"))

        for i, warning in enumerate(self.parse_warnings, 1):
            rows.append(("Warnings", f"Warning {i}", warning))
        return rows

    def update_stats_table(self, show=True):
        """Repopulate the stats dock table; optionally surface the dock."""
        rows = self.build_stats_rows()
        self.stats_table.setRowCount(len(rows))
        for row_index, (category, metric, value) in enumerate(rows):
            for col_index, text in enumerate((category, metric, value)):
                self.stats_table.setItem(row_index, col_index, QTableWidgetItem(str(text)))
        if rows and show:
            self.stats_dock.show()
            self.stats_dock.raise_()

    def export_stats_csv(self):
        rows = self.build_stats_rows()
        if not rows:
            QMessageBox.information(self, "No Statistics", "Load a device folder first.")
            return
        clean_name = re.sub(r'[\\/*?:"<>|]', '_', self.device_folder or "session")
        default_path = os.path.join(self.label_folder.text() or "", f"{clean_name}_stats.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export Statistics", default_path,
                                              "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["Category", "Metric", "Value"])
                writer.writerows(rows)
            print(f"Statistics exported to {path}")
            self.statusBar().showMessage(f"Statistics exported to {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to write CSV: {e}")

    def get_data_extents(self):
        """Earliest and latest timestamp across all loaded datasets, or None."""
        tmin = tmax = None
        for df in self.data.values():
            if df.empty or 'time_stamp' not in df.columns:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df['time_stamp']):
                continue
            col = df['time_stamp'].dropna()
            if col.empty:
                continue
            lo, hi = col.min(), col.max()
            if tmin is None or lo < tmin:
                tmin = lo
            if tmax is None or hi > tmax:
                tmax = hi
        return (tmin, tmax) if tmin is not None else None

    def update_summary_panel(self):
        """Show SN, firmware, data extents, record counts and warnings after load."""
        lines = []
        id_parts = []
        if self.serial_number:
            id_parts.append(f"SN {self.serial_number}")
        if self.firmware_info:
            id_parts.append(self.firmware_info)
        if id_parts:
            lines.append("  |  ".join(id_parts))

        extents = self.get_data_extents()
        if extents:
            t0 = extents[0].tz_convert(self.tz_offset)
            t1 = extents[1].tz_convert(self.tz_offset)
            span_hours = (t1 - t0).total_seconds() / 3600
            lines.append(f"{t0:%Y-%m-%d %H:%M} \u2192 {t1:%Y-%m-%d %H:%M}  ({span_hours:.1f} h)")

        counts = [f"{key}: {len(df):,}" for key, df in self.data.items() if len(df)]
        if counts:
            lines.append("   ".join(counts))

        if getattr(self, 'session_stats', None):
            stats_text = self.format_session_stats(self.session_stats)
            if stats_text:
                lines.append(stats_text)

        if self.parse_warnings:
            lines.append("\u26a0 " + "; ".join(self.parse_warnings))

        self.summary_label.setText("\n".join(lines) if lines else "No data loaded")

    def prefill_date_range(self):
        """Pre-fill the date-range picker with the loaded data extents."""
        extents = self.get_data_extents()
        if not extents:
            return
        for edit, ts in ((self.range_start_edit, extents[0]), (self.range_end_edit, extents[1])):
            local = ts.tz_convert(self.tz_offset)
            edit.blockSignals(True)
            edit.setDateTime(QDateTime.fromString(local.strftime('%Y-%m-%d %H:%M:%S'), 'yyyy-MM-dd HH:mm:ss'))
            edit.blockSignals(False)
        # Keep settings in sync with the freshly filled widgets (no rebuild)
        self.settings['range_start'] = self.range_start_edit.dateTime().toString('yyyy-MM-dd HH:mm:ss')
        self.settings['range_end'] = self.range_end_edit.dateTime().toString('yyyy-MM-dd HH:mm:ss')

    def show_tof_analysis(self):
        """Open the TOF dioptric analysis window from the current session stats."""
        tof_stats = self.session_stats.get('tof') if self.session_stats else None
        if not tof_stats:
            QMessageBox.information(
                self, "TOF Analysis",
                "No TOF dioptric statistics available.\n\nLoad a device folder with "
                "TOF data first, and make sure veet_tof_bands.py is next to this script.")
            return
        self.tof_window = TOFAnalysisWindow(tof_stats, self.device_folder)
        self.tof_window.show()

    def update_selected_plots(self):
         self.selected_plots = [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def show_plot_window(self):
        # If device_folder is still "Unknown_Device", make one last attempt to get serial number
        if self.device_folder == "Unknown_Device":
            serial_number = self.get_serial_number_from_inf(self.label_folder.text())
            if serial_number:
                self.device_folder = f"Unknown_Device (SN: {serial_number})"
        
        # Show loading indicator and terminal feedback for plot generation
        plot_types = [label for selected, label in zip(self.selected_plots, self.plot_labels) if selected]
        print(f"Generating plots: {', '.join(plot_types)}")
        print("Creating interactive plot windows...")
        
        # Create and show progress dialog for plot generation
        progress = QProgressDialog("Generating interactive plots...\nThis may take several seconds for large datasets or multiple plots.", "Cancel", 0, 0, self)
        progress.setWindowTitle("Generating VEET Plots")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)  # Disable cancel for now
        progress.show()
        
        # Process events to make sure dialog shows immediately
        QApplication.processEvents()
        
        try:
            self.plot_window = PlotWindow(
                self.selected_plots, 
                self.plot_labels, 
                self.settings,
                self.tz_offset, 
                self.data, 
                self.device_folder,
                self.ALS_wave_col_names, 
                self.PHO_wave_col_names,
                self.ALSgainRatios, 
                self.PHOgainRatios, 
                self.phoToScale, 
                self.phoScales,
                self.firmware_info, 
                self.additional_info,
                self.inf_line,
                self.serial_number,
                log_events=self.log_events,
                wear_windows=self.compute_wear_windows()
            )
            progress.close()
            print("Plot windows generated successfully")
            self.plot_window.show()
        except Exception as e:
            progress.close()
            print(f"Error generating plots: {e}")
            QMessageBox.warning(self, "Plot Error", f"Error generating plots: {str(e)}")

def main():
    app = QApplication(sys.argv)

    # Show a splash screen immediately; the heavy library imports and UI
    # construction happen behind it with status text so startup never looks hung
    pixmap = QPixmap(420, 160)
    pixmap.fill(QColor('#f0f0f0'))
    splash = QSplashScreen(pixmap)
    splash.showMessage("VEET Data Charting\n\nLoading libraries...",
                       Qt.AlignCenter, QColor('#303030'))
    splash.show()
    app.processEvents()

    _ensure_core_modules()

    splash.showMessage("VEET Data Charting\n\nBuilding interface...",
                       Qt.AlignCenter, QColor('#303030'))
    app.processEvents()

    main_window = MainWindow()
    main_window.show()
    splash.finish(main_window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()