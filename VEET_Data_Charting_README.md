# VEET Data Charting Tool v1.14

A comprehensive interactive visualization tool for VEET (Wearable Environmental Exposure Tracker) device data analysis. This Python application processes the CSV data files from the device and generates detailed, interactive plots for comprehensive sensor data analysis.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-pandas%2C%20matplotlib%2C%20PySide6-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### **Enhanced GUI Interface**
- User-friendly graphical interface with progress dialogs for data loading and plot generation
- Visual feedback during long operations to improve user experience
- Intuitive checkbox selection for plot types
- Advanced time range and timezone controls
- Real-time configuration options

### **Nine Comprehensive Plot Types**
1. **Battery Voltage** - Comprehensive device state markers and power analysis
2. **Battery Voltage Rate of Change** - Discharge rate analysis over time
3. **TOF (Time-of-Flight) Distance** - Average distance measurements
4. **ALS Lux** - Calculated luminance values with optional log scaling
5. **ALS Basic Counts** - Gain & integration time normalized wavelength sensor data (<400nm, 400-700nm, >700nm)
6. **PHO Basic Counts** - Gain & integration time normalized photosensor data (415-910nm)
7. **IMU Translational Acceleration** - 3-axis accelerometer data
8. **System Temperature** - Temperature readings from IMU sensor
9. **Sensor Sample Interval** - Timing analysis showing sensor acquisition rates

### **Advanced Battery Analysis**
- Deep Sleep events visualization
- CPU speed change indicators (300MHz/12MHz modes)
- Charge rate status (Fast/Slow/No charge)
- VBUS connection monitoring
- USB connection state tracking
- Disk mount status indicators
- App Disk Ready state markers
- Reboot event detection

### **Interactive Features**
- **Ctrl+Click** anywhere on plots to copy coordinates to clipboard
- Zoom with axis-specific controls (hold Y or X while zooming)
- Draggable legends for optimal viewing
- Interactive PHO Basic Counts legends - click to show/hide data series
- Real-time coordinate display on hover

### **Data Processing & Visualization**
- Flexible file handling with various naming conventions
- Automatic gain and integration time normalization for ALS and PHO data
- Logarithmic or linear scale options on ALS and PHO data
- Advanced time range controls with offset capabilities
- Timezone adjustment (GMT-12 to GMT+14)
- Batch processing for multiple device folders (depricated)

### **Export Capabilities**
- Individual plot export as PNG images
- Batch export for multiple device datasets (depricated)
- Automatic filename generation from plot titles

## 📋 Requirements

### Dependencies
```
pandas>=1.3.0
matplotlib>=3.4.0
PySide6>=6.0.0
numpy>=1.20.0
pytz>=2021.1
```
See full requirements list in requirements.txt in python script folder

### Python Version
- Python 3.7 or higher

## 🛠️ Installation

1. **Clone or download** this repository to your local machine
2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python VEET_Data_Charting_1.14.py
   ```

## 📁 Data File Requirements

The script expects the following files in your device data folder:

```
Device_Folder/
├── *log.csv              # Any file ending with "log.csv"
└── *Sensor_Data.csv      # Any file ending with "Sensor_Data.csv"
```

**Example valid filenames:**
- `2025-08-03 23-46 log.csv`
- `2025-08-03 23-46 Sensor_Data.csv`
- `device_001_log.csv`
- `device_001_Sensor_Data.csv`

Note:  the script can also directly read from a VEET device drive, however the 
data transfer time greatly increases load time.

## 🖥️ Usage Guide

### **Quick Start**
1. **Launch** the application
2. **Click** "Select Folder & Load Data" button
3. **Browse** to your device data folder containing the CSV files
4. **Select** desired plot types using checkboxes
5. **Configure** timezone and time range options (as needed)
6. **Click** "Show Plot" to generate visualizations

### **Advanced Configuration**

#### **Time Range Controls**
- **Hours to display**: Limit the X-axis to a specific time window
- **Hours offset from end**: Start the display window before the latest data point
- **Timezone**: Select from GMT-12 to GMT+14 for proper time representation

#### **Scaling Options**
- **Log Scale**: Available for ALS Lux, ALS Basic Counts, and PHO Basic Counts
- Enhanced visualization of data with wide dynamic ranges

#### **Export Options**
- **Individual Export**: Save current plot as PNG image
- **Batch Export**: Process multiple device folders automatically (depricated feature)

## 📊 Plot Information Sources

### **Plot Titles**
- Automatically derived from the selected folder name
- Falls back to "Unknown_Device" with serial number if folder name is indeterminate

### **System Information Box**
- **Firmware Version**: Extracted from 5th column of INF line in Sensor_Data.csv
- **Device Configuration**: Sensor intervals and settings from last 6 values of INF line
- **Automatic Extraction**: Data parsed from the first INF line found in Sensor_Data.csv

## 🎯 Interactive Plot Features

### **Navigation**
- **Zoom**: Use matplotlib toolbar controls.  Left mouse click zooms, Right mouse click expands
- **Axis-Specific Zoom**: Hold Y or X key while zooming to only adjust that axis
- **Pan**: Click and drag to move around the plot. 
- **Axis-Specific Pan**: Hold Y or X key while panning to only adjust that axis 

### **Data Inspection**
- **Coordinate Copy**: Ctrl+Click anywhere to copy coordinates to clipboard
- **Hover Display**: Real-time coordinate display in status bar
- **Legend Interaction**: Click PHO legend items to toggle data series

### **Legend Management**
- **Draggable**: Click and drag legends to optimal positions
- **Interactive**: Toggle data series visibility (PHO Basic Counts)
- **Organized Markers**: Battery plot includes comprehensive device state indicators

## ⚙️ Configuration Options

The script includes extensive configurability through the settings dictionary:

### **Data Processing** (Development Controls Only)
- `applyALSgain`: Enable/disable ALS gain normalization
- `scalePHO`: Enable/disable PHO scaling
- `applyPHOgain`: Enable/disable PHO gain normalization

### **Visualization** (Development Controls Only)
- `useLogScale`: Default scaling behavior for different plot types
- `plot_options`: Line width, marker style, and size customization
- `time_zone`: Default timezone offset settings

### **Filtering** (Development Controls Only)
- `filters`: Data filtering options for specific time ranges
- `conf_thresh`: Confidence threshold settings

## 🔧 Troubleshooting

### **Common Issues**

**Data not plotting:**
- Check filter settings in the configuration
- Verify file naming matches expected patterns (*log.csv, *Sensor_Data.csv)
- Ensure files are not empty or corrupted

**Performance issues:**
- Large datasets may take several seconds to load
- Plot generation time increases with multiple plots
- Consider limiting time range for better performance

**Missing plots:**
- Ensure corresponding data exists in the CSV files
- Check for data in the expected columns
- Verify sensor data is properly formatted

### **File Format Requirements**

**Sensor_Data.csv:**
- Must contain INF line for device information
- Column structure should match expected headers
- Timestamps should be in Unix epoch format

**log.csv:**
- Event logging data for battery analysis
- STATE entries for device state visualization
- Compatible with Deep Sleep and reboot event detection

## 📝 Version History

### v1.14 (Current)
- **FIXED**: Integration time normalization for both ALS and PHO basic counts calculations
- **IMPROVED**: PHO and ALS basic counts now properly account for both gain and integration time
- **ENHANCED**: True normalized basic counts ensure comparable measurements across different sensor configurations
- **UPDATED**: Documentation to reflect gain & integration time normalization

### v1.13
- Enhanced GUI interface improvements
- Updated plot type naming (ALS Basic Counts, PHO Basic Counts)
- Improved system information extraction
- Enhanced battery state visualization
- Interactive legend functionality for PHO plots
- Flexible file naming support

## 🔗 Distribution Locations

### Packages


## 🤝 Contributing

For bug reports, feature requests, or contributions, please contact the development team @ Sherpa Design.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Tools


## ⚡ Performance Notes

- **Data Loading**: Can take several seconds for large datasets
- **Plot Generation**: Processing time increases with multiple plots and complex visualizations
- **Memory Usage**: Keep available RAM in mind for very large datasets
- **Export Operations**: Batch exports may take time depending on number of devices

## 📧 Support

For technical support or questions about VEET data analysis, please refer to the project documentation or contact the development team.


