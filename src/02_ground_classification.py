# The next step is to generate a DTM from the ground points. In the Whian dataset the ground point percentage was (6.1%) which isn't alot.
# The elevation range was steep 186m - 389m (from sea level)
# area of dataset is 400ha or 2km x 2km

import laspy 
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
RAW_DATA_PATH = Path("data/raw/whian.laz")
OUTPUT_DTM_PATH = Path("outputs/dtm/whian_whian_dtm_2m.tif")
OUTPUT_REPORT_PATH = Path("outputs/reports/dtm_generation_report.json")
OUTPUT_PREVIEW_PATH = Path("outputs/previews/dtm_preview.png")

# output dirs
OUTPUT_DTM_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

#CRS for Whian Whian
CRS_EPSG = 7856 # GDA2020 / MGA Zone 56 (NSW standard)

def extract_ground_points(las):
    print("Extracting ground points...")

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    classification = np.array(las.classification)

    # filter ground points (class 2)
    ground_mask = classification == 2
    ground_x = x[ground_mask]
    ground_y = y[ground_mask]
    ground_z = z[ground_mask]

    print(f"    Total points: {len(las):,}")
    print(f"    Ground points: {len(ground_x):,} ({len(ground_x)/len(las)*100:.1f}%)")

    # Ensure there is sufficient ground points
    if len(ground_x) < 1000:
        raise ValueError(f"Insufficient ground points: {len(ground_x):,}. Cannot generate reliable DTM")
    
    # Check spatial distribution
    print(f"    Ground point density: {len(ground_x) / ((x.max()-x.min()) * (y.max()-y.min())):.3f} pts/m2")

    return ground_x, ground_y, ground_z, x.min(), x.max(), y.min(), y.max()

def generate_dtm_grid(min_x, max_x, min_y, max_y, resolution=2.0):
    # Create regular grid for DTM interpolation
    print(f"Creating {resolution}m resolution grid...")

    # Create grid coordinates (2m resolution for stability with sparse data)
    grid_x = np.arange(min_x, max_x + resolution, resolution)
    grid_y = np.arange(min_y, max_y + resolution, resolution)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    print(f"    Grid dimensions: {grid_xx.shape[1]} x {grid_yy.shape[0]} pixels")
    print(f"    Total pixels: {grid_xx.size:,}")

    return grid_xx, grid_yy, resolution