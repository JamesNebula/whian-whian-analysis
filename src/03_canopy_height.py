# Generate canopy height model and ecological metrics from normalized point heights
# Identify high value conservation zones based on structural complexity

import laspy 
import numpy as np
import rasterio 
from rasterio.transform import from_origin
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = Path("data/raw/whian.laz")
DTM_PATH = Path("outputs/dtm/whian_whian_dtm_2m.tif")
OUTPUT_CHM_PATH = Path("outputs/chm/whian_whian_chm_2m.tif")
OUTPUT_COMPLEXITY_PATH = Path("outputs/chm/whian_whian_complexity_2m.tif")
OUTPUT_CONSERVATION_PATH = Path("outputs/chm/whian_whian_conservation_priority.tif")
OUTPUT_3D_HTML = Path("outputs/visualizations/canopy_3d.html")
OUTPUT_REPORT_PATH = Path("outputs/reports/chm_analysis_report.json")
OUTPUT_PREVIEW_PATH = Path("outputs/previews/chm_preview.png")

# Create output directories
for p in [OUTPUT_CHM_PATH, OUTPUT_COMPLEXITY_PATH, OUTPUT_CONSERVATION_PATH, 
          OUTPUT_3D_HTML, OUTPUT_REPORT_PATH, OUTPUT_PREVIEW_PATH]:
    p.parent.mkdir(parents=True, exist_ok=True)

# CRS for whian
CRS_EPSG = 7856

def load_dtm(dtm_path):
    print("     Loading DTM...")
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    print(f"    DTM dimensions: {dtm.shape[1]} x {dtm.shape[0]} pixels")
    print(f"    Elevation range: {np.nanmin(dtm):.1f}m - {np.nanmax(dtm):.1f}m")
    return dtm, transform, crs, bounds

def normalize_heights(las, dtm, transform):
    # Normalize point height above ground using DTM
    print("\nNormalizing point heights above ground...")

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    # convert XY coords to pixel indices in DTM
    cols = ((x - transform.c) / transform.a).astype(int)
    rows = ((y - transform.f) / transform.e).astype(int)

    # clip indices to valid DTM range
    cols = np.clip(cols, 0, dtm.shape[1] - 1)
    rows = np.clip(rows, 0, dtm.shape[0] - 1)

    # extract dtm elevation at each point location
    ground_elevations = dtm[rows, cols]

    # calc height above ground
    height_above_ground = z - ground_elevations

    # Filter out invalid heights (neg values = ground/below-ground points)
    valid_mask = height_above_ground >= 0
    normalized_heights = height_above_ground[valid_mask]
    normalized_x = x[valid_mask]
    normalized_y = y[valid_mask]
    normalized_z = z[valid_mask]

    print(f"    Points above ground: {len(normalized_heights):,} ({len(normalized_heights)/len(las)*100:.1f}%)")
    print(f"    Height Range: {normalized_heights.min():.1f}m - {normalized_heights.max():.1f}m")
    print(f"    Mean canopy height: {normalized_heights.mean():.1f}m")

    return normalized_x, normalized_y, normalized_heights, normalized_z

def generate_chm(normalized_x, normalized_y, normalized_heights, dtm_shape, transform, resolution=2.0):
    print("\n Generating Canopy Height Model...")

    # Convert coords to pixel space
    cols = ((normalized_x - transform.c) / transform.a).astype(int)
    rows = ((normalized_y - transform.f) / transform.e).astype(int)
    cols = np.clip(cols, 0, dtm_shape[1] - 1)
    rows = np.clip(rows, 0, dtm_shape[0] - 1)

    # Initialize CHM with zeros (ground level)
    chm = np.zeros(dtm_shape, dtype=np.float32)

    # Rasterize using max height per pixel
    for col, row, height in zip(cols, rows, normalized_heights):
        if height > chm[row, col]:
            chm[row, col] = height

    # Fill small gaps using 3x3 maximum filter
    chm_filled = ndimage.maximum_filter(chm, size=3)
    chm[chm == 0] = chm_filled[chm == 0]

    # Quality metrics
    valid_pixels = np.sum(chm > 0)
    coverage_pct = (valid_pixels / chm.size) * 100

    print(f"    CHM coverage: {coverage_pct:.1f}%")
    print(f"    Max canopy height: {chm.max():.1f}m")
    print(f"    Mean canopy height: {chm[chm > 0].mean():.1f}m")

    return chm

def calculate_vertical_complexity(normalized_x, normalized_y, normalized_heights, dtm_shape, transform, resolution=2.0):
    # Calculate the vertical height complexity index (VCI)
    # Count height strata per 2m squared pixel, higher strata count = higher biodiversity potential

    print("\n Calculating Vertical Complexity Index...")

    # Define height strata thresholds
    strata_thresholds = [0, 2, 5, 10, 20, 100]

    # Convert coords to pixel space
    cols = ((normalized_x - transform.c) / transform.a).astype(int)
    rows = ((normalized_y - transform.f) / transform.e).astype(int)
    cols = np.clip(cols, 0, dtm_shape[1] - 1)
    rows = np.clip(rows, 0, dtm_shape[0] - 1)

    # Initialize complexity raster
    complexity = np.zeros(dtm_shape, dtype=np.uint8)

    # For each pixel, count strata present
    pixel_dict = {}
    for col, row, height in zip(cols, rows, normalized_heights):
        key = (row, col)
        if key not in pixel_dict:
            pixel_dict[key] = []
        pixel_dict[key].append(height)

    # Calc strata count per pixel
    for (row, col), heights in pixel_dict.items():
        strata_present = 0
        for i in range(len(strata_thresholds) - 1):
            lower = strata_thresholds[i]
            upper = strata_thresholds[i + 1]
            if any(lower <= h < upper for h in heights):
                strata_present += 1
        complexity[row, col] = strata_present

    complexity_filled = ndimage.maximum_filter(complexity, size=5)
    complexity[complexity == 0] = complexity_filled[complexity == 0]

    print(f"    Complexity Range: {complexity.min()} - {complexity.max()} strata")
    print(f"    Mean complexity: {complexity[complexity > 0].mean():.1f} strata/pixel")

    return complexity

def identify_conservation_priority(chm, complexity):
    # Identify high-priority conservation zones using multi-criteria analysis
    print("\nIdentifying Conservation Priority Zones...")

    # Criteria weights
    height_weight = 0.4 # tall trees
    complexity_weight = 0.6 # structural complexity

    # Normalize inputs to 0-1 scale
    chm_norm = (chm - chm.min()) / (chm.max() - chm.min() + 1e-6)
    complexity_norm = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)

    # Calculate priority score
    priority = (height_weight * chm_norm) + (complexity_weight * complexity_norm)

    # Classify into priority levels (0-100scale)
    priority_score = (priority * 100).astype(np.uint8)

    # Create categorical map: 1=Low, 2=Medium, 3=High, 4=Critical
    conservation_map = np.zeros_like(priority_score, dtype=np.uint8)
    conservation_map[priority_score >= 75] = 4 #critical
    conservation_map[(priority_score >= 50) & (priority_score < 75)] = 3 #high
    conservation_map[(priority_score >= 25) & (priority_score < 50)] = 2 #medium
    conservation_map[priority_score < 25] = 1 #low

    # Stats
    total_pixels = conservation_map.size
    critical_area = np.sum(conservation_map == 4)
    high_area = np.sum(conservation_map == 3)

    print(f"    Critical priority zones: {critical_area/total_pixels*100:.1f}% ({critical_area*4/10000:.1f} ha)")
    print(f"    High priority zones: {high_area/total_pixels*100:.1f}% ({high_area*4/10000:.1f} ha)")

    return conservation_map, priority_score
