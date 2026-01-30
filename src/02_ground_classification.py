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

def interpolate_dtm(ground_x, ground_y, ground_z, grid_xx, grid_yy, method='linear'):
    # Interpolate ground points to regular grid

    print(f"Interpolating ground surface using '{method}' method")

    # Flatten grid
    grid_points = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))
    ground_points = np.column_stack((ground_x, ground_y))

    # Interpolate (using fill_value=np.nan) to identify areas without ground data
    interpolated = griddata(
        points=ground_points,
        values=ground_z,
        xi = grid_points,
        method=method,
        fill_value=np.nan
    )

    # Reshape to 2D grid
    dtm = interpolated.reshape(grid_xx.shape)

    # Quality metrics
    valid_pixels = np.sum(~np.isnan(dtm))
    coverage_pct = (valid_pixels / dtm.size) * 100

    print(f"    Valid Pixels: {valid_pixels:,} ({coverage_pct:.1f}% coverage)")

    if coverage_pct < 80:
        print(f"    Low coverage ({coverage_pct:.1f}%) - filling gaps with nearest neighbour...")

        filled = griddata(
            points=ground_points,
            values=ground_z,
            xi=grid_points,
            method='nearest',
            fill_value=ground_z.mean()
        )
        dtm[np.isnan(dtm)] = filled[np.isnan(dtm)]
        print(f"    Gap-filled coverage: 100%")
    
    return dtm

def save_dtm_geotiff(dtm, min_x, max_y, resolution, crs_epsg, output_path):
    print(f"Saving DTM to GeoTIFF...")

    height, width = dtm.shape

    # Create affine transform (top-left origin)
    transform = from_origin(min_x, max_y, resolution, resolution)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtm.dtype,
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
        nodata=np.nan,
        compress='lzw'
    ) as dst:
        dst.write(dtm, 1)

    print(f"    Saved: {output_path}")
    print(f"    CRS: EPSG: {crs_epsg} (GDA2020 MGA ZONE 56)")
    print(f"    Resolution: {resolution}m")

def create_dtm_preview(dtm, output_path, min_x, max_y, resolution):
    print("Generating DTM preview...")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot DTM with terrain appropriate colormap
    im = ax.imshow(
        dtm, 
        cmap='terrain',
        origin='upper',
        extent=[min_x, min_x + dtm.shape[1] * resolution, 
                max_y - dtm.shape[0] * resolution, max_y], # type: ignore
        interpolation='bilinear'
    )

    # Add contour lines for terrain detail
    contours = ax.contour(
        dtm, 
        levels=np.arange(np.nanmin(dtm), np.nanmax(dtm), 10), #10m contours
        colors='black',
        alpha=0.3,
        lindwidths=0.5,
        origin='upper',
        extent=[min_x, min_x + dtm.shape[1] * resolution,
                max_y - dtm.shape[0] * resolution, max_y]
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt='%d')

    ax.set_title('Whian Whian Digital Terrain Model\n2m Resolution | GDA2020 MGA Zone 56',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, pad=0.1, fraction=0.025)
    cbar.set_label('Elevation (m ASL)', rotation=270, labelpad=25, fontsize=12)

    # Add scale bar
    scale_bar_length = 500 # 500 meters
    scale_bar_x = min_x + 200
    scale_bar_y = max_y - (dtm.shape[0] * resolution) + 100

    ax.plot([scale_bar_x, scale_bar_x + scale_bar_length],
            [scale_bar_y, scale_bar_y],
            'k-', linewidth=3)
    ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 50,
            f'{scale_bar_length}m',
            ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"    Preview Saved: {output_path}")

def validate_dtm(dtm, original_min_z, original_max_z):
    # Validate DTM against original point cloud stats
    print("Validating DTM quality...")

    dtm_min = np.nanmin(dtm)
    dtm_max = np.nanmax(dtm)
    dtm_mean = np.nanmean(dtm)
    dtm_std = np.nanstd(dtm)

    print(f"    DTM Elevation Range: {dtm_min:.2f}m - {dtm_max:.2f}m")
    print(f"    Original Point Range: {original_min_z:.2f}m - {original_max_z:.2f}m")

    # Validation checks
    issues = []
    if dtm_min < original_min_z - 5:
        issues.append(f"DTM minimum ({dtm_min:.1f}m) is {original_min_z - dtm_min:.1f}m below raw data minimum")
    if dtm_max > original_max_z + 5:
        issues.append(f"DTM Maximum ({dtm_max:.1f}m) is {dtm_max - original_max_z:.1f}m above raw data maximum")

    # Terrain roughness check (steeper areas have higher std)
    if dtm_std < 10:
        print(f"    Low terrain variation (σ={dtm_std:.1f}m) - unexpected for Nightcap Range")
    else:
        print(f"    Healthy terrain variation (σ={dtm_std:.1f}m) - consistent with mountainous rainforest")
    
    if not issues:
        print("     DTM validation passed")
    else:
        print("     Validation warnings:")
        for issue in issues:
            print(f"    Issue: {issue}")
    
    return {
        "dtm_min_m": float(dtm_min),
        "dtm_max_m": float(dtm_max),
        "dtm_mean_m": float(dtm_mean),
        "dtm_std_m": float(dtm_std),
        "validation_issues": issues,
        "validation_passed": len(issues) == 0
    }

def save_generation_report(report_data, output_path):
    print("\nSaving generation report...")

    report = {
        "project": "Whian Whian Forest Analysis",
        "date_processed": datetime.now().isoformat(),
        "input_data": {
            "filename": "whian.laz",
            "total_points": 6261585,
            "ground_points_extracted": report_data["ground_points_count"],
            "ground_point_percentage": report_data["ground_point_percentage"]
        },
        "dtm_parameters": {
            "resolution_m": report_data["resolution"],
            "interpolation_method": report_data["interpolation_method"],
            "grid_dimensions": f"{report_data['grid_width']} x {report_data['grid_height']} pixels",
            "crs_epsg": CRS_EPSG
        },
        "dtm_statistics": report_data["dtm_stats"],
        "quality_assessment": {
            "coverage_percentage": report_data["coverage_pct"],
            "validation_passed": report_data["dtm_stats"]["validation_passed"],
            "notes": [
                "2m resolution used for stability with low ground points (6.1% coverage)",
                "Linear interpolation preserves steep terrain features without artifacts",
                "Gap filling applied to areas with insufficient ground points",
                "Contour lines added to preview for terrain readability"
            ]
        },
        "outputs": {
            "dtm_geotiff": str(OUTPUT_DTM_PATH),
            "preview_image": str(OUTPUT_PREVIEW_PATH),
            "report_json": str(output_path)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"    Report saved: {output_path}")
    return report

def print_summary_console(report):
    dtm_params = report["dtm_parameters"]
    dtm_stats = report["dtm_statistics"]
    qa = report["quality_assessment"]

    print("="*80 + "\n")
    print("\n   INPUT DATA")
    print(f"Ground points used: {report['input_data']['ground_points_extracted']:,}"
          f"({report['input_data']['ground_point_percentage']:.1f}%)")
    
    print("\n   DTM SPECIFICATIONS")
    print(f"    Resolution: {dtm_params['resolution_m']}m")
    print(f"    Dimensions: {dtm_params['grid_dimensions']}")
    print(f"    CRS: EPSG: {dtm_params['crs_epsg']}")
    print(f"    Interpolation: {dtm_params['interpolation_method']}")

    print("\n   TERRAIN STATISTICS")
    print(f"    Elevation Range: {dtm_stats['dtm_min_m']:.1f}m - {dtm_stats['dtm_max_m']:.1f}m")
    print(f"    Mean Elevation: {dtm_stats['dtm_mean_m']:.1f}m")
    print(f"    Terrain Roughness: {dtm_stats['dtm_std_m']:.1f}m")

    print("\n   QUALITY ASSESSMENT")
    print(f"    Coverage: {qa['coverage_percentage']:.1f}%")
    print(f"    Validation: {'PASSED' if qa['validation_passed'] else 'WARNING'}")

    print("\n   OUTPUTS GENERATED")
    for key, value in report['outputs'].items():
        print(f"    {key:15s} {value}")

    print("="*80 + "\n")

def main():
    print("Loading File...")
    las = laspy.read(RAW_DATA_PATH)
    print(f"    Loaded {len(las):,} points\n")

    #Extract ground points
    ground_x, ground_y, ground_z, min_x, max_x, min_y, max_y = extract_ground_points(las)

    # Create interpolation grid
    grid_xx, grid_yy, resolution = generate_dtm_grid(min_x, max_x, min_y, max_y, resolution=2.0)

    # Interpolate DTM surface
    dtm = interpolate_dtm(ground_x, ground_y, ground_z, grid_xx, grid_yy, method='linear')

    # Validate DTM
    dtm_stats = validate_dtm(dtm, np.array(las.z).min(), np.array(las.z).max())

    # Save as GeoTIFF
    save_dtm_geotiff(dtm, min_x, max_y, resolution, CRS_EPSG, OUTPUT_DTM_PATH)

    # Create preview visualisation
    create_dtm_preview(dtm, OUTPUT_PREVIEW_PATH, min_x, max_y, resolution)

    # Generate report
    report_data = {
        "ground_points_count": len(ground_x),
        "ground_point_percentage": len(ground_x) / len(las) * 100,
        "resolution": resolution,
        "interpolation_method": 'linear',
        "grid_width": grid_xx.shape[1],
        "grid_height": grid_xx.shape[0],
        "coverage_pct": (np.sum(~np.isnan(dtm)) / dtm.size) * 100,
        "dtm_stats": dtm_stats
    }

    report = save_generation_report(report_data, OUTPUT_REPORT_PATH)

    # Print console summary
    print_summary_console(report)

if __name__ == "__main__":
    main()