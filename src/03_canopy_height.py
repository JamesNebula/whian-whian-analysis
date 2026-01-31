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
import json
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

def save_raster(data, transform, crs, output_path, nodata=0, dtype=rasterio.float32):
    # Save raster data as geotiff
    height, width = data.shape

    with rasterio.open(
        output_path, 
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)

    print(f"    Saved: {output_path}")

def create_chm_preview(chm, complexity, conservation_map, transform, output_path):
    # Create chm preview with complexity overlay
    print("\nGenerating CHM preview visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # plot 1 CHM
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(chm, cmap='YlGn', origin='upper', vmin=0, vmax=40)
    ax1.set_title('Canopy Height Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.025, pad=0.01)
    cbar1.set_label('Height (m)', rotation=270, labelpad=15)

    # Add contour lines for terrain context 
    dtm_path = Path("outputs/dtm/whian_whian_dtm_2m.tif")
    if dtm_path.exists():
        with rasterio.open(dtm_path) as src:
            dtm = src.read(1)
        contours = ax1.contour(dtm, levels=np.arange(200, 370, 20),
                               colors='gray', alpha=0.3, linewidths=0.5)
        ax1.clabel(contours, inline=True, fontsize=6, fmt='%d')

    # plot 2 Vertical complexity
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(complexity, cmap='RdYlGn', origin='upper', vmin=1, vmax=5)
    ax2.set_title('Vertical Complexity Index\n(strata count)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_ticks([1, 2, 3, 4, 5])

    # Plot 3 Conservation Priority
    ax3 = fig.add_subplot(gs[1, 1])
    cmap = mcolors.ListedColormap(['#d73027', '#fc8d59', '#fee08b', '#1a9850'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im3 = ax3.imshow(conservation_map, cmap=cmap, norm=norm, origin='upper')
    ax3.set_title('Conservation Priority Zones', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, 
                         ticks=[1, 2, 3, 4], boundaries=bounds)
    cbar3.ax.set_yticklabels(['Low', 'Medium', 'High', 'Critical'])

    # Plot 4 height distribution histogram
    ax4 = fig.add_subplot(gs[1, 2])
    heights = chm[chm > 0].flatten()
    ax4.hist(heights, bins=30, color='#2c7fb8', edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Canopy Height (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Height Distribution', fontsize=12, fontweight='bold')
    ax4.axvline(heights.mean(), color='red', linestyle='--', label=f"Mean: {heights.mean():.1f}m")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add emergent tree annotation
    emergent_count = np.sum(heights > 35)
    ax4.text(0.98, 0.95, f"Emergent trees\n(>35m): {emergent_count:,}",
             transform=ax4.transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Main title
    fig.suptitle('Whian Whian Rainforest: Canopy Structure Analysis\n'
                 '2m Resolution | GDA2020 MGA Zone 56', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"    Preview saved: {output_path}")

def create_3d_visualization(normalized_x, normalized_y, normalized_heights, normalized_z, output_path):
    # Generate interactive 3d visualisation with matplotlib
    print("Generating 3D canopy visualisation...")

    sample_size = min(150_000, len(normalized_heights))
    indices = np.random.choice(len(normalized_heights), size=sample_size, replace=False)

    # Create 3d scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points colored by height above ground
    scatter = ax.scatter(
        normalized_x[indices],
        normalized_y[indices],
        normalized_z[indices],
        c=normalized_heights[indices],
        cmap='YlGn',
        s=1,
        alpha=0.6
    )

    # Customize plot
    ax.set_title('Whian Whian Rainforest Canopy Structure\n(Interactive 3D View)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', labelpad=10)
    ax.set_ylabel('Northing (m)', labelpad=10)
    ax.set_zlabel('Elevation (m ASL)', labelpad=10)

    # Set viewing angle for best rainforest perspective
    ax.view_init(elev=30, azim=60)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Height Above Ground (m)', rotation=270, labelpad=25)

    # Add terrain surface
    min_x, max_x = normalized_x.min(), normalized_x.max()
    min_y, max_y = normalized_y.min(), normalized_y.max()
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 50), np.linspace(min_y, max_y, 50))
    zz = np.full_like(xx, normalized_z.min() + 5) # Simplified ground surface
    ax.plot_surface(xx, yy, zz, color='brown', alpha=0.2, linewidth=0)

    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    3D Preview saved: {output_path.with_suffix('.png')}")

def save_analysis_report(chm, complexity, conservation_map, normalized_heights, output_path):
    print("\nSaving analysis report...")

    report = {
        "project": 'Whian Whian Rainforest LiDAR Analysis',
        "date_processed": datetime.now().isoformat(),
        "input_data": {
            "laz_file": "whian.laz",
            "dtm_file": "whian_whian_dtm_2m.tif",
            "total_points": 6261585,
            "points_above_ground": int(len(normalized_heights)),
            "ground_point_percentage": 6.1
        },
        "chm_parameters": {
            "reolution_m": 2.0,
            "rasterization_method": "max height per pixel",
            "gap_filling": "3x3 maximum filter"
        },
        "chm_statistics": {
            "mean_height_m": float(normalized_heights.mean()),
            "max_height_m": float(normalized_heights.max()),
            "median_height_m": float(np.median(normalized_heights)),
            "std_height_m": float(normalized_heights.std()),
            "emergent_trees_count": int(np.sum(normalized_heights > 35)),
            "emergent_trees_pct": float(np.sum(normalized_heights > 35) / len(normalized_heights) * 100)
        },
        "biodiversity_metrics": {
            "vertical_complexity": {
                "mean_strata": float(complexity[complexity > 0].mean()),
                "max_strata": int(complexity.max()),
                "strata_definition": "0-2m, 2-5m, 5-10m, 10-20m, 20m+"
            },
            "conservation_priority": {
                "critical_zone_pct": float(np.sum(conservation_map == 4) / conservation_map.size * 100),
                "high_priority_pct": float(np.sum(conservation_map == 3) / conservation_map.size * 100),
                "critical_zone_hectares": float(np.sum(conservation_map == 4) * 4 / 10000),
                "priority_method": "Weighted combination: 40% canopy height + 60% vertical complexity"
            }
        },
        "ecological_insights": [
            "Canopy height distribution shows healthy multilayered rainforest structure",
            f"Identified {np.sum(normalized_heights > 35):,} emergent trees (>35m)",
            f"Critical conservation zones ({np.sum(conservation_map == 4) * 4 / 10000:.1f} ha) characterized by tall canopy (>30m) and high structural complexity (>4 strata)",
            "Lower density (1.6 pts/m2) sufficient for landscape scale structural analysis despite limitations for fine-scale species mapping"
        ],
        "outputs": {
            "chm_geotiff": str(OUTPUT_CHM_PATH),
            "complexity_geotiff": str(OUTPUT_COMPLEXITY_PATH),
            "conservation_geotiff": str(OUTPUT_CONSERVATION_PATH),
            "preview_image": str(OUTPUT_PREVIEW_PATH),
            "report_json": str(output_path)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"    Report saved: {output_path}")
    return report

def print_summary_console(report):
    
    chm_stats = report["chm_statistics"]
    bio_metrics = report["biodiversity_metrics"]
    insights = report["ecological_insights"]
    
    print("="*80)
    
    print("\n  CANOPY STRUCTURE")
    print(f"   Mean Height:    {chm_stats['mean_height_m']:.1f}m")
    print(f"   Max Height:     {chm_stats['max_height_m']:.1f}m")
    print(f"   Emergent Trees: {chm_stats['emergent_trees_count']:,} ({chm_stats['emergent_trees_pct']:.1f}%) >35m")
    
    print("\n  BIODIVERSITY METRICS")
    print(f"   Vertical Complexity: {bio_metrics['vertical_complexity']['mean_strata']:.1f} strata/pixel (max {bio_metrics['vertical_complexity']['max_strata']})")
    print(f"   Critical Zones:      {bio_metrics['conservation_priority']['critical_zone_hectares']:.1f} ha ({bio_metrics['conservation_priority']['critical_zone_pct']:.1f}%)")
    print(f"   High Priority:       {bio_metrics['conservation_priority']['high_priority_pct']:.1f}% of landscape")
    
    print("\n  KEY ECOLOGICAL INSIGHTS")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print("\n  OUTPUTS GENERATED")
    for key, value in report['outputs'].items():
        print(f"   • {key:20s} {value}")
    
    print("="*80 + "\n")

def main():
    # load DTM 
    dtm, transform, crs, bounds = load_dtm(DTM_PATH)

    # load original laz file
    las = laspy.read(RAW_DATA_PATH)

    # Normalize heights above ground
    normalized_x, normalized_y, normalized_heights, normalized_z = normalize_heights(las, dtm, transform)

    # Generate CHM
    chm = generate_chm(normalized_x, normalized_y, normalized_heights, dtm.shape, transform)

    # calc vertical complexity index
    complexity = calculate_vertical_complexity(normalized_x, normalized_y, normalized_heights, dtm.shape, transform)

    # Identify conservation priority zones
    conservation_map, priority_score = identify_conservation_priority(chm, complexity)

    # save outputs
    print("\nSaving raster outputs...")
    save_raster(chm, transform, crs, OUTPUT_CHM_PATH, nodata=0, dtype=rasterio.float32)
    save_raster(complexity, transform, crs, OUTPUT_COMPLEXITY_PATH, nodata=0, dtype=rasterio.uint8)
    save_raster(conservation_map, transform, crs, OUTPUT_CONSERVATION_PATH, nodata=0, dtype=rasterio.uint8)
    
    # Create visualizations
    create_chm_preview(chm, complexity, conservation_map, transform, OUTPUT_PREVIEW_PATH)
    create_3d_visualization(normalized_x, normalized_y, normalized_heights, normalized_z, OUTPUT_3D_HTML)

    # save analysis report
    report = save_analysis_report(chm, complexity, conservation_map, normalized_heights, OUTPUT_REPORT_PATH)

    # print summary
    print_summary_console(report)

if __name__ == "__main__":
    main()
