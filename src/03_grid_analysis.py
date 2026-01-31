"""
Grid-Based Height Analysis
"""
import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_PATH = Path("data/raw/whian.laz")
OUTPUT_HEATMAP = Path("outputs/analysis/height_heatmap.png")
OUTPUT_GRID_STATS = Path("outputs/analysis/grid_statistics.json")

# Create output directories
OUTPUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)


def load_points(filepath):
    """Load all points from LAZ file."""
    print("🔍 Loading LAZ file...")
    las = laspy.read(filepath)
    
    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)
    
    print(f"   Loaded {len(las):,} points")
    print(f"   Area: {(x.max()-x.min()) * (y.max()-y.min()) / 10000:.1f} hectares")
    print(f"   Elevation range: {z.min():.1f}m – {z.max():.1f}m\n")
    
    return x, y, z


def create_grid(x, y, z, cell_size=50):
    """
    Divide area into grid cells and calculate mean height per cell.
    
    Args:
        cell_size: Size of each grid cell in meters
    """
    print(f"  Creating {cell_size}m grid cells...")
    
    # Calculate grid boundaries
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create grid edges
    x_edges = np.arange(x_min, x_max + cell_size, cell_size)
    y_edges = np.arange(y_min, y_max + cell_size, cell_size)
    
    print(f"   Grid dimensions: {len(x_edges)-1} × {len(y_edges)-1} cells")
    print(f"   Total cells: {(len(x_edges)-1) * (len(y_edges)-1):,}")
    
    # Initialize grid arrays
    mean_heights = np.zeros((len(y_edges)-1, len(x_edges)-1))
    point_counts = np.zeros((len(y_edges)-1, len(x_edges)-1), dtype=int)
    
    # Assign points to grid cells and calculate statistics
    print("   Calculating mean heights per cell...")
    
    for i in range(len(x)):
        # Find which cell this point belongs to
        x_idx = int((x[i] - x_min) / cell_size)
        y_idx = int((y[i] - y_min) / cell_size)
        
        # Check bounds (edge cases)
        if 0 <= x_idx < mean_heights.shape[1] and 0 <= y_idx < mean_heights.shape[0]:
            # Accumulate for mean calculation
            if point_counts[y_idx, x_idx] == 0:
                mean_heights[y_idx, x_idx] = z[i]
            else:
                # Running average
                n = point_counts[y_idx, x_idx]
                mean_heights[y_idx, x_idx] = (mean_heights[y_idx, x_idx] * n + z[i]) / (n + 1)
            
            point_counts[y_idx, x_idx] += 1
    
    # Calculate coverage statistics
    cells_with_data = np.sum(point_counts > 0)
    total_cells = mean_heights.size
    coverage_pct = (cells_with_data / total_cells) * 100
    
    print(f"   Cells with data: {cells_with_data:,} ({coverage_pct:.1f}%)")
    print(f"   Empty cells: {total_cells - cells_with_data:,}")
    
    return mean_heights, point_counts, x_edges, y_edges, coverage_pct


def create_heatmap(mean_heights, x_edges, y_edges, coverage_pct, output_path):
    """heatmap visualization of mean heights."""
    print("\n🎨 Creating heatmap visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap
    im = ax.imshow(
        mean_heights,
        cmap='terrain',  # Natural terrain colors
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], # type: ignore
        aspect='equal',
        interpolation='nearest'
    )
    
    # Customize plot
    ax.set_title(f'Whian Whian Rainforest: Mean Height per {int(x_edges[1]-x_edges[0])}m Cell\n'
                 f'Grid Coverage: {coverage_pct:.1f}%', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Mean Height (m)', rotation=270, labelpad=25, fontsize=12)
    
    # Add grid lines (every 5 cells for readability)
    grid_step = 5
    for i in range(0, len(x_edges), grid_step):
        ax.axvline(x_edges[i], color='white', alpha=0.3, linewidth=0.5)
    for i in range(0, len(y_edges), grid_step):
        ax.axhline(y_edges[i], color='white', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved heatmap: {output_path}")
    return True


def identify_tall_zones(mean_heights, point_counts, x_edges, y_edges, percentile=90):
    """
    Identify cells with heights in the top percentile.
    """
    print("\n🎯 Identifying tall canopy zones...")
    
    # Only consider cells with data
    valid_heights = mean_heights[point_counts > 0]
    
    if len(valid_heights) == 0:
        print("   ⚠️  No valid cells found")
        return []
    
    # Calculate threshold
    threshold = np.percentile(valid_heights, percentile)
    print(f"   Height threshold (top {100-percentile}%): {threshold:.1f}m")
    
    # Find cells above threshold
    tall_mask = (mean_heights >= threshold) & (point_counts > 0)
    tall_cells = np.sum(tall_mask)
    tall_area = tall_cells * (x_edges[1]-x_edges[0])**2 / 10000  # Convert to hectares
    
    print(f"   Tall canopy cells: {tall_cells:,}")
    print(f"   Area: {tall_area:.1f} hectares ({tall_cells/mean_heights.size*100:.1f}% of grid)")
    
    return tall_mask, threshold, tall_area


def create_tall_zones_map(mean_heights, tall_mask, x_edges, y_edges, threshold, tall_area, output_path):
    """visualization highlighting tall canopy zones."""
    print("\n🗺️  Creating tall zones visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot all cells with base color
    im = ax.imshow(
        mean_heights,
        cmap='Greens',
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], # type: ignore
        aspect='equal',
        alpha=0.7,
        interpolation='nearest'
    )
    
    # Overlay tall zones in red
    tall_overlay = np.zeros_like(mean_heights)
    tall_overlay[tall_mask] = 1
    
    ax.imshow(
        tall_overlay,
        cmap=mcolors.ListedColormap(['none', '#d73027']),  # Transparent, then red
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], # type: ignore
        aspect='equal',
        alpha=0.6,
        interpolation='nearest'
    )
    
    # Customize
    ax.set_title(f'Whian Whian Rainforest: Tall Canopy Zones\n'
                 f'Cells with mean height ≥ {threshold:.1f}m ({tall_area:.1f} ha)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', alpha=0.6, label=f'Tall Canopy (≥{threshold:.1f}m)'),
        Patch(facecolor='green', alpha=0.3, label='Other Areas')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved tall zones map: {output_path}")
    return True


def save_statistics(mean_heights, point_counts, tall_mask, threshold, tall_area, output_path):
    """Save grid statistics to JSON."""
    import json
    from datetime import datetime
    
    stats = {
        "project": "Whian Whian Rainforest Grid Analysis",
        "date_processed": datetime.now().isoformat(),
        "grid_parameters": {
            "cell_size_m": int(x_edges[1] - x_edges[0]) if 'x_edges' in locals() else 50, # type: ignore
            "total_cells": int(mean_heights.size),
            "cells_with_data": int(np.sum(point_counts > 0)),
            "coverage_percentage": float((np.sum(point_counts > 0) / mean_heights.size) * 100)
        },
        "height_statistics": {
            "mean_height_all_cells_m": float(np.mean(mean_heights[point_counts > 0])),
            "min_height_m": float(np.min(mean_heights[point_counts > 0])),
            "max_height_m": float(np.max(mean_heights[point_counts > 0])),
            "std_height_m": float(np.std(mean_heights[point_counts > 0]))
        },
        "tall_zones": {
            "threshold_m": float(threshold),
            "cells_count": int(np.sum(tall_mask)),
            "area_hectares": float(tall_area),
            "percentage_of_grid": float((np.sum(tall_mask) / mean_heights.size) * 100)
        },
        "summary": [
            f"Analyzed {mean_heights.size:,} grid cells across Whian Whian rainforest",
            f"Mean canopy height: {np.mean(mean_heights[point_counts > 0]):.1f}m",
            f"Identified {np.sum(tall_mask):,} cells ({tall_area:.1f} ha) with tall canopy (≥{threshold:.1f}m)",
            "Grid coverage: {:.1f}% of cells contained LiDAR points".format((np.sum(point_counts > 0) / mean_heights.size) * 100)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Saved statistics: {output_path}")
    return stats


def print_summary(stats):
   
    print("\n" + "="*70)
    print("GRID-BASED ANALYSIS")
    print("="*70)
    
    print("\n  GRID ANALYSIS SUMMARY")
    grid = stats["grid_parameters"]
    height = stats["height_statistics"]
    tall = stats["tall_zones"]
    
    print(f"   Grid Size: {grid['cell_size_m']}m × {grid['cell_size_m']}m cells")
    print(f"   Total Cells: {grid['total_cells']:,}")
    print(f"   Coverage: {grid['coverage_percentage']:.1f}%")
    
    print(f"\n  HEIGHT STATISTICS")
    print(f"   Mean Height: {height['mean_height_all_cells_m']:.1f}m")
    print(f"   Range: {height['min_height_m']:.1f}m – {height['max_height_m']:.1f}m")
    print(f"   Std Dev: {height['std_height_m']:.1f}m")
    
    print(f"\n  TALL CANOPY ZONES")
    print(f"   Threshold: {tall['threshold_m']:.1f}m")
    print(f"   Area: {tall['area_hectares']:.1f} hectares")
    print(f"   Grid Coverage: {tall['percentage_of_grid']:.1f}%")
    
    print("\n  OUTPUTS GENERATED")
    print(f"   • Height heatmap: {OUTPUT_HEATMAP}")
    print(f"   • Tall zones map: {OUTPUT_HEATMAP.with_name('tall_zones.png')}")
    print(f"   • Statistics JSON: {OUTPUT_GRID_STATS}")
    
    print("="*70 + "\n")


def main():
    
    print("\n" + "="*70)
    print("GRID-BASED HEIGHT ANALYSIS")
    print("="*70 + "\n")
    
    # Step 1: Load points
    x, y, z = load_points(RAW_DATA_PATH)
    
    # Step 2: Create grid and calculate statistics
    mean_heights, point_counts, x_edges, y_edges, coverage_pct = create_grid(x, y, z, cell_size=50)
    
    # Step 3: Create heatmap
    create_heatmap(mean_heights, x_edges, y_edges, coverage_pct, OUTPUT_HEATMAP)
    
    # Step 4: Identify tall canopy zones
    tall_mask, threshold, tall_area = identify_tall_zones(mean_heights, point_counts, x_edges, y_edges, percentile=90)
    
    # Step 5: Create tall zones visualization
    create_tall_zones_map(mean_heights, tall_mask, x_edges, y_edges, threshold, tall_area, 
                         OUTPUT_HEATMAP.with_name('tall_zones.png'))
    
    # Step 6: Save statistics
    stats = save_statistics(mean_heights, point_counts, tall_mask, threshold, tall_area, OUTPUT_GRID_STATS)
    
    # Step 7: Print summary
    print_summary(stats)


if __name__ == "__main__":
    main()