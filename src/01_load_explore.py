import laspy
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
RAW_DATA_PATH = Path('data/raw/whian.laz')
OUTPUT_REPORT_PATH = Path('outputs/reports/data_summary.json')
OUTPUT_PREVIEW_PATH = Path('outputs/previews/point_preview.png')

# Output dirs
OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_laz_file(filepath):
    print(f"Loading LAZ: {filepath}")
    try:
        las = laspy.read(filepath)
        print(f"Successfully loaded {len(las):,} points\n")
        return las
    except FileNotFoundError:
        raise FileNotFoundError(f"LAZ file not found at: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading LAZ file: {e}")
    

def extract_metadata(las):
    print("Extracting metadata")

    metadata = {
        "file_info": {
            "filename": Path(las.header.file_source_id).name if hasattr(las.header, 'file_source_id') else 'unknown',
            "point_count": len(las),
            "point_format": las.header.point_format.id,
            "version": f"{las.header.version_major}.{las.header.version_minor}",
            "created_date": str(datetime.now()),
        },
        "spatial_extent": {
            "min_x": float(las.x.min()),
            "max_x": float(las.x.max()),
            "min_y": float(las.y.min()),
            "max_y": float(las.y.max()),
            "min_z": float(las.z.min()),
            "max_z": float(las.z.max()),
            "width_m": float(las.x.max() - las.x.min()),
            "length_m": float(las.y.max() - las.y.min()),
            "area_hectares": float((las.x.max() - las.x.min()) * (las.y.max() - las.y.min()) / 10000),
        },
        "point_density": {
            "total_points": len(las),
            "area_m2": float((las.x.max() - las.x.min()) * (las.y.max() - las.y.min())),
            "points_per_m2": float(len(las) / ((las.x.max() - las.x.min()) * (las.y.max() - las.y.min()))),
        },
        "height_statistics": {
            "min_height_m": float(las.z.min()),
            "max_height_m": float(las.z.max()),
            "mean_height_m": float(las.z.mean()),
            "median_height_m": float(np.median(las.z)),
            "std_height_m": float(las.z.std()),
            "percentiles": {
                "p10": float(np.percentile(las.z, 10)),
                "p25": float(np.percentile(las.z, 25)),
                "p50": float(np.percentile(las.z, 50)),
                "p75": float(np.percentile(las.z, 75)),
                "p90": float(np.percentile(las.z, 90))
            }
        },
        "classifications": {},
        "extra_dimensions": list(las.point_format.dimension_names)
    }

    # extract classification counts if possible
    if 'classification' in las.point_format.dimension_names:
        classes, counts = np.unique(las.classification, return_counts=True)
        metadata["classifications"] = {
            str(int(cls)): int(cnt) for cls, cnt in zip(classes, counts)
        }
        print(f"    Found {len(classes)} classification types")

    if 'return_number' in las.point_format.dimension_names:
        returns, counts = np.unique(las.return_number, return_counts=True)
        metadata['return_statistics'] = {
            str(int(r)): int(cnt) for r, cnt in zip(returns, counts)
        }
        print(f"    Return numbers: {dict(zip(returns, counts))}")

    print("Metadata extraction complete\n")
    return metadata

def validate_data_quality(metadata):
    print("Validating data quality...")
    issues = []
    recommendations = []

    density = metadata['point_density']['points_per_m2']

    if density < 2:
        issues.append(f"Low Density: {density:.1f} pts/m2 may miss fine canopy structure")
        recommendations.append("Consider higher density data for species-level analysis")
    elif density < 5:
        recommendations.append("Adequate density for canopy height modelling")
    else:
        recommendations.append("Excellent density for detailed 3D vegetation structure analysis")

    # Check height range
    height_range = metadata["height_statistics"]["max_height_m"] - metadata["height_statistics"]["min_height_m"]
    if height_range < 10:
        issues.append(f"Unusual height range: Only {height_range:.1f}m variation - may be ground-only")
    elif metadata["height_statistics"]["max_height_m"] > 30:
        recommendations.append("Good canopy height range detected - suitable for emergent tree analysis")

    # Check for ground points (needed for DTM)
    if metadata["classifications"]:
        ground_points = metadata["classifications"].get("2", 0)
        ground_pct = (ground_points / metadata["point_density"]["total_points"]) * 100
        if ground_pct < 5:
            issues.append(f"Low ground points: Only {ground_pct:.1f}% classified as ground")
            recommendations.append("May need ground classification algorithm (Prog TIN possible option)")

    print("Quality validation complete\n")
    return issues, recommendations

def create_preview_plot(las, output_path):
    # Top down scatter plot coloured by height
    print("Generating preview visualization...")

    sample_size = min(100_000, len(las))
    indices = np.random.choice(len(las), size=sample_size, replace=False)

    x_sample = las.x[indices]
    y_sample = las.y[indices]
    z_sample = las.z[indices]

    x_norm = (x_sample - x_sample.min()) / (x_sample.max() - x_sample.min())
    y_norm = (y_sample - y_sample.min()) / (y_sample.max() - y_sample.min())

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x_norm, y_norm, c=z_sample, cmap='viridis', s=1, alpha=0.6)

    ax.set_title("Whian Whian Rainforest LiDAR Preview\n(Top-down view coloured by height)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Normalized X Coordinate')
    ax.set_ylabel('Normalized Y Coordinate')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Height (m)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Preview saved to: {output_path}\n")

def save_summary_report(metadata, issues, recommendations, output_path):
    print("Saving summary report...")

    report = {
        "project": "Whian Whian Rainforest LiDAR Analysis",
        "date_processed": datetime.now().isoformat(),
        "metadata": metadata,
        "quality_assessment": {
            "issues_found": issues,
            "recommendations": recommendations
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_path}\n")
    return report

def print_summary_console(report):
    # Print summary to console
    meta = report["metadata"]
    qa = report["quality_assessment"]
    
    print("\n" + "="*70)
    print("WHIAN WHIAN LIDAR DATA SUMMARY REPORT")
    print("="*70)
    
    print("\n  SPATIAL EXTENT")
    print(f"   Area: {meta['spatial_extent']['area_hectares']:.1f} hectares")
    print(f"   Dimensions: {meta['spatial_extent']['width_m']:.0f}m × {meta['spatial_extent']['length_m']:.0f}m")
    
    print("\n  POINT CLOUD STATISTICS")
    print(f"   Total Points: {meta['point_density']['total_points']:,}")
    print(f"   Density: {meta['point_density']['points_per_m2']:.1f} points/m²")
    print(f"   Height Range: {meta['height_statistics']['min_height_m']:.1f}m – {meta['height_statistics']['max_height_m']:.1f}m")
    
    if meta["classifications"]:
        print("\n   CLASSIFICATIONS")
        for cls, count in sorted(meta["classifications"].items()):
            pct = (count / meta['point_density']['total_points']) * 100
            cls_name = {
                "0": "Never Classified", "1": "Unassigned", "2": "Ground", 
                "3": "Low Veg", "4": "Med Veg", "5": "High Veg",
                "6": "Building", "7": "Low Point", "8": "Reserved",
                "9": "Water", "10": "Rail", "11": "Road",
                "13": "Wire - Guard", "14": "Wire - Conductor", "15": "Transmission Tower",
                "16": "Wire Connector", "17": "Bridge Deck", "18": "High Noise"
            }.get(cls, f"Class {cls}")
            print(f"   {cls_name:20s}: {count:>8,} ({pct:>5.1f}%)")
    
    if qa["issues_found"]:
        print("\n  QUALITY ISSUES")
        for issue in qa["issues_found"]:
            print(f"   • {issue}")
    
    print("\n  RECOMMENDATIONS")
    for rec in qa["recommendations"]:
        print(f"   • {rec}")
    
    print("="*70 + "\n")