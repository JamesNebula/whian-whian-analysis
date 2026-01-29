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

