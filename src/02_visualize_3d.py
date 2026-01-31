"""
3D Visualization of Whian Whian Rainforest
"""

import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_PATH = Path("data/raw/whian.laz")
OUTPUT_3D_PNG = Path("outputs/visualizations/canopy_3d.png")
OUTPUT_3D_ROTATING_GIF = Path("outputs/visualizations/canopy_3d_rotating.gif")  # Optional
OUTPUT_HEIGHT_HIST = Path("outputs/visualizations/height_histogram.png")

# Create output directories
OUTPUT_3D_PNG.parent.mkdir(parents=True, exist_ok=True)


def load_and_sample_points(filepath, sample_size=3000000):
    
    print("🔍 Loading LAZ file...")
    las = laspy.read(filepath)
    print(f"   Loaded {len(las):,} points")
    
    # Sample for visualization 
    if len(las) > sample_size:
        print(f"   Sampling {sample_size:,} points for visualization...")
        indices = np.random.choice(len(las), size=sample_size, replace=False)
        x = np.array(las.x)[indices]
        y = np.array(las.y)[indices]
        z = np.array(las.z)[indices]
    else:
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z)
    
    print(f"   Ready to visualize {len(x):,} points\n")
    return x, y, z


def create_3d_scatter(x, y, z, output_path):
    
    print("🎨 Creating 3D visualization...")
    
    # Create figure with professional styling
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by height (z-coordinate)
    scatter = ax.scatter(
        x, y, z,
        c=z,                    # Color by height
        cmap='YlGn',            # Yellow-Green colormap (natural forest colors)
        s=1,                    # Small points for detail
        alpha=0.6,              # Slight transparency
        linewidths=0            # No borders on points
    )
    
    # Customize plot appearance
    ax.set_title('Whian Whian Rainforest Point Cloud\n3D Visualization', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Easting (m)', fontsize=11, labelpad=10)
    ax.set_ylabel('Northing (m)', fontsize=11, labelpad=10)
    ax.set_zlabel('Elevation (m)', fontsize=11, labelpad=10)
    
    # Set viewing angle for best forest perspective
    ax.view_init(elev=25, azim=-60)  # Slightly elevated, angled view
    
    # Adjust axis limits to focus on data
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min() - 10, z.max() + 10])  # Add padding on z-axis
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Elevation (m)', rotation=270, labelpad=25, fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save PNG
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved 3D visualization: {output_path}")
    print(f"   📏 Elevation range: {z.min():.1f}m – {z.max():.1f}m")
    return True


def create_height_histogram(z, output_path):
    
    print("\n📊 Creating height distribution histogram...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(z, bins=50, color='#2c7fb8', 
                               edgecolor='white', linewidth=0.5, alpha=0.8)
    
    # Customize
    ax.set_title('Whian Whian Rainforest: Height Distribution', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Elevation (m)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    
    # Add vertical line for mean
    mean_height = z.mean()
    ax.axvline(mean_height, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_height:.1f}m')
    
    # Add statistics text box
    stats_text = (
        f'Total Points: {len(z):,}\n'
        f'Min Height: {z.min():.1f}m\n'
        f'Max Height: {z.max():.1f}m\n'
        f'Mean Height: {mean_height:.1f}m\n'
        f'Std Dev: {z.std():.1f}m'
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved histogram: {output_path}")
    return True


def create_2d_topdown(x, y, z, output_path):
   
    print("\n🗺️  Creating top-down view...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot colored by height
    scatter = ax.scatter(x, y, c=z, cmap='terrain', s=0.5, alpha=0.7)
    
    ax.set_title('Whian Whian Rainforest: Top-Down View\nColored by Elevation', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Elevation (m)', rotation=270, labelpad=20, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved top-down view: {output_path}")
    return True


def print_summary(x, y, z):
    
    print("\n" + "="*70)
    print("3D VISUALIZATION")
    print("="*70)
    
    print("\n  VISUALIZATION SUMMARY")
    print(f"   Points visualized: {len(x):,}")
    print(f"   Area covered: {(x.max()-x.min()) * (y.max()-y.min()) / 10000:.1f} hectares")
    print(f"   Elevation range: {z.min():.1f}m – {z.max():.1f}m")
    print(f"   Mean elevation: {z.mean():.1f}m")
    
    print("\n  OUTPUTS GENERATED")
    print(f"   • 3D scatter plot: {OUTPUT_3D_PNG}")
    print(f"   • Height histogram: {OUTPUT_HEIGHT_HIST}")
    
    print("="*70 + "\n")


def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("3D VISUALIZATION")
    print("="*70 + "\n")
    
    # Step 1: Load and sample points
    x, y, z = load_and_sample_points(RAW_DATA_PATH, sample_size=3000000)
    
    # Step 2: Create 3D scatter plot
    create_3d_scatter(x, y, z, OUTPUT_3D_PNG)
    
    # Step 3: Create height histogram
    create_height_histogram(z, OUTPUT_HEIGHT_HIST)
    
    # Step 4: Create top-down view 
    create_2d_topdown(x, y, z, OUTPUT_3D_PNG.with_name('topdown_view.png'))
    
    # Step 5: Print summary
    print_summary(x, y, z)


if __name__ == "__main__":
    main()