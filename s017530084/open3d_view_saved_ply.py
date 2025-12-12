#!/usr/bin/env python3
"""
Simple Open3D viewer for PLY files saved by simple_infer_main.py.

Loads any of the following if present:
- <index>_points.ply         (PointCloud - LiDAR points)
- <index>_pred.ply           (PointCloud - Predictions)

Usage:
  python open3d_view_saved_ply.py --dir results_kitti/20251209_234600 --index 0
  python open3d_view_saved_ply.py --dir results_kitti/20251209_234600 --index 0 --compare

Install Open3D via:
  pip install open3d
"""

import argparse
import os
import sys
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install with `pip install open3d`.\n")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_if_exists(path: str, loader, name: str):
    """Load a geometry with the given loader if the path exists."""
    if os.path.exists(path):
        try:
            obj = loader(path)
            print(f"[LOAD] {name}: {path}")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({path}): {e}")
    else:
        print(f"[SKIP] {name} not found: {path}")
    return None


def visualize_with_matplotlib(geoms, labels, output_path):
    """Fallback visualization using matplotlib (no X11/OpenGL required)."""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is not available for fallback visualization")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different geometries
    default_colors = [
        [0.5, 0.5, 0.5],  # Gray for point cloud
        [1.0, 0.0, 0.0],  # Red for predictions
        [0.0, 1.0, 0.0],  # Green for ground truth
    ]
    
    all_points = []
    for idx, (geom, label) in enumerate(zip(geoms, labels)):
        if isinstance(geom, o3d.geometry.PointCloud):
            points = np.asarray(geom.points)
            if len(points) == 0:
                continue
            
            all_points.append(points)
            
            # Get colors if available
            if geom.has_colors():
                colors = np.asarray(geom.colors)
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=1, alpha=0.6, label=label)
            else:
                color = default_colors[idx % len(default_colors)]
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=[color], s=1, alpha=0.6, label=label)
        
        elif isinstance(geom, o3d.geometry.LineSet):
            # Draw 3D bounding boxes as wireframes
            points = np.asarray(geom.points)
            lines = np.asarray(geom.lines)
            
            if len(points) == 0 or len(lines) == 0:
                continue
            
            all_points.append(points)
            
            # Use bright red color for predictions (already set when loading)
            color = default_colors[idx % len(default_colors)]
            # For LineSets, always use the color from the geometry (which we set to red)
            if geom.has_colors():
                colors = np.asarray(geom.colors)
                # Use the first color (should be red [1,0,0] after paint_uniform_color)
                if len(colors.shape) == 2 and len(colors) > 0:
                    line_color = colors[0] if colors.shape[0] == len(lines) else colors[lines[0][0]]
                    # Ensure color is in [0,1] range
                    if line_color.max() > 1.0:
                        line_color = line_color / 255.0
                else:
                    line_color = color
            else:
                line_color = color
            
            # Draw all lines with the determined color
            for i, line in enumerate(lines):
                p1, p2 = points[line[0]], points[line[1]]
                ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                         color=line_color, linewidth=1, alpha=1.0, 
                         label=label if i == 0 else "")
    
    if not all_points:
        raise RuntimeError("No valid point clouds to visualize")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Point Cloud Visualization')
    
    # Set equal aspect ratio - combine all points to get proper bounds
    if all_points:
        all_pts_combined = np.vstack(all_points)
        min_vals = all_pts_combined.min(axis=0)
        max_vals = all_pts_combined.max(axis=0)
        ranges = max_vals - min_vals
        max_range = ranges.max()
        
        if max_range > 0:
            centers = (min_vals + max_vals) / 2
            # Add padding to ensure everything is visible
            padding = max_range * 0.05
            ax.set_xlim(centers[0] - max_range/2 - padding, centers[0] + max_range/2 + padding)
            ax.set_ylim(centers[1] - max_range/2 - padding, centers[1] + max_range/2 + padding)
            ax.set_zlim(centers[2] - max_range/2 - padding, centers[2] + max_range/2 + padding)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved visualization to: {output_path} (using matplotlib)")


def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for saved PLY outputs")
    parser.add_argument("--dir", default="results_kitti/20251209_234600",
                        help="Folder containing PLY files (default: results_kitti/20251209_234600)")
    parser.add_argument("--index", type=int, default=0,
                        help="Sample index, e.g. 0, 1, 2 (default: 0)")
    parser.add_argument("--compare", action="store_true",
                        help="Show both point cloud and predictions together")
    parser.add_argument("--width", type=int, default=1440,
                        help="Viewer window width (default: 1440)")
    parser.add_argument("--height", type=int, default=900,
                        help="Viewer window height (default: 900)")
    parser.add_argument("--save-image", type=str, default=None,
                        help="Save visualization as image instead of opening window (e.g., --save-image output.png)")
    parser.add_argument("--headless", action="store_true",
                        help="Use headless rendering (for servers without display)")
    parser.add_argument("--all", action="store_true",
                        help="Process all PLY files in directory (loops through all frames)")
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)
    
    # If --all flag is set, find all available indices
    if args.all:
        import glob
        points_files = glob.glob(os.path.join(base_dir, "*_points.ply"))
        indices = []
        for pf in points_files:
            basename = os.path.basename(pf)
            try:
                idx = int(basename.split("_")[0])
                indices.append(idx)
            except ValueError:
                continue
        indices = sorted(set(indices))
        if not indices:
            print(f"[ERROR] No PLY files found in {base_dir}")
            return 1
        print(f"[INFO] Found {len(indices)} frames to process: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    else:
        indices = [args.index]
    
    # Process each index
    for index in indices:
        points_path = os.path.join(base_dir, f"{index}_points.ply")
        pred_path = os.path.join(base_dir, f"{index}_pred.ply")

        geoms = []
        pcd_points = None
        ls_pred = None
        pcd_pred = None

        # Load point cloud
        pcd_points = load_if_exists(points_path, o3d.io.read_point_cloud, "Point cloud")
        if pcd_points is not None:
            # Color point cloud gray if no colors
            if not pcd_points.has_colors():
                pcd_points.paint_uniform_color([0.5, 0.5, 0.5])
            geoms.append(pcd_points)
            print(f"  Loaded {len(pcd_points.points)} points")

        # Load predictions (try as LineSet first, then fallback to PointCloud)
        ls_pred = load_if_exists(pred_path, o3d.io.read_line_set, "Predictions (LineSet)")
        if ls_pred is not None:
            # Force bright red color for predictions to make them clearly visible
            ls_pred.paint_uniform_color([1.0, 0.0, 0.0])
            if args.compare or pcd_points is None:
                geoms.append(ls_pred)
            print(f"  Loaded {len(ls_pred.points)} box corners, {len(ls_pred.lines)} edges ({len(ls_pred.points)//8} boxes)")
        else:
            # Fallback: try as point cloud
            pcd_pred = load_if_exists(pred_path, o3d.io.read_point_cloud, "Predictions (PointCloud)")
            if pcd_pred is not None:
                # Color predictions red if no colors
                if not pcd_pred.has_colors():
                    pcd_pred.paint_uniform_color([1.0, 0.0, 0.0])
                if args.compare or pcd_points is None:
                    geoms.append(pcd_pred)
                print(f"  Loaded {len(pcd_pred.points)} prediction points")

        if not geoms:
            print(f"\n[SKIP] Frame {index}: No geometries loaded.")
            print(f"  Tried paths:\n    {points_path}\n    {pred_path}")
            continue

        window_title = f"PLY Viewer: Sample {index}"
        has_pred = (ls_pred is not None) or (pcd_pred is not None)
        if args.compare and pcd_points and has_pred:
            window_title += " (Point Cloud + Predictions)"
        elif has_pred and not pcd_points:
            window_title += " (Predictions only)"

        # Determine output mode
        save_image = args.save_image
        if args.headless and save_image is None:
            # Auto-generate image filename in headless mode
            save_image = os.path.join(base_dir, f"{index}_visualization.png")
        elif args.all and save_image is None:
            # Auto-generate image filename when processing all
            save_image = os.path.join(base_dir, f"{index}_visualization.png")
        
        if args.all:
            print(f"\n[PROCESSING] Frame {index}/{indices[-1]}...")
        
        if save_image:
            # Headless rendering: save as image
            print(f"\n[INFO] Rendering visualization (headless mode)...")
            try:
                vis = o3d.visualization.Visualizer()
                if not vis.create_window(visible=False, width=args.width, height=args.height):
                    raise RuntimeError("Failed to create visualization window")
                
                for geom in geoms:
                    vis.add_geometry(geom)
                
                # Set up a good viewing angle
                ctr = vis.get_view_control()
                if ctr is not None:
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_lookat([0.0, 0.0, 0.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.7)
                
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(save_image, do_render=True)
                vis.destroy_window()
                print(f"[SUCCESS] Saved visualization to: {save_image}")
            except Exception as e:
                print(f"\n[WARN] Open3D headless rendering failed: {e}")
                print("[INFO] Falling back to matplotlib-based visualization (no X11/OpenGL required)...")
                
                # Fallback to matplotlib
                try:
                    labels = []
                    if pcd_points:
                        labels.append("Point Cloud")
                    if ls_pred is not None or pcd_pred is not None:
                        if args.compare or not pcd_points:
                            labels.append("Predictions")
                    
                    visualize_with_matplotlib(geoms, labels, save_image)
                except Exception as e2:
                    print(f"\n[ERROR] Matplotlib fallback also failed: {e2}")
                    print("[INFO] Options:")
                    print("  1. Install matplotlib: pip install matplotlib")
                    print("  2. Install OSMesa: sudo apt-get install libosmesa6-dev")
                    print("  3. Use X11 forwarding: ssh -X user@host (then run without --headless)")
                    print("  4. Download PLY files and view locally with CloudCompare or MeshLab")
                    if not args.all:
                        sys.exit(1)
                    continue
        else:
            # Interactive mode: open window (only for single frame)
            if args.all:
                print(f"[SKIP] Frame {index}: Interactive mode not available with --all flag")
                print(f"       Use --headless or --save-image for batch processing")
                continue
            
            print("\n[INFO] Opening viewer. Controls: mouse to rotate, scroll to zoom, 'Q' to exit.")
            try:
                o3d.visualization.draw_geometries(
                    geoms,
                    window_name=window_title,
                    width=args.width,
                    height=args.height,
                )
            except Exception as e:
                print(f"\n[ERROR] Failed to open viewer: {e}")
                print("[INFO] This is likely a headless server. Auto-saving image instead...")
                
                # Auto-save as image using matplotlib fallback
                auto_save_path = os.path.join(base_dir, f"{index}_visualization.png")
                try:
                    labels = []
                    if pcd_points:
                        labels.append("Point Cloud")
                    if ls_pred is not None or pcd_pred is not None:
                        if args.compare or not pcd_points:
                            labels.append("Predictions")
                    
                    visualize_with_matplotlib(geoms, labels, auto_save_path)
                except Exception as e2:
                    print(f"\n[ERROR] Matplotlib fallback also failed: {e2}")
                    print("[INFO] Try using --headless or --save-image <filename>")
                    print(f"       Example: python {sys.argv[0]} --dir {args.dir} --index {index} --headless")
                    sys.exit(1)
    
    if args.all:
        print(f"\n[SUCCESS] Processed {len(indices)} frames")


if __name__ == "__main__":
    main()