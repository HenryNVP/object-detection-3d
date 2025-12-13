#!/usr/bin/env python3
"""
Create a video from visualization images.

Takes the first N images from a directory and creates a video with specified
frame duration.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def natural_sort_key(text):
    """Extract numeric value for natural sorting."""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', text)]


def get_image_files(directory, pattern="*_visualization.png"):
    """Get sorted list of image files matching pattern."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all matching files
    image_files = sorted(dir_path.glob(pattern), key=lambda p: natural_sort_key(p.name))
    
    if not image_files:
        # Try alternative patterns (KITTI uses *_multiview.jpg, nuScenes uses *_visualization.png)
        alt_patterns = ["*_multiview.jpg", "*_multiview.jpeg", "*.png", "*.jpg", "*.jpeg"]
        for alt_pattern in alt_patterns:
            image_files = sorted(dir_path.glob(alt_pattern), key=lambda p: natural_sort_key(p.name))
            if image_files:
                print(f"Found images matching pattern: {alt_pattern}")
                break
    
    return image_files


def create_video_ffmpeg(image_files, output_path, fps=2.0, codec='libx264', quality='medium'):
    """Create video using ffmpeg."""
    if not image_files:
        raise ValueError("No image files provided")
    
    # Create temporary file list for ffmpeg
    temp_list_file = Path(output_path).parent / "temp_image_list.txt"
    
    try:
        with open(temp_list_file, 'w') as f:
            for img_file in image_files:
                # Use absolute path and escape special characters
                abs_path = img_file.resolve()
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1.0/fps}\n")
            # Repeat last frame to ensure it's shown
            if image_files:
                abs_path = image_files[-1].resolve()
                f.write(f"file '{abs_path}'\n")
        
        # Build ffmpeg command
        if quality == 'high':
            crf = '18'
        elif quality == 'medium':
            crf = '23'
        else:  # low
            crf = '28'
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-i', str(temp_list_file),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Scale to even dimensions for H.264
            '-c:v', codec,
            '-crf', crf,
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # Output frame rate
            str(output_path)
        ]
        
        print(f"Running ffmpeg command...")
        print(f"  Input: {len(image_files)} images")
        print(f"  Output: {output_path}")
        print(f"  FPS: {fps} (frame duration: {1.0/fps:.2f}s)")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("ffmpeg error:", result.stderr, file=sys.stderr)
            raise RuntimeError(f"ffmpeg failed with return code {result.returncode}")
        
        print(f"✓ Video created successfully: {output_path}")
        return True
        
    finally:
        # Clean up temp file
        if temp_list_file.exists():
            temp_list_file.unlink()


def create_video_opencv(image_files, output_path, fps=2.0):
    """Create video using OpenCV (fallback if ffmpeg not available)."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not available. Install with: pip install opencv-python")
    
    if not image_files:
        raise ValueError("No image files provided")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        raise ValueError(f"Could not read image: {image_files[0]}")
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    
    print(f"Creating video with OpenCV...")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps} (frame duration: {1.0/fps:.2f}s)")
    print(f"  Frames: {len(image_files)}")
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read {img_file}, skipping", file=sys.stderr)
            continue
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
    
    video_writer.release()
    print(f"✓ Video created successfully: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create a video from visualization images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create video from first 30 images in current directory
  python create_video_from_images.py --dir ./results_nuscenes_second/20251211_071652 --num-frames 30

  # Custom frame duration (0.3 seconds per frame)
  python create_video_from_images.py --dir ./results --num-frames 30 --fps 3.33

  # Use OpenCV instead of ffmpeg
  python create_video_from_images.py --dir ./results --num-frames 30 --use-opencv
        """
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory containing visualization images'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=30,
        help='Number of frames to include (default: 30)'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=2.0,
        help='Frames per second (default: 2.0, which is 0.5s per frame)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path (default: <dir>/video.mp4)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_visualization.png',
        help='File pattern to match (default: *_visualization.png)'
    )
    
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Use OpenCV instead of ffmpeg (slower, but doesn\'t require ffmpeg)'
    )
    
    parser.add_argument(
        '--quality',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Video quality (only for ffmpeg, default: medium)'
    )
    
    args = parser.parse_args()
    
    # Get image files
    try:
        image_files = get_image_files(args.dir, args.pattern)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not image_files:
        print(f"Error: No images found in {args.dir} matching pattern {args.pattern}", file=sys.stderr)
        sys.exit(1)
    
    # Take first N frames
    selected_files = image_files[:args.num_frames]
    print(f"Found {len(image_files)} total images, using first {len(selected_files)}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.dir) / "video.mp4"
    
    # Create video
    try:
        if args.use_opencv:
            create_video_opencv(selected_files, output_path, fps=args.fps)
        else:
            # Try ffmpeg first
            try:
                create_video_ffmpeg(selected_files, output_path, fps=args.fps, quality=args.quality)
            except FileNotFoundError:
                print("ffmpeg not found, falling back to OpenCV...", file=sys.stderr)
                create_video_opencv(selected_files, output_path, fps=args.fps)
            except RuntimeError as e:
                print(f"ffmpeg failed: {e}", file=sys.stderr)
                print("Falling back to OpenCV...", file=sys.stderr)
                create_video_opencv(selected_files, output_path, fps=args.fps)
    
    except Exception as e:
        print(f"Error creating video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

