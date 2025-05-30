#ConvertTiffToVideo.py
"""
Script to create a video from multi single-page TIFF files with frame averaging.
Reduces noise by averaging consecutive frames before writing to video.
"""


import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path

def load_tiff_files(input_dir, pattern="*.tif"):
    """Load and sort TIFF file paths."""
    tiff_files = glob.glob(os.path.join(input_dir, pattern))
    if not tiff_files:
        # Try alternative extensions
        tiff_files = glob.glob(os.path.join(input_dir, "*.tiff"))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    # Sort files naturally (handles numeric sequences properly)
    tiff_files.sort(key=lambda x: os.path.basename(x))
    print(f"Found {len(tiff_files)} TIFF files")
    return tiff_files

def average_frames(frames):
    """Average a list of frames to reduce noise."""
    if len(frames) == 1:
        return frames[0]
    
    # Convert to float32 for averaging to prevent overflow
    averaged = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        averaged += frame.astype(np.float32)
    
    averaged /= len(frames)
    return averaged.astype(np.uint8)

def create_video_from_tiffs(input_dir, output_path, fps=10, avg_frames=1, codec='mp4v'):
    """
    Create video from TIFF files with optional frame averaging.
    
    Args:
        input_dir: Directory containing TIFF files
        output_path: Output video file path
        fps: Output video frame rate
        avg_frames: Number of consecutive frames to average (1 = no averaging)
        codec: Video codec (mp4v, XVID, etc.)
    """
    
    # Load TIFF file paths
    tiff_files = load_tiff_files(input_dir)
    
    # Read first image to get dimensions and color info
    first_image = cv2.imread(tiff_files[0], cv2.IMREAD_UNCHANGED)
    if first_image is None:
        raise ValueError(f"Could not read first TIFF file: {tiff_files[0]}")
    
    height, width = first_image.shape[:2]
    
    # Debug: Print image info
    print(f"First image shape: {first_image.shape}")
    print(f"First image dtype: {first_image.dtype}")
    print(f"First image min/max values: {first_image.min()}/{first_image.max()}")
    
    # Handle different image types (grayscale vs color)
    if len(first_image.shape) == 2:
        # Grayscale
        is_color = False
        print(f"Processing grayscale images: {width}x{height}")
    elif len(first_image.shape) == 3 and first_image.shape[2] == 3:
        # Color RGB
        is_color = True
        print(f"Processing color images: {width}x{height}x{first_image.shape[2]}")
    elif len(first_image.shape) == 3 and first_image.shape[2] == 4:
        # RGBA - convert to RGB
        is_color = True
        first_image = cv2.cvtColor(first_image, cv2.COLOR_BGRA2BGR)
        print(f"Processing RGBA images (converting to RGB): {width}x{height}")
    else:
        print(f"Warning: Unusual image format with shape {first_image.shape}")
        is_color = len(first_image.shape) == 3
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)
    
    if not video_writer.isOpened():
        raise ValueError(f"Could not open video writer with codec {codec}")
    
    print(f"Creating video with {fps} FPS, averaging every {avg_frames} frames")
    
    frame_buffer = []
    frames_processed = 0
    
    try:
        for i, tiff_path in enumerate(tiff_files):
            # Read image preserving original format
            img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not read {tiff_path}, skipping...")
                continue
            
            # Handle different color formats consistently
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Convert RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img.shape) == 2 and is_color:
                # Convert grayscale to color if we expect color
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and not is_color:
                # Convert color to grayscale if we expect grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Ensure consistent dimensions
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            # Add to buffer
            frame_buffer.append(img)
            
            # Process when buffer is full or at end of files
            if len(frame_buffer) == avg_frames or i == len(tiff_files) - 1:
                # Average frames in buffer
                averaged_frame = average_frames(frame_buffer)
                
                # Write to video
                video_writer.write(averaged_frame)
                frames_processed += 1
                
                # Clear buffer
                frame_buffer = []
                
                # Progress update
                if frames_processed % 10 == 0:
                    print(f"Processed {frames_processed} output frames ({i+1}/{len(tiff_files)} input files)")
    
    finally:
        video_writer.release()
    
    print(f"Video created successfully: {output_path}")
    print(f"Total output frames: {frames_processed}")
    print(f"Video duration: {frames_processed/fps:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description="Create video from TIFF files with noise reduction")
    parser.add_argument("input_dir", help="Directory containing TIFF files")
    parser.add_argument("output", help="Output video file path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate (default: 10)")
    parser.add_argument("--avg-frames", type=int, default=3, 
                       help="Number of consecutive frames to average for noise reduction (default: 3)")
    parser.add_argument("--codec", default="mp4v", 
                       help="Video codec (default: mp4v, alternatives: XVID, H264)")
    parser.add_argument("--pattern", default="*.tif", 
                       help="File pattern to match (default: *.tif)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    if args.avg_frames < 1:
        print("Error: avg-frames must be >= 1")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        create_video_from_tiffs(
            args.input_dir, 
            args.output, 
            fps=args.fps,
            avg_frames=args.avg_frames,
            codec=args.codec
        )
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python tiff_to_video.py /path/to/tiff/files output.mp4 --fps 10 --avg-frames 3")
        print("\nFor help: python tiff_to_video.py --help")
    else:
        exit(main())