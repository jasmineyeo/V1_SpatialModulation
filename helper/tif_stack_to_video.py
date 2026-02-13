import numpy as np
import cv2
import os
import tifffile
from tqdm import tqdm

def convert_single_tif_to_video(tif_file, output_file='output.mp4', rec_frame_rate=20,
                               playback_speed=2, frames_avg=2, start_frame=0, num_frames=None):
    """
    Convert a single TIF file to a video file

    Parameters:
    tif_file - path to the TIF file
    output_file - name/path of output video file
    rec_frame_rate - original recording frame rate in Hz
    playback_speed - multiplier for playback speed
    frames_avg - number of frames to average for smoother video
    start_frame - first frame to include
    num_frames - number of frames to convert (None means all frames)
    """

    # Check if TIF file exists
    if not os.path.exists(tif_file):
        print(f"Error: TIF file not found: {tif_file}")
        return

    print(f"Opening TIF file: {os.path.basename(tif_file)}...", flush=True)

    # Load frames using tifffile with progress bar
    with tifffile.TiffFile(tif_file) as tif:
        total_frames = len(tif.pages)
        print(f"Total frames in file: {total_frames}", flush=True)
        print("Loading frames into memory...", flush=True)

        # Load frames with progress bar
        all_frames = []
        for page in tqdm(tif.pages, desc="Loading frames", unit="frame", ncols=80):
            all_frames.append(page.asarray())

        print("Converting to array...", flush=True)
        all_frames = np.array(all_frames)
    
    # Set up video parameters
    video_framerate = rec_frame_rate * playback_speed
    frame_size = (all_frames[0].shape[1], all_frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Determine end frame
    if num_frames is None:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + num_frames, total_frames)
    
    if start_frame >= total_frames:
        print(f"Error: start_frame ({start_frame}) is greater than total frames ({total_frames})")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the video writer
    out = cv2.VideoWriter(output_file, fourcc, video_framerate, frame_size, isColor=False)
    
    print(f"Converting frames {start_frame} to {end_frame-1}...")
    print(f"Video framerate: {video_framerate} fps")
    
    # Process frames with averaging
    for i in range(start_frame, end_frame):
        start_idx = max(0, i - frames_avg + 1)
        frames_to_average = all_frames[start_idx:i+1]
        avg_frame = np.mean(frames_to_average, axis=0).astype(np.int16)
        
        # Normalize the averaged frame to 8-bit
        norm_frame = cv2.normalize(avg_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out.write(norm_frame)
    
    # Release the video writer
    out.release()
    print(f"Conversion complete. The output file is {output_file}")
    
    # Display full path
    full_path = os.path.abspath(output_file)
    print(f"The output file was saved at: {full_path}")
    
    return full_path

tif_file = r"D:\V1_SpatialModulation\2p\V1window\JSY061_ChronicImaging_Axonal\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001\TSeries-02022026-1804-001_registered.tif"
output_path =r"D:\V1_SpatialModulation\2p\V1window\JSY061_ChronicImaging_Axonal\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001\TSeries-02022026-1804-001_registered.mp4"

output_path = convert_single_tif_to_video(
    tif_file=tif_file,
    output_file=output_path,
    rec_frame_rate=10.0477,
    # rec_frame_rate=15.11,
    # rec_frame_rate=7.49,
    playback_speed=5,
    frames_avg=10,
    start_frame=0,
    num_frames=500
)

print(f"Video saved to: {output_path}")