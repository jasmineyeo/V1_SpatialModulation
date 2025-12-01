import cv2

def trim_video_opencv(input_path, output_path, start_frame, num_frames, fps=None):
    """
    Trims a video to a specific number of frames using OpenCV.
    
    Parameters:
    input_path (str): Path to the input video file.
    output_path (str): Path to save the trimmed video file.
    start_frame (int): The frame number to start from (0-indexed).
    num_frames (int): The number of frames to keep.
    fps (float, optional): Output framerate. If None, uses input video's fps.
    
    Returns:
    None
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use input fps if not specified
    output_fps = fps if fps is not None else input_fps
    
    print(f"Input video: {total_frames} frames, {input_fps:.2f} fps, {width}x{height}")
    
    # Check if start_frame and num_frames are valid
    if start_frame >= total_frames:
        print(f"Error: start_frame ({start_frame}) exceeds total frames ({total_frames})")
        cap.release()
        return
    
    end_frame = min(start_frame + num_frames, total_frames)
    actual_frames = end_frame - start_frame
    
    print(f"Trimming from frame {start_frame} to {end_frame} ({actual_frames} frames)")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    while frame_count < actual_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{actual_frames} frames")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Video trimmed successfully! Output: {output_path}")
    print(f"Output video: {actual_frames} frames at {output_fps:.2f} fps")

# Example usage
input_path = r"\\goard-nas1\Goard_Lab\Pregnancy_Project\Mini2p\AnalyzedVideos\NSW130\251006_JSY_NSW130_G4\251006_JSY_NSW130_G4_rec1_0001.mp4"
output_path = r"C:\Users\jasmineyeo\Desktop\trimmedMini2p\251006_JSY_NSW130_G4_rec1_0001_trimmed_empty.mp4"
start_frame = 0  # Start at frame 0
num_frames = 113  # Number of frames to keep
fps = 7.52  # Optional: specify output framerate

trim_video_opencv(input_path, output_path, start_frame, num_frames, fps)