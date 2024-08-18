import os
import random

import cv2

def generate_image_data_from_video(video_folder, output_folder, num_frames=50):
    # Get a list of all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    for video_file in video_files:
        # Open the video file
        video_path = os.path.join(video_folder, video_file)
        print(f'Processing video: {video_path}')
        video = cv2.VideoCapture(video_path)

        # Get the total duration of the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Total frames: {total_frames}')
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        # Generate random frame indices
        frame_indices = random.sample(range(int(duration * fps)), num_frames)

        # Extract and save the frames
        for frame_index in frame_indices:
            # Set the frame position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read the frame
            _, frame = video.read()

            # Save the frame
            frame_path = os.path.join(output_folder, f'{video_file}_{frame_index}.jpg')
            os.makedirs(output_folder, exist_ok=True)
            cv2.imwrite(frame_path, frame)

        # Release the video file
        video.release()

# Example usage
# video_folder = '../../data/archive/real/no_medical_mask/'
# output_folder = '../../data/frames/real/'
video_folder = '../../data/archive/silicone/no_medical_mask/'
output_folder = '../../data/frames/silicone/'
create_dataset(video_folder, output_folder)