import os
import random
from tqdm import tqdm

import cv2
from deepface import DeepFace

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

# video_folder = '../../data/archive/real/no_medical_mask/'
# output_folder = '../../data/frames/real/'
# video_folder = '../../data/archive/silicone/no_medical_mask/'
# output_folder = '../../data/frames/silicone/'
# generate_image_data_from_video(video_folder, output_folder)

def get_most_confident_face(detections):
    most_confident_face = None
    max_confidence = 0

    for face in detections:
        confidence = face['confidence']

        if confidence > max_confidence:
            most_confident_face = face
            max_confidence = confidence

    return most_confident_face if max_confidence > 0.5 else None

def generate_face_data_from_images(image_folder, output_folder):
    # Walk through the image folder
    for root, _, files in os.walk(image_folder):
        for file in tqdm(files):
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)

                # Detect faces using DeepFace's YOLOv8
                detections = DeepFace.extract_faces(image_path, detector_backend='yolov8', enforce_detection=False)

                face = get_most_confident_face(detections)

                # Save the detected faces
                if face is not None:
                    relative_path = os.path.relpath(root, image_folder)
                    face_output_folder = os.path.join(output_folder, relative_path)
                    os.makedirs(face_output_folder, exist_ok=True)
                    face_path = os.path.join(face_output_folder, f'{os.path.splitext(file)[0]}_face.jpg')
                    cv2.imwrite(face_path, (face['face'] * 255).astype(int))


image_folder = '../../data/frames'
output_folder = '../../data/silicone_faces'
generate_face_data_from_images(image_folder, output_folder)
