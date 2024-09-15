import argparse

import cv2
import numpy as np
import onnxruntime as ort
from deepface import DeepFace

from src.models.silicone_mask import SiliconeMaskModel
from src.models.AENet import Predictor


SPOOF_TH = 0.5
SILICONE_TH = 0.5
IMG_SIZE = 224
ROUND_DIGITS = 4


def get_scalar(nested_list):
    while isinstance(nested_list, (list, np.ndarray)) and len(nested_list) > 0:
        nested_list = nested_list[0]
    return nested_list


def best_face(faces):
    if len(faces) > 0:
        # Select the most confidently extracted face
        return max(faces, key=lambda x: x["confidence"])
    else:
        return None


def process_frame_silicone(frame, silicon_mask_model, is_keras):
    preproc_img = SiliconeMaskModel.preprocess(frame)
    if is_keras:
        prediction = silicon_mask_model.predict(preproc_img)
    else:
        input_layer = silicon_mask_model.get_inputs()[0].name
        prediction = silicon_mask_model.run(None, {input_layer: preproc_img})

    del preproc_img
    return get_scalar(prediction)


def process_frame_deepface(frame):
    faces = DeepFace.extract_faces(
        frame,
        detector_backend="yolov8",
        enforce_detection=False,
        anti_spoofing=True,
    )

    face = best_face(faces)
    del faces
    return face


def process_frame_aenet(frame, aenet_pred):
    if frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    if not isinstance(frame.dtype, np.uint8):
        frame = (frame * 255).astype(np.uint8)
    return aenet_pred.predict([frame])[0][1]


def process_frame_all(frame, silicon_model, aenet_pred, is_keras):
    sil_pred = float(process_frame_silicone(frame, silicon_model, is_keras))
    dp_pred = process_frame_deepface(frame)
    ae_pred = process_frame_aenet(dp_pred["face"], aenet_pred)

    print(
        f"DeepFace FAS predictor. Is it spoof? {dp_pred['is_real']}. "
        f"Score: {round(dp_pred['antispoof_score'], ROUND_DIGITS)}. [1 = Spoof, 0 = Live]."
    )
    print(
        f"Is subject wearing silicon mask? {sil_pred > SILICONE_TH}. "
        f"Score: {round(sil_pred, ROUND_DIGITS)}. [1 = Mask, 0 = No Mask]."
    )
    print(
        f"AENet FAS predictor. Is it spoof? {ae_pred > SPOOF_TH}. "
        f"Score: {round(ae_pred, ROUND_DIGITS)}. [1 = Spoof, 0 = Live]."
    )
    print("\n")


def main(media_path, model_path):
    is_keras = model_path.endswith(".keras")

    silicon_mask_model = (
        SiliconeMaskModel(model_path)
        if is_keras
        else ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    )

    aenet_pred = Predictor()

    if media_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        process_frame_all(media_path, silicon_mask_model, aenet_pred, is_keras)
        return

    cap = cv2.VideoCapture(media_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame_all(frame, silicon_mask_model, aenet_pred, is_keras)

        cv2.imshow("Frame", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run liveness detection on an image or video."
    )
    parser.add_argument("media_path", help="Path to the image or video file")
    parser.add_argument(
        "--model_path",
        default="../ckpt/best_silicone_mask_model.keras",
        help="Path to the model file to run inference from",
    )
    args = parser.parse_args()

    main(args.media_path, args.model_path)
