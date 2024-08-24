import argparse
import os

import cv2
import onnxruntime as ort

from models.liveness import LivenessModel as SiliconMaskModel


SPOOF_TH = 0.5


def get_scalar(nested_list):
    while isinstance(nested_list, list):
        nested_list = nested_list[0]
    return nested_list


def process_frame(frame, silicon_mask_model, is_keras):
    preproc_img = SiliconMaskModel.preprocess(frame)
    if is_keras:
        prediction = silicon_mask_model.predict(preproc_img)
    else:
        input_layer = silicon_mask_model.get_inputs()[0].name
        prediction = silicon_mask_model.run(None, {input_layer: preproc_img})

    prediction = get_scalar(prediction)
    print(f"\nPrediction: {prediction}")
    print("Live/Real") if prediction < SPOOF_TH else print("Spoof/Attack")

    del preproc_img

    return prediction


def main(media_path, model_path):

    is_keras = model_path.endswith(".keras")

    silicon_mask_model = (
        SiliconMaskModel(model_path)
        if is_keras
        else ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    )

    if media_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        return process_frame(media_path, silicon_mask_model, is_keras)

    cap = cv2.VideoCapture(media_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, silicon_mask_model, is_keras)

        cv2.imshow('Frame', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run liveness detection on an image."
    )
    parser.add_argument("media_path", help="Path to the image or video file")
    parser.add_argument(
        "--model_path",
        default="../ckpt/[frames] best_model.keras",
        help="Path to the model file to run inference from",
    )
    args = parser.parse_args()

    main(args.media_path, args.model_path)
