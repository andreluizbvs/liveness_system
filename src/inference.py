import argparse
from models.liveness import LivenessModel


def main(img_path, model_path):
    liveness_model = LivenessModel(model_path)
    preproc_img = liveness_model.preprocess(img_path)
    prediction = liveness_model.predict(preproc_img)[0][0]

    print(f"\nPrediction: {prediction}")

    if prediction < 0.5:
        print("Live/Real") # Class 0 
    else:
        print("Spoof/Attack") # Class 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run liveness detection on an image."
    )
    parser.add_argument("img_path", help="Path to the image file")
    parser.add_argument(
        "--model_path",
        default="../ckpt/best_model.keras",
        help="Path to the model file to run inference from",
    )
    args = parser.parse_args()

    main(args.img_path, args.model_path)
