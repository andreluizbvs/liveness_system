import sys

from models.liveness import LivenessModel

def main(img_path, model_path):
    liveness_model = LivenessModel(model_path)
    preprocessed_image = LivenessModel.preprocess(img_path)
    prediction = liveness_model.predict(preprocessed_image)
    
    if prediction > 0.5:
        print("Live/Real")
    else:
        print("Spoof/Attack")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python inference.py <image_path> <model_path>")
        sys.exit(1)
    
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    main(img_path, model_path)