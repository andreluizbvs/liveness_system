import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from deepface import DeepFace
from tqdm import tqdm

from dataloader.celeba_spoof import get_data
from src.models.silicone_mask import SiliconeMaskModel
from src.models.depth import FaceDepthModel
from src.models.AENet import Predictor
from src.liveness_inference import best_face


def show_metrics(model_name, dataset_name, accuracy, precision, recall, f1):
    print(
        f"\nEvaluation Metrics of the {model_name} Model "
        f"in the {dataset_name} test dataset:\n\n"
        f"Accuracy: {round(accuracy * 100, 2)}%\n"
        f"Precision: {round(precision * 100, 2)}%\n"
        f"Recall: {round(recall * 100, 2)}%\n"
        f"F1-Score: {round(f1 * 100, 2)}%\n"
    )


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


def evaluate_model(model, test_dataset):
    y_true, y_pred = zip(
        *[
            (
                y_test.numpy(),
                (np.array(model.predict(X_test)) > 0.5).astype(int),
            )
            for X_test, y_test in test_dataset
        ]
    )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return get_metrics(y_true, y_pred)


def evaluate_celeba_spoof_dataset_depthcls(X_test, y_test):
    model = FaceDepthModel("../ckpt/best_face_depth_model.keras")
    y_pred = []
    for img in tqdm(X_test):
        img = FaceDepthModel.preprocess(img)
        y = model.predict(img)
        y = (y > 0.5).astype(int)
        y_pred.append(y)

    y_pred = np.array(y_pred)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    show_metrics("DepthCls", "CelebA-Spoof", accuracy, precision, recall, f1)


def evaluate_celeba_spoof_dataset_silicone(X_test, y_test):
    model = SiliconeMaskModel("../ckpt/best_silicone_mask_model.keras")
    y_pred = []
    for img in tqdm(X_test):
        img = SiliconeMaskModel.preprocess(img)
        y = model.predict(img)
        y = (y_pred > 0.5).astype(int)
        y_pred.append(y)
    
    y_pred = np.array(y_pred)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    show_metrics("AENet", "CelebA-Spoof", accuracy, precision, recall, f1)


def evaluate_celeba_spoof_dataset_deepface(X_test, y_test):
    y_pred = []
    for img in tqdm(X_test):
        faces = DeepFace.extract_faces(
            img,
            detector_backend="yolov8",
            enforce_detection=False,
            anti_spoofing=True,
        )
        face = best_face(faces)
        if face is None:
            y_pred.append(0)
            continue

        y_pred.append(int(face["is_real"]))

    y_pred = np.array(y_pred)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    show_metrics(
        "DeepFace's Spoof Detection",
        "CelebA-Spoof",
        accuracy,
        precision,
        recall,
        f1,
    )


def evaluate_celeba_spoof_dataset_aenet(X_test, y_test):
    y_pred = []
    predictor = Predictor("../ckpt/ckpt_iter.pth.tar")
    for img in tqdm(X_test):
        if not isinstance(img, np.ndarray):
            img = img.cpu().numpy()
        img = predictor.preprocess_data(img)
        y = predictor.predict(img)[0][1]
        y = (y < 0.5).astype(int)
        y_pred.append(y)

    y_pred = np.array(y_pred)
    print(y_pred)
    print(y_test)
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    show_metrics("AENet", "CelebA-Spoof", accuracy, precision, recall, f1)


def main():
    _, _, X_test, _, _, y_test = get_data()
    evaluate_celeba_spoof_dataset_aenet(X_test, y_test)
    evaluate_celeba_spoof_dataset_silicone(X_test, y_test)
    evaluate_celeba_spoof_dataset_deepface(X_test, y_test)
    evaluate_celeba_spoof_dataset_depthcls(X_test, y_test)


if __name__ == "__main__":
    main()
