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
from src.liveness_inference import best_face


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


def evaluate_celeba_spoof_dataset(model):
    _, _, X_test, _, _, y_test = get_data()
    X_test = SiliconeMaskModel.preprocess(X_test)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(y_test.shape)
    return get_metrics(y_test, y_pred)


def evaluate_celeba_spoof_dataset_deepface():
    _, _, X_test, _, _, y_test = get_data()
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
    return get_metrics(y_test, y_pred)


def main():
    print("Evaluating FAS Model")
    print("Accuracy, Precision, Recall, F1-Score:")
    print(evaluate_celeba_spoof_dataset_deepface())


if __name__ == "__main__":
    main()
