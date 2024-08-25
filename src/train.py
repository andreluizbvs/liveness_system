import argparse

from dataloader.dataset import create_dataset, create_dataset_from_split
from dataloader.celeba_spoof import get_data
from models.liveness import LivenessModel
from models.adversarial_attack import AdversarialModel
from utils.security import identify_vulnerabilities, mitigate_vulnerabilities


amount_lives = amount_spoofs = 5000


def main(data_path, model_path, epochs, patience, combine):
    print("Loading architecture...")
    liveness_model = LivenessModel(model_path, combine_frame_and_face=combine)

    print("Loading data...")
    if "silicon" in data_path:
        train_dataset, val_dataset, test_dataset = create_dataset(
            data_path,
            image_size=(liveness_model.img_size, liveness_model.img_size),
            combine_frame_and_face=combine,
        )
    else:
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_data(
            amount_lives, amount_spoofs
        )
        print("Create dataset from split...")
        train_dataset, val_dataset, test_dataset = create_dataset_from_split(
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            image_size=(liveness_model.img_size, liveness_model.img_size),
            combine_frame_and_face=combine,
        )

    print(f"Shape of one sample: {train_dataset.take(1)}")

    liveness_model.train(
        train_dataset, val_dataset, epochs=epochs, patience=patience
    )

    results = liveness_model.evaluate(test_dataset)
    f1 = (
        2.0
        * (results["precision"] * results["recall"])
        / (results["precision"] + results["recall"])
    )
    print(
        "Evaluation Metrics in the test dataset:\n"
        f"Accuracy: {round(results['accuracy'] * 100, 2)}%\n"
        f"Precision: {round(results['precision'] * 100, 2)}%\n"
        f"Recall: {round(results['recall'] * 100, 2)}%\n"
        f"F1-Score: {round(f1 * 100, 2)}%\n"
    )

    # # Test adversarial attacks
    # adversarial_model = AdversarialModel(liveness_model)
    # X_adv = adversarial_model.generate_adversarial_examples(X_val)
    # y_adv_pred = adversarial_model.test_adversarial_examples(X_adv)
    # adv_accuracy, adv_precision, adv_recall = evaluate_model(y_val, y_adv_pred)
    # print(f'Adversarial Accuracy: {adv_accuracy}, Precision: {adv_precision}, Recall: {adv_recall}')

    # # Security measures
    # identify_vulnerabilities()
    # mitigate_vulnerabilities()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the liveness detection model."
    )
    parser.add_argument(
        "--data_path",
        default="../data/silicone_faces",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the model directory to resume training from",
    )
    parser.add_argument(
        "--epochs", default=150, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--patience",
        default=20,
        help="Number of epochs to wait for early stopping",
    )
    parser.add_argument(
        "--combine",
        default=False,
        help="Combine frame and face features",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_path,
        args.epochs,
        args.patience,
        args.combine,
    )
