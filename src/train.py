import argparse

from dataloader.dataset import create_dataset
from models.liveness import LivenessModel
from models.adversarial_attack import AdversarialModel
from utils.security import identify_vulnerabilities, mitigate_vulnerabilities


def main(data_path, model_path, combine):
    # Load architecture
    liveness_model = LivenessModel(model_path, combine_frame_and_face=combine)

    # Load data
    train_dataset, val_dataset, test_dataset = create_dataset(
        data_path, 
        image_size=(liveness_model.img_size, liveness_model.img_size),
        combine_frame_and_face=combine,
    )

    liveness_model.train(train_dataset, val_dataset, epochs=5)

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
        "--combine",
        default=False,
        help="Combine frame and face features",
    )
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.combine)
