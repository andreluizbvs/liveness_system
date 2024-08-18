import argparse

from dataloader.dataset import create_dataset
from evaluate import evaluate_model
from models.liveness import LivenessModel
from models.adversarial_attack import AdversarialModel
from utils.security import identify_vulnerabilities, mitigate_vulnerabilities


def main(data_path, model_path):
    # Train liveness detection model
    liveness_model = LivenessModel(model_path)
    
    # Load data
    train_dataset, val_dataset, test_dataset = create_dataset(
        data_path, 
        image_size=(liveness_model.img_size, liveness_model.img_size)
    )

    liveness_model.train(train_dataset, val_dataset, epochs=10)

    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(
        liveness_model, test_dataset
    )

    print(
        "Evaluation Metrics in the test dataset:\n"
        f"Accuracy: {round(accuracy * 100, 2)}%\n"
        f"Precision: {round(precision * 100, 2)}%\n"
        f"Recall: {round(recall * 100, 2)}%\n"
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
        default="../data/frames", 
        help="Path to the data directory"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to the model directory to continue training",
    )
    args = parser.parse_args()

    main(args.data_path, args.model_path)
