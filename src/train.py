import argparse

import matplotlib.pyplot as plt

from dataloader.dataset import create_dataset, create_dataset_from_split
from dataloader.celeba_spoof import get_data
from src.models.silicone_mask import SiliconeMaskModel
from src.models.depth import FaceDepthModel
from models.adversarial_attack import AdversarialAttack
from utils.security import identify_vulnerabilities
from utils.metrics import f1_score


AMOUNT_LIVES = AMOUNT_SPOOFS = 5000
SHOW_SAMPLES = False


def main(data_path, model_path, model_name, epochs, patience, combine):
    print("Loading architecture...")

    if model_name == "silicone":
        model = SiliconeMaskModel(model_path, combine_frame_and_face=combine)
    elif model_name == "depth":
        model = FaceDepthModel(model_path)
    else:
        raise ValueError("Model name not recognized")

    print("Loading data...")
    if "silicon" in data_path:
        train_dataset, val_dataset, test_dataset = create_dataset(
            data_path,
            image_size=(model.img_size, model.img_size),
            combine_frame_and_face=combine,
        )
    else:
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_data(
            AMOUNT_LIVES, AMOUNT_SPOOFS
        )
        print("Create dataset from split...")
        train_dataset, val_dataset, test_dataset = create_dataset_from_split(
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            image_size=(model.img_size, model.img_size),
            combine_frame_and_face=combine,
        )

    # Get one image from train_dataset and show it
    if SHOW_SAMPLES:
        for images, _ in train_dataset.take(1):
            # Display the image
            for image in images:
                plt.imshow(image.numpy())
                plt.title("Image")
                plt.axis("off")
                plt.show()
                input()

    model.train(train_dataset, val_dataset, epochs=epochs, patience=patience)

    results = model.evaluate(test_dataset)
    f1 = f1_score(results["precision"], results["recall"])
    print(
        "Evaluation Metrics in the test dataset:\n"
        f"Accuracy: {round(results['accuracy'] * 100, 2)}%\n"
        f"Precision: {round(results['precision'] * 100, 2)}%\n"
        f"Recall: {round(results['recall'] * 100, 2)}%\n"
        f"F1-Score: {round(f1 * 100, 2)}%\n"
    )

    if "silicone_video_frames" not in data_path:
        print("Adversarial attacks are only available for the silicone video frame datase.")
        print("Please run: python train.py --data_path ../data/silicone_video_frames --model_name silicone")
        print("Exiting...")
        exit(1)

    # Test adversarial attacks
    adversarial_model = AdversarialAttack(model)

    train_dataset = train_dataset.unbatch()
    val_dataset = val_dataset.unbatch()
    test_dataset = test_dataset.unbatch()

    X_train, y_train = zip(*train_dataset.as_numpy_iterator())
    X_valid, y_valid = zip(*val_dataset.as_numpy_iterator())
    X_test, y_test = zip(*test_dataset.as_numpy_iterator())

    X_train_adv = adversarial_model.generate_adversarial_examples(X_train)
    X_val_adv = adversarial_model.generate_adversarial_examples(X_valid)
    X_test_adv = adversarial_model.generate_adversarial_examples(X_test)

    train_dataset_adv, val_dataset_adv, test_dataset_adv = (
        create_dataset_from_split(
            X_train_adv,
            X_val_adv,
            X_test_adv,
            y_train,
            y_valid,
            y_test,
            image_size=(model.img_size, model.img_size),
            combine_frame_and_face=combine,
        )
    )
    results_pre_finetuning = model.evaluate(test_dataset_adv)

    # TODO: Security measures - identify the types of attack the model is most vulnerable
    identify_vulnerabilities(model, test_dataset_adv)

    # Mitigate vulnerabilities
    # Fine-tune the model with the adversarial examples
    coef = 2
    model.train(
        train_dataset_adv,
        val_dataset_adv,
        epochs=(epochs // coef),
        patience=(patience // coef),
    )
    results_post_finetuning = model.evaluate(test_dataset_adv)

    print(
        "Evaluation metrics comparison before and after fine-tuning:\n"
        f"Accuracy before: {round(results_pre_finetuning['accuracy'] * 100, 2)}% "
        f"Accuracy after: {round(results_post_finetuning['accuracy'] * 100, 2)}%\n"
        f"Precision before: {round(results_pre_finetuning['precision'] * 100, 2)}% "
        f"Precision after: {round(results_post_finetuning['precision'] * 100, 2)}%\n"
        f"Recall before: {round(results_pre_finetuning['recall'] * 100, 2)}% "
        f"Recall after: {round(results_post_finetuning['recall'] * 100, 2)}%\n"
        f"F1-Score before: {round(f1_score(results_pre_finetuning['precision'], results_pre_finetuning['recall']) * 100, 2)}% "
        f"F1-Score after: {round(f1_score(results_post_finetuning['precision'], results_post_finetuning['recall']) * 100, 2)}%\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the silicone mask classification model or the face depth-based classification model."
    )
    parser.add_argument(
        "--data_path",
        default="../data/silicone_faces",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--model_name",
        default="silicone",
        help="Model name to train",
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
        args.model_name,
        args.epochs,
        args.patience,
        args.combine,
    )
