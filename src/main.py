import numpy as np

from dataloader.dataloader import create_dataset
from models.liveness import LivenessModel
from models.adversarial_attack import AdversarialModel
from evaluate import evaluate_model
from utils.security import identify_vulnerabilities, mitigate_vulnerabilities

def split_features_labels(dataset):
    def _split_features_labels(image, label):
        return image, label
    return dataset.map(_split_features_labels)

def dataset_to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset.as_numpy_iterator():
        images.append(image)
        labels.append(label)
        
    # Ensure all images have the same shape
    images = np.array([np.array(img) for img in images])
    labels = np.array(labels)
    
    return images, labels

def main():
    # Load data
    train_dataset, test_dataset = create_dataset('../data/frames')

    # Train liveness detection model
    liveness_model = LivenessModel()
    liveness_model.train(train_dataset, test_dataset)

    test_dataset = split_features_labels(test_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    # # Evaluate model
    y_pred = liveness_model(X_test)
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print(
        f'Accuracy: {accuracy},'
        f'Precision: {precision},'
        f'Recall: {recall},'
        f'F1-Score: {f1}'
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

if __name__ == '__main__':
    main()