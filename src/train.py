import sys

import numpy as np

from dataloader.dataloader import create_dataset
from models.liveness import LivenessModel
from models.adversarial_attack import AdversarialModel
from evaluate import evaluate_model
from utils.security import identify_vulnerabilities, mitigate_vulnerabilities


def main(data_path = '../data/frames'):
    # Load data
    train_dataset, test_dataset = create_dataset(data_path)

    # Train liveness detection model
    liveness_model = LivenessModel()
    liveness_model.train(train_dataset, test_dataset, epochs=1)

    # # Evaluate model
    y_test, y_pred = [], []
    for X_test, y in test_dataset:
        pred = np.array(liveness_model(X_test))
        pred = (pred > 0.5).astype(int)
        y_test.append(y.numpy())
        y_pred.append(pred)
    
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)

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
    if len(sys.argv) > 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    main(data_path)