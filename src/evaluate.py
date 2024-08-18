import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_model(model, test_dataset):

    y_true, y_pred = zip(*[
        (y_test.numpy(), (np.array(model.predict(X_test)) > 0.5).astype(int))
        for X_test, y_test in test_dataset
    ])

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1