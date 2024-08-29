# F1- Score
def f1_score(precision, recall):
    """
    F1 score metric
    """
    return 2.0 * (precision * recall) / (precision + recall)