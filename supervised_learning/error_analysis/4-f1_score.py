#!/usr/bin/env python3
"""A function that calculates the F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion: np.ndarray) -> List[float]:
    """A function that calculates the F1 score of a confusion
    matrix"""
    classes = confusion.shape[0]
    f1_scores = []
    for i in range(classes):
        tp = confusion[i][i]
        fp = sum(confusion[:, i]) - tp
        fn = sum(confusion[i, :]) - tp
        p = precision(confusion)[i]
        r = sensitivity(confusion)[i]
        if p + r != 0:
            f1_scores[i] = 2 * (p * r) / (p + r)
    return f1_scores  
