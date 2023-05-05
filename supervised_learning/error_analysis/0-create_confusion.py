#!/usr/bin/env python3
"""Convert one-hot labels and logits to class indices"""
import numpy as np

def create_confusion_matrix(labels, logits):
    """Convert one-hot labels and logits to class indices"""
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)
    
    # Create confusion matrix
    confusion = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.int32)
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1
        
    return confusion
