"""
Evaluation metrics computation
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    """Compute accuracy and F1 scores"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    
    return {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

def get_predictions(trainer, test_dataset):
    """Get predictions from trainer"""
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    return y_true, y_pred, predictions
