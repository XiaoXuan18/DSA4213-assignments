"""
Error analysis functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import config

def analyze_errors(y_true, y_pred, test_dataset, id2label=config.ID2LABEL):
    """
    Analyze misclassified samples
    
    Returns:
        error_cases: DataFrame with misclassified samples
        error_count_per_class: Count of errors per category
    """
    
    # Create DataFrame with predictions
    df_errors = pd.DataFrame({
        "text": test_dataset["description"],
        "true_label": y_true,
        "predicted_label": y_pred
    })
    
    # Filter misclassified samples
    error_cases = df_errors[df_errors["true_label"] != df_errors["predicted_label"]].copy()
    
    # Map labels to names
    error_cases["true_label_name"] = error_cases["true_label"].map(id2label)
    error_cases["predicted_label_name"] = error_cases["predicted_label"].map(id2label)
    
    # Count errors per class
    error_count_per_class = (
        error_cases["true_label_name"]
        .value_counts()
        .rename_axis("True Label")
        .reset_index(name="Error Count")
    )
    
    return error_cases, error_count_per_class

def print_error_summary(error_cases, error_count_per_class, model_name="Model"):
    """Print error analysis summary"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Error Analysis")
    print(f"{'='*60}")
    print(f"\nTotal misclassified samples: {len(error_cases)}")
    print(f"\nError count per category:")
    print(error_count_per_class.to_string(index=False))
    print(f"\n{'='*60}\n")

def get_error_examples(error_cases, n=10):
    """Get first n error examples"""
    return error_cases[["text", "true_label_name", "predicted_label_name"]].head(n)

def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix"""
    return confusion_matrix(y_true, y_pred)
