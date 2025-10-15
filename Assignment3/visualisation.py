"""
Visualisation functions for data exploration and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_text_length_histogram(df_merge):
    """Plot histogram of text lengths"""
    plt.figure(figsize=(8, 5))
    plt.hist(df_merge["text length"], bins=10, color="skyblue", edgecolor="black")
    plt.title("Histogram of Text Lengths")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    print(f"Max text length: {df_merge['text length'].max()}")

def plot_category_distribution(df_merge):
    """Plot bar chart of category distribution"""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_merge, x='category', order=df_merge['category'].value_counts().index)
    plt.title("Number of Samples per Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    print("\nCategory counts:")
    print(df_merge['category'].value_counts())

def plot_training_curves(loss_tracker, test_metrics, training_args, train_dataset_size, title="Loss Across Epochs"):
    """Plot training, validation, and test loss"""
    
    # Calculate average train loss per epoch
    steps_per_epoch = train_dataset_size // training_args['per_device_train_batch_size']
    logging_interval = training_args['logging_steps']
    logs_per_epoch = steps_per_epoch // logging_interval
    
    train_loss_per_epoch = [
        np.mean(loss_tracker.train_losses[i*logs_per_epoch:(i+1)*logs_per_epoch])
        for i in range(len(loss_tracker.val_losses))
    ]
    
    # Validation loss per epoch
    val_loss_per_epoch = loss_tracker.val_losses
    
    # Test loss (flat line)
    test_loss = test_metrics["eval_loss"]
    test_loss_per_epoch = [test_loss] * len(train_loss_per_epoch)
    
    # Plot
    epochs = range(1, len(train_loss_per_epoch) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_per_epoch, marker='o', label="Train Loss")
    plt.plot(epochs, val_loss_per_epoch, marker='o', label="Validation Loss")
    plt.plot(epochs, test_loss_per_epoch, marker='o', label="Test Loss (final)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title(title)
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, id2label, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    labels = [id2label[i] for i in sorted(set(y_true) | set(y_pred))]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def plot_comparison_table(results_dict):
    """Print comparison table of all experiments"""
    import pandas as pd
    
    comparison_df = pd.DataFrame(results_dict).T
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(comparison_df.to_string())
    print("="*80)
    
    return comparison_df
