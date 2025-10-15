"""
Utility helper functions
"""

import json
import pandas as pd
from model_setup import count_trainable_params

def save_results_to_json(results_dict, filename="experiment_results.json"):
    """Save experiment results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {filename}")

def save_error_cases_to_csv(error_cases, filename="error_cases.csv"):
    """Save error cases to CSV"""
    error_cases.to_csv(filename, index=False)
    print(f"Error cases saved to {filename}")

def collect_experiment_results(model, trainer, train_time, val_metrics, test_metrics, loss_tracker, model_name):
    """
    Collect all results from an experiment into a dictionary
    
    Returns:
        Dictionary with all relevant metrics
    """
    total_params, trainable_params = count_trainable_params(model)
    
    results = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "percent_trainable": round(trainable_params / total_params * 100, 2),
        "train_time_mins": round(train_time / 60, 2),
        "val_loss": round(val_metrics.get('eval_loss', 0), 4),
        "val_accuracy": round(val_metrics.get('eval_accuracy', 0), 4),
        "val_f1_micro": round(val_metrics.get('eval_f1_micro', 0), 4),
        "val_f1_macro": round(val_metrics.get('eval_f1_macro', 0), 4),
        "test_loss": round(test_metrics.get('eval_loss', 0), 4),
        "test_accuracy": round(test_metrics.get('eval_accuracy', 0), 4),
        "test_f1_micro": round(test_metrics.get('eval_f1_micro', 0), 4),
        "test_f1_macro": round(test_metrics.get('eval_f1_macro', 0), 4),
        "train_losses_sample": loss_tracker.train_losses[:5],
        "val_losses_per_epoch": loss_tracker.val_losses,
    }
    
    return results

def print_comparison_insights(results_dict):
    """Print insights comparing different experiments"""
    print("\n" + "="*80)
    print("INSIGHTS & COMPARISONS")
    print("="*80)
    
    if "Full Fine-tuning" in results_dict and "LoRA" in results_dict:
        full = results_dict["Full Fine-tuning"]
        lora = results_dict["LoRA"]
        
        print("\n--- LoRA vs Full Fine-tuning ---")
        print(f"Parameter Reduction: {full['trainable_params']:,} → {lora['trainable_params']:,}")
        print(f"  ({lora['percent_trainable']:.2f}% trainable with LoRA)")
        print(f"\nTraining Time: {full['train_time_mins']:.2f} mins → {lora['train_time_mins']:.2f} mins")
        time_reduction = (1 - lora['train_time_mins'] / full['train_time_mins']) * 100
        print(f"  ({time_reduction:.1f}% reduction)")
        print(f"\nTest Accuracy: {full['test_accuracy']:.4f} → {lora['test_accuracy']:.4f}")
        acc_diff = (lora['test_accuracy'] - full['test_accuracy']) * 100
        print(f"  ({acc_diff:+.2f}% difference)")
    
    if "Full Fine-tuning (30%)" in results_dict and "LoRA (30%)" in results_dict:
        full_30 = results_dict["Full Fine-tuning (30%)"]
        lora_30 = results_dict["LoRA (30%)"]
        full = results_dict["Full Fine-tuning"]
        lora = results_dict["LoRA"]
        
        print("\n--- 30% Dataset Performance Retention ---")
        full_retention = full_30['test_accuracy'] / full['test_accuracy'] * 100
        lora_retention = lora_30['test_accuracy'] / lora['test_accuracy'] * 100
        print(f"Full Fine-tuning: {full['test_accuracy']:.4f} → {full_30['test_accuracy']:.4f}")
        print(f"  (Retained {full_retention:.1f}% of performance)")
        print(f"LoRA: {lora['test_accuracy']:.4f} → {lora_30['test_accuracy']:.4f}")
        print(f"  (Retained {lora_retention:.1f}% of performance)")
        
        if lora_retention > full_retention:
            print(f"\n✓ LoRA shows better generalization in low-resource settings!")
    
    print("\n" + "="*80)
