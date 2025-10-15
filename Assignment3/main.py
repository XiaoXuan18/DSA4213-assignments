"""
Main execution script for DistilBERT Fine-tuning Experiments
Compares Full Fine-tuning vs LoRA on AG News dataset
"""

# =============================
# Imports
# =============================
import config
from data_preprocessing import (
    load_and_merge_data,
    split_data,
    tokenize_datasets,
    create_subset
)
from visualisation import (
    plot_text_length_histogram,
    plot_category_distribution,
    plot_training_curves,
    plot_confusion_matrix,
    plot_comparison_table
)
from model_setup import setup_full_model, setup_lora_model
from training import train_model, evaluate_and_print
from evaluation import compute_metrics, get_predictions
from error_analysis import (
    analyze_errors,
    print_error_summary,
    get_error_examples
)
from utils import (
    collect_experiment_results,
    save_results_to_json,
    save_error_cases_to_csv,
    print_comparison_insights
)

# =============================
# Main Execution
# =============================

def main():
    print("="*80)
    print("DISTILBERT FINE-TUNING EXPERIMENT")
    print("Comparing Full Fine-tuning vs LoRA on AG News Dataset")
    print("="*80)
    
    # Set seed for reproducibility
    config.set_seed()
    
    # =============================
    # 1. Data Loading & Preprocessing
    # =============================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    
    df_merge = load_and_merge_data()
    
    # Visualizations
    plot_text_length_histogram(df_merge)
    plot_category_distribution(df_merge)
    
    # Split data
    train_dataset, val_dataset, test_dataset = split_data(df_merge)
    
    # Tokenize
    train_tokenized, val_tokenized, test_tokenized, tokenizer = tokenize_datasets(
        train_dataset, val_dataset, test_dataset
    )
    
    # Create 30% subset for creativity experiment
    train_tokenized_30 = create_subset(train_tokenized)
    
    # =============================
    # 2. Full Fine-Tuning (100% data)
    # =============================
    print("\n" + "="*80)
    print("STEP 2: FULL FINE-TUNING (100% DATA)")
    print("="*80)
    
    model_full = setup_full_model()
    
    trainer_full, train_time_full, loss_tracker_full = train_model(
        model=model_full,
        train_dataset=train_tokenized,
        val_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        output_dir=config.OUTPUT_DIRS['full']
    )
    
    val_metrics_full, test_metrics_full = evaluate_and_print(
        trainer_full, val_tokenized, test_tokenized, "Full Fine-tuning"
    )
    
    # Plot training curves
    plot_training_curves(
        loss_tracker_full, test_metrics_full, config.TRAINING_ARGS,
        len(train_tokenized), "Full Fine-Tuning: Loss Across Epochs"
    )
    
    # Error analysis
    y_true_full, y_pred_full, _ = get_predictions(trainer_full, test_tokenized)
    error_cases_full, error_count_full = analyze_errors(y_true_full, y_pred_full, test_dataset)
    print_error_summary(error_cases_full, error_count_full, "Full Fine-tuning")
    plot_confusion_matrix(y_true_full, y_pred_full, config.ID2LABEL, "Confusion Matrix: Full Fine-Tuning")
    
    # Collect results
    results_full = collect_experiment_results(
        model_full, trainer_full, train_time_full, val_metrics_full,
        test_metrics_full, loss_tracker_full, "Full Fine-tuning"
    )
    
    # =============================
    # 3. LoRA Fine-Tuning (100% data)
    # =============================
    print("\n" + "="*80)
    print("STEP 3: LORA FINE-TUNING (100% DATA)")
    print("="*80)
    
    model_lora = setup_lora_model()
    
    trainer_lora, train_time_lora, loss_tracker_lora = train_model(
        model=model_lora,
        train_dataset=train_tokenized,
        val_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        output_dir=config.OUTPUT_DIRS['lora']
    )
    
    val_metrics_lora, test_metrics_lora = evaluate_and_print(
        trainer_lora, val_tokenized, test_tokenized, "LoRA"
    )
    
    # Plot training curves
    plot_training_curves(
        loss_tracker_lora, test_metrics_lora, config.TRAINING_ARGS,
        len(train_tokenized), "LoRA: Loss Across Epochs"
    )
    
    # Error analysis
    y_true_lora, y_pred_lora, _ = get_predictions(trainer_lora, test_tokenized)
    error_cases_lora, error_count_lora = analyze_errors(y_true_lora, y_pred_lora, test_dataset)
    print_error_summary(error_cases_lora, error_count_lora, "LoRA")
    plot_confusion_matrix(y_true_lora, y_pred_lora, config.ID2LABEL, "Confusion Matrix: LoRA")
    
    # Collect results
    results_lora = collect_experiment_results(
        model_lora, trainer_lora, train_time_lora, val_metrics_lora,
        test_metrics_lora, loss_tracker_lora, "LoRA"
    )
    
    # =============================
    # 4. Full Fine-Tuning (30% data)
    # =============================
    print("\n" + "="*80)
    print("STEP 4: FULL FINE-TUNING (30% DATA - CREATIVITY)")
    print("="*80)
    
    model_full_30 = setup_full_model()
    
    trainer_full_30, train_time_full_30, loss_tracker_full_30 = train_model(
        model=model_full_30,
        train_dataset=train_tokenized_30,
        val_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        output_dir=config.OUTPUT_DIRS['full_30']
    )
    
    val_metrics_full_30, test_metrics_full_30 = evaluate_and_print(
        trainer_full_30, val_tokenized, test_tokenized, "Full Fine-tuning (30%)"
    )
    
    # Plot training curves
    plot_training_curves(
        loss_tracker_full_30, test_metrics_full_30, config.TRAINING_ARGS,
        len(train_tokenized_30), "Full Fine-Tuning (30%): Loss Across Epochs"
    )
    
    # Error analysis
    y_true_full_30, y_pred_full_30, _ = get_predictions(trainer_full_30, test_tokenized)
    error_cases_full_30, error_count_full_30 = analyze_errors(y_true_full_30, y_pred_full_30, test_dataset)
    print_error_summary(error_cases_full_30, error_count_full_30, "Full Fine-tuning (30%)")
    plot_confusion_matrix(y_true_full_30, y_pred_full_30, config.ID2LABEL, "Confusion Matrix: Full Fine-Tuning (30%)")
    
    # Collect results
    results_full_30 = collect_experiment_results(
        model_full_30, trainer_full_30, train_time_full_30, val_metrics_full_30,
        test_metrics_full_30, loss_tracker_full_30, "Full Fine-tuning (30%)"
    )
    
    # =============================
    # 5. LoRA Fine-Tuning (30% data)
    # =============================
    print("\n" + "="*80)
    print("STEP 5: LORA FINE-TUNING (30% DATA - CREATIVITY)")
    print("="*80)
    
    model_lora_30 = setup_lora_model()
    
    trainer_lora_30, train_time_lora_30, loss_tracker_lora_30 = train_model(
        model=model_lora_30,
        train_dataset=train_tokenized_30,
        val_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        output_dir=config.OUTPUT_DIRS['lora_30']
    )
    
    val_metrics_lora_30, test_metrics_lora_30 = evaluate_and_print(
        trainer_lora_30, val_tokenized, test_tokenized, "LoRA (30%)"
    )
    
    # Plot training curves
    plot_training_curves(
        loss_tracker_lora_30, test_metrics_lora_30, config.TRAINING_ARGS,
        len(train_tokenized_30), "LoRA (30%): Loss Across Epochs"
    )
    
    # Error analysis
    y_true_lora_30, y_pred_lora_30, _ = get_predictions(trainer_lora_30, test_tokenized)
    error_cases_lora_30, error_count_lora_30 = analyze_errors(y_true_lora_30, y_pred_lora_30, test_dataset)
    print_error_summary(error_cases_lora_30, error_count_lora_30, "LoRA (30%)")
    plot_confusion_matrix(y_true_lora_30, y_pred_lora_30, config.ID2LABEL, "Confusion Matrix: LoRA (30%)")
    
    # Collect results
    results_lora_30 = collect_experiment_results(
        model_lora_30, trainer_lora_30, train_time_lora_30, val_metrics_lora_30,
        test_metrics_lora_30, loss_tracker_lora_30, "LoRA (30%)"
    )
    
    # =============================
    # 6. Final Comparison & Analysis
    # =============================
    print("\n" + "="*80)
    print("STEP 6: FINAL COMPARISON & ANALYSIS")
    print("="*80)
    
    all_results = {
        "Full Fine-tuning": results_full,
        "LoRA": results_lora,
        "Full Fine-tuning (30%)": results_full_30,
        "LoRA (30%)": results_lora_30
    }
    
    # Print comparison table
    comparison_df = plot_comparison_table(all_results)
    
    # Print insights
    print_comparison_insights(all_results)
    
    # Save results
    save_results_to_json(all_results, "experiment_results.json")
    save_error_cases_to_csv(error_cases_full, "error_cases_full.csv")
    save_error_cases_to_csv(error_cases_lora, "error_cases_lora.csv")
    save_error_cases_to_csv(error_cases_full_30, "error_cases_full_30.csv")
    save_error_cases_to_csv(error_cases_lora_30, "error_cases_lora_30.csv")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    print("\nAll results have been saved.")
    print("Check the output directories for model checkpoints.")

if __name__ == "__main__":
    main()
