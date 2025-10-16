# To use this code:
1. Create a new folder
2. Copy all 9 Python files into that folder
3. Run in your terminal: python main.py

# Project Structure

Assignment3/
├── main.py                    # Main execution script
├── config.py                  # Configuration and hyperparameters
├── data_preprocessing.py      # Data loading and preprocessing
├── model_setup.py            # Model initialisation (LoRA & Full)
├── training.py               # Training functions and callbacks
├── evaluation.py             # Evaluation metrics
├── visualisation.py          # Plotting functions
├── error_analysis.py         # Error case analysis
├── utils.py                  # Helper utilities
└── README.md                 # This file

# Module Descriptions
config.py
- Central configuration file
- Contains all hyperparameters, paths, and constants

data_preprocessing.py
- load_and_merge_data(): Loads AG News dataset
- splot_data(): Splits into train/val/test (80/10/10)
- tokenize_datasets(): Tokenises text using DistilBERT tokenizer
- create_subset(): Creates 30% subset for creativity experiment

model_setup.py
- setup_full_model(): Initialize model for full fine-tuning
- setup_lora_model(): Initialize model with LoRA adapters
- count_trainable_params(): Count trainable parameters

training.py
- LossTrackerCallback: Custom callback to track train/val loss
- train_model(): Main training function
- evaluate_and_print(): Evaluate and display results

evaluation.py
- compute_metrics(): Calculate accuracy, F1-micro, F1-macro
- get_predictions(): Get model predictions

visualisation.py
- plot_text_length_histogram(): Visualize text length distribution
- plot_category_distribution(): Visualize class balance
- plot_training_curves(): Plot train/val/test loss curves
- plot_confusion_matrix(): Visualize confusion matrix
- plot_comparison_table(): Display results comparison

error_analysis.py
- analyze_errors(): Extract and categorize misclassifications
- print_error_summary(): Display error statistics
- get_error_examples(): Get sample error cases
- compute_confusion_matrix(): Create confusion matrix for evaluation

utils.py
- collect_experiment_results(): Package all metrics
- save_results_to_json(): Save results to JSON
- save_error_cases_to_csv(): Export error cases
- print_comparison_insights(): Display comparative insights

main.py
Orchestrates entire experiment workflow
Runs all 4 experiments:
1. Full Fine-tuning (100% data)
2. LoRA (100% data)
3. Full Fine-tuning (30% data)
4. LoRA (30% data)
