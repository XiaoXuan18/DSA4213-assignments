"""
Training functions and callbacks
"""

import time
from transformers import Trainer, TrainingArguments, TrainerCallback
import config

class LossTrackerCallback(TrainerCallback):
    """Custom callback to track training & validation loss"""
    def __init__(self, val_dataset):
        self.train_losses = []
        self.val_losses = []
        self.val_dataset = val_dataset
        self.trainer_ref = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        # Track validation loss at the end of each epoch
        val_metrics = self.trainer_ref.evaluate(self.val_dataset)
        self.val_losses.append(val_metrics["eval_loss"])

def train_model(model, train_dataset, val_dataset, compute_metrics, output_dir, training_args_dict=None, seed=config.SEED):
    """
    Train model and return trainer, metrics, and training time
    
    Args:
        model: The model to train
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset
        compute_metrics: Metrics computation function
        output_dir: Directory to save results
        training_args_dict: Dictionary of training arguments (optional)
        seed: Random seed
    
    Returns:
        trainer: Trained Trainer object
        train_time: Training time in seconds
        loss_tracker: Loss tracking callback
    """
    
    # Use default config if not provided
    if training_args_dict is None:
        training_args_dict = config.TRAINING_ARGS.copy()
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args_dict['num_train_epochs'],
        per_device_train_batch_size=training_args_dict['per_device_train_batch_size'],
        per_device_eval_batch_size=training_args_dict['per_device_eval_batch_size'],
        learning_rate=training_args_dict['learning_rate'],
        weight_decay=training_args_dict['weight_decay'],
        logging_steps=training_args_dict['logging_steps'],
        save_strategy=training_args_dict['save_strategy'],
        eval_strategy=training_args_dict['eval_strategy'],
        load_best_model_at_end=training_args_dict['load_best_model_at_end'],
        report_to=[],
        seed=seed,
    )
    
    # Setup loss tracker
    loss_tracker = LossTrackerCallback(val_dataset=val_dataset)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_tracker],
    )
    loss_tracker.trainer_ref = trainer
    
    # Train and measure time
    print(f"\nStarting training... (output: {output_dir})")
    start = time.perf_counter()
    trainer.train()
    end = time.perf_counter()
    train_time = end - start
    
    print(f"Training completed in {train_time/60:.2f} minutes")
    
    return trainer, train_time, loss_tracker

def evaluate_and_print(trainer, val_dataset, test_dataset, model_name="Model"):
    """Evaluate on validation and test sets and print results"""
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*60}")
    
    # Validation
    val_metrics = trainer.evaluate(val_dataset)
    print("\n--- Validation ---")
    print(f"Loss: {val_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"Accuracy: {val_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"F1 (micro): {val_metrics.get('eval_f1_micro', 'N/A'):.4f}")
    print(f"F1 (macro): {val_metrics.get('eval_f1_macro', 'N/A'):.4f}")
    
    # Test
    test_metrics = trainer.evaluate(test_dataset)
    print("\n--- Test ---")
    print(f"Loss: {test_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"Accuracy: {test_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"F1 (micro): {test_metrics.get('eval_f1_micro', 'N/A'):.4f}")
    print(f"F1 (macro): {test_metrics.get('eval_f1_macro', 'N/A'):.4f}")
    
    print(f"{'='*60}\n")
    
    return val_metrics, test_metrics
