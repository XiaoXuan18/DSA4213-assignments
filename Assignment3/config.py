"""
Configuration file for DistilBERT fine-tuning experiment
"""

import torch
import random
import numpy as np

# =============================
# Random Seeds for Reproducibility
# =============================
SEED = 50

def set_seed(seed=SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================
# Device Configuration
# =============================
DEVICE = torch.device("cpu")

# =============================
# Model Configuration
# =============================
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 4
MAX_LENGTH = 128

# =============================
# Label Mapping
# =============================
LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business", 
    4: "Science/Tech"
}

ID2LABEL = LABEL_MAP
LABEL2ID = {v: k for k, v in LABEL_MAP.items()}

# =============================
# LoRA Configuration
# =============================
LORA_CONFIG = {
    'r': 8,
    'lora_alpha': 32,
    'target_modules': ["q_lin", "v_lin"],
    'lora_dropout': 0.1,
    'bias': "none",
    'task_type': "SEQ_CLS"
}

# =============================
# Training Configuration
# =============================
TRAINING_ARGS = {
    'num_train_epochs': 3,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'logging_steps': 50,
    'save_strategy': "epoch",
    'eval_strategy': "epoch",
    'load_best_model_at_end': False,
}

# =============================
# Data Split Configuration
# =============================
TEST_SIZE = 0.2
VAL_TEST_SPLIT = 0.5
SUBSET_RATIO = 0.3  # For creativity experiment

# =============================
# Output Directories
# =============================
OUTPUT_DIRS = {
    'full': "./results_full",
    'lora': "./results_lora",
    'full_30': "./results_full_30",
    'lora_30': "./results_lora_30"
}
