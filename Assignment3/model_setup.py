"""
Model setup functions for Full Fine-tuning and LoRA
"""

import torch
from transformers import DistilBertForSequenceClassification
from peft import LoraConfig, get_peft_model
import config

def setup_full_model(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS, device=config.DEVICE):
    """Setup model for full fine-tuning"""
    print(f"\nSetting up Full Fine-tuning model...")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)
    
    total, trainable = count_trainable_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    
    return model

def setup_lora_model(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS, device=config.DEVICE, lora_config_dict=config.LORA_CONFIG):
    """Setup model with LoRA adapters"""
    print(f"\nSetting up LoRA model...")
    
    # Load base model
    base_model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['lora_alpha'],
        target_modules=lora_config_dict['target_modules'],
        lora_dropout=lora_config_dict['lora_dropout'],
        bias=lora_config_dict['bias'],
        task_type=lora_config_dict['task_type']
    )
    
    # Wrap with LoRA
    model_lora = get_peft_model(base_model, lora_config)
    model_lora.to(device)
    
    # Ensure classifier head is trainable
    for name, param in model_lora.named_parameters():
        if "classifier" in name or "pre_classifier" in name or "pre_class" in name:
            param.requires_grad = True
    
    total, trainable = count_trainable_params(model_lora)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    
    return model_lora

def count_trainable_params(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_model_modules(model, max_lines=200):
    """Print model module names for debugging"""
    print(f"\n----- Model named modules (first {max_lines} shown) -----")
    for i, (name, module) in enumerate(model.named_modules()):
        if i < max_lines:
            print(i, name, type(module))
        else:
            break
    print("----- end module listing -----\n")
