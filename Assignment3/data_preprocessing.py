"""
Data loading and preprocessing functions
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import config

def load_and_merge_data():
    """Load AG News dataset and merge train/test"""
    print("Loading AG News dataset...")
    dataset = load_dataset("sh0416/ag_news")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset['test'])
    
    # Merge into one table
    df_merge = pd.concat([train_df, test_df], ignore_index=True)
    
    # Map numeric labels to category names
    df_merge["category"] = df_merge["label"].map(config.LABEL_MAP)
    
    # Create text length column
    df_merge["text length"] = df_merge["description"].apply(lambda x: len(x.split()))
    
    print(f"Total samples: {len(df_merge)}")
    print(f"Label distribution:\n{df_merge['category'].value_counts()}")
    
    return df_merge

def split_data(df_merge, test_size=config.TEST_SIZE, val_test_split=config.VAL_TEST_SPLIT, seed=config.SEED):
    """Split data into train, validation, and test sets"""
    print("\nSplitting data...")
    
    # First split: train vs (val + test)
    train_dataset, temp_dataset = train_test_split(
        df_merge, 
        test_size=test_size, 
        stratify=df_merge['label'], 
        random_state=seed
    )
    
    # Second split: val vs test
    val_dataset, test_dataset = train_test_split(
        temp_dataset, 
        test_size=val_test_split, 
        stratify=temp_dataset["label"], 
        random_state=seed
    )
    
    print(f"Train: {train_dataset.shape[0]}, Val: {val_dataset.shape[0]}, Test: {test_dataset.shape[0]}")
    
    # Convert back to HuggingFace Dataset format
    train_dataset = Dataset.from_pandas(train_dataset.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_dataset.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_dataset.reset_index(drop=True))
    
    return train_dataset, val_dataset, test_dataset

def tokenize_datasets(train_dataset, val_dataset, test_dataset, model_name=config.MODEL_NAME, max_length=config.MAX_LENGTH):
    """Tokenize all datasets"""
    print("\nTokenizing datasets...")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    def preprocess(batch):
        tokens = tokenizer(
            batch["description"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        tokens["labels"] = batch["label"]
        return tokens
    
    # Apply tokenization
    train_tokenized = train_dataset.map(preprocess, batched=True)
    val_tokenized = val_dataset.map(preprocess, batched=True)
    test_tokenized = test_dataset.map(preprocess, batched=True)
    
    # Set PyTorch format
    columns = ["input_ids", "attention_mask", "labels"]
    train_tokenized.set_format(type="torch", columns=columns)
    val_tokenized.set_format(type="torch", columns=columns)
    test_tokenized.set_format(type="torch", columns=columns)
    
    print("Tokenization complete!")
    
    return train_tokenized, val_tokenized, test_tokenized, tokenizer

def create_subset(train_dataset_tokenised, subset_ratio=config.SUBSET_RATIO, seed=config.SEED):
    """Create stratified subset of training data"""
    print(f"\nCreating {subset_ratio*100}% subset...")
    
    labels = np.array(train_dataset_tokenised["labels"])
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    subset_indices = []
    
    # Stratified sampling
    for label, count in zip(unique_labels, counts):
        label_indices = np.where(labels == label)[0]
        n_select = int(count * subset_ratio)
        np.random.seed(seed)
        selected = np.random.choice(label_indices, size=n_select, replace=False)
        subset_indices.extend(selected)
    
    np.random.seed(seed)
    np.random.shuffle(subset_indices)
    
    train_subset = train_dataset_tokenised.select(subset_indices)
    
    print(f"Original train size: {len(train_dataset_tokenised)}")
    print(f"{subset_ratio*100}% subset size: {len(train_subset)}")
    print("Label distribution:", np.unique(train_subset["labels"], return_counts=True))
    
    return train_subset
