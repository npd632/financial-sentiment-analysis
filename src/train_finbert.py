#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_finbert.py
Description: Loads FinancialPhraseBank dataset, preprocesses text, fine-tunes the FinBERT model,
             evaluates performance, and saves the model, tokenizer, and metrics.
"""

import os
import re
import json
import yaml
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EvalPrediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentDataset(torch.utils.data.Dataset):
    """Custom Dataset class for Hugging Face Trainer."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_config(config_path="config.yaml"):
    """Loads YAML configuration file."""
    logger.info(f"Loading config from {config_path}...")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        raise

def clean_text(text):
    """
    Cleans raw text by removing HTML tags, URLs, special characters, 
    and extra whitespaces, and converting to lowercase.
    (Kept identical to baseline cleaning for consistency).
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers (keeping only alphabet and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(df, text_col="Sentence", label_col="Sentiment"):
    """Validates and cleans the dataframe."""
    logger.info("Preprocessing and cleaning data...")
    # Make a copy to avoid modifications to original
    df = df.copy()
    
    # Check if columns exist
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Required columns {text_col} and/or {label_col} not found in dataset.")
        
    # Drop rows with null values in text or label
    df = df.dropna(subset=[text_col, label_col])
    
    # Clean the text column
    df["cleaned_text"] = df[text_col].apply(clean_text)
    
    # Filter out empty clean text
    df = df[df["cleaned_text"] != ""]
    
    logger.info(f"Preprocessed dataset contains {len(df)} samples.")
    return df

def compute_metrics(eval_pred: EvalPrediction):
    """Compute basic evaluation metrics for Trainer callbacks."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, preds)
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted
    }

def train_and_evaluate(config):
    """Main pipeline to load, tokenize, fine-tune, evaluate and save FinBERT model."""
    # Retrieve configuration parameters
    try:
        phrasebank_path = config["data"]["phrasebank_path"]
        train_split = config["data"]["train_split"]
        random_seed = config["data"]["random_seed"]
        
        pretrained_model = config["models"]["finbert"]["pretrained_model"]
        max_length = config["models"]["finbert"]["max_length"]
        batch_size = config["models"]["finbert"]["batch_size"]
        epochs = config["models"]["finbert"]["epochs"]
        learning_rate = float(config["models"]["finbert"]["learning_rate"])
        models_save_dir = config["models"]["finbert"]["save_path"]
        
        metrics_save_path = config["evaluation"]["metrics_module2_path"]
    except KeyError as e:
        logger.error(f"Missing required configuration key in config: {e}")
        raise

    # Load dataset
    if not os.path.exists(phrasebank_path):
        raise FileNotFoundError(f"Dataset not found at: {phrasebank_path}")
    
    logger.info(f"Loading dataset from {phrasebank_path}...")
    df = pd.read_csv(phrasebank_path)
    
    # Preprocess dataset
    df = preprocess_data(df)
    
    # Map labels to numeric IDs
    label2id = {"neutral": 0, "positive": 1, "negative": 2}
    id2label = {0: "neutral", 1: "positive", 2: "negative"}
    df["label_id"] = df["Sentiment"].map(label2id)
    
    # Split dataset into train and test sets
    logger.info(f"Splitting dataset: train_split={train_split}, random_seed={random_seed}")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label_id"],
        test_size=(1.0 - train_split),
        random_state=random_seed,
        stratify=df["label_id"]
    )
    
    # Load BertTokenizer
    logger.info(f"Loading tokenizer: {pretrained_model}")
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    
    # Tokenize input texts
    logger.info("Tokenizing datasets...")
    train_encodings = tokenizer(list(X_train_raw), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=max_length)
    
    # Create PyTorch datasets
    train_dataset = SentimentDataset(train_encodings, list(y_train))
    test_dataset = SentimentDataset(test_encodings, list(y_test))
    
    # Load BertForSequenceClassification
    logger.info(f"Loading pretrained model: {pretrained_model}")
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Ensure models save directory exists
    os.makedirs(models_save_dir, exist_ok=True)
    
    # Set training arguments
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(models_save_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=random_seed,
        fp16=torch.cuda.is_available() # Use mixed precision if GPU is available
    )
    
    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    logger.info("Starting training (fine-tuning)...")
    trainer.train()
    
    # Evaluate model on test set to get final detailed metrics
    logger.info("Evaluating fine-tuned model...")
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    
    # Compute per-class metrics
    unique_label_ids = [0, 1, 2]
    prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
        labels, preds, labels=unique_label_ids, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for idx, label_id in enumerate(unique_label_ids):
        label_name = id2label[label_id]
        per_class_metrics[label_name] = {
            "precision": float(prec_class[idx]),
            "recall": float(rec_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx])
        }
        
    evaluation_metrics = {
        "finbert": {
            "accuracy": float(accuracy),
            "macro_avg": {
                "precision": float(prec_macro),
                "recall": float(rec_macro),
                "f1_score": float(f1_macro)
            },
            "weighted_avg": {
                "precision": float(prec_weighted),
                "recall": float(rec_weighted),
                "f1_score": float(f1_weighted)
            },
            "per_class": per_class_metrics
        }
    }
    
    logger.info(f"FinBERT - Accuracy: {accuracy:.4f}, Weighted F1: {f1_weighted:.4f}, Macro F1: {f1_macro:.4f}")
    
    # Save the best model and tokenizer
    logger.info(f"Saving best model and tokenizer to {models_save_dir}...")
    trainer.save_model(models_save_dir)
    tokenizer.save_pretrained(models_save_dir)
    
    # Ensure evaluation metrics folder exists
    metrics_dir = os.path.dirname(metrics_save_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
        
    # Save metrics to JSON
    logger.info(f"Saving evaluation metrics to {metrics_save_path}...")
    try:
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_metrics, f, indent=4)
        logger.info("Metrics successfully saved.")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")
        raise

if __name__ == "__main__":
    try:
        config = load_config()
        train_and_evaluate(config)
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed with error: {e}", exc_info=True)
