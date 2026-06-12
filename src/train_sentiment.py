#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_sentiment.py
Description: Loads FinancialPhraseBank dataset, preprocesses text, trains Naive Bayes and SVM baseline models,
             evaluates performance, and saves the models and metrics.
"""

import os
import re
import json
import yaml
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

def train_and_evaluate(config):
    """Main pipeline to load, preprocess, train, evaluate and save models."""
    # Retrieve configuration parameters
    try:
        phrasebank_path = config["data"]["phrasebank_path"]
        train_split = config["data"]["train_split"]
        random_seed = config["data"]["random_seed"]
        
        tfidf_max_features = config["models"]["baseline"]["tfidf_max_features"]
        svm_kernel = config["models"]["baseline"]["svm_kernel"]
        naive_bayes_alpha = config["models"]["baseline"]["naive_bayes_alpha"]
        models_save_dir = config["models"]["baseline"]["save_path"]
        
        metrics_save_path = config["evaluation"]["metrics_module1_path"]
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
    
    # Split dataset into train and test sets
    logger.info(f"Splitting dataset: train_split={train_split}, random_seed={random_seed}")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["Sentiment"],
        test_size=(1.0 - train_split),
        random_state=random_seed,
        stratify=df["Sentiment"]
    )
    
    # TF-IDF Vectorization
    logger.info(f"Fitting TF-IDF Vectorizer with max_features={tfidf_max_features}")
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # Define models
    models = {
        "naive_bayes": MultinomialNB(alpha=naive_bayes_alpha),
        "svm": SVC(kernel=svm_kernel, probability=True, random_state=random_seed)
    }
    
    # Dictionary to save results
    evaluation_metrics = {}
    
    # Ensure models save directory exists
    os.makedirs(models_save_dir, exist_ok=True)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate macro and weighted precision, recall, f1
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        
        # Calculate per-class precision, recall, f1
        unique_labels = sorted(list(y_test.unique()))
        prec_class, rec_class, f1_class, support_class = precision_recall_fscore_support(
            y_test, y_pred, labels=unique_labels, average=None
        )
        
        per_class_metrics = {}
        for idx, label in enumerate(unique_labels):
            per_class_metrics[label] = {
                "precision": float(prec_class[idx]),
                "recall": float(rec_class[idx]),
                "f1_score": float(f1_class[idx]),
                "support": int(support_class[idx])
            }
            
        evaluation_metrics[model_name] = {
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
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Weighted F1: {f1_weighted:.4f}, Macro F1: {f1_macro:.4f}")
        
        # Save model to disk
        model_filename = os.path.join(models_save_dir, f"{model_name}.pkl")
        logger.info(f"Saving {model_name} model to {model_filename}...")
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
            
    # Save vectorizer as well
    vectorizer_filename = os.path.join(models_save_dir, "tfidf_vectorizer.pkl")
    logger.info(f"Saving TF-IDF Vectorizer to {vectorizer_filename}...")
    with open(vectorizer_filename, "wb") as f:
        pickle.dump(vectorizer, f)
        
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
