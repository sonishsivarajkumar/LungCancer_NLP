# scripts/train_ml_ner.py

import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Parent directory
sys.path.append(parent_dir)

import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import load_data_from_files

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_ner_training.log"),
            logging.StreamHandler()
        ]
    )

def preprocess_data(df):
    """Preprocess the data for ML model training."""
    # Remove rows with missing values
    df = df.dropna(subset=['entity_text', 'entity_type'])

    # Convert text to lowercase
    df['entity_text'] = df['entity_text'].str.lower()

    # Encode entity types
    label_encoder = LabelEncoder()
    df['entity_type_encoded'] = label_encoder.fit_transform(df['entity_type'])

    return df, label_encoder

def create_and_train_models(X_train, y_train):
    """Create and train multiple ML models."""
    models = {}

    # Random Forest
    rf_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        ('classifier', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])

    # XGBoost
    xgb_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        ('classifier', OneVsRestClassifier(xgb.XGBClassifier(use_label_encoder=False, 
                                                           eval_metric='mlogloss',
                                                           random_state=42)))
    ])

    # Train models
    logging.info("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    models['random_forest'] = rf_pipeline

    logging.info("Training XGBoost model...")
    xgb_pipeline.fit(X_train, y_train)
    models['xgboost'] = xgb_pipeline

    return models

def train_attribute_classifiers(df):
    """Train separate classifiers for attributes."""
    attribute_models = {}

    # Train Status_Certainty classifier for treatment entities
    treatment_entities = ["Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
                          "Immunotherapy", "Targeted_Therapy"]
    treatment_df = df[df['entity_type'].isin(treatment_entities)]

    if not treatment_df.empty and 'Status_Certainty' in treatment_df['attributes'].iloc[0]:
        X = treatment_df['entity_text']
        y = treatment_df['attributes'].apply(lambda x: x.get('Status_Certainty', 'Unknown'))

        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        attribute_models['Status_Certainty'] = pipeline

    # Train Certainty classifier for response entities
    response_entities = ["Complete_Response", "Partial_Response", 
                         "Stable_Disease", "Progressive_Disease"]
    response_df = df[df['entity_type'].isin(response_entities)]

    if not response_df.empty and 'Certainty' in response_df['attributes'].iloc[0]:
        X = response_df['entity_text']
        y = response_df['attributes'].apply(lambda x: x.get('Certainty', 'Unknown'))

        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        attribute_models['Certainty'] = pipeline

    return attribute_models

def main():
    setup_logging()
    logging.info("Starting ML-based NER training")

    # Load train files
    with open('data/train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f.readlines()]

    # Load data using the function from utils.data_loader
    df = load_data_from_files(train_files)

    # Preprocess data
    df_processed, label_encoder = preprocess_data(df)

    # Prepare data
    X_train = df_processed['entity_text']
    y_train = df_processed['entity_type_encoded']

    # Train models
    models = create_and_train_models(X_train, y_train)

    # Train attribute classifiers
    attribute_models = train_attribute_classifiers(df_processed)

    # Create directory for saving models
    os.makedirs('./models/ml_models', exist_ok=True)

    # Save models and label encoder
    for model_name, model in models.items():
        joblib.dump(model, f'./models/ml_models/ner_{model_name}_model.joblib')

    for attr_name, model in attribute_models.items():
        joblib.dump(model, f'./models/ml_models/ner_{attr_name.lower()}_model.joblib')

    joblib.dump(label_encoder, './models/ml_models/ner_label_encoder.joblib')

    logging.info("ML-based NER models have been saved")

if __name__ == "__main__":
    main()
