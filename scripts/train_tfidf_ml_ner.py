# scripts/train_tfidf_ml_ner.py
import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

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
    df = df.dropna(subset=['text', 'entity_type'])
    
    # Convert text to lowercase
    df['text'] = df['text'].str.lower()
    
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
    treatment_mask = df['entity_type'].isin([
        "Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
        "Immunotherapy", "Targeted_Therapy"
    ])
    
    if treatment_mask.any():
        treatment_df = df[treatment_mask]
        if 'Status_Certainty' in treatment_df.columns:
            X = treatment_df['text']
            y = treatment_df['Status_Certainty']
            
            pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            pipeline.fit(X, y)
            attribute_models['Status_Certainty'] = pipeline
    
    # Train Certainty classifier for response entities
    response_mask = df['entity_type'].isin([
        "Complete_Response", "Partial_Response", 
        "Stable_Disease", "Progressive_Disease"
    ])
    
    if response_mask.any():
        response_df = df[response_mask]
        if 'Certainty' in response_df.columns:
            X = response_df['text']
            y = response_df['Certainty']
            
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

    # Define the base folder path and subfolders
    base_folder = r"C:\Users\sivarajkumars2\OneDrive - UPMC\Documents\Lung_cancer_project\Annotation_shared\LC_Annotations\Annotated Files\Round_4\Subhash"
    folders = [
        "Completed Partial Response",
        "Completed Progressive Disease",
        "Completed RECIST complete response",
        "Completed Stable Disease",
    ]

    # Load data using the function from rule-based NER
    from train_rule_based_ner import load_xml_data
    df = load_xml_data(base_folder, folders)
    
    # Preprocess data
    df_processed, label_encoder = preprocess_data(df)
    
    # Split the data
    X = df_processed['text']
    y = df_processed['entity_type_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    # Save train/test split for evaluation
    np.save('./models/ml_models/X_test.npy', X_test)
    np.save('./models/ml_models/y_test.npy', y_test)
    
    logging.info("ML-based NER models have been saved")

if __name__ == "__main__":
    main()