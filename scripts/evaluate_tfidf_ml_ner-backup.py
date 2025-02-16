# scripts/evaluate_tfidf_ml_ner.py

import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from utils.data_loader import load_data_from_files

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_ner_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def load_models():
    """Load all trained ML models."""
    models = {}
    try:
        models['random_forest'] = joblib.load('./models/ml_models/ner_random_forest_model.joblib')
        models['xgboost'] = joblib.load('./models/ml_models/ner_xgboost_model.joblib')
        models['label_encoder'] = joblib.load('./models/ml_models/ner_label_encoder.joblib')
        
        # Load attribute models
        status_path = './models/ml_models/ner_status_certainty_model.joblib'
        if os.path.exists(status_path):
            models['Status_Certainty'] = joblib.load(status_path)
        
        cert_path = './models/ml_models/ner_certainty_model.joblib'
        if os.path.exists(cert_path):
            models['Certainty'] = joblib.load(cert_path)
        
        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def evaluate_entity_classification(model, X_test, y_test, label_encoder, model_name):
    """Evaluate entity classification performance."""
    y_pred = model.predict(X_test)
    
    # Convert encoded labels back to original classes
    y_test_classes = label_encoder.inverse_transform(y_test)
    y_pred_classes = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    report = classification_report(y_test_classes, y_pred_classes, zero_division=0)
    cm = confusion_matrix(y_test_classes, y_pred_classes, labels=label_encoder.classes_)
    
    # Plot confusion matrix
    results_dir = os.path.join('results', 'ml', 'ner')
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"ml_ner_{model_name.lower()}_confusion_matrix.png"))
    plt.close()
    
    # Save classification report
    report_file = os.path.join(results_dir, f"ml_ner_{model_name.lower()}_classification_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)

    return report

def evaluate_attributes(model, X_test, y_test, attr_name):
    """Evaluate attribute classification performance."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Calculate per-class accuracy
    accuracy_per_class = {}
    for label in set(y_test):
        mask = (y_test == label)
        if mask.any():
            accuracy = (y_pred[mask] == y_test[mask]).mean()
            accuracy_per_class[label] = accuracy
    
    return report, accuracy_per_class

def main():
    setup_logging()
    logging.info("Starting ML-based NER evaluation (TF-IDF)")
    
    # Load test files
    test_files_path = 'data/test_files.txt'
    if not os.path.exists(test_files_path):
        logging.error("Test files list 'test_files.txt' does not exist.")
        return
    
    with open(test_files_path, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]

    # Load test data
    df_test = load_data_from_files(test_files)
    df_test = df_test.dropna(subset=['entity_text', 'entity_type'])
    df_test['entity_text'] = df_test['entity_text'].str.lower()
    
    # Load models
    models = load_models()
    label_encoder = models['label_encoder']
    df_test['entity_type_encoded'] = label_encoder.transform(df_test['entity_type'])
    
    X_test = df_test['entity_text']
    y_test = df_test['entity_type_encoded']
    
    # Evaluate entity classification for each model
    for model_name in ['random_forest', 'xgboost']:
        logging.info(f"Evaluating {model_name} model...")
        report = evaluate_entity_classification(
            models[model_name], X_test, y_test, 
            label_encoder, model_name
        )
        
        print(f"\nClassification Report - {model_name}:")
        print(report)
    
    # Evaluate attributes
    attribute_results = {}
    
    # Evaluate Status_Certainty for treatment entities
    treatment_entities = ["Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
                          "Immunotherapy", "Targeted_Therapy"]
    treatment_mask = df_test['entity_type'].isin(treatment_entities)
    if treatment_mask.any() and 'Status_Certainty' in df_test.columns:
        treatment_df = df_test[treatment_mask]
        X_attr = treatment_df['entity_text']
        y_attr = treatment_df['Status_Certainty']
        if 'Status_Certainty' in models:
            report, accuracies = evaluate_attributes(models['Status_Certainty'], X_attr, y_attr, 'Status_Certainty')
            attribute_results['Status_Certainty'] = {
                'report': report,
                'accuracies': accuracies
            }
    
    # Evaluate Certainty for response entities
    response_entities = ["Complete_Response", "Partial_Response", 
                         "Stable_Disease", "Progressive_Disease"]
    response_mask = df_test['entity_type'].isin(response_entities)
    if response_mask.any() and 'Certainty' in df_test.columns:
        response_df = df_test[response_mask]
        X_attr = response_df['entity_text']
        y_attr = response_df['Certainty']
        if 'Certainty' in models:
            report, accuracies = evaluate_attributes(models['Certainty'], X_attr, y_attr, 'Certainty')
            attribute_results['Certainty'] = {
                'report': report,
                'accuracies': accuracies
            }
    
    # Save attribute evaluation results
    results_dir = os.path.join('results', 'ml', 'ner')
    os.makedirs(results_dir, exist_ok=True)
    for attr_name, results in attribute_results.items():
        with open(os.path.join(results_dir, f"ml_ner_{attr_name.lower()}_evaluation.txt"), 'w') as f:
            f.write(f"Classification Report for {attr_name}:\n")
            f.write(results['report'])
            f.write("\nPer-class Accuracies:\n")
            for label, acc in results['accuracies'].items():
                f.write(f"{label}: {acc:.3f}\n")
    
    logging.info("Evaluation completed and results saved")

if __name__ == "__main__":
    main()