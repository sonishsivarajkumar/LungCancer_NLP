# evaluate_bioclinicalbert_ml_ner.py

import sys
import os

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data_from_files
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModel
import torch

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/bioclinicalbert_ner_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def load_models():
    """Load all trained NER models and attribute models."""
    models = {}
    try:
        models['random_forest'] = joblib.load('./models/bioclinicalbert_ml_models/ner_random_forest_model.joblib')
        models['xgboost'] = joblib.load('./models/bioclinicalbert_ml_models/ner_xgboost_model.joblib')
        models['lightgbm'] = joblib.load('./models/bioclinicalbert_ml_models/ner_lightgbm_model.joblib')
        models['adaboost'] = joblib.load('./models/bioclinicalbert_ml_models/ner_adaboost_model.joblib')
        models['gradient_boosting'] = joblib.load('./models/bioclinicalbert_ml_models/ner_gradient_boosting_model.joblib')
        models['logistic_regression'] = joblib.load('./models/bioclinicalbert_ml_models/ner_logistic_regression_model.joblib')
        
        # SVM model may be absent if commented out in training
        svm_path = './models/bioclinicalbert_ml_models/ner_svm_model.joblib'
        if os.path.exists(svm_path):
            models['svm'] = joblib.load(svm_path)
        
        models['label_encoder'] = joblib.load('./models/bioclinicalbert_ml_models/ner_label_encoder.joblib')
        
        # Attribute models if present
        attr_status_path = './models/bioclinicalbert_ml_models/ner_status_certainty_model.joblib'
        if os.path.exists(attr_status_path):
            models['Status_Certainty'] = joblib.load(attr_status_path)
        
        attr_cert_path = './models/bioclinicalbert_ml_models/ner_certainty_model.joblib'
        if os.path.exists(attr_cert_path):
            models['Certainty'] = joblib.load(attr_cert_path)

        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def preprocess_data(df):
    df = df.dropna(subset=['text', 'entity_type'])
    df['text'] = df['text'].str.lower()
    return df

def encode_text_in_batches(text_list, tokenizer, model, batch_size=16, max_length=128):
    embeddings = []
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            outputs = model(**encoded_input)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

def multiclass_roc(y_test_binarized, y_proba, n_classes):
    fpr_micro, tpr_micro, thresholds = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
    return fpr_micro, tpr_micro, thresholds

def evaluate_models(models, X_test_embeddings, y_test, label_encoder, df):
    results_dir = os.path.join('results', 'bioclinicalbert_ner')
    os.makedirs(results_dir, exist_ok=True)
    master_report_path = os.path.join(results_dir, 'bioclinicalbert_ner_evaluation_classification_reports.txt')

    y_test_classes = label_encoder.inverse_transform(y_test)
    classes = label_encoder.classes_
    n_classes = len(classes)

    y_test_binarized = label_binarize(y_test, classes=range(n_classes))

    with open(master_report_path, 'w') as master_report_file:
        for model_name, model_obj in models.items():
            if model_name in ['label_encoder', 'Status_Certainty', 'Certainty']:
                continue

            logging.info(f"Evaluating {model_name} model...")
            y_proba = model_obj.predict_proba(X_test_embeddings)
            y_pred = model_obj.predict(X_test_embeddings)
            y_pred_classes = label_encoder.inverse_transform(y_pred)

            # Classification report
            report = classification_report(y_test_classes, y_pred_classes, zero_division=0)
            print(f"\nClassification Report - {model_name}:")
            print(report)
            master_report_file.write(f"Classification Report - {model_name}:\n")
            master_report_file.write(report + "\n\n")

            # Confusion matrix
            cm = confusion_matrix(y_test_classes, y_pred_classes, labels=classes)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"bioclinicalbert_ner_{model_name}_confusion_matrix.png"))
            plt.close()

            # Micro-average ROC/PR
            fpr_micro, tpr_micro, _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)

            plt.figure()
            plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Micro-Average ROC - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(results_dir, f"bioclinicalbert_ner_{model_name}_roc_curve.png"))
            plt.close()

            precision_micro, recall_micro, _ = precision_recall_curve(y_test_binarized.ravel(), y_proba.ravel())
            ap_micro = average_precision_score(y_test_binarized, y_proba, average='micro')

            plt.figure()
            plt.step(recall_micro, precision_micro, where='post', color='b', alpha=0.2)
            plt.fill_between(recall_micro, precision_micro, step='post', alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Micro-average Precision-Recall curve: AP={ap_micro:.2f} - {model_name}')
            plt.savefig(os.path.join(results_dir, f"bioclinicalbert_ner_{model_name}_pr_curve.png"))
            plt.close()

            master_report_file.write(f"Micro-Average AUC-ROC - {model_name}: {roc_auc_micro:.4f}\n")
            master_report_file.write(f"Micro-Average Average Precision (AUC-PR) - {model_name}: {ap_micro:.4f}\n\n")

            # Misclassified samples
            misclassified_indices = np.where(y_pred != y_test)[0]
            misclassified_samples = df.iloc[misclassified_indices].copy()
            misclassified_samples['true_label'] = y_test_classes[misclassified_indices]
            misclassified_samples['predicted_label'] = y_pred_classes[misclassified_indices]
            misclassified_samples.to_csv(os.path.join(results_dir, f"bioclinicalbert_ner_{model_name}_misclassified_samples.csv"), index=False)

    logging.info("Main entity classification evaluation completed")

def evaluate_attribute_classifiers(models, df):
    attribute_results = {}
    tokenizer = AutoTokenizer.from_pretrained('./models/bioclinicalbert_ml_models/tokenizer/')
    model = AutoModel.from_pretrained('./models/bioclinicalbert_ml_models/model/').cpu()

    def encode_attribute_text(X):
        return encode_text_in_batches(X.tolist(), tokenizer, model)

    if 'Status_Certainty' in models:
        treatment_mask = df['entity_type'].isin([
            "Cancer_Surgery", "Radiotherapy", "Chemotherapy",
            "Immunotherapy", "Targeted_Therapy"
        ])
        if treatment_mask.any():
            treatment_df = df[treatment_mask]
            if 'Status_Certainty' in treatment_df.columns:
                X = treatment_df['text']
                y_true = treatment_df['Status_Certainty']
                X_embeddings = encode_attribute_text(X)
                y_pred = models['Status_Certainty'].predict(X_embeddings)
                report = classification_report(y_true, y_pred, zero_division=0)
                attribute_results['Status_Certainty'] = report

    if 'Certainty' in models:
        response_mask = df['entity_type'].isin([
            "Complete_Response", "Partial_Response",
            "Stable_Disease", "Progressive_Disease"
        ])
        if response_mask.any():
            response_df = df[response_mask]
            if 'Certainty' in response_df.columns:
                X = response_df['text']
                y_true = response_df['Certainty']
                X_embeddings = encode_attribute_text(X)
                y_pred = models['Certainty'].predict(X_embeddings)
                report = classification_report(y_true, y_pred, zero_division=0)
                attribute_results['Certainty'] = report

    return attribute_results

def main():
    setup_logging()
    logging.info("Starting BioClinicalBERT-based NER evaluation")

    # Use data/test_files.txt to load test data
    with open('data/test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]

    from utils.data_loader import load_data_from_files
    df_test = load_data_from_files(test_files)
    df_test = preprocess_data(df_test)

    models = load_models()
    label_encoder = models['label_encoder']

    print(f"Classes: {label_encoder.classes_}")

    tokenizer = AutoTokenizer.from_pretrained('./models/bioclinicalbert_ml_models/tokenizer/')
    model = AutoModel.from_pretrained('./models/bioclinicalbert_ml_models/model/').cpu()
    logging.info("BioClinicalBERT model loaded for evaluation on CPU.")

    # Prepare test data embeddings
    X_test = df_test['text']
    y_test = label_encoder.transform(df_test['entity_type'])
    logging.info("Encoding test data...")
    X_test_embeddings = encode_text_in_batches(X_test.tolist(), tokenizer, model)

    # Evaluate main entity classification models
    evaluate_models(models, X_test_embeddings, y_test, label_encoder, df_test)

    # Evaluate attribute classifiers
    attribute_results = evaluate_attribute_classifiers(models, df_test)

    results_dir = os.path.join('results', 'bioclinicalbert_ner')
    master_report_path = os.path.join(results_dir, 'bioclinicalbert_ner_evaluation_classification_reports.txt')
    with open(master_report_path, 'a') as f:
        for attr_name, report in attribute_results.items():
            f.write(f"Attribute Classification Report - {attr_name}:\n")
            f.write(report + "\n\n")

    logging.info("Evaluation of attribute classifiers completed")
    logging.info("All evaluations completed and results saved")

if __name__ == "__main__":
    main()