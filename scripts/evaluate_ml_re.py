# scripts/evaluate_ml_re.py

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
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data_from_files_with_relations
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
from transformers import AutoTokenizer, AutoModel
import torch

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_re_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def preprocess_data(df):
    """
    Preprocess the data for RE model evaluation.
    """
    df['combined_text'] = (
        df['ent1_text'] + ' ' +
        df['context'] + ' ' +
        df['ent2_text'] + ' ' +
        df['ent1_type'] + ' ' +
        df['ent2_type']
    )
    df['combined_text'] = df['combined_text'].fillna('')
    return df

def encode_text_in_batches(text_list, tokenizer, model, batch_size=16):
    embeddings = []
    model.eval()
    device = torch.device('cpu')  # Ensure the device is CPU
    model.to(device)
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            encoded_input = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=128
            )
            # Move tensors to CPU
            encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
            outputs = model(**encoded_input)
            # Use the [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

def load_models():
    """Load all trained ML RE models."""
    models = {}
    try:
        models['random_forest'] = joblib.load('./models/ml_models/re_model/re_random_forest_model.joblib')
        models['xgboost'] = joblib.load('./models/ml_models/re_model/re_xgboost_model.joblib')
        models['lightgbm'] = joblib.load('./models/ml_models/re_model/re_lightgbm_model.joblib')
        models['adaboost'] = joblib.load('./models/ml_models/re_model/re_adaboost_model.joblib')
        models['gradient_boosting'] = joblib.load('./models/ml_models/re_model/re_gradient_boosting_model.joblib')
        models['svm'] = joblib.load('./models/ml_models/re_model/re_svm_model.joblib')
        models['logistic_regression'] = joblib.load('./models/ml_models/re_model/re_logistic_regression_model.joblib')
        models['label_encoder'] = joblib.load('./models/ml_models/re_model/re_label_encoder.joblib')
        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def evaluate_models(models, X_test_embeddings, y_test, label_encoder, df_processed):
    """
    Evaluate each model and save the results.
    """
    results_dir = os.path.join('results', 'ml', 're')
    os.makedirs(results_dir, exist_ok=True)
    master_report_path = os.path.join(results_dir, 'ml_re_evaluation_classification_reports.txt')

    with open(master_report_path, 'w') as master_report_file:
        for model_name, model_obj in models.items():
            if model_name == 'label_encoder':
                continue

            logging.info(f"Evaluating {model_name} model...")
            y_proba = model_obj.predict_proba(X_test_embeddings)

            # Determine optimal threshold based on F1 score
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba[:, 1])
            f1_scores = [
                2 * (p * r) / (p + r) if (p + r) > 0 else 0
                for p, r in zip(precision, recall)
            ]
            optimal_idx = np.argmax(f1_scores)
            if optimal_idx < len(thresholds):
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5  # Default threshold if not found
            logging.info(
                f"Optimal threshold for {model_name} is {optimal_threshold:.4f} "
                f"with F1 score {f1_scores[optimal_idx]:.4f}"
            )

            # Use optimal threshold for final predictions
            y_pred_optimal = (y_proba[:, 1] >= optimal_threshold).astype(int)
            y_pred_labels = label_encoder.inverse_transform(y_pred_optimal)
            y_true_labels = label_encoder.inverse_transform(y_test)

            # Classification report
            report = classification_report(
                y_true_labels, y_pred_labels, zero_division=0
            )
            print(f"\nClassification Report - {model_name}:")
            print(report)

            # Append classification report to the master report file
            master_report_file.write(f"Classification Report - {model_name}:\n")
            master_report_file.write(report + "\n")
            master_report_file.write(f"Optimal Threshold - {model_name}: {optimal_threshold:.4f}\n\n")

            # Confusion matrix
            cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d',
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_,
                        cmap='Blues')
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"ml_re_{model_name}_confusion_matrix.png"))
            plt.close()

            # ROC Curve and AUC
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(results_dir, f"ml_re_{model_name}_roc_curve.png"))
            plt.close()

            # Precision-Recall Curve and Average Precision Score
            average_precision = average_precision_score(y_test, y_proba[:, 1])

            plt.figure()
            plt.step(recall, precision, where='post', color='b', alpha=0.2)
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall curve: AP={average_precision:.2f} - {model_name}')
            plt.savefig(os.path.join(results_dir, f"ml_re_{model_name}_pr_curve.png"))
            plt.close()

            # Append AUC scores to the master report
            master_report_file.write(f"AUC-ROC Score - {model_name}: {roc_auc:.4f}\n")
            master_report_file.write(f"Average Precision Score (AUC-PR) - {model_name}: {average_precision:.4f}\n\n")

            # Save misclassified samples for analysis
            misclassified_indices = np.where(y_pred_optimal != y_test)[0]
            misclassified_samples = df_processed.iloc[misclassified_indices]
            misclassified_samples['true_label'] = y_true_labels[misclassified_indices]
            misclassified_samples['predicted_label'] = y_pred_labels[misclassified_indices]
            misclassified_samples.to_csv(os.path.join(results_dir, f"ml_re_{model_name}_misclassified_samples.csv"), index=False)

    logging.info("Evaluation completed and results saved")

def main():
    setup_logging()
    logging.info("Starting ML-based RE evaluation")

    # Load test files
    test_files_path = 'data/test_files.txt'
    if not os.path.exists(test_files_path):
        logging.error(f"Test files list '{test_files_path}' does not exist.")
        return

    with open(test_files_path, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]

    print(f"Number of test files: {len(test_files)}")

    # Load data
    df_relations = load_data_from_files_with_relations(test_files)

    if df_relations.empty:
        logging.error("No relations found in the test data. Please ensure that relations are properly annotated.")
        return

    print(f"Sample test data:\n{df_relations.head()}")

    # Preprocess data
    df_processed = preprocess_data(df_relations)

    # Prepare data
    X_test = df_processed['combined_text']
    y_test_labels = df_processed['relation']

    print(f"Number of test samples: {len(X_test)}")

    # Load models
    models = load_models()
    label_encoder = models['label_encoder']
    y_test = label_encoder.transform(y_test_labels)

    print(f"Classes: {label_encoder.classes_}")

    # Load tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('./models/ml_models/re_model/tokenizer/')
    model = AutoModel.from_pretrained('./models/ml_models/re_model/model/').cpu()
    print("Model loaded successfully in offline mode (CPU).")

    # Encode test data
    logging.info("Encoding test data...")
    X_test_embeddings = encode_text_in_batches(X_test.tolist(), tokenizer, model)

    # Evaluate models
    evaluate_models(models, X_test_embeddings, y_test, label_encoder, df_processed)

if __name__ == "__main__":
    main()
