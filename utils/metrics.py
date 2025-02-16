# utils/metrics.py

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

def flatten_predictions(y_true, y_pred, idx2label):
    """Flatten the list of sequences into a single list of labels."""
    y_true_flat = [idx2label[idx] for seq in y_true for idx in seq]
    y_pred_flat = [idx2label[idx] for seq in y_pred for idx in seq]
    return y_true_flat, y_pred_flat

def generate_classification_report(y_true, y_pred, labels, output_path):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(output_path, 'w') as f:
        f.write(report)
    logging.info(f"Classification report saved to {output_path}")
    print(report)

def plot_confusion_matrix_fn(y_true, y_pred, labels, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")

def save_error_examples(df_test, y_true, y_pred, output_dir, label2idx, idx2label):
    """Save examples of false positives and false negatives for each entity type."""
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = set(y_true)
    unique_labels.discard('O')  # Remove 'O' if present
    for label in unique_labels:
        false_positives = df_test[(y_pred == label) & (y_true != label)]
        false_negatives = df_test[(y_pred != label) & (y_true == label)]
        
        with open(os.path.join(output_dir, f'{label}_errors.txt'), 'w') as f:
            f.write(f"Entity Type: {label}\n")
            f.write("---------------------\n")
            f.write(f"False Positives: {len(false_positives)}\n")
            if not false_positives.empty:
                f.write("Examples of False Positives:\n")
                for _, row in false_positives.head(5).iterrows():
                    f.write(f"Text: {row['text']}\n")
                    f.write(f"Predicted as: {label}\n")
                    f.write(f"True label: {row['entity_type']}\n\n")
            else:
                f.write("No False Positives.\n\n")
            
            f.write(f"False Negatives: {len(false_negatives)}\n")
            if not false_negatives.empty:
                f.write("Examples of False Negatives:\n")
                for _, row in false_negatives.head(5).iterrows():
                    f.write(f"Text: {row['text']}\n")
                    f.write(f"Predicted as: Other\n")
                    f.write(f"True label: {label}\n\n")
            else:
                f.write("No False Negatives.\n\n")
            f.write("\n")
    logging.info(f"Error examples saved to {output_dir}")
