# scripts/train_biobert_ml_re.py

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
from utils.data_loader import load_data_from_files_with_relations
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Commented out SMOTE to avoid additional memory usage
# from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure the model and tokenizer are loaded in CPU mode
tokenizer = AutoTokenizer.from_pretrained('./models/pretrained/Bio_ClinicalBERT/')
model = AutoModel.from_pretrained('./models/pretrained/Bio_ClinicalBERT/').cpu()
print("Model loaded successfully in offline mode (CPU).")

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_re_training.log"),
            logging.StreamHandler()
        ]
    )

def preprocess_data(df):
    """
    Preprocess the data for RE model training.
    """
    # Combine features into a single text
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

def create_and_train_models(X_train_embeddings, y_train, scale_pos_weight):
    """
    Create and train multiple ML models for RE with hyperparameter tuning.
    """
    models = {}
    best_params = {}

    # Random Forest Classifier with class weighting
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }

    rf_classifier = RandomForestClassifier(random_state=42)
    logging.info("Performing Grid Search for Random Forest...")
    rf_grid_search = GridSearchCV(
        rf_classifier,
        rf_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    rf_grid_search.fit(X_train_embeddings, y_train)
    best_rf_classifier = rf_grid_search.best_estimator_
    models['random_forest'] = best_rf_classifier
    best_params['random_forest'] = rf_grid_search.best_params_
    logging.info(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")

    # XGBoost Classifier with scale_pos_weight
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [1, scale_pos_weight]
    }

    xgb_classifier = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    logging.info("Performing Grid Search for XGBoost...")
    xgb_grid_search = GridSearchCV(
        xgb_classifier,
        xgb_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    xgb_grid_search.fit(X_train_embeddings, y_train)
    best_xgb_classifier = xgb_grid_search.best_estimator_
    models['xgboost'] = best_xgb_classifier
    best_params['xgboost'] = xgb_grid_search.best_params_
    logging.info(f"Best parameters for XGBoost: {xgb_grid_search.best_params_}")

    # LightGBM Classifier
    lgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [-1, 10],
        'learning_rate': [0.01, 0.1],
        'class_weight': [None, 'balanced']
    }

    lgb_classifier = lgb.LGBMClassifier(random_state=42)
    logging.info("Performing Grid Search for LightGBM...")
    lgb_grid_search = GridSearchCV(
        lgb_classifier,
        lgb_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    lgb_grid_search.fit(X_train_embeddings, y_train)
    best_lgb_classifier = lgb_grid_search.best_estimator_
    models['lightgbm'] = best_lgb_classifier
    best_params['lightgbm'] = lgb_grid_search.best_params_
    logging.info(f"Best parameters for LightGBM: {lgb_grid_search.best_params_}")

    # AdaBoost Classifier
    ada_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    ada_classifier = AdaBoostClassifier(random_state=42)
    logging.info("Performing Grid Search for AdaBoost...")
    ada_grid_search = GridSearchCV(
        ada_classifier,
        ada_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    ada_grid_search.fit(X_train_embeddings, y_train)
    best_ada_classifier = ada_grid_search.best_estimator_
    models['adaboost'] = best_ada_classifier
    best_params['adaboost'] = ada_grid_search.best_params_
    logging.info(f"Best parameters for AdaBoost: {ada_grid_search.best_params_}")

    # Gradient Boosting Classifier
    gb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.1]
    }

    gb_classifier = GradientBoostingClassifier(random_state=42)
    logging.info("Performing Grid Search for Gradient Boosting...")
    gb_grid_search = GridSearchCV(
        gb_classifier,
        gb_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    gb_grid_search.fit(X_train_embeddings, y_train)
    best_gb_classifier = gb_grid_search.best_estimator_
    models['gradient_boosting'] = best_gb_classifier
    best_params['gradient_boosting'] = gb_grid_search.best_params_
    logging.info(f"Best parameters for Gradient Boosting: {gb_grid_search.best_params_}")

    # SVM Classifier with class weighting
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear'],
        'class_weight': ['balanced']
    }

    svm_classifier = SVC(probability=True, random_state=42)
    logging.info("Performing Grid Search for SVM...")
    svm_grid_search = GridSearchCV(
        svm_classifier,
        svm_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    svm_grid_search.fit(X_train_embeddings, y_train)
    best_svm_classifier = svm_grid_search.best_estimator_
    models['svm'] = best_svm_classifier
    best_params['svm'] = svm_grid_search.best_params_
    logging.info(f"Best parameters for SVM: {svm_grid_search.best_params_}")

    # Logistic Regression Classifier with class weighting
    lr_param_grid = {
        'C': [0.1, 1, 10],
        'class_weight': ['balanced'],
        'max_iter': [1000]
    }

    lr_classifier = LogisticRegression(random_state=42)
    logging.info("Performing Grid Search for Logistic Regression...")
    lr_grid_search = GridSearchCV(
        lr_classifier,
        lr_param_grid,
        cv=3,
        scoring='f1',
        n_jobs=1  # Sequential processing
    )
    lr_grid_search.fit(X_train_embeddings, y_train)
    best_lr_classifier = lr_grid_search.best_estimator_
    models['logistic_regression'] = best_lr_classifier
    best_params['logistic_regression'] = lr_grid_search.best_params_
    logging.info(f"Best parameters for Logistic Regression: {lr_grid_search.best_params_}")

    return models, best_params

def main():
    setup_logging()
    logging.info("Starting ML-based RE training")

    # Load train files
    train_files_path = 'data/train_files.txt'
    if not os.path.exists(train_files_path):
        logging.error(f"Train files list '{train_files_path}' does not exist.")
        return

    with open(train_files_path, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]

    print(f"Number of training files: {len(train_files)}")

    # Load data
    df_relations = load_data_from_files_with_relations(train_files)

    if df_relations.empty:
        logging.error("No relations found in the data. Please ensure that relations are properly annotated.")
        return

    print(f"Sample data:\n{df_relations.head()}")

    # Preprocess data
    df_processed = preprocess_data(df_relations)

    # Encode labels
    label_encoder = LabelEncoder()
    df_processed['relation_encoded'] = label_encoder.fit_transform(df_processed['relation'])

    # Prepare data
    X = df_processed['combined_text']
    y = df_processed['relation_encoded']

    print(f"Number of samples: {len(X)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")

    # Split into training and validation sets with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Initialize ClinicalBERT tokenizer and model in CPU mode
    tokenizer = AutoTokenizer.from_pretrained('./models/pretrained/Bio_ClinicalBERT/')
    model = AutoModel.from_pretrained('./models/pretrained/Bio_ClinicalBERT/').cpu()
    print("Model loaded successfully in offline mode (CPU).")
    
    # Encode training data
    logging.info("Encoding training data...")
    X_train_embeddings = encode_text_in_batches(
        X_train.tolist(), tokenizer, model
    )

    # Handle class imbalance using class weights
    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)
    scale_pos_weight = negative_count / positive_count

    # Create and train models
    models, best_params = create_and_train_models(
        X_train_embeddings, y_train, scale_pos_weight
    )

    # Validate models
    logging.info("Validating models on the validation set...")
    # Encode validation data
    logging.info("Encoding validation data...")
    X_val_embeddings = encode_text_in_batches(
        X_val.tolist(), tokenizer, model
    )

    # Prepare master report file
    results_dir = os.path.join('results', 'ml', 're')
    os.makedirs(results_dir, exist_ok=True)
    master_report_path = os.path.join(
        results_dir, 'ml_re_training_classification_reports.txt'
    )

    with open(master_report_path, 'w') as master_report_file:
        for model_name, model_obj in models.items():
            y_proba = model_obj.predict_proba(X_val_embeddings)

            # Determine optimal threshold based on F1 score
            precision, recall, thresholds = precision_recall_curve(
                y_val, y_proba[:, 1]
            )
            f1_scores = [
                2 * (p * r) / (p + r) if (p + r) > 0 else 0
                for p, r in zip(precision, recall)
            ]
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            logging.info(
                f"Optimal threshold for {model_name} is {optimal_threshold:.4f} "
                f"with F1 score {f1_scores[optimal_idx]:.4f}"
            )

            # Use optimal threshold for final predictions
            y_pred_optimal = (y_proba[:, 1] >= optimal_threshold).astype(int)
            y_true = y_val
            y_pred_labels = label_encoder.inverse_transform(y_pred_optimal)
            y_true_labels = label_encoder.inverse_transform(y_true)
            report = classification_report(
                y_true_labels, y_pred_labels, zero_division=0
            )
            print(f"\nValidation Classification Report - {model_name}:")
            print(report)

            # Append classification report and best parameters to master report file
            master_report_file.write(
                f"Validation Classification Report - {model_name}:\n"
            )
            master_report_file.write(report + "\n")
            master_report_file.write(
                f"Best Parameters - {model_name}: {best_params[model_name]}\n"
            )
            master_report_file.write(
                f"Optimal Threshold - {model_name}: {optimal_threshold:.4f}\n\n"
            )

    # Create directory for saving models
    os.makedirs('./models/ml_models/re_model', exist_ok=True)

    # Save models and label encoder
    for model_name, model_obj in models.items():
        model_path = f'./models/ml_models/re_model/re_{model_name}_model.joblib'
        joblib.dump(model_obj, model_path)
        logging.info(f"Saved {model_name} model to {model_path}")

    label_encoder_path = './models/ml_models/re_model/re_label_encoder.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    logging.info(f"Saved label encoder to {label_encoder_path}")

    # Save tokenizer and model for embeddings
    tokenizer.save_pretrained('./models/ml_models/re_model/tokenizer/')
    model.save_pretrained('./models/ml_models/re_model/model/')
    logging.info("Saved tokenizer and model for embeddings")

    logging.info("ML-based RE models have been saved")

if __name__ == "__main__":
    main()
