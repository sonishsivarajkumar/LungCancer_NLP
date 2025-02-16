# scripts/train_tfidf_ml_ner.py

import os
import logging
import joblib
import numpy as np
import pandas as pd

from utils.data_loader import load_data_from_files
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC  # Uncomment if you want to include SVM

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
    df = df.dropna(subset=['text', 'entity_type'])
    df['text'] = df['text'].str.lower()
    label_encoder = LabelEncoder()
    df['entity_type_encoded'] = label_encoder.fit_transform(df['entity_type'])
    return df, label_encoder

def train_attribute_classifiers(df):
    """Train separate classifiers for attributes using TF-IDF."""
    attribute_models = {}
    
    def train_attr_model(X, y):
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X, y)
        return pipeline

    # Status_Certainty for treatment entities
    treatment_mask = df['entity_type'].isin([
        "Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
        "Immunotherapy", "Targeted_Therapy"
    ])
    if treatment_mask.any() and 'Status_Certainty' in df.columns:
        treatment_df = df[treatment_mask].dropna(subset=['Status_Certainty'])
        if not treatment_df.empty:
            X = treatment_df['text']
            y = treatment_df['Status_Certainty']
            attribute_models['Status_Certainty'] = train_attr_model(X, y)
    
    # Certainty for response entities
    response_mask = df['entity_type'].isin([
        "Complete_Response", "Partial_Response", 
        "Stable_Disease", "Progressive_Disease"
    ])
    if response_mask.any() and 'Certainty' in df.columns:
        response_df = df[response_mask].dropna(subset=['Certainty'])
        if not response_df.empty:
            X = response_df['text']
            y = response_df['Certainty']
            attribute_models['Certainty'] = train_attr_model(X, y)

    return attribute_models

def create_and_train_models(X_train, y_train):
    """Create and train multiple ML models with grid search."""
    models = {}
    scoring = 'f1_macro'
    n_jobs = 1
    cv = 3
    verbose_level = 2  # Increase verbosity to see grid search progress

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=5000)

    logging.info("Performing Grid Search for Random Forest...")
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10],
        'classifier__min_samples_split': [2, 5],
        'classifier__class_weight': ['balanced']
    }
    rf_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(RandomForestClassifier(random_state=42)))
    ])
    rf_gs = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    rf_gs.fit(X_train, y_train)
    logging.info(f"Random Forest best params: {rf_gs.best_params_}")
    logging.info(f"Random Forest best score: {rf_gs.best_score_}")
    models['random_forest'] = rf_gs.best_estimator_

    logging.info("Performing Grid Search for XGBoost...")
    xgb_param_grid = {
        'classifier__estimator__n_estimators': [100],
        'classifier__estimator__max_depth': [3,6],
        'classifier__estimator__learning_rate': [0.1],
        'classifier__estimator__scale_pos_weight': [1]
    }
    xgb_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss', random_state=42)))
    ])
    xgb_gs = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    xgb_gs.fit(X_train, y_train)
    logging.info(f"XGBoost best params: {xgb_gs.best_params_}")
    logging.info(f"XGBoost best score: {xgb_gs.best_score_}")
    models['xgboost'] = xgb_gs.best_estimator_

    logging.info("Performing Grid Search for LightGBM...")
    lgb_param_grid = {
        'classifier__estimator__n_estimators': [100, 200],
        'classifier__estimator__max_depth': [-1,10],
        'classifier__estimator__learning_rate': [0.1]
    }
    lgb_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(lgb.LGBMClassifier(random_state=42)))
    ])
    lgb_gs = GridSearchCV(lgb_pipeline, lgb_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    lgb_gs.fit(X_train, y_train)
    logging.info(f"LightGBM best params: {lgb_gs.best_params_}")
    logging.info(f"LightGBM best score: {lgb_gs.best_score_}")
    models['lightgbm'] = lgb_gs.best_estimator_

    logging.info("Performing Grid Search for AdaBoost...")
    ada_param_grid = {
        'classifier__estimator__n_estimators': [50, 100],
        'classifier__estimator__learning_rate': [0.1, 1.0]
    }
    ada_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(AdaBoostClassifier(random_state=42)))
    ])
    ada_gs = GridSearchCV(ada_pipeline, ada_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    ada_gs.fit(X_train, y_train)
    logging.info(f"AdaBoost best params: {ada_gs.best_params_}")
    logging.info(f"AdaBoost best score: {ada_gs.best_score_}")
    models['adaboost'] = ada_gs.best_estimator_

    logging.info("Performing Grid Search for Gradient Boosting...")
    gb_param_grid = {
        'classifier__estimator__n_estimators': [100],
        'classifier__estimator__max_depth': [3,6],
        'classifier__estimator__learning_rate': [0.1]
    }
    gb_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(GradientBoostingClassifier(random_state=42)))
    ])
    gb_gs = GridSearchCV(gb_pipeline, gb_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    gb_gs.fit(X_train, y_train)
    logging.info(f"Gradient Boosting best params: {gb_gs.best_params_}")
    logging.info(f"Gradient Boosting best score: {gb_gs.best_score_}")
    models['gradient_boosting'] = gb_gs.best_estimator_

    logging.info("Performing Grid Search for Logistic Regression...")
    lr_param_grid = {
        'classifier__estimator__C': [0.1, 1, 10],
        'classifier__estimator__class_weight': ['balanced'],
        'classifier__estimator__max_iter': [1000]
    }
    lr_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', OneVsRestClassifier(LogisticRegression(random_state=42)))
    ])
    lr_gs = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    lr_gs.fit(X_train, y_train)
    logging.info(f"Logistic Regression best params: {lr_gs.best_params_}")
    logging.info(f"Logistic Regression best score: {lr_gs.best_score_}")
    models['logistic_regression'] = lr_gs.best_estimator_

    # Uncomment if you want SVM
    # logging.info("Performing Grid Search for SVM...")
    # svm_param_grid = {
    #     'classifier__estimator__C': [0.1, 1],
    #     'classifier__estimator__kernel': ['linear'],
    #     'classifier__estimator__class_weight': ['balanced']
    # }
    # svm_pipeline = Pipeline([
    #     ('vectorizer', vectorizer),
    #     ('classifier', OneVsRestClassifier(SVC(probability=True, random_state=42)))
    # ])
    # svm_gs = GridSearchCV(svm_pipeline, svm_param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose_level)
    # svm_gs.fit(X_train, y_train)
    # logging.info(f"SVM best params: {svm_gs.best_params_}")
    # logging.info(f"SVM best score: {svm_gs.best_score_}")
    # models['svm'] = svm_gs.best_estimator_

    return models

def main():
    setup_logging()
    logging.info("Starting ML-based NER training (TF-IDF with Multiple Models and Grid Search)")

    # Load train and test files
    with open('data/train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    with open('data/test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]

    # Load training data
    df_train = load_data_from_files(train_files)
    df_train_processed, label_encoder = preprocess_data(df_train)
    X_train = df_train_processed['text']
    y_train = df_train_processed['entity_type_encoded']

    # Load test data for saving arrays
    df_test = load_data_from_files(test_files)
    df_test = df_test.dropna(subset=['text', 'entity_type'])
    df_test['text'] = df_test['text'].str.lower()
    df_test['entity_type_encoded'] = label_encoder.transform(df_test['entity_type'])
    X_test = df_test['text']
    y_test = df_test['entity_type_encoded']

    # Create and train models with grid search
    models = create_and_train_models(X_train, y_train)

    # Train attribute classifiers
    attribute_models = train_attribute_classifiers(df_train_processed)

    # Create directory for saving models
    os.makedirs('./models/ml_models', exist_ok=True)
    
    # Save models and label encoder
    for model_name, model_obj in models.items():
        joblib.dump(model_obj, f'./models/ml_models/ner_{model_name}_model.joblib')
    for attr_name, model_obj in attribute_models.items():
        joblib.dump(model_obj, f'./models/ml_models/ner_{attr_name.lower()}_model.joblib')
    joblib.dump(label_encoder, './models/ml_models/ner_label_encoder.joblib')
    
    # Save train/test split for evaluation
    np.save('./models/ml_models/X_test.npy', X_test)
    np.save('./models/ml_models/y_test.npy', y_test)
    
    logging.info("ML-based NER models (TF-IDF with Grid Search) have been saved")

if __name__ == "__main__":
    main()