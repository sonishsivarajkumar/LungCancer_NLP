# scripts/evaluate_rule_based_ner.py

import os
import yaml
import pandas as pd
import re
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from train_rule_based_ner import load_xml_data

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/rule_based_ner_evaluation.log"),
            logging.StreamHandler()
        ]
    )

def load_patterns():
    print(os.getcwd())
    with open(os.path.join(os.getcwd(), 'models', 'rule_based_models', 'rule_based_ner_patterns.yaml')) as f:
        patterns = yaml.safe_load(f)
    
    # Compile patterns
    return {entity_type: re.compile(pattern, re.I) for entity_type, pattern in patterns.items()}

def classify_text(text, pattern_dict):
    for entity_type, pattern in pattern_dict.items():
        matches = pattern.findall(text)
        for match in matches:
            # Check for negation
            negation_pattern = re.compile(r'\b(no evidence of|no signs of|without)\b', re.I)
            if not negation_pattern.search(text, 0, text.find(match)):
                return entity_type
    return "Other" 


def evaluate_attributes(true_df, pred_df, attribute_mapping):
    evaluations = []
    
    for entity_type, attributes in attribute_mapping.items():
        entity_mask = true_df["entity_type"] == entity_type
        for attr in attributes:
            if attr in true_df.columns and attr in pred_df.columns:
                true_values = true_df[entity_mask][attr]
                pred_values = pred_df[entity_mask][attr]
                
                if not true_values.empty:
                    correct = (true_values == pred_values).sum()
                    total = len(true_values)
                    accuracy = correct / total if total > 0 else 0
                    
                    evaluations.append({
                        "entity_type": entity_type,
                        "attribute": attr,
                        "accuracy": accuracy,
                        "total_instances": total
                    })
    
    return pd.DataFrame(evaluations)

def main():
    setup_logging()
    logging.info("Starting rule-based NER evaluation")

    # Define the base folder path and subfolders
    base_folder = r"C:\Users\sivarajkumars2\OneDrive - UPMC\Documents\Lung_cancer_project\Annotation_shared\LC_Annotations\Annotated Files\Round_4\Subhash"
    folders = [
        "Completed Partial Response",
        "Completed Progressive Disease",
        "Completed RECIST complete response",
        "Completed Stable Disease",
    ]
    
    # Create results directory
    results_dir = os.path.join('results', 'rule_based', 'ner')
    os.makedirs(results_dir, exist_ok=True)

    # Load test data
    df_test = load_xml_data(base_folder, folders)
    
    # Load patterns
    pattern_dict = load_patterns()
    
    # Predict entities
    df_test["predicted_entity_type"] = df_test["text"].apply(
        lambda x: classify_text(x, pattern_dict)
    )
    
    # Entity-level evaluation
    true_labels = df_test["entity_type"]
    pred_labels = df_test["predicted_entity_type"]
    
    # Save classification report
    report = classification_report(true_labels, pred_labels)
    report_file = os.path.join(results_dir, 'rule_based_ner_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("Entity-Level Classification Report:\n")
        f.write(report)
    
    print("\nEntity-Level Classification Report:")
    print(report)
    
    # Confusion Matrix
    entity_types = sorted(df_test["entity_type"].unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=entity_types)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=entity_types, 
                yticklabels=entity_types)
    plt.title("Entity-Level Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_file = os.path.join(results_dir, 'rule_based_ner_confusion_matrix.png')
    plt.savefig(cm_file)
    plt.close()
    
    # Attribute-level evaluation
    attribute_mapping = {
        "Cancer_Surgery": ["Status_Certainty", "Combi"],
        "Radiotherapy": ["Status_Certainty", "Combi"],
        "Chemotherapy": ["Status_Certainty", "Combi"],
        "Immunotherapy": ["Status_Certainty", "Combi"],
        "Targeted_Therapy": ["Status_Certainty", "Combi"],
        "Complete_Response": ["Certainty"],
        "Partial_Response": ["Certainty"],
        "Stable_Disease": ["Certainty"],
        "Progressive_Disease": ["Certainty"]
    }
    
    attr_eval = evaluate_attributes(df_test, df_test, attribute_mapping)
    print("\nAttribute-Level Evaluation:")
    print(attr_eval)
    
    # Save attribute evaluation results
    attr_eval_file = os.path.join(results_dir, 'rule_based_ner_attribute_evaluation.csv')
    attr_eval.to_csv(attr_eval_file, index=False)
    
    # Error Analysis
    logging.info("Performing error analysis")
    df_test['correct'] = df_test['entity_type'] == df_test['predicted_entity_type']
    
    # False Positives: predicted != 'Other' and predicted != true
    false_positives = df_test[(df_test['predicted_entity_type'] != 'Other') & 
                              (df_test['predicted_entity_type'] != df_test['entity_type'])]
    fp_file = os.path.join(results_dir, 'false_positives.csv')
    false_positives.to_csv(fp_file, index=False)
    
    # False Negatives: true != 'Other' and predicted == 'Other'
    false_negatives = df_test[(df_test['entity_type'] != 'Other') & 
                              (df_test['predicted_entity_type'] == 'Other')]
    fn_file = os.path.join(results_dir, 'false_negatives.csv')
    false_negatives.to_csv(fn_file, index=False)
    
    # Misclassifications: predicted != true
    misclassifications = df_test[df_test['correct'] == False]
    misclassifications_file = os.path.join(results_dir, 'misclassifications.csv')
    misclassifications.to_csv(misclassifications_file, index=False)
    
    # Counts of errors per entity type
    error_counts = misclassifications.groupby(['entity_type', 'predicted_entity_type']).size().reset_index(name='count')
    error_counts_file = os.path.join(results_dir, 'error_counts.csv')
    error_counts.to_csv(error_counts_file, index=False)
    
    # Per-entity-type error analysis
    entity_types = df_test['entity_type'].unique()
    error_analysis = []

    for entity in entity_types:
        entity_mask = df_test['entity_type'] == entity
        total = entity_mask.sum()
        correct = df_test[entity_mask & df_test['correct']].shape[0]
        false_neg = df_test[entity_mask & (df_test['predicted_entity_type'] == 'Other')].shape[0]
        false_pos = df_test[(~entity_mask) & (df_test['predicted_entity_type'] == entity)].shape[0]
        accuracy = correct / total * 100 if total > 0 else 0
        error_analysis.append({
            'Entity Type': entity,
            'Total Instances': total,
            'Correct Predictions': correct,
            'False Positives': false_pos,
            'False Negatives': false_neg,
            'Accuracy (%)': accuracy,
        })

    error_analysis_df = pd.DataFrame(error_analysis)
    error_analysis_file = os.path.join(results_dir, 'error_analysis_per_entity.csv')
    error_analysis_df.to_csv(error_analysis_file, index=False)
    
    # Generate error analysis report
    logging.info("Generating error analysis report")
    error_report_file = os.path.join(results_dir, 'error_analysis_report.txt')
    with open(error_report_file, 'w') as f:
        f.write("Error Analysis Report\n")
        f.write("=====================\n\n")
        for entity in entity_types:
            f.write(f"Entity Type: {entity}\n")
            f.write("---------------------\n")
            entity_mask = df_test['entity_type'] == entity
            total = entity_mask.sum()
            correct = df_test[entity_mask & df_test['correct']].shape[0]
            false_neg = df_test[entity_mask & (df_test['predicted_entity_type'] == 'Other')].shape[0]
            false_pos = df_test[(~entity_mask) & (df_test['predicted_entity_type'] == entity)].shape[0]
            accuracy = correct / total * 100 if total > 0 else 0
            f.write(f"Total Instances: {total}\n")
            f.write(f"Correct Predictions: {correct}\n")
            f.write(f"False Positives: {false_pos}\n")
            f.write(f"False Negatives: {false_neg}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n\n")
            
            # Examples of False Positives
            false_pos_examples = false_positives[false_positives['predicted_entity_type'] == entity].head(5)
            if not false_pos_examples.empty:
                f.write("Examples of False Positives:\n")
                for idx, row in false_pos_examples.iterrows():
                    f.write(f"Text: {row['text']}\n")
                    f.write(f"Predicted as: {row['predicted_entity_type']}\n")
                    f.write(f"True label: {row['entity_type']}\n")
                    f.write("\n")
            else:
                f.write("No False Positives.\n\n")
            
            # Examples of False Negatives
            false_neg_examples = false_negatives[false_negatives['entity_type'] == entity].head(5)
            if not false_neg_examples.empty:
                f.write("Examples of False Negatives:\n")
                for idx, row in false_neg_examples.iterrows():
                    f.write(f"Text: {row['text']}\n")
                    f.write(f"Predicted as: {row['predicted_entity_type']}\n")
                    f.write(f"True label: {row['entity_type']}\n")
                    f.write("\n")
            else:
                f.write("No False Negatives.\n\n")
            
            f.write("\n")

    # Print summary of errors
    num_false_positives = len(false_positives)
    num_false_negatives = len(false_negatives)
    num_total = len(df_test)
    logging.info(f"Total instances: {num_total}")
    logging.info(f"Number of false positives: {num_false_positives}")
    logging.info(f"Number of false negatives: {num_false_negatives}")
    
    logging.info("Error analysis completed and results saved")
    logging.info("Evaluation completed and results saved")

if __name__ == "__main__":
    main()
