# inference_re.py

import os
import logging
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/re_inference.log"),
            logging.StreamHandler()
        ]
    )

def load_models():
    """Load all trained RE models."""
    models = {}
    try:
        # Load ML models
        models['vectorizer'] = joblib.load('./models/ml_models/re_vectorizer.joblib')
        models['random_forest'] = joblib.load('./models/ml_models/re_random_forest.joblib')
        models['xgboost'] = joblib.load('./models/ml_models/re_xgboost.joblib')
        
        # Load BERT model
        models['bert'] = AutoModelForSequenceClassification.from_pretrained(
            './models/bert_models/re'
        )
        models['bert_tokenizer'] = AutoTokenizer.from_pretrained(
            './models/bert_models/re'
        )
        
        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def prepare_pair_text(treatment_entity, response_entity):
    """Prepare text representation of entity pair."""
    return (f"{treatment_entity['entity_type']} : {treatment_entity['text']} "
            f"[SEP] {response_entity['entity_type']} : {response_entity['text']}")

def predict_relation_ml(pair_text, models, model_name):
    """Predict relation using ML model."""
    try:
        # Transform text
        X = models['vectorizer'].transform([pair_text])
        
        # Get prediction and probability
        model = models[model_name]
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return prediction, probability
    
    except Exception as e:
        logging.error(f"Error in ML prediction: {str(e)}")
        return None, None

def predict_relation_bert(pair_text, models):
    """Predict relation using BERT model."""
    try:
        # Tokenize input
        inputs = models['bert_tokenizer'](
            pair_text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get prediction
        models['bert'].eval()
        with torch.no_grad():
            outputs = models['bert'](**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            probability = probabilities[0, 1].item()
        
        return prediction, probability
    
    except Exception as e:
        logging.error(f"Error in BERT prediction: {str(e)}")
        return None, None

def find_entities(text, ner_results):
    """Find treatment and response entities in text using NER results."""
    treatment_types = [
        "Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
        "Immunotherapy", "Targeted_Therapy"
    ]
    response_types = [
        "Complete_Response", "Partial_Response", 
        "Stable_Disease", "Progressive_Disease"
    ]
    
    treatments = [e for e in ner_results if e['entity_type'] in treatment_types]
    responses = [e for e in ner_results if e['entity_type'] in response_types]
    
    return treatments, responses

def process_text(text, ner_results, re_models):
    """Process text to extract relations between entities."""
    treatments, responses = find_entities(text, ner_results)
    relations = []
    
    for treatment in treatments:
        for response in responses:
            pair_text = prepare_pair_text(treatment, response)
            
            # Get predictions from all models
            rf_pred, rf_prob = predict_relation_ml(pair_text, re_models, 'random_forest')
            xgb_pred, xgb_prob = predict_relation_ml(pair_text, re_models, 'xgboost')
            bert_pred, bert_prob = predict_relation_bert(pair_text, re_models)
            
            # Ensemble prediction (majority voting)
            predictions = [rf_pred, xgb_pred, bert_pred]
            probabilities = [rf_prob, xgb_prob, bert_prob]
            
            if any(pred is None for pred in predictions):
                continue
                
            ensemble_pred = int(sum(predictions) > len(predictions)/2)
            avg_prob = np.mean([p for p in probabilities if p is not None])
            
            if ensemble_pred == 1:
                relations.append({
                    'treatment_text': treatment['text'],
                    'treatment_type': treatment['entity_type'],
                    'response_text': response['text'],
                    'response_type': response['entity_type'],
                    'confidence': avg_prob,
                    'rf_prediction': rf_pred,
                    'xgb_prediction': xgb_pred,
                    'bert_prediction': bert_pred
                })
    
    return relations

def format_output(relations):
    """Format relations for output."""
    return pd.DataFrame(relations)

def main():
    setup_logging()
    logging.info("Starting Relation Extraction inference")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load models
    models = load_models()
    
    # Example texts and their NER results
    sample_cases = [
        {
            "text": "Patient received chemotherapy with carboplatin and showed complete response.",
            "ner_results": [
                {"text": "chemotherapy with carboplatin", "entity_type": "Chemotherapy", "start": 16, "end": 44},
                {"text": "complete response", "entity_type": "Complete_Response", "start": 56, "end": 72}
            ]
        },
        {
            "text": "After immunotherapy treatment, partial response was observed.",
            "ner_results": [
                {"text": "immunotherapy", "entity_type": "Immunotherapy", "start": 6, "end": 19},
                {"text": "partial response", "entity_type": "Partial_Response", "start": 31, "end": 46}
            ]
        }
    ]
    
    # Process each case
    for i, case in enumerate(sample_cases):
        logging.info(f"Processing text {i+1}")
        
        # Extract relations
        relations = process_text(case['text'], case['ner_results'], models)
        results_df = format_output(relations)
        

        # Save results
        output_file = f"results/text_{i+1}_relations.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\nResults for text {i+1}:")
        print(results_df)
        
        # Generate detailed report
        report_file = f"results/text_{i+1}_relation_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Relation Extraction Report for Text {i+1}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Original Text: {case['text']}\n\n")
            
            f.write("Entities Found:\n")
            f.write("--------------\n")
            for entity in case['ner_results']:
                f.write(f"Type: {entity['entity_type']}\n")
                f.write(f"Text: {entity['text']}\n")
                f.write(f"Span: {entity['start']}~{entity['end']}\n\n")
            
            f.write("Relations Found:\n")
            f.write("---------------\n")
            if not relations:
                f.write("No relations detected.\n")
            else:
                for j, relation in enumerate(relations, 1):
                    f.write(f"Relation {j}:\n")
                    f.write(f"Treatment: {relation['treatment_text']} ({relation['treatment_type']})\n")
                    f.write(f"Response: {relation['response_text']} ({relation['response_type']})\n")
                    f.write(f"Confidence: {relation['confidence']:.3f}\n")
                    f.write("Model Predictions:\n")
                    f.write(f"  Random Forest: {relation['rf_prediction']}\n")
                    f.write(f"  XGBoost: {relation['xgb_prediction']}\n")
                    f.write(f"  BERT: {relation['bert_prediction']}\n\n")
        
        logging.info(f"Results saved to {output_file} and {report_file}")

    # Generate summary of model agreement
    logging.info("Generating model agreement summary")
    all_results = []
    for case in sample_cases:
        relations = process_text(case['text'], case['ner_results'], models)
        all_results.extend(relations)
    
    if all_results:
        agreement_stats = {
            'full_agreement': 0,
            'partial_agreement': 0,
            'disagreement': 0
        }
        
        for relation in all_results:
            predictions = [
                relation['rf_prediction'],
                relation['xgb_prediction'],
                relation['bert_prediction']
            ]
            
            if len(set(predictions)) == 1:
                agreement_stats['full_agreement'] += 1
            elif len(set(predictions)) == 2:
                agreement_stats['partial_agreement'] += 1
            else:
                agreement_stats['disagreement'] += 1
        
        # Save agreement statistics
        with open('results/model_agreement_summary.txt', 'w') as f:
            f.write("Model Agreement Summary\n")
            f.write("=====================\n\n")
            total = len(all_results)
            f.write(f"Total Relations Analyzed: {total}\n\n")
            f.write(f"Full Agreement (all models): {agreement_stats['full_agreement']} ")
            f.write(f"({agreement_stats['full_agreement']/total*100:.1f}%)\n")
            f.write(f"Partial Agreement (2/3 models): {agreement_stats['partial_agreement']} ")
            f.write(f"({agreement_stats['partial_agreement']/total*100:.1f}%)\n")
            f.write(f"Disagreement: {agreement_stats['disagreement']} ")
            f.write(f"({agreement_stats['disagreement']/total*100:.1f}%)\n")
    
    logging.info("Inference completed")

if __name__ == "__main__":
    main()