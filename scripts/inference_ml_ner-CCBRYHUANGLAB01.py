# scripts/inference_ml_ner.py

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ml_ner_inference.log"),
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
        models['Status_Certainty'] = joblib.load('./models/ml_models/ner_status_certainty_model.joblib')
        models['Certainty'] = joblib.load('./models/ml_models/ner_certainty_model.joblib')
        return models
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise

def get_entity_predictions(text, model, label_encoder):
    """Get entity predictions for a given text."""
    try:
        # Predict entity type
        pred_encoded = model.predict([text])[0]
        entity_type = label_encoder.inverse_transform([pred_encoded])[0]
        
        # Get prediction probability
        pred_proba = model.predict_proba([text])[0]
        confidence = pred_proba.max()
        
        return entity_type, confidence
    except Exception as e:
        logging.error(f"Error in entity prediction: {str(e)}")
        return None, None

def get_attribute_predictions(text, entity_type, attribute_models):
    """Get attribute predictions for a given text and entity type."""
    attributes = {}
    
    try:
        if entity_type in ["Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
                          "Immunotherapy", "Targeted_Therapy"]:
            attributes["Status_Certainty"] = attribute_models["Status_Certainty"].predict([text])[0]
            attributes["Combi"] = "No"  # Default value
            
        elif entity_type in ["Complete_Response", "Partial_Response", 
                           "Stable_Disease", "Progressive_Disease"]:
            attributes["Certainty"] = attribute_models["Certainty"].predict([text])[0]
    except Exception as e:
        logging.error(f"Error in attribute prediction: {str(e)}")
    
    return attributes

def process_text(text, models, model_name='random_forest'):
    """Process a single text input and return extracted entities with attributes."""
    entities = []
    
    # Split text into potential entity spans (simple tokenization)
    words = text.split()
    max_span_length = 5  # Maximum number of words to consider as an entity
    
    for i in range(len(words)):
        for j in range(i + 1, min(i + max_span_length + 1, len(words) + 1)):
            span_text = ' '.join(words[i:j])
            
            # Get entity predictions
            entity_type, confidence = get_entity_predictions(
                span_text, 
                models[model_name], 
                models['label_encoder']
            )
            
            # If entity is predicted with sufficient confidence
            if entity_type and confidence > 0.5 and entity_type != "Other":
                # Get attribute predictions
                attributes = get_attribute_predictions(
                    span_text,
                    entity_type,
                    models
                )
                
                # Calculate span indices
                start = text.find(span_text)
                if start != -1:
                    entities.append({
                        "text": span_text,
                        "start": start,
                        "end": start + len(span_text),
                        "entity_type": entity_type,
                        "confidence": float(confidence),
                        "attributes": attributes
                    })
    
    # Remove overlapping entities, keep the ones with higher confidence
    entities.sort(key=lambda x: (x['confidence'], -len(x['text'])), reverse=True)
    filtered_entities = []
    used_spans = set()
    
    for entity in entities:
        span_range = set(range(entity['start'], entity['end']))
        if not (span_range & used_spans):  # No overlap with existing entities
            filtered_entities.append(entity)
            used_spans.update(span_range)
    
    return filtered_entities

def format_output(entities):
    """Format entities for output."""
    formatted = []
    for i, entity in enumerate(entities):
        formatted_entity = {
            "id": f"T{i}",
            "text": entity["text"],
            "entity_type": entity["entity_type"],
            "span": f"{entity['start']}~{entity['end']}",
            "confidence": entity["confidence"]
        }
        formatted_entity.update(entity["attributes"])
        formatted.append(formatted_entity)
    
    return pd.DataFrame(formatted)

def main():
    setup_logging()
    logging.info("Starting ML-based NER inference")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load models
    models = load_models()
    
    # Example texts for inference
    sample_texts = [
        "Patient received chemotherapy with carboplatin and showed complete response.",
        "After immunotherapy treatment, partial response was observed.",
        "The targeted therapy with capmatinib resulted in stable disease."
    ]
    
    # Process each text
    for i, text in enumerate(sample_texts):
        logging.info(f"Processing text {i+1}")
        
        # Get predictions from both models
        for model_name in ['random_forest', 'xgboost']:
            entities = process_text(text, models, model_name)
            results_df = format_output(entities)
            
            # Save results
            output_file = f"results/text_{i+1}_{model_name}_entities.csv"
            results_df.to_csv(output_file, index=False)
            
            print(f"\nResults for text {i+1} using {model_name}:")
            print(results_df)
            
            logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()