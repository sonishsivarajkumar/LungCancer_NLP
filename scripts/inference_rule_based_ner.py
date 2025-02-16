# inference_rule_based_ner.py

import os
import yaml
import re
import logging
import pandas as pd

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/rule_based_ner_inference.log"),
            logging.StreamHandler()
        ]
    )

def load_patterns():
    try:
        with open('./models/rule_based_models/rule_based_ner_patterns.yaml', 'r') as f:
            patterns = yaml.safe_load(f)
        return {entity_type: re.compile(pattern, re.I) for entity_type, pattern in patterns.items()}
    except Exception as e:
        logging.error(f"Error loading patterns: {str(e)}")
        raise

def get_default_attributes(entity_type):
    if entity_type in ["Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
                      "Immunotherapy", "Targeted_Therapy"]:
        return {
            "Status_Certainty": "Confirmed_Present",
            "Combi": "No"
        }
    elif entity_type in ["Complete_Response", "Partial_Response", 
                        "Stable_Disease", "Progressive_Disease"]:
        return {
            "Certainty": "Confirmed"
        }
    return {}

def extract_entities(text, pattern_dict):
    entities = []
    
    # Find all possible matches for each entity type
    for entity_type, pattern in pattern_dict.items():
        for match in pattern.finditer(text):
            entity = {
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "entity_type": entity_type,
                "attributes": get_default_attributes(entity_type)
            }
            entities.append(entity)
    
    # Sort entities by start position and handle overlapping
    entities.sort(key=lambda x: (x['start'], -x['end']))
    filtered_entities = []
    last_end = -1
    
    for entity in entities:
        if entity['start'] >= last_end:
            filtered_entities.append(entity)
            last_end = entity['end']
    
    return filtered_entities

def format_output(entities):
    formatted = []
    for i, entity in enumerate(entities):
        formatted_entity = {
            "id": f"T{i}",
            "entity_type": entity["entity_type"],
            "text": entity["text"],
            "span": f"{entity['start']}~{entity['end']}"
        }
        formatted_entity.update(entity["attributes"])
        formatted.append(formatted_entity)
    
    return pd.DataFrame(formatted)

def process_text(text, pattern_dict):
    """Process a single text input and return extracted entities."""
    try:
        entities = extract_entities(text, pattern_dict)
        return format_output(entities)
    except Exception as e:
        logging.error(f"Error processing text: {str(e)}")
        return pd.DataFrame()

def main():
    setup_logging()
    logging.info("Starting rule-based NER inference")

    # Load patterns
    pattern_dict = load_patterns()

    # Example usage with sample texts
    sample_texts = [
        "Patient received chemotherapy with carboplatin and showed complete response.",
        "After immunotherapy treatment, partial response was observed.",
        "The targeted therapy with capmatinib resulted in stable disease."
    ]

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Process each text and save results
    for i, text in enumerate(sample_texts):
        logging.info(f"Processing text {i+1}")
        
        # Extract entities
        results_df = process_text(text, pattern_dict)
        
        # Save results
        if not results_df.empty:
            output_file = f"results/text_{i+1}_entities.csv"
            results_df.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")
            
            print(f"\nResults for text {i+1}:")
            print(results_df)
        else:
            logging.warning(f"No entities found in text {i+1}")

if __name__ == "__main__":
    main()