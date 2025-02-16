# scipts/train_rule_based_ner.py

import os
import xml.etree.ElementTree as ET
import pandas as pd
import re
import yaml
import logging
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

stop_words = set([
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'to', 'with', 'from', 'of', 'in', 
    'on', 'at', 'for', 'by', 'about', 'as', 'into', 'like', 'through', 'after',
    'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under',
    'around', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may',
    'might', 'must', 'this', 'that', 'these', 'those', 'then', 'there', 'here',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'but', 'or', 'because', 'as', 'until', 
    'while'
])


def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/rule_based_ner_training.log"),
            logging.StreamHandler()
        ]
    )

def load_xml_data(base_folder, folders):
    entities = []
    for folder_name in folders:
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.exists(folder_path):
            logging.warning(f"Folder does not exist: {folder_path}")
            continue

        xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
        for xml_file in xml_files:
            xml_file_path = os.path.join(folder_path, xml_file)
            try:
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                text_element = root.find("TEXT")
                document_text = text_element.text if text_element is not None else ""
                
                tags_element = root.find("TAGS")
                if tags_element is not None:
                    for annotation in tags_element:
                        if annotation.tag != "Treatment_Response":
                            entity = extract_entity(annotation, xml_file, document_text)
                            if entity:
                                entities.append(entity)
            except Exception as e:
                logging.error(f"Error processing {xml_file}: {str(e)}")
    
    return pd.DataFrame(entities)

def extract_entity(annotation, xml_file, document_text):
    tag = annotation.tag
    attrib = annotation.attrib
    
    entity = {
        "file_name": xml_file,
        "entity_id": attrib.get("id", ""),
        "entity_type": tag,
        "spans": attrib.get("spans", ""),
        "text": attrib.get("text", ""),
        "document_content": document_text,
    }
    
    if tag in ["Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
               "Immunotherapy", "Targeted_Therapy"]:
        entity["Status_Certainty"] = attrib.get("Status_Certainty", "Confirmed_Present")
        entity["Combi"] = attrib.get("Combi", "No")
    elif tag in ["Complete_Response", "Partial_Response", 
                 "Stable_Disease", "Progressive_Disease"]:
        entity["Certainty"] = attrib.get("Certainty", "Confirmed")
    
    return entity

from collections import Counter

def extract_keywords(df, entity_type):
    texts = df[df["entity_type"] == entity_type]["text"].str.lower()
    keywords = set()
    for text in texts:
        # Extract n-grams (up to trigrams)
        words = re.findall(r"\b\w+\b", text)
        for n in range(1, 4):
            ngrams = zip(*[words[i:] for i in range(n)])
            for ngram in ngrams:
                phrase = ' '.join(ngram)
                # Apply filtering here
                if all(word not in stop_words and len(word) > 2 for word in ngram):
                    keywords.add(phrase)
    return keywords



def create_patterns(train_df):
    entity_types = train_df["entity_type"].unique()
    pattern_dict = {}
    
    for entity_type in entity_types:
        keywords = extract_keywords(train_df, entity_type)
        if keywords:
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            pattern = r"\b(" + "|".join(map(re.escape, sorted_keywords)) + r")\b"
            pattern_dict[entity_type] = pattern
            logging.info(f"Created pattern for {entity_type} with {len(keywords)} keywords")
    
    return pattern_dict

def main():
    setup_logging()
    logging.info("Starting rule-based NER training")

    # Define the base folder path and subfolders
    base_folder = r"C:\Users\sivarajkumars2\OneDrive - UPMC\Documents\Lung_cancer_project\Annotation_shared\LC_Annotations\Annotated Files\Round_4\Subhash"
    folders = [
        "Completed Partial Response",
        "Completed Progressive Disease",
        "Completed RECIST complete response",
        "Completed Stable Disease",
    ]
    
    # Load and preprocess data
    df_entities = load_xml_data(base_folder, folders)
    df_entities = df_entities.dropna(subset=["entity_type", "text"])
    
    # Split data
    train_df, val_df = train_test_split(df_entities, test_size=0.2, random_state=42)
    
    # Create patterns
    pattern_dict = create_patterns(train_df)
    
    # Save patterns
    os.makedirs('./models/rule_based_models', exist_ok=True)
    with open('./models/rule_based_models/rule_based_ner_patterns.yaml', 'w') as f:
        yaml.dump(pattern_dict, f)
    
    logging.info("Rule-based NER patterns have been saved")

if __name__ == "__main__":
    main()