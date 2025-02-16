# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import torch
import yaml
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title='Lung Cancer NER and RE Pipeline',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    st.title('Lung Cancer NER and RE Pipeline')
    st.markdown("""
    This application extracts entities (treatments and responses) from clinical notes 
    and identifies relationships between them using various models.
    """)

def check_model_availability():
    """Check which models are available and return their status."""
    model_status = {
        'rule_based': False,
        'ml': False,
        'bert': False
    }
    
    # Check rule-based models
    if os.path.exists('./models/rule_based_models/rule_based_ner_patterns.yaml'):
        model_status['rule_based'] = True
    
    # Check ML models
    ml_files = [
        './models/ml_models/ner_random_forest_model.joblib',
        './models/ml_models/ner_xgboost_model.joblib'
    ]
    if any(os.path.exists(f) for f in ml_files):
        model_status['ml'] = True
    
    # Check BERT models
    if os.path.exists('./models/bert_models'):
        model_status['bert'] = True
    
    return model_status

def load_rule_based_models():
    """Load rule-based NER models."""
    try:
        with open('./models/rule_based_models/rule_based_ner_patterns.yaml', 'r') as file:
            patterns = yaml.safe_load(file)
            # Compile patterns
            compiled_patterns = {
                entity_type: re.compile(pattern, re.I) 
                for entity_type, pattern in patterns.items()
            }
        return {'patterns': compiled_patterns}
    except Exception as e:
        st.warning(f"Error loading rule-based models: {str(e)}")
        return None

def load_ml_models():
    """Load ML-based models."""
    try:
        models = {}
        if os.path.exists('./models/ml_models/ner_random_forest_model.joblib'):
            models['random_forest'] = joblib.load('./models/ml_models/ner_random_forest_model.joblib')
        if os.path.exists('./models/ml_models/ner_xgboost_model.joblib'):
            models['xgboost'] = joblib.load('./models/ml_models/ner_xgboost_model.joblib')
        return models
    except Exception as e:
        st.warning(f"Error loading ML models: {str(e)}")
        return None

def load_re_models():
    """Load relation extraction models."""
    try:
        models = {}
        if os.path.exists('./models/ml_models/re_random_forest.joblib'):
            models['random_forest'] = joblib.load('./models/ml_models/re_random_forest.joblib')
        if os.path.exists('./models/ml_models/re_xgboost.joblib'):
            models['xgboost'] = joblib.load('./models/ml_models/re_xgboost.joblib')
        return models
    except Exception as e:
        st.warning(f"Error loading RE models: {str(e)}")
        return None

def get_default_attributes(entity_type):
    """Get default attributes for an entity type."""
    attributes = {}
    if entity_type in [
        'Cancer_Surgery', 'Radiotherapy', 'Chemotherapy', 
        'Immunotherapy', 'Targeted_Therapy'
    ]:
        attributes['Status_Certainty'] = 'Confirmed_Present'
        attributes['Combi'] = 'No'
    elif entity_type in [
        'Complete_Response', 'Partial_Response', 
        'Stable_Disease', 'Progressive_Disease'
    ]:
        attributes['Certainty'] = 'Confirmed'
    return attributes

def extract_entities_rule_based(text, patterns):
    """Extract entities using rule-based approach."""
    entities = []
    for entity_type, pattern in patterns.items():
        for match in pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'entity_type': entity_type,
                'confidence': 1.0,
                'attributes': get_default_attributes(entity_type)
            })
    return entities

def extract_entities_ml(text, models, model_name):
    """Extract entities using ML approach."""
    entities = []
    words = re.findall(r'\b\w+\b', text)
    
    for i in range(len(words)):
        for j in range(i + 1, min(i + 5, len(words) + 1)):
            phrase = ' '.join(words[i:j])
            
            # Create simple features (presence/absence of words)
            features = np.zeros((1, 100))  # Adjust size based on your model
            
            # Get prediction
            model = models[model_name.lower().replace(' ', '_')]
            try:
                prediction = model.predict(features)[0]
                confidence = np.max(model.predict_proba(features)[0])
                
                if confidence > 0.5:
                    start = text.find(phrase)
                    if start != -1:
                        entities.append({
                            'text': phrase,
                            'start': start,
                            'end': start + len(phrase),
                            'entity_type': prediction,
                            'confidence': float(confidence),
                            'attributes': get_default_attributes(prediction)
                        })
            except Exception as e:
                continue
    
    return entities

def extract_relationships(entities, re_models, model_name):
    """Extract relationships between entities."""
    relationships = []
    
    treatment_types = [
        "Cancer_Surgery", "Radiotherapy", "Chemotherapy", 
        "Immunotherapy", "Targeted_Therapy"
    ]
    response_types = [
        "Complete_Response", "Partial_Response", 
        "Stable_Disease", "Progressive_Disease"
    ]
    
    treatments = [e for e in entities if e['entity_type'] in treatment_types]
    responses = [e for e in entities if e['entity_type'] in response_types]
    
    if model_name in re_models:
        model = re_models[model_name.lower().replace(' ', '_')]
        
        for treatment in treatments:
            for response in responses:
                # Create simple features
                features = np.zeros((1, 100))  # Adjust size based on your model
                
                try:
                    prediction = model.predict(features)[0]
                    confidence = np.max(model.predict_proba(features)[0])
                    
                    if prediction == 1:
                        relationships.append({
                            'treatment_text': treatment['text'],
                            'treatment_type': treatment['entity_type'],
                            'response_text': response['text'],
                            'response_type': response['entity_type'],
                            'confidence': float(confidence)
                        })
                except Exception as e:
                    continue
    
    return relationships

def visualize_entities(text, entities):
    """Create a visual representation of extracted entities."""
    colors = {
        'Cancer_Surgery': '#FF9999',
        'Radiotherapy': '#99FF99',
        'Chemotherapy': '#9999FF',
        'Immunotherapy': '#FFFF99',
        'Targeted_Therapy': '#FF99FF',
        'Complete_Response': '#99FFFF',
        'Partial_Response': '#FFB366',
        'Stable_Disease': '#B366FF',
        'Progressive_Disease': '#66FFB3'
    }
    
    html_text = text
    for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
        color = colors.get(entity['entity_type'], '#CCCCCC')
        html_text = (
            html_text[:entity['start']] +
            f'<span style="background-color: {color};" title="{entity["entity_type"]}">' +
            html_text[entity['start']:entity['end']] +
            '</span>' +
            html_text[entity['end']:]
        )
    
    return html_text

def visualize_relationships(relationships):
    """Create a visual representation of the relationships."""
    if not relationships:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    treatments = list(set((r['treatment_text'], r['treatment_type']) for r in relationships))
    responses = list(set((r['response_text'], r['response_type']) for r in relationships))
    
    for i, (text, type_) in enumerate(treatments):
        ax.plot(0, i, 'bo', label=type_ if i == 0 else "")
        ax.text(0.1, i, text, ha='left', va='center')
    
    for i, (text, type_) in enumerate(responses):
        ax.plot(1, i, 'ro', label=type_ if i == 0 else "")
        ax.text(1.1, i, text, ha='left', va='center')
    
    for rel in relationships:
        t_idx = next(i for i, (t, _) in enumerate(treatments) if t == rel['treatment_text'])
        r_idx = next(i for i, (r, _) in enumerate(responses) if r == rel['response_text'])
        ax.plot([0, 1], [t_idx, r_idx], 'g--', alpha=0.3)
    
    ax.set_xlim(-0.5, 2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Treatments', 'Responses'])
    ax.set_yticks([])
    plt.legend()
    st.pyplot(fig)

def main():
    setup_page()
    
    # Check available models
    model_status = check_model_availability()
    
    # Show model status in sidebar
    st.sidebar.header('Available Models')
    st.sidebar.markdown(f"""
    - Rule-based: {'✓' if model_status['rule_based'] else '✗'}
    - Machine Learning: {'✓' if model_status['ml'] else '✗'}
    - BERT: {'✓' if model_status['bert'] else '✗'}
    """)
    
    # Initialize available model types
    available_ner_types = []
    if model_status['rule_based']:
        available_ner_types.append('Rule-based')
    if model_status['ml']:
        available_ner_types.append('Machine Learning')
    if model_status['bert']:
        available_ner_types.append('BERT')
    
    if not available_ner_types:
        st.error("No models are currently available. Please check model installation.")
        return
    
    # Model selection
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        ner_model_type = st.selectbox(
            'NER Model Type',
            available_ner_types
        )
    
    with col2:
        re_model_type = st.selectbox(
            'RE Model Type',
            ['Random Forest', 'XGBoost']
        )
    
    # ML model selection for NER
    ml_model = None
    if ner_model_type == 'Machine Learning':
        ml_model = st.sidebar.selectbox(
            'NER ML Model',
            ['Random Forest', 'XGBoost']
        )
    
    # Text input
    text = st.text_area(
        'Enter Clinical Note:',
        height=200,
        placeholder='Enter the clinical note text here...'
    )
    
    if st.button('Extract Entities and Relationships'):
        if text:
            try:
                # Extract entities
                entities = []
                
                if ner_model_type == 'Rule-based':
                    rule_based_models = load_rule_based_models()
                    if rule_based_models:
                        entities = extract_entities_rule_based(text, rule_based_models['patterns'])
                
                elif ner_model_type == 'Machine Learning':
                    ml_models = load_ml_models()
                    if ml_models:
                        entities = extract_entities_ml(text, ml_models, ml_model)
                
                # Display entities
                if entities:
                    st.subheader('Extracted Entities')
                    st.markdown(
                        visualize_entities(text, entities),
                        unsafe_allow_html=True
                    )
                    
                    df_entities = pd.DataFrame([
                        {
                            'Entity': e['text'],
                            'Type': e['entity_type'],
                            'Confidence': e.get('confidence', '-'),
                            **e.get('attributes', {})
                        }
                        for e in entities
                    ])
                    st.dataframe(df_entities)
                    
                    # Extract and display relationships
                    re_models = load_re_models()
                    if re_models:
                        relationships = extract_relationships(entities, re_models, re_model_type)
                        
                        st.subheader('Extracted Relationships')
                        if relationships:
                            df_relationships = pd.DataFrame([
                                {
                                    'Treatment': r['treatment_text'],
                                    'Treatment Type': r['treatment_type'],
                                    'Response': r['response_text'],
                                    'Response Type': r['response_type'],
                                    'Confidence': f"{r['confidence']:.2f}"
                                }
                                for r in relationships
                            ])
                            st.dataframe(df_relationships)
                            
                            # Visualize relationships
                            st.subheader('Relationship Visualization')
                            visualize_relationships(relationships)
                        else:
                            st.info('No relationships found between the extracted entities.')
                else:
                    st.info('No entities found in the text.')
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.error("Please check if all required model files are present and properly formatted.")
        else:
            st.warning('Please enter some text to analyze.')

if __name__ == "__main__":
    main()