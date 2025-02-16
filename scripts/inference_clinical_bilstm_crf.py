# scripts/inference_clinical_bilstm_crf.py

import os
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from utils.data_preprocessing import setup_logging_inference, load_mappings
from utils.clinical_ner_model import ClinicalBiLSTMCRF
from utils.metrics import flatten_predictions, generate_classification_report, plot_confusion_matrix_fn, save_error_examples
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re

def setup_logging_inference(log_file):
    """Setup logging configuration for inference."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class InferenceDataset(Dataset):
    """Custom Dataset for Inference."""

    def __init__(self, sentences, word2idx, max_len):
        self.sentences = sentences
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        sent_seq = [self.word2idx.get(w, self.word2idx['UNK']) for w in tokens]
        if len(sent_seq) < self.max_len:
            sent_seq += [self.word2idx['PAD']] * (self.max_len - len(sent_seq))
        else:
            sent_seq = sent_seq[:self.max_len]
        return torch.tensor(sent_seq, dtype=torch.long), tokens

def load_model_checkpoint(model, checkpoint_path, device):
    """Load the model weights."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")
    return model

def process_texts(model, sentences, word2idx, label2idx, idx2label, max_len, device):
    """Process and predict entities for a list of sentences."""
    dataset = InferenceDataset(sentences, word2idx, max_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []
    tokens_list = []

    with torch.no_grad():
        for batch, tokens in tqdm(loader, desc="Predicting"):
            batch = batch.to(device)
            loss, preds = model(batch)
            for pred_seq in preds:
                predictions.append(pred_seq)
    return predictions

def extract_entities(sentences, predictions, label2idx, idx2label, max_len):
    """Extract entities based on predictions."""
    entities = []
    for sentence, pred in zip(sentences, predictions):
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        start = None
        entity_type = None
        for idx, label in enumerate(pred):
            if idx >= len(tokens):
                break
            if label == 'O':
                if start is not None:
                    end = idx
                    entity_text = ' '.join(tokens[start:end])
                    entities.append({
                        "text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "confidence": None  # Placeholder for confidence
                    })
                    start = None
                    entity_type = None
                continue
            if label.startswith('B-'):
                if start is not None:
                    end = idx
                    entity_text = ' '.join(tokens[start:end])
                    entities.append({
                        "text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "confidence": None
                    })
                start = idx
                entity_type = label[2:]
            elif label.startswith('I-') and entity_type == label[2:]:
                continue
            else:
                if start is not None:
                    end = idx
                    entity_text = ' '.join(tokens[start:end])
                    entities.append({
                        "text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "confidence": None
                    })
                start = None
                entity_type = None
        # Catch any remaining entity
        if start is not None:
            end = len(tokens)
            entity_text = ' '.join(tokens[start:end])
            entities.append({
                "text": entity_text,
                "start": start,
                "end": end,
                "entity_type": entity_type,
                "confidence": None
            })
    return entities

def format_output(entities):
    """Format entities for output."""
    formatted = []
    for i, entity in enumerate(entities):
        formatted_entity = {
            "id": f"T{i}",
            "text": entity["text"],
            "entity_type": entity["entity_type"],
            "span": f"{entity['start']}~{entity['end']}",
            "confidence": entity["confidence"] if entity["confidence"] else 1.0  # Default confidence
        }
        formatted.append(formatted_entity)
    return pd.DataFrame(formatted)

def main():
    # Setup logging
    log_file = 'logs/clinical_bilstm_crf_inference.log'
    setup_logging_inference(log_file)
    logging.info("Starting ClinicalBiLSTMCRF NER inference")

    # Define paths
    model_dir = './models/clinical_bilstm_crf_models'
    checkpoint_path = os.path.join(model_dir, 'clinical_bilstm_crf_model.pth')
    mappings_path = model_dir  # Assuming mappings are saved in the same directory

    # Load mappings
    word2idx, idx2word, label2idx, idx2label = load_mappings(mappings_path)

    # Determine maximum sequence length
    max_len = 50  # Must match the training max_len

    # Initialize and load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'bert_model_name': 'emilyalsentzer/Bio_ClinicalBERT',
        'num_labels': len(label2idx),
        'lstm_hidden_size': 256,
        'num_lstm_layers': 2,
        'num_attention_heads': 8,
        'dropout': 0.1
    }
    model = ClinicalBiLSTMCRF(config)
    model = load_model_checkpoint(model, checkpoint_path, device)

    # Example texts for inference
    sample_texts = [
        "Patient received chemotherapy with carboplatin and showed complete response.",
        "After immunotherapy treatment, partial response was observed.",
        "The targeted therapy with capmatinib resulted in stable disease."
    ]

    # Process and predict
    predictions = process_texts(model, sample_texts, word2idx, label2idx, idx2label, max_len, device)

    # Extract entities
    entities = extract_entities(sample_texts, predictions, label2idx, idx2label, max_len)

    # Format output
    results_df = format_output(entities)

    # Save results
    os.makedirs('evaluation_results/clinical_bilstm_crf', exist_ok=True)
    output_file = f"evaluation_results/clinical_bilstm_crf/inference_results.csv"
    results_df.to_csv(output_file, index=False)

    logging.info(f"Inference results saved to {output_file}")
    print(f"\nResults for inference:")
    print(results_df)

if __name__ == "__main__":
    main()
