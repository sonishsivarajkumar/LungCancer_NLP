# scripts/visualize_attention.py

import os
import logging
import joblib
import numpy as np
import pandas as pd
import torch
from utils.data_preprocessing import setup_logging, load_mappings
from utils.clinical_ner_model import ClinicalBiLSTMCRF
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging_attention(log_file):
    """Setup logging configuration for attention visualization."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def visualize_attention(model, tokenizer, sentence, word2idx, idx2label, max_len, device):
    """Visualize attention weights for a given sentence."""
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_len]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids + [word2idx.get('PAD')] * (max_len - len(input_ids))
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = (input_tensor != word2idx['PAD']).long().to(device)

    with torch.no_grad():
        outputs = model.bert(input_ids=input_tensor, attention_mask=attention_mask)
        attentions = outputs.attentions  # List of attention weights from each layer

    # Example: Visualize attention from the last layer
    last_layer_attn = attentions[-1][0]  # Shape: (num_heads, seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=0).cpu().numpy()  # Shape: (seq_len, seq_len)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title('Average Attention Weights (Last BERT Layer)')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

def main():
    # Setup logging
    log_file = 'logs/attention_visualization.log'
    setup_logging_attention(log_file)
    logging.info("Starting Attention Visualization")

    # Define paths
    model_dir = './models/clinical_bilstm_crf_models'
    checkpoint_path = os.path.join(model_dir, 'clinical_bilstm_crf_model.pth')
    mappings_path = model_dir

    # Load mappings
    word2idx, idx2word, label2idx, idx2label = load_mappings(mappings_path)

    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {checkpoint_path}")

    # Example sentence
    sentence = "Patient received chemotherapy with carboplatin and showed complete response."

    # Visualize attention
    visualize_attention(model, tokenizer, sentence, word2idx, idx2label, max_len=50, device=device)

    logging.info("Attention Visualization completed successfully")

if __name__ == "__main__":
    main()
