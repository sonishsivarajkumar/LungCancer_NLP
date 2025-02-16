# utils/data_preprocessing.py

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import re

def setup_logging(log_file):
    """Setup logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_data(base_folder, folders):
    """Load and preprocess data from XML files."""
    from train_rule_based_ner import load_xml_data  # Ensure this import is correct
    df = load_xml_data(base_folder, folders)
    logging.info(f"Data loaded: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data for deep learning model training."""
    required_columns = ['text', 'entity_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns in DataFrame: {missing_columns}")
        raise KeyError(f"Missing columns: {missing_columns}")

    # Remove rows with missing values
    df = df.dropna(subset=required_columns)
    logging.info(f"Data after dropping NA: {df.shape}")

    # Convert text to lowercase
    df['text'] = df['text'].str.lower()

    # Prepare sentences and labels
    sentences = df['text'].tolist()
    labels = df['entity_type'].tolist()

    # Tokenize sentences
    tokenized_sentences = [re.findall(r'\b\w+\b', sentence) for sentence in sentences]

    # Build vocabulary
    words = set()
    for sent in tokenized_sentences:
        words.update(sent)
    words = sorted(list(words))
    logging.info(f"Unique words in vocabulary: {len(words)}")

    # Create word to index mapping
    word2idx = {w: i + 2 for i, w in enumerate(words)}  # Start indexing from 2
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1

    idx2word = {i: w for w, i in word2idx.items()}

    # Create label to index mapping
    labels_set = sorted(list(set(labels)))
    if 'O' not in labels_set:
        labels_set.insert(0, 'O')  # Ensure 'O' is present for non-entity tokens
    label2idx = {l: i for i, l in enumerate(labels_set)}
    idx2label = {i: l for l, i in label2idx.items()}

    logging.info(f"Labels: {label2idx}")

    return tokenized_sentences, labels, word2idx, idx2word, label2idx, idx2label

def encode_data(tokenized_sentences, labels, word2idx, label2idx, max_len):
    """Encode sentences and labels into sequences of indices."""
    X = []
    y = []
    for sent, label in zip(tokenized_sentences, labels):
        sent_seq = [word2idx.get(w, word2idx['UNK']) for w in sent]
        label_seq = [label2idx[label]] * len(sent_seq)  # Modify if labels are token-level
        X.append(sent_seq)
        y.append(label_seq)

    # Pad sequences
    X = pad_sequences(X, max_len)
    y = pad_sequences(y, max_len, padding_value=label2idx.get('O', 0))

    # Convert labels to torch tensors
    y = np.array(y)

    logging.info(f"Encoded X shape: {X.shape}")
    logging.info(f"Encoded y shape: {y.shape}")

    return X, y

def pad_sequences(sequences, max_len, padding_value=0):
    """Pad sequences to the same length."""
    padded = np.full((len(sequences), max_len), padding_value, dtype='int32')
    for idx, seq in enumerate(sequences):
        truncated = seq[:max_len]
        padded[idx, :len(truncated)] = truncated
    return padded

class NERDataset(Dataset):
    """Custom Dataset for NER."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def save_mappings(word2idx, idx2word, label2idx, idx2label, model_dir):
    """Save mappings using joblib."""
    joblib.dump(word2idx, os.path.join(model_dir, 'word2idx.joblib'))
    joblib.dump(idx2word, os.path.join(model_dir, 'idx2word.joblib'))
    joblib.dump(label2idx, os.path.join(model_dir, 'label2idx.joblib'))
    joblib.dump(idx2label, os.path.join(model_dir, 'idx2label.joblib'))
    logging.info("Mappings saved successfully.")

def load_mappings(model_dir):
    """Load word and label mappings."""
    word2idx = joblib.load(os.path.join(model_dir, 'word2idx.joblib'))
    idx2word = joblib.load(os.path.join(model_dir, 'idx2word.joblib'))
    label2idx = joblib.load(os.path.join(model_dir, 'label2idx.joblib'))
    idx2label = joblib.load(os.path.join(model_dir, 'idx2label.joblib'))
    return word2idx, idx2word, label2idx, idx2label
