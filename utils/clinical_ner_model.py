# utils/clinical_ner_model.py

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF
import logging

class ClinicalBiLSTMCRF(nn.Module):
    """Clinical NER model with Clinical-BERT, BiLSTM-CRF architecture."""

    def __init__(self, config):
        super(ClinicalBiLSTMCRF, self).__init__()
        self.config = config

        # BERT encoder
        self.bert = AutoModel.from_pretrained(config['bert_model_name'])
        self.bert_dropout = nn.Dropout(config['dropout'])

        # BiLSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=self.bert.config.hidden_size if i == 0 else config['lstm_hidden_size'] * 2,
                hidden_size=config['lstm_hidden_size'],
                bidirectional=True,
                batch_first=True
            ) for i in range(config['num_lstm_layers'])
        ])
        
        self.lstm_dropout = nn.Dropout(config['dropout'])
        self.layer_norm = nn.LayerNorm(config['lstm_hidden_size'] * 2)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config['lstm_hidden_size'] * 2,
            num_heads=config['num_attention_heads'],
            dropout=config['dropout'],
            batch_first=True
        )

        # Output layers
        self.classifier = nn.Linear(config['lstm_hidden_size'] * 2, config['num_labels'])
        self.crf = CRF(config['num_labels'], batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the model."""
        # BERT encoding
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        sequence_output = self.bert_dropout(sequence_output)

        # BiLSTM with residual connections
        lstm_output = sequence_output
        for i, lstm in enumerate(self.lstm_layers):
            new_lstm_output, _ = lstm(lstm_output)
            new_lstm_output = self.layer_norm(new_lstm_output)

            # Add residual connection if shapes match
            if lstm_output.shape == new_lstm_output.shape:
                lstm_output = lstm_output + new_lstm_output
            else:
                lstm_output = new_lstm_output

            lstm_output = self.lstm_dropout(lstm_output)

        # Multi-head attention
        attn_output, _ = self.attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = self.lstm_dropout(attn_output)

        # Classification layer
        emissions = self.classifier(attn_output)

        # CRF layer
        if labels is not None:
            # Training mode
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return loss, predictions
        else:
            # Inference mode
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute focal loss."""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
