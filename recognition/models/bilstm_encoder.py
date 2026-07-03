"""
bilstm_encoder.py

Isolated sign recognition PyTorch model using a BiLSTM encoder with temporal 
attention pooling. 

Input shape: (batch_size, seq_len, 285)
Output shape: (batch_size, num_classes)

Reason for architecture: BiLSTM is chosen over a Transformer for the initial 
baseline because INCLUDE has limited samples per class (~15-20), and 
attention-based Transformers tend to overfit heavily on small datasets 
without extensive pretraining, whereas BiLSTMs provide a more reliable 
temporal baseline for limited-data regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    Attention pooling over time steps so the model can learn which frames 
    in a sign are the most discriminative "peak" frames.
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attention(x) # (batch_size, seq_len, 1)
        
        if mask is not None:
            # Mask shape: (batch_size, seq_len)
            mask = mask.unsqueeze(-1)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum across seq_len
        pooled = torch.sum(x * attn_weights, dim=1) # (batch_size, hidden_dim)
        return pooled, attn_weights

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=285, hidden_dim=256, num_layers=2, num_classes=263, dropout=0.3):
        super(BiLSTMEncoder, self).__init__()
        
        # Initial projection to handle the high dimensional landmark input
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # BiLSTM output dim is hidden_dim * 2
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, 285)
        mask: (batch, seq_len) where 1 is valid frame, 0 is padding/invalid
        """
        x = self.proj(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Optionally pack padded sequence here if using rigorous variable lengths
        # For simplicity in this baseline, we process the padded sequence and use attention mask
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_dim * 2)
        
        pooled, attn_weights = self.attention(lstm_out, mask)
        
        logits = self.classifier(pooled)
        return logits
