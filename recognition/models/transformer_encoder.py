"""
transformer_encoder.py

Isolated sign recognition PyTorch model using a Transformer encoder.

Input shape: (batch_size, seq_len, 285)
Output shape: (batch_size, num_classes)

Reason for architecture: This Transformer architecture is implemented for Phase 2b 
as a direct performance and latency comparison against the BiLSTM baseline, 
specifically to evaluate whether self-attention over sequences provides 
better temporal modeling for sign landmarks when data is limited.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x shape: (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=285, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, num_classes=263, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, 285)
        mask: (batch, seq_len) where 1 is valid, 0 is invalid
        """
        batch_size = x.size(0)
        
        x = self.proj(x) # (batch, seq_len, d_model)
        
        # Append CLS token to the beginning of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (batch, seq_len+1, d_model)
        
        x = self.pos_encoder(x)
        
        if mask is not None:
            # Mask for PyTorch transformer requires True for positions that are NOT allowed to attend.
            # PyTorch expects shape (batch, seq_len) for src_key_padding_mask
            # 1 -> False (valid), 0 -> True (invalid/padding)
            # We also need to add a True (valid) mask for the CLS token
            cls_mask = torch.ones((batch_size, 1), device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat((cls_mask, mask), dim=1)
            padding_mask = (full_mask == 0) # True means ignore
        else:
            padding_mask = None
            
        out = self.transformer(x, src_key_padding_mask=padding_mask) # (batch, seq_len+1, d_model)
        
        # Use the CLS token output for classification
        cls_out = out[:, 0, :] # (batch, d_model)
        
        logits = self.classifier(cls_out)
        return logits
