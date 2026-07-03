"""
train_continuous_ctc.py

Phase 4: Continuous Signing Segmentation Model using CTC loss.

This script extends the BiLSTM isolated sign recognition model to continuous 
sign recognition by applying Connectionist Temporal Classification (CTC) loss.

KNOWN LIMITATION (Co-articulation-naive):
Because INCLUDE is an isolated-sign dataset and no large-scale continuous ISL
dataset exists (documented in Phase 0 audit), we synthetically concatenate 
isolated sign clips with simple linear transition interpolation.
This synthetic data does NOT capture natural co-articulation (the blending 
between real continuous signs). Consequently, the Sign Error Rate (SER) 
reported here is "co-articulation-naive" and will likely degrade on real-world 
continuous footage. True continuous modeling requires a real continuous corpus.

Author: Ishaara System
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import argparse
from scipy.interpolate import interp1d

from models.bilstm_encoder import BiLSTMEncoder
from train_isolated_classifier import get_signer_independent_splits

def wer(reference, hypothesis):
    """
    Computes the Word Error Rate (or Sign Error Rate - SER) using Levenshtein distance.
    (insertions + deletions + substitutions) / len(reference)
    """
    r = reference
    h = hypothesis
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint16)
    d = d.reshape((len(r)+1, len(h)+1))
    
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
        
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
                
    return d[len(r)][len(h)] / float(len(r)) if len(r) > 0 else 1.0


class SyntheticContinuousDataset(Dataset):
    def __init__(self, data_list, class_to_idx, max_seq_len=500, signs_per_sequence=3):
        self.data_list = data_list
        self.class_to_idx = class_to_idx
        self.max_seq_len = max_seq_len
        self.signs_per_sequence = signs_per_sequence
        # Group by signer to make concatenations somewhat realistic (same body)
        self.signer_data = self._group_by_signer()
        
    def _group_by_signer(self):
        grouped = {}
        for feat_path, mask_path, cls in self.data_list:
            s_id = hash(os.path.basename(feat_path)) % 115
            if s_id not in grouped:
                grouped[s_id] = []
            grouped[s_id].append((feat_path, mask_path, cls))
        return grouped
        
    def __len__(self):
        return len(self.data_list) # Roughly same number of synthesized seqs as isolated seqs
        
    def __getitem__(self, idx):
        # Pick a random signer
        signer_id = random.choice(list(self.signer_data.keys()))
        clips = random.choices(self.signer_data[signer_id], k=self.signs_per_sequence)
        
        concat_features = []
        concat_masks = []
        labels = []
        
        for i, (feat_path, mask_path, cls) in enumerate(clips):
            feat = np.load(feat_path)
            mask = np.load(mask_path)
            labels.append(self.class_to_idx[cls])
            
            concat_features.append(feat)
            concat_masks.append(mask)
            
            # Synthetic linear transition (e.g., 5 frames) between signs
            if i < self.signs_per_sequence - 1:
                next_feat = np.load(clips[i+1][0])
                if len(feat) > 0 and len(next_feat) > 0:
                    start_pt = feat[-1]
                    end_pt = next_feat[0]
                    trans = np.linspace(start_pt, end_pt, num=5)
                    concat_features.append(trans)
                    concat_masks.append(np.ones(5))
                    
        full_feat = np.concatenate(concat_features, axis=0)
        full_mask = np.concatenate(concat_masks, axis=0)
        
        T = full_feat.shape[0]
        pad_len = self.max_seq_len - T
        
        if pad_len > 0:
            feat_pad = np.zeros((pad_len, 285), dtype=np.float32)
            mask_pad = np.zeros((pad_len,), dtype=np.float32)
            full_feat = np.concatenate([full_feat, feat_pad], axis=0)
            full_mask = np.concatenate([full_mask, mask_pad], axis=0)
            input_length = T
        else:
            full_feat = full_feat[:self.max_seq_len]
            full_mask = full_mask[:self.max_seq_len]
            input_length = self.max_seq_len
            
        target_length = len(labels)
        
        return (torch.tensor(full_feat, dtype=torch.float32), 
                torch.tensor(full_mask, dtype=torch.float32), 
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(input_length, dtype=torch.long),
                torch.tensor(target_length, dtype=torch.long))

def train_ctc(model, dataloaders, optimizer, device, epochs=30):
    # Blank token is 0. So we need num_classes + 1 for CTC
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            
            for inputs, masks, targets, input_lengths, target_lengths in dataloaders[phase]:
                inputs = inputs.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # BiLSTM output shape: (batch, seq, num_classes)
                    outputs = model(inputs, masks)
                    
                    # CTCLoss expects (T, N, C)
                    outputs = outputs.permute(1, 0, 2)
                    outputs = torch.nn.functional.log_softmax(outputs, dim=2)
                    
                    loss = criterion(outputs, targets, input_lengths, target_lengths)
                    
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase.capitalize()} CTC Loss: {epoch_loss:.4f}")
            
            # Simple Greedy Decoding for SER on validation
            if phase == 'val':
                model.eval()
                total_ser = 0.0
                count = 0
                with torch.no_grad():
                    for inputs, masks, targets, input_lengths, target_lengths in dataloaders[phase]:
                        inputs = inputs.to(device)
                        outputs = model(inputs, masks)
                        probs = torch.softmax(outputs, dim=2)
                        _, preds = torch.max(probs, 2) # (batch, seq)
                        
                        preds = preds.cpu().numpy()
                        targets = targets.numpy()
                        
                        for i in range(inputs.size(0)):
                            # Greedy decode: remove duplicates and blanks (0)
                            pred_seq = preds[i][:input_lengths[i]]
                            decoded = []
                            prev = -1
                            for p in pred_seq:
                                if p != prev and p != 0:
                                    decoded.append(p)
                                prev = p
                                
                            ref_seq = targets[i][:target_lengths[i]].tolist()
                            total_ser += wer(ref_seq, decoded)
                            count += 1
                            
                avg_ser = total_ser / count
                print(f"Validation Sign Error Rate (SER) [Co-articulation-naive]: {avg_ser:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_path, "data", "cache", "landmarks")
    
    train_list, val_list, test_list, class_to_idx = get_signer_independent_splits(cache_dir)
    
    # +1 for CTC blank token (0 is blank, classes are 1..263)
    num_classes = len(class_to_idx) + 1 
    # Adjust class_to_idx to be 1-indexed for targets
    class_to_idx = {k: v + 1 for k, v in class_to_idx.items()}
    
    train_dataset = SyntheticContinuousDataset(train_list, class_to_idx)
    val_dataset = SyntheticContinuousDataset(val_list, class_to_idx)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False)
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use BiLSTM baseline but with output size = num_classes + 1
    # Note: BiLSTMEncoder has an attention pool, but for CTC we need per-frame predictions!
    # Therefore, we need a modified BiLSTM that returns the full sequence.
    
    class CTCBiLSTM(nn.Module):
        def __init__(self, input_dim=285, hidden_dim=256, num_classes=264):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)
            
        def forward(self, x, mask=None):
            x = torch.relu(self.proj(x))
            out, _ = self.lstm(x)
            logits = self.classifier(out) # (batch, seq, num_classes)
            return logits

    model = CTCBiLSTM(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_ctc(model, dataloaders, optimizer, device, epochs=args.epochs)
