"""
eval_accuracy.py

Evaluation script for isolated ISL sign recognition models.
Calculates Top-1 and Top-5 accuracy on the held-out test set, generates
a confusion matrix, and identifies the most frequently confused sign classes.

Author: Ishaara System
"""

import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train_isolated_classifier import get_signer_independent_splits, ISLDataset
from models.bilstm_encoder import BiLSTMEncoder
from models.transformer_encoder import TransformerEncoder

def evaluate(model, dataloader, device, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, masks, labels in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, masks)
            
            # Top-1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            top1_correct += torch.sum(preds == labels).item()
            
            # Top-5
            _, top5_preds = outputs.topk(5, 1, True, True)
            for i in range(labels.size(0)):
                if labels[i] in top5_preds[i]:
                    top5_correct += 1
                    
            total += labels.size(0)
            
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Identify top confusions
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))
                
    confusions.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 Confused Classes (True -> Predicted):")
    for i in range(min(5, len(confusions))):
        print(f"{confusions[i][0]} -> {confusions[i][1]} (Count: {confusions[i][2]})")
        
    return top1_acc, top5_acc, cm

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(20, 20))
    # If there are 263 classes, drawing labels might be too dense, limit if needed
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bilstm", choices=["bilstm", "transformer"])
    args = parser.parse_args()
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_path, "data", "cache", "landmarks")
    ckpt_path = os.path.join(base_path, "recognition", "checkpoints", f"best_{args.model}.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        exit(1)
        
    _, _, test_list, class_to_idx = get_signer_independent_splits(cache_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    num_classes = len(class_to_idx)
    
    test_dataset = ISLDataset(test_list, class_to_idx, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == "bilstm":
        model = BiLSTMEncoder(num_classes=num_classes)
    else:
        model = TransformerEncoder(num_classes=num_classes)
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    
    top1, top5, cm = evaluate(model, test_loader, device, class_names)
    
    save_dir = os.path.join(base_path, "eval")
    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, f"cm_{args.model}.png"))
