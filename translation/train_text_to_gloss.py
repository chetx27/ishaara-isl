"""
train_text_to_gloss.py

Phase 6: Text-to-Gloss Reverse Translation

Fine-tunes a small T5 model to translate grammatical English into ISL Glosses.
This feeds into the Phase 6 Lookup-and-Stitch generation pipeline.
"""

import os
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Use the same synthetic pairs but reverse the input/output logic
from train_gloss_to_text import SYNTHETIC_PAIRS, compute_bleu

class TextToGlossDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        
        for gloss, eng in pairs:
            # T5 expects prefix
            inp = "translate English to ISL: " + eng
            
            tokenized_inp = tokenizer(inp, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_tgt = tokenizer(gloss, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            
            self.inputs.append({
                "input_ids": tokenized_inp.input_ids.squeeze(),
                "attention_mask": tokenized_inp.attention_mask.squeeze(),
            })
            labels = tokenized_tgt.input_ids.squeeze()
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.targets.append(labels)
            
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        item = {key: val.clone().detach() for key, val in self.inputs[idx].items()}
        item["labels"] = self.targets[idx].clone().detach()
        return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    train_size = int(0.8 * len(SYNTHETIC_PAIRS))
    train_pairs = SYNTHETIC_PAIRS[:train_size]
    test_pairs = SYNTHETIC_PAIRS[train_size:]
    
    train_dataset = TextToGlossDataset(train_pairs, tokenizer)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, "translation", "models", "text_to_gloss")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    print("Training Text-to-Gloss model...")
    trainer.train()
    
    # Reverse pairs for qualitative eval
    rev_test_pairs = [(eng, gloss) for gloss, eng in test_pairs]
    compute_bleu(model, tokenizer, rev_test_pairs, device)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
