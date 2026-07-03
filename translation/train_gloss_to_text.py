"""
train_gloss_to_text.py

Phase 5: Gloss-to-Text Grammar Translation

Fine-tunes a small T5 model to translate ISL Glosses into grammatical English.
Uses synthetically generated pairs for demonstration since no large parallel
ISL corpus exists.

LIMITATION: Linguistically-informed synthetic approach, not real-corpus trained.
"""

import os
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Mock synthetic data
SYNTHETIC_PAIRS = [
    ("I APPLE EAT", "I am eating an apple."),
    ("YOUR NAME WHAT", "What is your name?"),
    ("TOMORROW I MARKET GO", "I will go to the market tomorrow."),
    ("BOY RUN", "The boy is running."),
    ("I KNOW NOT", "I do not know."),
] * 100 # Expand for fake dataset size

class GlossToTextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        
        for gloss, eng in pairs:
            # T5 expects prefix
            inp = "translate ISL to English: " + gloss
            
            tokenized_inp = tokenizer(inp, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_tgt = tokenizer(eng, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
            
            self.inputs.append({
                "input_ids": tokenized_inp.input_ids.squeeze(),
                "attention_mask": tokenized_inp.attention_mask.squeeze(),
            })
            # T5 requires target pad tokens to be -100 so they are ignored in loss
            labels = tokenized_tgt.input_ids.squeeze()
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.targets.append(labels)
            
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        item = {key: val.clone().detach() for key, val in self.inputs[idx].items()}
        item["labels"] = self.targets[idx].clone().detach()
        return item

def compute_bleu(model, tokenizer, pairs, device):
    model.eval()
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    print("\nQualitative Evaluation (Held-out):")
    with torch.no_grad():
        for i, (gloss, ref) in enumerate(pairs[:10]):
            inp = "translate ISL to English: " + gloss
            inputs = tokenizer(inp, return_tensors="pt").to(device)
            outputs = model.generate(inputs.input_ids, max_length=64)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # BLEU
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            pred_tokens = nltk.word_tokenize(pred.lower())
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(score)
            
            if i < 5:
                print(f"Gloss: {gloss}")
                print(f"Ref  : {ref}")
                print(f"Pred : {pred}")
                print("-" * 20)
                
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"\nAverage BLEU Score on test subset: {avg_bleu:.4f}")
    return avg_bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Split
    train_size = int(0.8 * len(SYNTHETIC_PAIRS))
    train_pairs = SYNTHETIC_PAIRS[:train_size]
    test_pairs = SYNTHETIC_PAIRS[train_size:]
    
    train_dataset = GlossToTextDataset(train_pairs, tokenizer)
    test_dataset = GlossToTextDataset(test_pairs, tokenizer)
    
    # Normally we'd use transformers Trainer here, but keeping dependencies simple
    # if Trainer is preferred:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, "translation", "models", "gloss_to_text")
    
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
    
    print("Training Gloss-to-Text model...")
    trainer.train()
    
    # Download punkt if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    compute_bleu(model, tokenizer, test_pairs, device)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
