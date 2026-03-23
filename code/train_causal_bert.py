#!/usr/bin/env python3
"""
Train Causal-BERT model for safety detection
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CausalBERTModel(nn.Module):
    """Causal-BERT with CCL loss"""
    def __init__(self, bert_model, num_labels=2, margin=1.0):
        super().__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.margin = margin

        # Projection head for CCL
        self.projection = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_ids, attention_mask, labels=None, return_embedding=False):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        logits = self.bert.classifier(pooled_output)

        loss = None
        if labels is not None:
            ce_loss = nn.CrossEntropyLoss()(logits, labels)
            loss = ce_loss

        if return_embedding:
            embedding = self.projection(pooled_output)
            return {"logits": logits, "loss": loss, "embedding": embedding}

        return {"logits": logits, "loss": loss}


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def compute_metrics(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs["logits"], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()

    metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Causal-BERT")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument("--train_data", type=str, default="./data/ccsd_v2/ccsd_train.json")
    parser.add_argument("--val_data", type=str, default="./data/ccsd_v2/ccsd_val.json")
    parser.add_argument("--test_data", type=str, default="./data/ccsd_v2/ccsd_test.json")
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/causal_bert")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) / f"causal_bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training Causal-BERT model...")
    logger.info(f"Output directory: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    bert_model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    model = CausalBERTModel(bert_model, num_labels=args.num_labels, margin=args.margin)
    model.to(device)

    # Load data
    logger.info("Loading datasets...")
    with open(args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets and dataloaders
    train_dataset = SimpleDataset(train_data, tokenizer, args.max_length)
    val_dataset = SimpleDataset(val_data, tokenizer, args.max_length)
    test_dataset = SimpleDataset(test_data, tokenizer, args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_val_f1 = 0
    best_model_state = None

    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Train
        train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_dataloader, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"Val FPR: {val_metrics['fpr']:.4f}, FNR: {val_metrics['fnr']:.4f}")

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args)
            }, output_dir / "best_model.pt")

            logger.info(f"New best model saved with Macro F1: {best_val_f1:.4f}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_dataloader, device)

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"  FPR: {test_metrics['fpr']:.4f}")
    logger.info(f"  FNR: {test_metrics['fnr']:.4f}")

    # Save results
    results = {
        "args": vars(args),
        "test_metrics": test_metrics,
        "best_val_f1": best_val_f1
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\nTraining completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
