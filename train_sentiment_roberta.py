#!/usr/bin/env python3
"""
Fine-tune RoBERTa for multi-class sentiment classification.

This script is based on FineTune.ipynb and covers:
- Load CSV data
- Train
- Evaluate (precision/recall/F1/ROC-AUC/accuracy)
- Save model + tokenizer for later inference
- Save metrics to a .txt file

Expected CSV columns (defaults):
- text_clean: preprocessed text
- label: integer class id (0..num_labels-1)

Example:
python train_sentiment_roberta.py \
  --train_csv train.csv \
  --test_csv test.csv \
  --output_dir roberta_sentiment_ckpt \
  --metrics_txt metrics.txt
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizerFast, set_seed

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)


@dataclass
class TrainConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"
    num_labels: int = 3
    max_len: int = 256
    train_batch_size: int = 64
    eval_batch_size: int = 32
    epochs: int = 3
    lr: float = 1e-5
    seed: int = 42
    text_col: str = "text_clean"
    label_col: str = "label"


class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: RobertaTokenizerFast, max_len: int, text_col: str, label_col: str):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.label_col = label_col

        if self.text_col not in self.df.columns:
            raise ValueError(f"Missing text column '{self.text_col}'. Available columns: {list(self.df.columns)}")
        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing label column '{self.label_col}'. Available columns: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.df.loc[idx, self.text_col])
        label = int(self.df.loc[idx, self.label_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # squeeze batch dimension
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            # token_type_ids exists for RoBERTa tokenizer, but it's all zeros; keep for compatibility
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


class RobertaClassifier(torch.nn.Module):
    """
    Matches the notebook's custom head:
    encoder: RobertaModel
    head: Linear(768->768)->ReLU->Dropout->Linear(768->num_labels)
    """
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs.last_hidden_state  # [bs, seq, hidden]
        cls = hidden_state[:, 0]                  # [bs, hidden]
        x = self.pre_classifier(cls)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    losses: List[float] = []
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = (correct / total) if total else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_labels: int) -> Dict[str, object]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    losses: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)
        losses.append(float(loss.item()))

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    y_proba = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_labels), dtype=float)

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0

    # precision/recall/f1: provide both macro and weighted (often useful for imbalanced labels)
    prf_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prf_weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # multi-class ROC-AUC (macro, one-vs-rest). May fail if only one class appears in y_true.
    roc_auc = None
    roc_auc_err = None
    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception as e:
        roc_auc_err = str(e)

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {
        "eval_loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(acc),
        "precision_macro": float(prf_macro[0]),
        "recall_macro": float(prf_macro[1]),
        "f1_macro": float(prf_macro[2]),
        "precision_weighted": float(prf_weighted[0]),
        "recall_weighted": float(prf_weighted[1]),
        "f1_weighted": float(prf_weighted[2]),
        "roc_auc_ovr_macro": None if roc_auc is None else float(roc_auc),
        "roc_auc_error": roc_auc_err,
        "classification_report": report,
    }


def save_artifacts(
    output_dir: str,
    model: torch.nn.Module,
    tokenizer: RobertaTokenizerFast,
    config: TrainConfig,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights + metadata for reproducible reload.
    torch.save(model.state_dict(), os.path.join(output_dir, "model_state_dict.pt"))

    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    # Save tokenizer in HF format for easy reload
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    # Save a small README on how to reload
    readme = f"""Reload example:

import json
import torch
from transformers import RobertaTokenizerFast
from train_roberta_sentiment import RobertaClassifier, get_device

ckpt_dir = r"{output_dir}"
device = get_device()

cfg = json.load(open(f"{output_dir}/train_config.json"))
tokenizer = RobertaTokenizerFast.from_pretrained(f"{output_dir}/tokenizer")
model = RobertaClassifier(cfg["model_name"], cfg["num_labels"])
model.load_state_dict(torch.load(f"{output_dir}/model_state_dict.pt", map_location=device))
model.to(device)
model.eval()
"""
    with open(os.path.join(output_dir, "README_reload.txt"), "w", encoding="utf-8") as f:
        f.write(readme)


def write_metrics(metrics_path: str, train_cfg: TrainConfig, train_hist: List[Dict[str, float]], eval_metrics: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("=== Train config ===\n")
    lines.append(json.dumps(asdict(train_cfg), indent=2))
    lines.append("\n\n=== Train history (per epoch) ===\n")
    for i, h in enumerate(train_hist, 1):
        lines.append(f"Epoch {i}: train_loss={h['train_loss']:.6f}, train_acc={h['train_acc']:.4f}")
    lines.append("\n\n=== Evaluation metrics ===\n")

    # user-requested metrics (also include macro/weighted versions)
    lines.append(f"Accuracy: {eval_metrics['accuracy']:.6f}")
    lines.append(f"Precision (macro): {eval_metrics['precision_macro']:.6f}")
    lines.append(f"Recall (macro): {eval_metrics['recall_macro']:.6f}")
    lines.append(f"F1-score (macro): {eval_metrics['f1_macro']:.6f}")
    if eval_metrics.get("roc_auc_ovr_macro") is not None:
        lines.append(f"ROC-AUC (OvR, macro): {eval_metrics['roc_auc_ovr_macro']:.6f}")
    else:
        lines.append(f"ROC-AUC (OvR, macro): N/A (error: {eval_metrics.get('roc_auc_error')})")

    lines.append("\n(Additional for imbalance) Precision/Recall/F1 (weighted):")
    lines.append(f"Precision (weighted): {eval_metrics['precision_weighted']:.6f}")
    lines.append(f"Recall (weighted): {eval_metrics['recall_weighted']:.6f}")
    lines.append(f"F1-score (weighted): {eval_metrics['f1_weighted']:.6f}")

    lines.append("\n\n=== Classification report ===\n")
    lines.append(eval_metrics["classification_report"])

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True, help="Path to train.csv")
    p.add_argument("--test_csv", required=True, help="Path to test.csv (evaluation)")
    p.add_argument("--output_dir", default="roberta_sentiment_ckpt", help="Directory to save model/tokenizer/config")
    p.add_argument("--metrics_txt", default="metrics.txt", help="Path to save metrics .txt")

    p.add_argument("--model_name", default="cardiffnlp/twitter-roberta-base-sentiment")
    p.add_argument("--num_labels", type=int, default=3)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--text_col", default="text_clean")
    p.add_argument("--label_col", default="label")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        max_len=args.max_len,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        text_col=args.text_col,
        label_col=args.label_col,
    )

    set_seed(cfg.seed)
    device = get_device()

    # Load data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    # Basic sanity checks
    for name, df in [("train", train_df), ("test", test_df)]:
        if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
            raise ValueError(
                f"{name}.csv must have columns '{cfg.text_col}' and '{cfg.label_col}'. "
                f"Got: {list(df.columns)}"
            )

    # Build tokenizer + datasets
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.model_name)

    train_ds = SentimentDataset(train_df, tokenizer, cfg.max_len, cfg.text_col, cfg.label_col)
    test_ds = SentimentDataset(test_df, tokenizer, cfg.max_len, cfg.text_col, cfg.label_col)

    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=0)

    # Model
    model = RobertaClassifier(cfg.model_name, cfg.num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Train
    train_hist: List[Dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        train_hist.append({"train_loss": train_loss, "train_acc": train_acc})
        print(f"[Epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.6f} train_acc={train_acc:.4f}")

    # Evaluate on test
    eval_metrics = eval_model(model, test_loader, device, cfg.num_labels)
    print("\n=== TEST METRICS ===")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {eval_metrics['precision_macro']:.4f} | Recall (macro): {eval_metrics['recall_macro']:.4f} | F1 (macro): {eval_metrics['f1_macro']:.4f}")
    if eval_metrics.get("roc_auc_ovr_macro") is not None:
        print(f"ROC-AUC (OvR, macro): {eval_metrics['roc_auc_ovr_macro']:.4f}")
    else:
        print(f"ROC-AUC: N/A ({eval_metrics.get('roc_auc_error')})")

    # Save artifacts + metrics
    save_artifacts(args.output_dir, model, tokenizer, cfg)
    write_metrics(args.metrics_txt, cfg, train_hist, eval_metrics)

    print(f"\nSaved model/tokenizer/config to: {args.output_dir}")
    print(f"Saved metrics to: {args.metrics_txt}")


if __name__ == "__main__":
    main()
