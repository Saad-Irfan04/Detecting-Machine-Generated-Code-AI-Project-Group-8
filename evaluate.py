# evaluate.py
import argparse
import os
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_dataset
from preprocess import tokenize_code_for_tfidf, tokenize_code, clean_code_for_bert

from model_baseline_A import TFIDFLogRegBaseline
from model_baseline_B import (
    Vocab, CodeDataset,
    BiLSTMClassifier, collate_batch,
    load_checkpoint
)
from model_baseline_C import BaselineC
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------
# Utility: Convert all numpy types to native Python types
# ------------------------------------------------------
def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj

# ------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------
def evaluate_tfidf(model_dir, df):
    model = TFIDFLogRegBaseline.load(os.path.join(model_dir, "tfidf_model.pkl"))

    X = df["code"].apply(tokenize_code_for_tfidf)
    y = df["label"]

    preds = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "preds": list(preds),
        "labels": list(y)
    }
    return convert_to_builtin_types(metrics)


def evaluate_bilstm(model_dir, df):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = Vocab.load(os.path.join(model_dir, "vocab.json"))
    model = BiLSTMClassifier(vocab_size=len(vocab))
    model, vocab = load_checkpoint(
        model,
        vocab_path=os.path.join(model_dir, "vocab.json"),
        model_state_path=os.path.join(model_dir, "model.pt")
    )
    model.to(device)
    model.eval()

    ds = CodeDataset(df["code"], df["label"], vocab)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx=0)
    )

    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, lengths)
            p = logits.argmax(dim=1).cpu().tolist()

            preds.extend(p)
            gold.extend(labels.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(gold, preds),
        "f1_macro": f1_score(gold, preds, average="macro"),
        "preds": preds,
        "labels": gold
    }
    return convert_to_builtin_types(metrics)


def evaluate_bert(model_dir, df, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds, gold = [], []

    for text, label in zip(df["code"], df["label"]):
        enc = tokenizer(
            clean_code_for_bert(text),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            pred = logits.argmax(dim=1).item()

        preds.append(pred)
        gold.append(label)

    metrics = {
        "accuracy": accuracy_score(gold, preds),
        "f1_macro": f1_score(gold, preds, average="macro"),
        "preds": preds,
        "labels": gold
    }
    return convert_to_builtin_types(metrics)


# ------------------------------------------------------
# Confusion Matrix Plotting
# ------------------------------------------------------
def save_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--data", type=str, default="data/test_full.parquet")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    df = load_dataset(args.data)
    model_dir = os.path.join("results", f"baseline_{args.model}")

    if args.model == "A":
        metrics = evaluate_tfidf(model_dir, df)
    elif args.model == "B":
        metrics = evaluate_bilstm(model_dir, df)
    elif args.model == "C":
        metrics = evaluate_bert(model_dir, df, max_length=args.max_length)

    print(metrics)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(
        metrics["labels"],
        metrics["preds"],
        os.path.join(model_dir, "confusion_matrix.png")
    )


if __name__ == "__main__":
    main()
