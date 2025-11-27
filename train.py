import os
os.environ["WANDB_DISABLED"] = "true"
import argparse
import os
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_train_valid
from preprocess import (
    tokenize_code_for_tfidf,
    tokenize_code,
    clean_code_for_bert
)

# Baselines
from model_baseline_A import TFIDFLogRegBaseline
from model_baseline_B import (
    Vocab, CodeDataset,
    BiLSTMClassifier, collate_batch,
    save_checkpoint
)
from model_baseline_C import BaselineC

from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer


# ------------------------------------------------------
# BERT Dataset Wrapper
# ------------------------------------------------------
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tok(
            clean_code_for_bert(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        enc["labels"] = int(self.labels[idx])
        return enc


# ------------------------------------------------------
# TRAINING PIPELINE
# ------------------------------------------------------
def train_tfidf(train_df, valid_df, save_dir):
    print("\nTraining TF-IDF + Logistic Regression...")

    model = TFIDFLogRegBaseline()

    X_train = train_df["code"].apply(tokenize_code_for_tfidf)
    y_train = train_df["label"]

    X_valid = valid_df["code"].apply(tokenize_code_for_tfidf)
    y_valid = valid_df["label"]

    model.fit(X_train, y_train)

    metrics = model.evaluate(X_valid, y_valid)
    print("\nValidation metrics:", metrics)

    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "tfidf_model.pkl"))

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_bilstm(train_df, valid_df, save_dir, epochs, batch_size):
    print("\nTraining Bi-LSTM...")

    # Filter out empty sequences to avoid runtime errors
    train_df = train_df[train_df["code"].str.strip().astype(bool)]
    valid_df = valid_df[valid_df["code"].str.strip().astype(bool)]

    train_tokens = [tokenize_code(c) for c in train_df["code"]]
    valid_tokens = [tokenize_code(c) for c in valid_df["code"]]

    train_labels = train_df["label"].tolist()
    valid_labels = valid_df["label"].tolist()

    vocab = Vocab()
    vocab.add_tokens_from_list(train_tokens)
    vocab.build()

    train_ds = CodeDataset(train_df["code"], train_labels, vocab)
    valid_ds = CodeDataset(valid_df["code"], valid_labels, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_idx=0)
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx=0)
    )

    model = BiLSTMClassifier(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            ids, lengths, labels = batch["input_ids"].to(device), batch["lengths"].to(device), batch["labels"].to(device)

            # Skip sequences with length <= 0
            non_empty_idx = (lengths > 0).nonzero(as_tuple=True)[0]
            if len(non_empty_idx) == 0:
                continue

            ids = ids[non_empty_idx]
            lengths = lengths[non_empty_idx]
            labels = labels[non_empty_idx]

            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.3f}")

    # ---- VALIDATION ----
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in valid_loader:
            ids, lengths, labels = batch["input_ids"].to(device), batch["lengths"].to(device), batch["labels"].to(device)

            # Skip sequences with length <= 0
            non_empty_idx = (lengths > 0).nonzero(as_tuple=True)[0]
            if len(non_empty_idx) == 0:
                continue

            ids = ids[non_empty_idx]
            lengths = lengths[non_empty_idx]
            labels = labels[non_empty_idx]

            logits = model(ids, lengths)
            p = logits.argmax(dim=1).cpu().tolist()

            preds.extend(p)
            gold.extend(labels.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(gold, preds),
        "f1_macro": f1_score(gold, preds, average="macro")
    }
    print("\nValidation metrics:", metrics)

    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(model, vocab, save_dir)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_bert(train_df, valid_df, save_dir, epochs, batch_size, max_length):
    print("\nTraining BERT...")

    base = BaselineC()
    tokenizer = base.get_tokenizer()
    model = base.get_model()

    train_ds = BERTDataset(train_df["code"], train_df["label"], tokenizer, max_length)
    valid_ds = BERTDataset(valid_df["code"], valid_df["label"], tokenizer, max_length)

    args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    learning_rate=2e-5,
    logging_steps=50,
    # fallback for old versions:
    do_eval=False,
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\nValidation metrics:", metrics)

    trainer.save_model(save_dir)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    train_df, valid_df = load_train_valid()

    save_dir = os.path.join("results", f"baseline_{args.model}")
    os.makedirs(save_dir, exist_ok=True)

    if args.model == "A":
        train_tfidf(train_df, valid_df, save_dir)

    elif args.model == "B":
        train_bilstm(train_df, valid_df, save_dir, args.epochs, args.batch_size)

    elif args.model == "C":
        train_bert(train_df, valid_df, save_dir, args.epochs, args.batch_size, args.max_length)


if __name__ == "__main__":
    main()