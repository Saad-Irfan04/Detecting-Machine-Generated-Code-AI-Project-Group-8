# run_experiments.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

from data_loader import load_train_valid  
from model_proposed import CodeBertClassifier, TOKENIZER, tokenize_batch_for_dataset

# Settings (tweak as needed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PARQUET = "data/train_full.parquet"   # change if different
VALID_PARQUET = "data/valid_full.parquet"
MAX_LEN = 256
EPOCHS = 3
TRAIN_BATCH = 8
EVAL_BATCH = 16
LR = 3e-5
SEED = 42

# helper to convert pandas DF -> HuggingFace style minimal dataset for simple DataLoader
class SimpleTorchDataset(torch.utils.data.Dataset):
    def _init_(self, df, tokenizer, max_length=256):
        self.codes = df["code"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.codes)

    def _getitem_(self, idx):
        enc = self.tokenizer(self.codes[idx],
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        # squeeze tensors
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}

def prepare_loaders(train_path=TRAIN_PARQUET, valid_path=VALID_PARQUET):
    train_df, valid_df = load_train_valid(train_path, valid_path)
    # optionally shuffle sample if you want to use smaller subset - keep full for final runs
    train_ds = SimpleTorchDataset(train_df, tokenizer=TOKENIZER, max_length=MAX_LEN)
    val_ds = SimpleTorchDataset(valid_df, tokenizer=TOKENIZER, max_length=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def train():
    torch.manual_seed(SEED)
    train_loader, val_loader = prepare_loaders()
    model = CodeBertClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        # validation
        model.eval()
        val_preds, val_trues = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(labels.cpu().numpy())
        avg_val = val_loss / len(val_loader)
        f1 = f1_score(val_trues, val_preds, average="macro")
        history["val_loss"].append(avg_val)
        history["val_f1"].append(f1)
        print(f"Epoch {epoch} — train_loss: {avg_train:.4f} — val_loss: {avg_val:.4f} — val_macro_f1: {f1:.4f}")

    # Save model and history
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "codebert_proposed.pth"))
    # Save history table
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(OUTPUT_DIR, "history.csv"), index=False)

    # Plots: loss and val_f1
    epochs = list(range(1, EPOCHS+1))
    plt.figure(figsize=(5,3.5))
    plt.plot(epochs, history["train_loss"], marker="o", linewidth=2)
    plt.plot(epochs, history["val_loss"], marker="o", linewidth=2)
    plt.title("Train / Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5,3.5))
    plt.plot(epochs, history["val_f1"], marker="o", linewidth=2)
    plt.title("Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "val_macro_f1.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    # Final validation confusion matrix and report
    cm = confusion_matrix(val_trues, val_preds)
    plt.figure(figsize=(5,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human","Machine"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    report = classification_report(val_trues, val_preds, digits=4)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write(report)

    print("Training complete. Artifacts saved to:", OUTPUT_DIR)
    print("Final Validation Macro-F1:", history["val_f1"][-1])

if _name_ == "_main_":
    train()