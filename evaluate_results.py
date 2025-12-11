# evaluate_results.py
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_train_valid
from model_proposed import CodeBertClassifier, TOKENIZER

TEST_PARQUET = "data/test_full.parquet"  # if you have a test file; otherwise use validation

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
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}

def evaluate(model_path="results/codebert_proposed.pth", test_parquet=TEST_PARQUET, device="cpu"):
    # If test_parquet not present, load validation from load_train_valid
    if os.path.exists(test_parquet):
        df = pd.read_parquet(test_parquet)
    else:
        # fallback: use validation split
        _, valid_df = load_train_valid()
        df = valid_df

    ds = SimpleTorchDataset(df, tokenizer=TOKENIZER, max_length=256)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = CodeBertClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            input_ids = b["input_ids"].to(device)
            attention_mask = b["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(b["label"].numpy())

    print("Macro-F1:", f1_score(trues, preds, average="macro"))
    print(classification_report(trues, preds, digits=4))

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(5,4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human","Machine"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    out = "results/test_confusion_matrix.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved confusion matrix to", out)

if _name_ == "_main_":
   evaluate()
