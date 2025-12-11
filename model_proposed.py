# model_proposed.py
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from preprocess import clean_code_for_bert

MODEL_NAME = "microsoft/codebert-base"

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden_state * mask
        summed = masked.sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        return summed / counts

class CodeBertClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.3, num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.pool = MeanPooling()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.pool(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))

# shared tokenizer helper
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch_for_dataset(batch, max_length=256):
    # expects batch to be a dict-like with batch["code"]
    cleaned = [clean_code_for_bert(x) for x in batch["code"]]
    return TOKENIZER(cleaned,
                     padding="max_length",
                     truncation=True,
                     max_length=max_length,
                     return_tensors="pt")
