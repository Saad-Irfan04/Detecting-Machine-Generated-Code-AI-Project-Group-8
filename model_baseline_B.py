import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any
import os
from preprocess import tokenize_code

#Vocab utilities
class Vocab:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq: int = 2, max_size: int = None):
        self.token2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.counter = {}
        self.min_freq = min_freq
        self.max_size = max_size

    def add_tokens_from_list(self, token_lists: List[List[str]]):
        for tokens in token_lists:
            for t in tokens:
                self.counter[t] = self.counter.get(t, 0) + 1

    def build(self):
        items = sorted(self.counter.items(), key=lambda x: (-x[1], x[0]))
        idx = len(self.token2idx)
        for token, freq in items:
            if freq < self.min_freq:
                continue
            if self.max_size is not None and idx >= self.max_size:
                break
            if token in self.token2idx:
                continue
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            idx += 1

    def __len__(self):
        return len(self.token2idx)

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.UNK_TOKEN])

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(t) for t in tokens]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
           token2idx = json.load(f)
        vocab = cls()
        vocab.token2idx = token2idx
        # Make sure idx2token keys are integers
        vocab.idx2token = {int(idx): token for token, idx in token2idx.items()}
        return vocab


#Dataset
class CodeDataset(Dataset):
    def __init__(self, codes: List[str], labels: List[int], vocab: Vocab = None, build_vocab: bool = False):
        """
        codes: list of raw code strings
        labels: list of int labels
        if build_vocab=True, expects vocab=None and will construct one (caller must pass tokenized lists separately)
        """
        self.raw_codes = codes
        self.labels = labels
        self.vocab = vocab

        self.token_lists = [tokenize_code(c) for c in self.raw_codes]

    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        if self.vocab is None:
            raise AssertionError("Vocab must be set before encoding dataset")
        token_ids = self.vocab.encode(tokens)
        return {"ids": token_ids, "length": len(token_ids), "label": int(self.labels[idx])}

#Collate function(padding)
def collate_batch(batch: List[Dict[str, Any]], pad_idx: int = 0, max_len: int = None) -> Dict[str, torch.Tensor]:
    lengths = [item["length"] for item in batch]
    if max_len is None:
        max_len = max(lengths)
    else:
        max_len = min(max_len, max(lengths))

    batch_size = len(batch)
    ids_padded = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.long)
    lengths_trunc = []

    for i, item in enumerate(batch):
        seq = item["ids"][:max_len]
        ids_padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        lengths_trunc.append(len(seq))
        labels[i] = item["label"]

    lengths_tensor = torch.tensor(lengths_trunc, dtype=torch.long)
    return {"input_ids": ids_padded, "lengths": lengths_tensor, "labels": labels}

#Bi-LSTM model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 256,
                 num_layers: int = 1, dropout: float = 0.3, num_classes: int = 2, pad_idx: int = 0, bidirectional: bool = True):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
      
        emb = self.emb(input_ids)  
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed)
        
        if self.num_directions == 2:
            last_forward = hn[-2, :, :]
            last_backward = hn[-1, :, :]
            h = torch.cat([last_forward, last_backward], dim=1)
        else:
            h = hn[-1, :, :]
        h = self.dropout(h)
        logits = self.fc(h)
        return logits

#Training/Save/Load helpers
def save_checkpoint(model: nn.Module, vocab: Vocab, path: str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    vocab.save(os.path.join(path, "vocab.json"))

def load_checkpoint(model: nn.Module, vocab_path: str, model_state_path: str):
    vocab = Vocab.load(vocab_path)
    model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
    return model, vocab