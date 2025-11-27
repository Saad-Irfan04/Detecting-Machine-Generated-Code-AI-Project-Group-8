# model_baseline_C.py
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertForSequenceClassification,
)

class BaselineC:
    """
    Wrapper for BERT fine-tuning baseline.
    Trainable using HuggingFace Trainer from train.py.
    """

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)

        # Basic BERT sequence classifier
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )

    def get_model(self):
        """Return the underlying HF model."""
        return self.model

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer