"""
preprocess.py
Shared preprocessing utilities for all baseline models.

This module supports:
1. TF-IDF + Logistic Regression   → tokenized string output
2. Bi-LSTM                        → token list output
3. BERT fine-tuning               → minimal cleaning

"""

import re
from typing import List



#basic cleaning  
def clean_code(text: str) -> str:
    """
    Basic safe cleaning for code.
    - Normalize newline characters
    - Replace tabs with 4 spaces (common formatting)
    - Strip leading/trailing empty whitespace
    - Remove null or invisible control chars
    """
    if text is None:
        return ""

    #normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    #replace tabs with spaces
    text = text.replace("\t", "    ")

    #remove null bytes or weird unicode control chars
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    return text.strip()



#universal code tokenizer (for TF-IDF + LSTM)

#Pattern explanation:
#  - Identifiers: [A-Za-z_]\w+
#  - Numbers: \d+(\.\d+)?
#  - Multi-char operators: ==, !=, <=, >=, ->, ::
#  - Single-symbol operators/punctuation: {}()[].,;+-/*%=<>
#  - Catch any non-whitespace char as fallback: \S
TOKEN_PATTERN = re.compile(
    r"""
    [A-Za-z_]\w+                      |  #variable/function/keyword
    \d+(\.\d+)?                       |  #integer/float literals
    ==|!=|<=|>=|->|::                 |  #multi-character operators
    [{}\[\]();,]                      |  #braces, semicolon, commas
    [+\-*/%=<>]                       |  #mathematical operators
    \S                                   #catch-all for any token-ish character
    """,
    re.VERBOSE
)


def tokenize_code(text: str) -> List[str]:
    """
    Tokenizes code into a list of tokens.
    Used by TF-IDF and Bi-LSTM.
    """
    text = clean_code(text)
    tokens = TOKEN_PATTERN.findall(text)

    #TOKEN_PATTERN.findall returns tuples for number-matching
    #extract only the actual token string.
    final_tokens = []
    for tok in tokens:
        if isinstance(tok, tuple):
            final_tokens.append(tok[0])  #keep the first element, e.g., '3.14'
        else:
            final_tokens.append(tok)

    return final_tokens



#TF-IDF specific preprocessing
def tokenize_code_for_tfidf(text: str) -> str:
    """
    Returns a string of space-separated tokens.
    Used ONLY by the TF-IDF + Logistic Regression baseline.
    """
    tokens = tokenize_code(text)
    return " ".join(tokens)



#BERT cleaning
def clean_code_for_bert(text: str) -> str:
    """
    BERT should handle tokenization itself.
    We only apply minimal cleaning (same as clean_code).
    IMPORTANT: Do NOT tokenize here. Let BERT tokenizer do that.
    """
    return clean_code(text)
