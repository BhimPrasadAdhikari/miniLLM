"""
Train the BPE tokenizer on FineWeb-Edu data.

Usage (from the project root):
    python -m scripts.train_tokenizer
    # or
    python scripts/train_tokenizer.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a plain script
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import stream_fineweb_for_tokenizer
from tokenizer.tokenizer import Tokenizer

# 1. Collect training data
tokenizer_text = stream_fineweb_for_tokenizer(
    target_mb=200,
    save_path="tokenizer_train.txt",
)

# 2. Train BPE
tok = Tokenizer()
tok.train(
    text=tokenizer_text,
    num_merges=32_000,
    verbose=True,
)

# 3. Add special tokens 
tok.add_special_tokens(["<|endoftext|>"])

print(f"\nVocab size : {len(tok.vocab)}")
print(f"EOT id     : {tok.special_tokens['<|endoftext|>']}")

tok.save("tokenizer.json")

test = "The researchers found that neural networks can learn complex representations."
ids  = tok.encode(test)
back = tok.decode(ids)

passed = back == test
print(f"\nRound-trip check : {'PASS' if passed else 'FAIL'}")
print(f"  original : {test}")
print(f"  decoded  : {back}")
print(f"  tokens   : {len(ids)}  (raw bytes: {len(test.encode())})")
print(f"  compression: {len(test.encode()) / len(ids):.2f}x")
