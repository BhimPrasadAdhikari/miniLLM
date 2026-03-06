# miniLLM

A small transformer language model trained from scratch in PyTorch.

## Architecture

- Transformer decoder (GPT-style)
- RoPE positional embeddings
- RMSNorm instead of LayerNorm
- SwiGLU feed-forward network
- Custom BPE tokenizer
- Custom AdamW optimizer

Default config: 6 layers · 512 d_model · 8 heads · 36M parameters

## Project Structure

```
model/
  components/
    transformerlm.py   # TransformerLM, TransformerBlock
    attention.py       # MultiHeadAttention
    rope.py            # RoPE frequency precomputation
    rms_norm.py        # RMSNorm
    swi_glu.py         # SwiGLU FFN
    AdamW.py           # Custom AdamW optimizer
    loss.py            # Cross-entropy loss
tokenizer/
  tokenizer.py         # BPE tokenizer (train / encode / decode / save / load)
data/
  dataset.py           # FineWeb-Edu streaming downloader
scripts/
  train_tokenizer.py   # Train and save the BPE tokenizer
  model_training.py    # Main training loop
  eval_tokenizer.py    # Tokenizer evaluation
```

## Training on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BhimPrasadAdhikari/miniLLM/blob/main/colab_train.ipynb)

```python
# In a Colab cell:
!git clone https://github.com/BhimPrasadAdhikari/miniLLM.git
%cd miniLLM
!pip install -r requirements.txt
!python -m scripts.model_training
```

## Requirements

```
pip install -r requirements.txt
```
