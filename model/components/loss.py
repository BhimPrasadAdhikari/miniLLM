import torch 
import torch.nn.functional as F 
from model.components import TransformerLM

def cross_entropy_loss(
    logits: torch.Tensor, 
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean cross-entropy loss over all positions. 
    Internally uses log-sum-exp stabilization.
    """

    B, T, V = logits.shape 

    logits_flat = logits.reshape(B*T, V)
    targets_flat = targets.reshape(B*T)

    loss = F.cross_entropy(logits_flat, targets_flat)

    return loss 

def compute_lm_loss(model: TransformerLM, token_ids: torch.Tensor,) -> torch.Tensor:
    """
    Full pipeline: takes a batch of token sequences, 
    runs forward pass, computes next-token prediction loss.
    """

    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    logits = model(inputs)  # (batch, seq_len, vocab_size)

    loss = cross_entropy_loss(logits, targets)

    return loss
