import torch 

import math
import time

import os

from model.components import TransformerLM, AdamW, compute_lm_loss


def get_lr(
    step: int,
    total_steps: int, 
    max_lr: float,
    min_lr: float, 
    warmup_steps: int,
) -> float:
    """
    Cosine decay schedule with linear warmup.
    Returns the learning rate for a given step.
    """

    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Phase 2: cosine decay form max_lr to min_lr 
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi*progress))

    return min_lr + (max_lr - min_lr) * cosine 

def clip_grad_norm(params, max_norm: float = 1.0) -> float:  # fix: was `> float`
    """
    Clips gradients so their global L2 norm <=max_norm. 
    Returns the norm  before clipping (useful for logging).
    """

    # Collect all gradients that exist
    grads = [p.grad for p in params if p.grad is not None]

    # Compute global norm across all parameter gradients
    total_norm = torch.sqrt(
        sum(g.pow(2).sum() for g in grads)  # fix: .sum() must be inside the generator
    ).item()

    # Scale factor: 1.0 if already within limit, < 1.0 if we need to shrink 
    clip_coef = max_norm / max(total_norm, max_norm)

    for g in grads:
        g.mul_(clip_coef)
    

    return total_norm 

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: AdamW, 
    step: int, 
    loss: float, 
    path: str, 
): 
    """Save full training state to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'loss': loss,
        'model_state': model.state_dict(),  # fix: missing comma
        'optimizer_state': {
            't': optimizer.t,
            'm': [m.clone() for m in optimizer.m],
            'v': [v.clone() for v in optimizer.v],
        }
    }, path)
    print(f" [ckpt] Saved -> {path}")
    

def load_checkpoint(
    model: torch.nn.Module, 
    optimizer: AdamW, 
    path: str, 
)-> int:
    """Load training state, Returns the step ro resume from"""
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state'])
    optimizer.t = ckpt['optimizer_state']['t']
    for i, (m, v) in enumerate(zip(
        ckpt['optimizer_state']['m'],
        ckpt['optimizer_state']['v']
    )):
        optimizer.m[i].copy_(m)
        optimizer.v[i].copy_(v)
    
    print(f"[ckpt] Resumed from step {ckpt['step']}")
    return ckpt['step']



def train(
    model: TransformerLM,  # fix: missing comma
    optimizer: AdamW,
    data: torch.Tensor,
    total_steps: int,
    batch_size: int,
    seq_len: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,  # fix: was missing, used inside function
    ckpt_every: int = 500,
    ckpt_dir: str = "checkpoints",  # fix: was missing, used inside function
    device: str = "cpu",
    start_step: int = 0,
):
    model.to(device)
    model.train()


    # track metrics 
    run_loss = 0.0
    step_time = 0.0 


    print(f"Training for {total_steps} steps | "  # fix: typo `toral_steps`
          f"batch={batch_size} | seq={seq_len} | device={device}\n")
    
    for step in range(start_step, total_steps):
        t0 = time.time()

        # Step 1: Learning rate update 
        lr = get_lr(step, total_steps, max_lr, min_lr, warmup_steps)
        optimizer.lr = lr # update lr in-place 

        # Step 2: Sample a random batch from data 
        # Each sample needs seq_len + 1 tokens (input + target shift).
        max_start = len(data) - seq_len - 1 
        starts = torch.randint(0, max_start, (batch_size,))
        batch = torch.stack([
            data[s: s + seq_len + 1] for s in starts
        ]).to(device)  # (batch_size, seq_len + 1)

        # Step 3: Forward pass + loss 
        optimizer.zero_grad()
        loss = compute_lm_loss(model, batch)

        # Step 4: Backward pass 
        loss.backward()

        # Step 5: Gadient clipping 
        grad_norm = clip_grad_norm(optimizer.params, max_norm=max_grad_norm)

        # Step 6: Optimizer step
        optimizer.step() 

        # Step 7: Logging 
        step_time = time.time() - t0 
        run_loss += loss.item() 

        tokens_per_sec = (batch_size * seq_len) / step_time 

        if step % 10 == 0:
            avg_loss = run_loss / (10 if step > 0 else 1)
            run_loss = 0.0
            print(
                f"step {step:5d} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.3f} | "
                f"{tokens_per_sec:.0f} tok/s"
            )
        

        # Step 8: Checkpointing 
        if step > 0 and step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, loss.item(),
                path=f"{ckpt_dir}/step_{step:06d}.pt"
            )
    
    print("\nTraining complete")
    return model


if __name__ == '__main__':
    import sys
    import os
    # Ensure project root is on the path when run directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from model.components import TransformerLM, AdamW
    from tokenizer.tokenizer import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tok = Tokenizer.load("tokenizer.json")
    vocab_size = len(tok.vocab)

    # Tokenize training corpus
    with open("tokenizer_train.txt", "r", encoding="utf-8") as f:
        text = f.read()
    token_ids = tok.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    print(f"Dataset: {len(data):,} tokens")

    # Model config
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        max_seq_len=512,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=3e-4)

    # Resume from latest checkpoint if one exists
    ckpt_dir = "checkpoints"
    start_step = 0
    if os.path.isdir(ckpt_dir):
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
        if ckpts:
            start_step = load_checkpoint(model, optimizer, os.path.join(ckpt_dir, ckpts[-1]))

    train(
        model=model,
        optimizer=optimizer,
        data=data,
        total_steps=5000,
        batch_size=16,
        seq_len=512,
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=200,
        max_grad_norm=1.0,
        ckpt_every=500,
        ckpt_dir=ckpt_dir,
        device=device,
        start_step=start_step,
    )
