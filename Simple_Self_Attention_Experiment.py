#!/usr/bin/env python3
"""
Small, fully-plottable self-attention example using
torch.nn.functional.scaled_dot_product_attention().

- Uses tiny tensors (T=4 tokens, d=3 embedding dim)
- Single head (you can extend to multi-head later)
- Prints Q, K, V, attention weights, and output
- Includes a minimal plotting helper (matplotlib)

Later you can add a causal mask via `attn_mask=...` in sdpa().
"""

import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_matrix(M: torch.Tensor, title: str, xlabels=None, ylabels=None):
    """Simple heatmap for a 2D tensor."""
    M = M.detach().cpu()
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.title(title)
    plt.colorbar()
    if xlabels is not None:
        plt.xticks(range(len(xlabels)), xlabels)
    if ylabels is not None:
        plt.yticks(range(len(ylabels)), ylabels)
    plt.tight_layout()


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.manual_seed(0)

    # Tiny shapes
    B = 1   # batch
    T = 4   # sequence length (tokens / timesteps)
    d = 3   # embedding dim (per head)

    # Example "token embeddings"
    x = torch.tensor(
        [[[1.0, 0.0, 1.0],
          [0.0, 2.0, 0.0],
          [1.0, 1.0, 0.0],
          [0.0, 0.0, 2.0]]]
    )  # [B, T, d]

    # Simple linear projections to get Q, K, V (small, hand-readable)
    Wq = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    Wk = torch.tensor([[1.0, 0.5, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])
    Wv = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.5],
                       [0.0, 0.0, 1.0]])

    # Project: [B,T,d] @ [d,d] -> [B,T,d]
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    # scaled_dot_product_attention expects [B, H, T, d] (or [*, T, d] depending on usage),
    # so we add a head dimension H=1:
    Qh = Q.unsqueeze(1)  # [B,1,T,d]
    Kh = K.unsqueeze(1)  # [B,1,T,d]
    Vh = V.unsqueeze(1)  # [B,1,T,d]

    # Standard (non-causal) self-attention: no mask
    # (dropout_p=0.0 for deterministic output)
    Oh = F.scaled_dot_product_attention(Qh, Kh, Vh, attn_mask=None, dropout_p=0.0, is_causal=False)
    O = Oh.squeeze(1)  # [B,T,d]

    # To plot attention weights, compute them explicitly (same math as SDPA):
    # scores = QK^T / sqrt(d)
    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d)      # [B,T,T]
    attn = scores.softmax(dim=-1)                           # [B,T,T]
    O_manual = attn @ V                                     # [B,T,d]

    print("\n=== Input x ===\n", x[0])
    print("\n=== Q ===\n", Q[0])
    print("\n=== K ===\n", K[0])
    print("\n=== V ===\n", V[0])

    print("\n=== scores = QK^T / sqrt(d) ===\n", scores[0])
    print("\n=== attention weights = softmax(scores) ===\n", attn[0])

    print("\n=== Output from scaled_dot_product_attention ===\n", O[0])
    print("\n=== Output from manual attention (should match) ===\n", O_manual[0])
    print("\nMax |difference|:", (O - O_manual).abs().max().item())

    # Plotting
    token_labels = [f"t{i}" for i in range(T)]
    plot_matrix(scores[0], "Attention scores (before softmax)", xlabels=token_labels, ylabels=token_labels)
    plot_matrix(attn[0], "Attention weights (after softmax)", xlabels=token_labels, ylabels=token_labels)

    # Also visualize output features per token
    plot_matrix(O[0], "Attention output O (per token x feature)", xlabels=[f"d{j}" for j in range(d)], ylabels=token_labels)

    plt.show()


if __name__ == "__main__":
    main()
