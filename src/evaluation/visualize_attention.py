import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.retrievers.models.encoder import TransformerEncoder, TOKENIZER

@torch.no_grad()
def plot_attention_maps(
    encoder: TransformerEncoder,
    text: str,
    save_path: str = "attention_maps",
    layers_idx: list[int] = [0],
    max_tokens: int = 32,
):
    """
    Run a forward pass on `text` through `encoder` and plot the multi-head self-attention
    heatmaps from `encoder.layers[layer_idx].multihead.attn`.
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    tokens = TOKENIZER(
        [text],
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    _ = encoder(input_ids, attention_mask)

    # attn shape: (batch_size=1, n_heads, seq_len, seq_len)
    attn_layers = [np.mean(encoder.layers[l].multihead.attn[0].detach().cpu().numpy(), axis = 0) for l in layers_idx] #mean across all the heads
    token_labels = TOKENIZER.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    n_plots = len(layers_idx)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), squeeze=False)
    
    for l in range(n_plots):
        layer_number = layers_idx[l] % len(encoder.layers)
        ax = axes[0, l]
        im = ax.imshow(attn_layers[l], cmap="viridis", aspect="auto")
        
        ax.set_title(f"Layer {layer_number}")
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(token_labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(f"outputs/plots/{save_path}_avghead_layers{layers_idx}_q{text[0]}.png", dpi=150)
    plt.close(fig)
    return save_path
