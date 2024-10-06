import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns

def get_cache_attention_weights(module, input, output, cache_attention_weights):
    attn_output, attn_output_weights = output
    cache_attention_weights.append(attn_output_weights.detach())


def get_prefetch_attention_weights(module, input, output, prefetch_attention_weights):
    attn_output, attn_output_weights = output
    prefetch_attention_weights.append(attn_output_weights.detach())


def visualize_joint_attention(
    model, cache_pc, prefetch_pc, prefetch_page, prefetch_offset
):
    cache_attention_weights = []
    prefetch_attention_weights = []

    for layer in model.transformer_encoder.cache_transformer_encoder.layers:
        layer.self_attn.register_forward_hook(get_cache_attention_weights)

    for layer in model.transformer_encoder.prefetch_transformer_encoder.layers:
        layer.self_attn.register_forward_hook(get_prefetch_attention_weights)

    # Run forward pass
    output = model(cache_pc, prefetch_pc, prefetch_page, prefetch_offset)

    # Visualize Cache Attention Weights
    attn_weights_cache = cache_attention_weights[
        0
    ]  # Shape: (batch_size * num_heads, seq_len, seq_len)
    attn_weights_cache_mean = attn_weights_cache.mean(
        dim=0
    )  # Shape: (seq_len, seq_len)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights_cache_mean.numpy(), cmap="viridis")
    plt.title("Cache Attention Weights (Averaged over Heads)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig('results/cache_attention_weights.png')

    # Visualize Prefetch Attention Weights
    attn_weights_prefetch = prefetch_attention_weights[
        0
    ]  # Shape: (batch_size * num_heads, seq_len, seq_len)
    attn_weights_prefetch_mean = attn_weights_prefetch.mean(
        dim=0
    )  # Shape: (seq_len, seq_len)

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights_prefetch_mean.numpy(), cmap="viridis")
    plt.title("Prefetch Attention Weights (Averaged over Heads)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.savefig('results/prefetch_attention_weights.png')
