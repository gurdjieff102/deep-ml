import numpy as np

def transformer_encoder_layer(
    X: np.ndarray,
    weights: dict,
    num_heads: int,
    eps: float = 1e-5
) -> np.ndarray:

    batch_size, seq_len, d_model = X.shape

    assert d_model % num_heads == 0

    # =========================================
    # 1. Multi-Head Self-Attention
    # =========================================

    # Project X -> Q, K, V
    Q = X @ weights["W_q"]   # (B, S, D)
    K = X @ weights["W_k"]   # (B, S, D)
    V = X @ weights["W_v"]   # (B, S, D)

    # Split into heads
    # (B, S, D) -> (B, H, S, head_dim)
    head_dim = d_model // num_heads

    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim)

    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    #
    # Q: (B, H, S, head_dim)
    # K: (B, H, S, head_dim)
    #
    # K.transpose: (B, H, head_dim, S)
    #
    # scores: (B, H, S, S)

    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(head_dim)

    # Stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)

    attention = np.exp(scores)
    attention = attention / np.sum(
        attention,
        axis=-1,
        keepdims=True
    )

    # Attention @ V
    # (B, H, S, S) @ (B, H, S, head_dim)
    # -> (B, H, S, head_dim)

    context = attention @ V

    # =========================================
    # 2. Concatenate heads
    # =========================================

    # (B, H, S, head_dim)
    # -> (B, S, H, head_dim)
    context = context.transpose(0, 2, 1, 3)

    # -> (B, S, D)
    context = context.reshape(batch_size, seq_len, d_model)

    # Output projection
    attention_output = context @ weights["W_o"]

    # =========================================
    # 3. Add & Norm
    # =========================================

    residual1 = X + attention_output

    mean1 = np.mean(residual1, axis=-1, keepdims=True)
    var1 = np.mean(
        (residual1 - mean1) ** 2,
        axis=-1,
        keepdims=True
    )

    norm1 = (residual1 - mean1) / np.sqrt(var1 + eps)

    norm1 = (
        norm1 * weights["gamma1"]
        + weights["beta1"]
    )

    # =========================================
    # 4. Feed-Forward Network
    # =========================================

    hidden = norm1 @ weights["W1"] + weights["b1"]

    # ReLU
    hidden = np.maximum(0, hidden)

    ffn_output = hidden @ weights["W2"] + weights["b2"]

    # =========================================
    # 5. Add & Norm
    # =========================================

    residual2 = norm1 + ffn_output

    mean2 = np.mean(residual2, axis=-1, keepdims=True)
    var2 = np.mean(
        (residual2 - mean2) ** 2,
        axis=-1,
        keepdims=True
    )

    norm2 = (residual2 - mean2) / np.sqrt(var2 + eps)

    output = (
        norm2 * weights["gamma2"]
        + weights["beta2"]
    )

    return output