import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.
    """
    
    # =========================================================
    # 1. LayerNorm 1
    # =========================================================
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    h = (x - mean) / np.sqrt(var + 1e-5)

    h = h * params["ln1_gamma"] + params["ln1_beta"]

    # =========================================================
    # 2. Multi-Head Causal Self-Attention
    # =========================================================
    
    W_q = params["W_q"]
    W_k = params["W_k"]
    W_v = params["W_v"]
    W_o = params["W_o"]

    # Q, K, V
    Q = h @ W_q
    K = h @ W_k
    V = h @ W_v

    batch, seq_len, emb_dim = x.shape
    d_head = emb_dim // num_heads

    # (B, S, E) -> (B, num_heads, S, d_head)
    Q = Q.reshape(batch, seq_len, num_heads, d_head)
    K = K.reshape(batch, seq_len, num_heads, d_head)
    V = V.reshape(batch, seq_len, num_heads, d_head)

    Q = Q.transpose(0, 2, 1, 3)
    K = K.transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)

    # Attention scores
    # (B, H, S, D) @ (B, H, D, S)
    # -> (B, H, S, S)
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_head)

    # =========================================================
    # Causal mask
    # =========================================================
    
    mask = np.triu(
        np.ones((seq_len, seq_len), dtype=bool),
        k=1
    )

    scores = np.where(mask, -np.inf, scores)

    # =========================================================
    # Softmax
    # =========================================================
    
    scores_max = np.max(scores, axis=-1, keepdims=True)

    exp_scores = np.exp(scores - scores_max)

    attention = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True
    )

    # =========================================================
    # Attention @ V
    # =========================================================
    
    # (B, H, S, S) @ (B, H, S, D)
    # -> (B, H, S, D)
    h = attention @ V

    # Back to (B, S, E)
    h = h.transpose(0, 2, 1, 3)
    h = h.reshape(batch, seq_len, emb_dim)

    # Output projection
    h = h @ W_o

    # =========================================================
    # Residual 1
    # =========================================================
    
    x = x + h

    # =========================================================
    # 3. LayerNorm 2
    # =========================================================
    
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    h = (x - mean) / np.sqrt(var + 1e-5)

    h = h * params["ln2_gamma"] + params["ln2_beta"]

    # =========================================================
    # 4. FFN
    # =========================================================
    
    h = h @ params["W_ff1"] + params["b_ff1"]

    # GELU
    h = 0.5 * h * (
        1 + np.tanh(
            np.sqrt(2 / np.pi) *
            (h + 0.044715 * h**3)
        )
    )

    h = h @ params["W_ff2"] + params["b_ff2"]

    # =========================================================
    # Residual 2
    # =========================================================
    
    x = x + h

    return x