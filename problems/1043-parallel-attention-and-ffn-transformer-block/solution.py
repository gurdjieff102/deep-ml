import numpy as np

def parallel_block(x, gamma, beta, Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    x = np.array(x, dtype=float)

    # LayerNorm
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + 1e-5)
    x_norm = gamma * x_norm + beta

    # Attention
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    scores = Q @ K.T
    scores = scores / np.sqrt(x.shape[-1])

    # softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores)
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    attn = attn_weights @ V
    attn = attn @ Wo

    # FFN
    ffn = x_norm @ W1 + b1
    ffn = np.maximum(ffn, 0)       # ReLU
    ffn = ffn @ W2 + b2

    # Parallel residual
    y = x + attn + ffn

    return y.tolist()