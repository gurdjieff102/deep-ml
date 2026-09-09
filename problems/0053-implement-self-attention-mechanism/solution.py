import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    """
    
    # 1. Compute attention scores
    d_k = K.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # 2. Apply softmax row-wise
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # 3. Weighted sum of V
    output = np.dot(attention_weights, V)

    return output