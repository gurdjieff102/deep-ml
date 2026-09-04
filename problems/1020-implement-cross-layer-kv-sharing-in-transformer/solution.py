import numpy as np

def cross_layer_kv_attention(x, layer_weights, n_kv_producing_layers):
    """
    Multi-layer causal self-attention with cross-layer KV sharing.
    """
    h = np.array(x, dtype=float)

    shared_K = None
    shared_V = None

    d_model = h.shape[-1]

    for i, weights in enumerate(layer_weights):

        # 1. Compute Q from current hidden state
        Q = h @ weights["Wq"]

        # 2. Compute or reuse K, V
        if i < n_kv_producing_layers:
            K = h @ weights["Wk"]
            V = h @ weights["Wv"]

            shared_K = K
            shared_V = V
        else:
            K = shared_K
            V = shared_V

        # 3. Scaled dot-product attention
        scores = Q @ K.T
        scores = scores / np.sqrt(d_model)

        # causal mask
        seq_len = h.shape[0]
        mask = np.triu(
            np.ones((seq_len, seq_len)),
            k=1
        )

        scores = scores + mask * (-1e9)

        # stable softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights = exp_scores / np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )

        attn = attn_weights @ V

        # 4. Residual connection
        h = h + attn

    return np.round(h, 4).tolist()