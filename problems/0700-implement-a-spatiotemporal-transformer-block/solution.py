import numpy as np

def spatiotemporal_transformer_block(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Forward pass of a spatiotemporal transformer block.
    
    Args:
        X: Input tensor of shape (T, N, D).
        params: Dict with attention and FFN weight matrices (see problem description).
    
    Returns:
        Output tensor of shape (T, N, D).
    """

    T, N, D = X.shape
    eps = 1e-5

    def layer_norm(x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)

        return (x - mean) / np.sqrt(var + eps)

    def attention(x, Wq, Wk, Wv, Wo):
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv

        scores = Q @ K.transpose(0, 2, 1)
        scores = scores / np.sqrt(D)

        # stable softmax
        scores = scores - np.max(
            scores,
            axis=-1,
            keepdims=True
        )

        attn = np.exp(scores)
        attn = attn / np.sum(
            attn,
            axis=-1,
            keepdims=True
        )

        out = attn @ V

        return out @ Wo

    # -----------------------------------------
    # 1. Spatial Self-Attention
    # -----------------------------------------

    x_norm = layer_norm(X)

    spatial_out = attention(
        x_norm,
        params["Wqs"],
        params["Wks"],
        params["Wvs"],
        params["Wos"]
    )

    X = X + spatial_out

    # -----------------------------------------
    # 2. Temporal Self-Attention
    # -----------------------------------------

    x_norm = layer_norm(X)

    # (T, N, D) -> (N, T, D)
    x_temporal = x_norm.transpose(1, 0, 2)

    temporal_out = attention(
        x_temporal,
        params["Wqt"],
        params["Wkt"],
        params["Wvt"],
        params["Wot"]
    )

    # (N, T, D) -> (T, N, D)
    temporal_out = temporal_out.transpose(1, 0, 2)

    X = X + temporal_out

    # -----------------------------------------
    # 3. Feed-Forward Network
    # -----------------------------------------

    x_norm = layer_norm(X)

    hidden = x_norm @ params["W1"]
    hidden = np.maximum(0, hidden)   # ReLU

    ffn_out = hidden @ params["W2"]

    X = X + ffn_out

    return X