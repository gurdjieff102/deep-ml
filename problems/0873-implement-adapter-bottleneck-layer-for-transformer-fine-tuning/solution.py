import numpy as np

def adapter_forward(
    x: np.ndarray,
    W_down: np.ndarray,
    b_down: np.ndarray,
    W_up: np.ndarray,
    b_up: np.ndarray
) -> list:

    # 1. Down projection
    h = x @ W_down + b_down

    # 2. ReLU
    h = np.maximum(0, h)

    # 3. Up projection
    h = h @ W_up + b_up

    # 4. Residual connection
    output = x + h

    return output.tolist()