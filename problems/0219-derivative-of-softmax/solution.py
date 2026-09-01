import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	x = np.array(x, dtype=float)
	x = x - np.max(x, axis=-1, keepdims=True)
	x = np.exp(x)
	x = x / np.sum(x, axis=-1, keepdims=True)
	res = np.diag(x) - np.outer(x, x)
	return res