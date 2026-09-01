import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores = np.array(scores, dtype=float)
	max_v = np.max(scores, axis=-1, keepdims=True)
	res = scores - np.log(np.sum(np.exp(scores)))
	return res
	# Your code here
	# pass