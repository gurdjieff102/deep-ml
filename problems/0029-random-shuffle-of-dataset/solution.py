import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
	indices = np.random.permutation(len(X))
	return X[indices], y[indices]
	# pass