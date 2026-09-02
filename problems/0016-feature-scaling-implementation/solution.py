import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)

	standard = (data - mean) / std

	min_val = np.min(data, axis=0)
	max_val = np.max(data, axis=0)
	normalize = (data - min_val) / (max_val - min_val)

	# Your code here
	return standard, normalize