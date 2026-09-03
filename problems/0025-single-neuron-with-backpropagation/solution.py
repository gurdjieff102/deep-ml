import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = initial_weights.copy()
	bias = initial_bias
	mse_values = []
	for _ in range(epochs):
		z = features @ weights + bias
		pred = 1 / (1 + np.exp(-z))
		mse = np.mean((pred - labels) ** 2)
		mse_values.append(mse)

		error = pred - labels
		dz = 2 * error * pred * (1 - pred)
		dw = np.mean(features * dz[:, np.newaxis], axis=0)
		db = np.mean(dz)

		weights -= learning_rate * dw
		bias -= learning_rate * db
	
	updated_weights = np.round(weights, 4)
	updataed_bias = round(bias, 4)
	return updated_weights, updataed_bias, mse_values
