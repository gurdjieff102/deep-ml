import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	prob = []
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)

	z = features @ weights + bias
	prob = 1/(1+np.exp(-z))
	mse = np.mean((prob - labels) ** 2)
	return np.round(prob, 4).tolist(), round(mse, 4)