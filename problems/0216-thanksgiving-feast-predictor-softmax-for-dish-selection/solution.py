import numpy as np
def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
	scores = np.array(preference_scores, dtype=float)
	scores = scores - np.max(scores, axis=-1, keepdims=True)
	exp_x = np.exp(scores)
	res = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
	return res
	