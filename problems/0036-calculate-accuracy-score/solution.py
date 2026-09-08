import numpy as np

def accuracy_score(y_true, y_pred):
	accurate = np.sum(y_true == y_pred)
	return accurate / len(y_true)
	# Your code here
	# pass