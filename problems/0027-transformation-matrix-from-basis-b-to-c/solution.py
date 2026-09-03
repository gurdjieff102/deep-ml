import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	b = np.array(B, dtype=float)
	c = np.array(C, dtype=float)
	p = np.linalg.inv(c) @ b 
	return p.tolist()