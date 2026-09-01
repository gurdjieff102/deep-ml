import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    q = np.array(q_values)
    q /= temperature
    q = q - np.max(q, axis=-1, keepdims=True)
    q = np.exp(q)
    q = q / np.sum(q)
    np.random.seed(seed)
    action = np.random.choice(len(q), p=q)
    return q.tolist(), int(action)