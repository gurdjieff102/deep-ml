import numpy as np

def softmax_policy_gradient_step(
    theta: np.ndarray,
    states: list,
    actions: list,
    weights: list,
    alpha: float
) -> tuple:

    theta = theta.copy()
    gradients = np.zeros_like(theta, dtype=float)
    action_probs_list = []

    for state, action, weight in zip(states, actions, weights):
        logits = theta[state]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        action_probs_list.append(probs.copy())

        gradient = -probs
        gradient[action] += 1
        gradients[state] += weight * gradient

    gradients /= len(states)
    theta += alpha * gradients

    return theta, action_probs_list