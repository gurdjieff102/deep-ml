import numpy as np
def compare_softmax(scores: list) -> dict:
    scores = np.array(scores)
    exp_score = np.exp(scores)
    navie = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
    if np.any(np.isnan(navie)):
        navie = np.full_like(navie, np.nan, dtype=float)

    exp_score = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    # exp_score = np.exp(exp_score)
    exp_score = exp_score / np.sum(exp_score)

    if np.any(np.isnan(navie)):
        max_diff = np.float32('nan')
    else:
        max_diff = np.max(np.abs(exp_score - navie))
    return {
        'naive': [round(float(val), 6) for val in navie],
        'stable': [round(float(val), 6) for val in exp_score],
        'max_abs_diff': round(float(max_diff), 6)

    }