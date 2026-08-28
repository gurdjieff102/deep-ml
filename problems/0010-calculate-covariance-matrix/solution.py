import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    x_mean = torch.mean(v_t, dim=-1, keepdim=True)
    center = v_t - x_mean
    cov = center @ center.T / (v_t.shape[1] - 1)
    return cov
    # pass
