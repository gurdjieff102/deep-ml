import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    if a.shape[-1] != b.shape[0]:
        return torch.tensor(-1)
    return torch.matmul(a, b)
    