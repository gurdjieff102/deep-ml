import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    if T_t.shape[0] != A_t.shape[0] or A_t.shape[1] != S_t.shape[0]:
        return torch.tensor(-1)

    if T_t.shape[0] != T_t.shape[1] or S_t.shape[0] != S_t.shape[1]:
        return torch.tensor(-1)
    if torch.linalg.det(T_t) == 0 or torch.linalg.det(S_t) == 0:
        return torch.tensor(-1)

    return torch.linalg.inv(T_t).T @ A_t @ S_t
    

