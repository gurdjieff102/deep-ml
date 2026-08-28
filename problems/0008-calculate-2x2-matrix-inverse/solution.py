import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    if det == 0:
        return None

    m = torch.as_tensor(matrix, dtype=torch.float)
    return torch.linalg.inv(matrix)
    # Your code here
    # passx