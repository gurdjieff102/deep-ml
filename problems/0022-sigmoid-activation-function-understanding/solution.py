import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z = torch.tensor(z, dtype=torch.float32)
    return round((1 / (torch.exp(-z) + 1)).item(), 4)
