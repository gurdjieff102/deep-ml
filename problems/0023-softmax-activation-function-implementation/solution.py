import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
  scores = torch.tensor(scores)
  scores = scores - scores.max()
  exp_score = scores.exp()
  return (exp_score / exp_score.sum()).tolist()
   
