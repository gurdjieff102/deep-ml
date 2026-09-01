import torch

def softmax(t, dim):
    t = torch.tensor(t, dtype=torch.float32)
    t = t - torch.max(t, dim=dim, keepdim=True).values
    exp_t = torch.exp(t)
    res = exp_t / torch.sum(exp_t, dim=dim, keepdim=True)
    return res
    
    # TODO: subtract max along dim, exp, then normalize
    # pass
