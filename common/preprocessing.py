import torch

# Transform original FDF [0, 1] by log or BCT to obtain better prediction
def log_transform(y, eps=1e-6):
    return torch.log(y + eps)

def bct_transform(y, lamb=0.1):
    return (torch.pow(y, lamb) - 1.0) / lamb

def inverse_bct(x, lamb=0.1):
    return torch.pow((lamb * x + 1.0) / lamb, 1.0 / lamb)

def normalize_nonnegative(x, dim=(1,2,3), eps=1e-12):
    x = x.clamp(min=0.0)
    s = x.sum(dim=dim, keepdim=True) + eps
    return x / s
