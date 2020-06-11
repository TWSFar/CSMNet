import torch
import numpy as np


def MyEncoder(obj):
    if isinstance(obj, list):
        return torch.tensor(obj, dtype=torch.float32)
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)
    elif isinstance(obj, torch.Tensor):
        return obj.float()
    else:
        return obj


if __name__=="__main__":
    a = MyEncoder([1, 2])
    b = MyEncoder(np.array([1, 2]))
    c = MyEncoder(torch.tensor([1, 2]))
    pass