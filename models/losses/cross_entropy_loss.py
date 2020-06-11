import torch
import torch.nn as nn
from .utils import MyEncoder


class CrossEntropyLoss(object):
    def __init__(self, weight=None, ignore_index=-1):
        self.ignore_index = ignore_index
        self.weight = MyEncoder(weight)

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        device = logit.device
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean').to(device)

        loss = criterion(logit, target.long())

        return loss

if __name__ == "__main__":
    torch.manual_seed(1)
    loss = CrossEntropyLoss()
    a = torch.tensor([[[[1.]], [[1.5]]]])
    a2 = torch.tensor([[[[1.]]]])
    b = torch.tensor([[[[2.]]]])
    print(loss(a, b))
    pass
