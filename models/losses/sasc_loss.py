"""Spatial Abastraction Loss
arXiv: https://arxiv.org/pdf/1903.00853.pdf (CVPR2019)
"""
import torch
import torch.nn as nn


class SALoss(object):
    def __init__(self, reduction="mean", levels=4):
        self.reduction = reduction
        self.levels = levels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, pred, target):
        loss = []
        if pred.shape != target.shape:
            target = target.reshape_as(pred)
        device = pred.device
        criterion = nn.MSELoss(reduction=self.reduction).to(device)
        loss.append(criterion(pred, target))
        for i in range(self.levels-1):
            pred = self.pool(pred)
            target = self.pool(target)
            loss.append(criterion(pred, target))

        return torch.stack(loss).sum()


class SCLoss(object):
    def __call__(self, pred, target):
        if pred.shape != target.shape:
            target = target.reshape_as(pred)
        pred_2 = pred.pow(2)
        target_2 = target.pow(2)
        sum_pXt = (pred * target).sum(dim=(1, 2, 3))
        mul_p2Xt2 = pred_2.sum(dim=(1, 2, 3)) * target_2.sum(dim=(1, 2, 3))
        loss = (1 - sum_pXt / (mul_p2Xt2).sqrt())

        return loss.mean()


class SASCLoss(object):
    def __init__(self, reduction="mean", levels=4):
        self.sal = SALoss(reduction, levels)
        self.scl = SCLoss()

    def __call__(self, pred, target):
        saloss = self.sal(pred, target)
        scloss = self.scl(pred, target)

        return saloss + scloss


if __name__ == "__main__":
    # torch.manual_seed(1)
    loss_c = SCLoss()
    loss_a = SALoss()
    loss_ac = SASCLoss()
    a = torch.rand(16, 1, 30, 40)  * 20
    a.requires_grad = True
    b = torch.rand(16, 30, 40) * 20
    loss_a(a, b).backward()
    # print(loss_a(a, b))
    loss_c(a, b).backward()
    loss_ac(a, b).backward()
    pass
