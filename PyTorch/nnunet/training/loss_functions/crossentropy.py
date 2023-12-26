from torch import nn, Tensor
import torch


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        # return super().forward(input, target.long())
        loss_func = nn.CrossEntropyLoss()
        cross_loss = loss_func(input, target.long())
        return cross_loss