import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, preds, targets):
        return 
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class Loss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy_with_logits(preds, targets)
        
        preds = F.sigmoid(preds)
        dice = dice_loss(preds, targets)
        
        loss = bce * self.weights[0] + dice * self.weights[1]
        return loss