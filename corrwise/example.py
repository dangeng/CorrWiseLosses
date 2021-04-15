import torch
import torch.nn as nn

from corr_wise import CorrWise

# Dummy images
pred = torch.randn(1,3,256,256)
target = torch.randn(1,3,256,256)

# Losses
base_loss = nn.L1Loss()
loss_fn = CorrWise(base_loss, 
                   backward_warp=True,
                   return_warped=False,
                   padding_mode='reflection',
                   scale_clip=.1)

# Calc loss
loss = loss_fn(pred, target)
