import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_utils import warp, normalize_flow

import numpy as np
import random
from torchvision.transforms import RandomResizedCrop
from torchvision.utils import save_image

import pdb

def sobel(flow):
    k_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).float().repeat(flow.shape[1], 1, 1, 1).to(flow.device)
    k_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).float().repeat(flow.shape[1], 1, 1, 1).to(flow.device)
    g_x = F.conv2d(flow, k_x, groups=flow.shape[1], padding=1)
    g_y = F.conv2d(flow, k_y, groups=flow.shape[1], padding=1)
    return g_x, g_y

def robust_l1(x):
    return (x ** 2 + 0.001 ** 2) ** 0.5

def smoothness(flow, image, edge_constant=150.0, second_order=False):
    img_gx, img_gy = sobel(image)
    weights_x = torch.exp(-torch.mean(torch.abs(edge_constant * img_gx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(edge_constant * img_gy), dim=1, keepdim=True))

    # Compute second derivatives of the predicted smoothness.
    flow_gx, flow_gy = sobel(flow)
    if second_order:
        flow_gx, _ = sobel(flow_gx)
        _, flow_gy = sobel(flow_gy)

    # Compute weighted smoothness
    return (torch.mean(weights_x * robust_l1(flow_gx)) + torch.mean(weights_y * robust_l1(flow_gy))) / 2.0


class CorrWise(nn.Module):
    def __init__(self, 
                 base_loss, 
                 epsilon=0.1,
                 flow_method='RAFT',
                 return_info=True,
                 detach=True,
                 padding_mode='reflection',
                 reg_flow_mag=False,
                 reg_flow_grad_mag=False,
                 flow_reg_norm=1,
                 edge_aware=0,
                 edge_penalty=0.0,
                 device=None):
        '''
        base_loss [func] : 
            takes in two torch images, and calculates a distance

        epsilon [float] :
            Scale the flow by (1-\epsilon). \epsilon = 1 reduces to the base loss

        flow_method [nn.Module] OR [string] : 
            if [nn.Module] : takes in two images normalized to 
            [0,1], returns flow with units of pixels

            if [string] : One of ['RAFT', 'RAFT-KITTI']

        return_info [bool] : 
            if true, return info dict in forward

        detach [bool] :
            if true, then detach the flow prediction so no 
            gradients are calculated

        padding_mode [string] :
            one of ['zeros', 'border', 'reflection']. 
            Determines how warping is padded. 
            See:
            https://pytorch.org/docs/stable/nn.functional.html#grid-sample

        reg_flow_mag [float] :
            if non-zero, then average magnitude of flow for regularization
            with `reg_flow_mag` as scale factor

        reg_flow_grad_mag [float] :
            if non-zero, then average magnitude of gradient of flow 
            for regularization with `reg_flow_mag` as scale factor

        flow_reg_norm [int] :
            int that specifies the norm used for flow 
            and flow gradient regularization
            
        edge_penalty [float] :
            constant to multiply the out of bounds flow loss by

        device [int] :
            device to use, if None, then don't move tensors around
        '''

        super(CorrWise, self).__init__()

        if flow_method == 'RAFT':
            from flow_utils import RAFT
            self.flow_method = RAFT()
        elif flow_method == 'RAFT-KITTI':
            from flow_utils import RAFT
            self.flow_method = RAFT(model='kitti')

        self.base_loss = base_loss
        self.epsilon = epsilon
        self.return_info = return_info
        self.detach = detach
        self.padding_mode = padding_mode
        self.reg_flow_mag = reg_flow_mag
        self.flow_reg_norm = flow_reg_norm
        self.reg_flow_grad_mag = reg_flow_grad_mag
        self.device = device
        self.edge_aware = edge_aware
        self.edge_penalty = edge_penalty
        
        if self.reg_flow_mag or self.reg_flow_grad_mag:
            assert not self.detach, 'Must set `no_detach` to true to regularize flow magnitude!'

    def calc_reg_losses(self, flow, im2):
        '''
        Given flow and the target image, calculate the regularization losses
            Returns a dict of the reg losses
        '''
        # Scale flow
        _, _, h, w = flow.shape
        size = torch.tensor([w, h]).float().to(flow.device)
        scaled_flow = flow / (-1 + size)[:, None, None]

        info = {}
        # Calculate flow magnitude
        if self.reg_flow_mag:
            mag = torch.linalg.norm(scaled_flow, ord=self.flow_reg_norm, dim=1).mean()
            info['flow_mag'] = mag

        # Calculate flow grad magnitude
        if self.reg_flow_grad_mag:
            flow_grad_x, flow_grad_y = sobel(scaled_flow)
            if self.edge_aware > 0:
                info['flow_grad_mag'] = smoothness(scaled_flow, im2, second_order=(self.edge_aware==2))
            else:
                mag_x = torch.linalg.norm(flow_grad_x, ord=self.flow_reg_norm, dim=1).mean()
                mag_y = torch.linalg.norm(flow_grad_y, ord=self.flow_reg_norm, dim=1).mean()
                info['flow_grad_mag'] = (mag_x + mag_y) / 2.

            info['flow_grad_x'] = flow_grad_x
            info['flow_grad_y'] = flow_grad_y

        # Calculate edge penalty
        if self.edge_penalty > 0:
            edge_penalty_loss = torch.clamp(torch.abs(flow) - 1, min=0).mean()
            info['edge_penalty'] = edge_penalty_loss

        return info

    def calculate_flow(self, im1, im2):
        '''
        Calculates flow between im1 and im2 (relative pixel-scale)
        '''
        # Calculate flow
        if self.detach:
            with torch.no_grad():
                flow = self.flow_method(im2, im1)
                flow = flow.detach()    # Prob not necessary
        else:
            flow = self.flow_method(im2, im1)

        return flow

    def align_and_compare(self, im1, im2, label=None):
        '''
        Aligns im1 and im2, and then calculates a loss
            Also does regularization on the flow (flow scaling)
            and flow reg losses (flow mag, flow grad mag)

        Returns
            base loss, dict containing useful info
        '''
        info = {}

        # Get flow im1 -> im2
        flow = self.calculate_flow(im1, im2)
        info['flow'] = flow

        # Calculate regularization and add to info
        reg = self.calc_reg_losses(flow, im2)
        info = {**info, **reg}

        # Scale flow
        flow = flow * (1 - self.epsilon)
        
        # Normalize flow to [-1, 1], absolute flow
        flow = normalize_flow(flow)
        
        # Warp im1 to im2
        warped = warp(im1, flow, padding_mode=self.padding_mode)
        info['warped'] = warped

        # Add label to info keys
        relabeled_info = {}
        for k,v in info.items():
            relabeled_info[f'{label}_{k}'] = v

        # Calculate loss
        return self.base_loss(warped, im2), relabeled_info

    def forward(self, pred, target):
        b,c,h,w = pred.shape

        # If specified, do calculations on this device
        if self.device:
            old_device = pred.device
            self.flow_method = self.flow_method.to(self.device)
            
            pred = pred.to(self.device)
            target = target.to(self.device)

        # Align and compare in both directions
        forward_loss, forward_info = self.align_and_compare(pred, target, label='forward')
        backward_loss, backward_info = self.align_and_compare(target, pred, label='backward')

        # merge info dicts
        info = {**forward_info, **backward_info}

        # Calculate regularization
        zero = torch.tensor(0, device=target.device)

        flow_mag_reg = .5 * (info.get('forward_flow_mag', zero) + \
                             info.get('backward_flow_mag', zero))
        flow_mag_reg = self.reg_flow_mag * flow_mag_reg

        flow_grad_mag_reg = .5 * (info.get('forward_flow_grad_mag', zero) + \
                                  info.get('backward_flow_grad_mag', zero))
        flow_grad_mag_reg = self.reg_flow_grad_mag * flow_grad_mag_reg

        edge_penalty = .5 * (info.get('forward_edge_penalty', zero) + \
                             info.get('backward_edge_penalty', zero))
        edge_penalty = self.edge_penalty * edge_penalty 

        # Calculate full loss
        base_loss = forward_loss + backward_loss
        loss = base_loss + flow_mag_reg + flow_grad_mag_reg + edge_penalty

        # Add losses to info
        info['base_loss'] = forward_loss + backward_loss
        info['flow_mag_reg'] = flow_mag_reg
        info['flow_grad_mag_reg'] = flow_grad_mag_reg
        info['edge_penalty'] = edge_penalty

        # If specified, move tensors back to original gpu
        if self.device:
            loss = loss.to(old_device)
            
        if self.return_info:
            return loss, info
        else:
            return loss

# Debugging
if __name__ == '__main__':
    im1 = torch.randn(1,3,400,400,requires_grad=True)
    im2 = torch.randn(1,3,400,400,requires_grad=True)

    base_loss = nn.L1Loss()
    loss = CorrWise(base_loss)

    l = loss(im1, im2)
