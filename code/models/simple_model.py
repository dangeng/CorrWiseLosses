import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks

import sys
sys.path.append('./corrwise')
from corrwise import CorrWise

class SimpleModel(BaseModel):
    def name(self):
        return 'SimpleModel'
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.corrwise = opt.corrwise

        ##### define networks
        # Generator network
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, upsample=opt.upsample)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:

            # define loss functions
            self.criterionPW = PixelwiseLosses(opt)
            if self.corrwise:
                self.criterionPW = CorrWise(self.criterionPW,
                                            epsilon=opt.epsilon,
                                            flow_method=self.opt.flow_method,
                                            return_info=True,
                                            detach=not self.opt.no_detach,
                                            padding_mode=self.opt.padding_mode,
                                            reg_flow_mag=opt.reg_flow_mag,
                                            reg_flow_grad_mag=opt.reg_flow_grad_mag,
                                            flow_reg_norm=opt.flow_reg_norm,
                                            edge_aware=opt.eas,
                                            edge_penalty=opt.edge_penalty)
                self.criterionPW.to(self.opt.gpu_ids[0])

            # initialize optimizers
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params,
                                                lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
                                                    
    def forward(self, context, target, infer=False):
        # Pred
        pred = self.netG.forward(context)
        
        # Pixelwise loss
        if self.corrwise:
            loss, info = self.criterionPW(pred, target)
        else:
            loss= self.criterionPW(pred, target)
            info = None
            
        # Only return the fake_B image if necessary to save BW
        return [ loss,
                 None if not infer else pred,
                 info ]

    def inference(self, context):
        with torch.no_grad():
            pred = self.netG.forward(context)
        return pred

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)

class InferenceModel(SimpleModel):
    def forward(self, inp):
        return self.inference(inp)

class PixelwiseLosses(nn.Module):
    # Helper class to average multiple losses
    # Useful so we only run alignment step once when using
    # a corrwise loss
    def __init__(self, opt):
        super(PixelwiseLosses, self).__init__()
        
        self.loss_fns = nn.ModuleList()
        
        if opt.perceptual:
            perceptual_loss_fn = networks.VGGLoss(opt)
            self.loss_fns.append(perceptual_loss_fn)
        if opt.mse:
            self.loss_fns.append(nn.MSELoss())
        if opt.l1:
            self.loss_fns.append(nn.L1Loss())
            
    def forward(self, x, y):
        # Return average of pixelwise losses
        loss = 0
        
        for loss_fn in self.loss_fns:
            loss += loss_fn(x, y)
            
        return loss / len(self.loss_fns)
