import torch
import torch.nn as nn
import functools

from torchvision import models

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
    
def get_norm_layer_3d(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class SliceLast(torch.nn.Module):
    # Return last temporal slice in video
    def __init__(self):
        super(SliceLast, self).__init__()

    def forward(self, x):
        return x[:, :, -1]

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, 
             n_local_enhancers=1, n_blocks_local=3, norm='batch', gpu_ids=[], upsample=False):    
    if netG == 'global':    
        norm_layer = get_norm_layer(norm_type=norm)     
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, 
                                    n_blocks_global, norm_layer)       
    elif netG == 'local':        
        norm_layer = get_norm_layer(norm_type=norm)     
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'frame':
        norm_layer = get_norm_layer(norm_type=norm)     
        norm_layer_3d = get_norm_layer_3d(norm_type=norm)     
        netG = FrameGenerator(input_nc, output_nc, ngf, n_downsample_global, 
                                    n_blocks_global, norm_layer_3d, norm_layer, upsample=upsample)
    elif netG == 'identity':
        netG = IdentityFrameGenerator()
        
    else:
        raise('generator not implemented!')
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class VGGLoss(nn.Module):
    def __init__(self, opt):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             


# Simple identity frame generator for debugging
# Takes in c frames, predicts next frame
class IdentityFrameGenerator(nn.Module):
    def __init__(self):
        super(IdentityFrameGenerator, self).__init__()
        
        # Dummy parameter to make optimizer happy
        #self.register_parameter('dummy', torch.zeros(1))
        self.dummy = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return x[:,:,-1,:,:] + self.dummy
        
# Frame generator
# Takes in c frames, predicts next frame
# Architecture (roughly): 3D convs -> 3D resblocks -> 
# slice last temporal frame -> 2D convs -> 2D transposed convs
class FrameGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, 
                 norm_layer_3d=nn.BatchNorm3d, 
                 norm_layer_2d=nn.BatchNorm2d, 
                 padding_type='reflect',
                 upsample=None):
        assert(n_blocks >= 0)
        super(FrameGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReplicationPad3d([3, 3, 3, 3, 1, 1]), 
                 nn.Conv3d(input_nc, ngf, kernel_size=[3, 7, 7], padding=0), 
                 norm_layer_3d(ngf), 
                 activation]
        
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            #stride = 2
            stride = [1, 2, 2]
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=stride, 
                                padding=1),
                      norm_layer_3d(ngf * mult * 2), activation]

        mult = 2**n_downsampling
        for i in range(2):
            model += [ResnetBlock3D(ngf * mult, padding_type=padding_type, 
                                    activation=activation, norm_layer=norm_layer_3d)]
        model.append(SliceLast())
        for i in range(n_blocks - 2):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, 
                                  activation=activation, norm_layer=norm_layer_2d)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            stride = 2
            if upsample is not None:
                model += [nn.Upsample(scale_factor=2, mode=upsample),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2), 
                                    kernel_size=3, stride=1, 
                                    padding=1),
                          norm_layer_2d(int(ngf * mult / 2)), activation]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 
                                             kernel_size=3, stride=stride, 
                                             padding=1, output_padding=1),
                          norm_layer_2d(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), 
                  nn.Tanh()] 

        self.model = nn.Sequential(*model)

    def forward(self, input):
        '''
        input - Tensor of frames of shape (B, C, T, H, W)
        output - Tensor of predicted frame, shape (B, C, H, W)
        '''

        out = self.model(input)             
        return out
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
        
class ResnetBlock3D(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock3D, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect' or padding_type == 'replicate':
            # If reflect, just use replication
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect' or padding_type == 'replicate':
            # If reflect, just use replication
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice_idxs = [2,7,12,21,30]
            
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(self.slice_idxs[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[0], self.slice_idxs[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[1], self.slice_idxs[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[2], self.slice_idxs[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[3], self.slice_idxs[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
