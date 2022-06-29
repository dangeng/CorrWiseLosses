import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # config file
        self.parser.add_argument('--config', type=str, default='', help='path to config file')        
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')        
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='simple', help='which model to use')
        self.parser.add_argument('--norm', type=str, default='batch', help='Type of normalization to use in network [batch, instance]')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--dataloader', type=str, default='kitti', help='which dataloader to use [toy, kitti]')

        # input/output sizes       
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--context_length', type=int, default=3, help='# images of context')
        self.parser.add_argument('--target_length', type=int, default=1, help='# images to predict (1 is just next frame prediction)')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str) 
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--frames_skip', type=int, default=1, help='Number of frames to skip in the video')
        self.parser.add_argument('--target_jitter', type=int, default=0, help='Number of frames to jitter the last target frame. Used to implement the "any" experiment.')
        self.parser.add_argument('--last_context_target', action='store_true', help='if true, targets are last context frame (only implemented for KITTI)')        

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='frame', help='selects model to use for netG [global, local, frame, identity]')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG') 
        self.parser.add_argument('--n_blocks_global', type=int, default=6, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--upsample', type=str, default='nearest', help='use upsampling instead of transposed convs [bilinear, nearest]')

        self.initialized = True

    def parse(self, save=True, ignore_errors=False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        # Get args specified by config
        self.opt = util.yaml_config(self.opt, self.opt.config, ignore_errors=(not self.isTrain or ignore_errors))
        
        self.opt.isTrain = self.isTrain   # train or test
        if not self.isTrain:
            self.opt.phase = 'test'
        
        # Parse gpu ids if it's a string
        if type(self.opt.gpu_ids) is not list:
            str_ids = self.opt.gpu_ids.split(',')
            self.opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
