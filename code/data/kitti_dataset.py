'''
Almost exactly the same as kitti_dpg_dataset, but loads kitti raw images instead of 
kitti flow images at 320x640 resolution, to match DPG resolution
'''
import os.path
from pathlib import Path
from data.base_dataset import BaseDataset
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip
from PIL import Image
import numpy as np
import torch

class KITTIDataset(BaseDataset):
    '''
    Outputs
        Context - (b,c,t,h,w) tensor of context frames
        Target - (b,c,t,h,w) tensor of target frames
    '''
    def initialize(self, opt):
        self.opt = opt  
        self.root = Path(opt.dataroot)
        
        ### Paths and params
        self.data_dir = Path(opt.dataroot) / opt.phase
        self.data_dir = self.data_dir.expanduser()

        # Get all seq names
        self.seq_names = os.listdir(self.data_dir)
        
        # Get seq lengths
        self.seq_lens = {}
        for seq_name in self.seq_names:
            self.seq_lens[seq_name] = len(os.listdir(self.data_dir / seq_name))

        # Save context and target lengths
        self.context_length = self.opt.context_length
        self.target_length = self.opt.target_length
        self.return_length = self.context_length + self.target_length
        
        # Get skip and "any" params
        self.frames_skip = self.opt.frames_skip
        self.target_jitter = self.opt.target_jitter
        
        # Get effective length
        # example (len=3, skip=3, jitter=0)
        # (Xoo)(Xoo)X = (len-1)*skip + 1
        self.eff_len = (self.return_length - 1) * self.frames_skip + \
                                              1 + self.target_jitter
        
        # Arbitrary dataset size
        self.dataset_size = 40000

        # Flip transform
        self.flip = not opt.no_flip
        self.flip_transform = RandomHorizontalFlip(p=1.0)

        # Make transform
        self.transform = Compose([ToTensor(),
                                  Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])
        
    def load_image(self, path):
        im = Image.open(path).convert('RGB')
        return self.transform(im)
        
    def get_seq_length(self, seq_name):
        return self.seq_lens[seq_name]

    def get_frame_path(self, seq_name, idx):
        return self.data_dir / seq_name / f'{idx:010}.png'

    def __getitem__(self, index):
        '''
        Just return a random sequence
        '''

        # Get random set/seq/start_idx
        seq_name = np.random.choice(self.seq_names)
        num_frames = self.get_seq_length(seq_name)
        seq_start = np.random.randint(0, num_frames + 1 - self.eff_len)

        img_path = str(self.get_frame_path(seq_name, seq_start))

        # Load all images
        idxs_to_load = [seq_start + self.frames_skip * idx for idx in range(self.return_length)]
        idxs_to_load[-1] += np.random.randint(self.target_jitter + 1)   # Jitter last target
        paths_to_load = [self.get_frame_path(seq_name, idx) for idx in idxs_to_load]
        images = [self.load_image(path) for path in paths_to_load]

        # Perform random flip
        if self.flip and np.random.rand() < .5:
            images = [self.flip_transform(im) for im in images]
                  
        # Split into context and target
        context_images = images[:self.context_length]
        target_images = images[-self.target_length:]
        
        # Stack to create temporal dimension
        context_images = torch.stack(context_images, dim=1)
        target_images = torch.stack(target_images, dim=1)

        # Not actually the image path, but the last part of the path is used
        # as a unique identifier, so we should make it unique
        img_path = img_path.replace('/0', '_0')

        # Make target image the last context frame (for debugging)
        if self.opt.last_context_target:
            target_images = context_images[:,-1:]
        
        # Return
        input_dict = {'context': context_images, 
                      'targets': target_images,
                      'path': img_path
        }

        return input_dict

    def __len__(self):
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'KITTI'
