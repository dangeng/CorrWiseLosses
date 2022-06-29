from pathlib import Path

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, CenterCrop

class ToyDataset(Dataset):
    def initialize(self, opt):
        self.opt = opt

        self.bg = Image.open(self.opt.bg_path)
        self.obj = Image.open(self.opt.obj_path)
        self.x_offset = float(opt.x_offset)
        self.y_offset = float(opt.y_offset)
        self.x_center = float(opt.x_center)
        self.y_center = float(opt.y_center)

        # Crop background to 512x256
        self.bg = CenterCrop((256, 512))(self.bg).convert('RGB')

        # Arbitrary size
        self.dataset_size = 500

        # Input buffer
        self.input = None

        # Transform
        self.transform = Compose([ToTensor(),
                                  Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])

    def composite(self, obj, bg, loc):
        '''
        Composites obj onto bg at location loc
        loc - (x,y) in scale [0,1]
        '''

        # copy images
        obj = obj.copy()
        bg = bg.copy()

        obj_w, obj_h = obj.size
        bg_w, bg_h = bg.size

        # Get location
        loc = np.array(loc)
        loc *= np.array([bg_w, bg_h]) - np.array([obj_w, obj_h])
        loc = loc.astype(int)

        # composite and return
        bg.paste(obj, tuple(loc), obj)
        return bg

    def getitem_helper(self, x_offset, y_offset):
        im = self.composite(self.obj, self.bg, (x_offset, y_offset))
        im = self.transform(im)

        if self.input is None:
            self.input = torch.randn_like(im)

        input_dict = {'context': self.input, 
                      'target': im, 
                      'path': ''}

        return input_dict

    def get_centered_image(self):
        return self.getitem_helper(self.x_center, self.y_center)['target']

    def __getitem__(self, index):
        # choose random obj offset
        x_offset = np.random.uniform(self.x_center - self.x_offset, 
                                     self.x_center + self.x_offset)
        y_offset = np.random.uniform(self.y_center - self.y_offset, 
                                     self.y_center + self.y_offset)

        return self.getitem_helper(x_offset, y_offset)

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'ToyDataset'
