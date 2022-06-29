from __future__ import print_function
from PIL import Image
import numpy as np
import os
import yaml

from subprocess import check_output


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping, f"Duplicate key '{key}' in config file"
            mapping.append(key)
        return super().construct_mapping(node, deep)


def yaml_config(opt, yaml_path, ignore_errors=False):
    '''
    Update an argparse namespace 
    with options from a yaml file
    '''
    with open(yaml_path, 'rb') as f:
        yaml_opts = yaml.load(f, Loader=UniqueKeyLoader)

    for k,v in yaml_opts.items():
        # Check if opt actually has the options in the yaml
        # This is kind of ugly, b/c we have configs that contain both
        # test and train options, so things go kind of haywire...
        # The solution is to just not check during test time, and
        # ignore the test options at train time, lol
        if not ignore_errors and k != 'how_many' and k != 'which_epoch':
            assert hasattr(opt, k), f"'{k}' in your config file isn't a valid option! Is there a typo?"
        opt.__setattr__(k, v)
    assert opt.name == os.path.split(yaml_path)[1][:-4], f"Config name '{opt.name}' doesn't match filename {os.path.split(yaml_path)[1]}"
    return opt

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # Converts a Tensor into a Numpy array
    # |imtype|: the desired type of the converted numpy array
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

