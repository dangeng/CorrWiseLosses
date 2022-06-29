'''
Evaluates generated images from running `test.py`
using various metrics
'''
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from util.util import yaml_config
import lpips
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default=None, 
                         help='Path to config file. Overrides all options')        
parser.add_argument('--name', type=str, default='', 
                         help='name of experiment')
parser.add_argument('--eval_fn', type=str, default='lpips', 
                         help='One of [lpips, mse, l1, ssim]')
parser.add_argument('--results_dir', type=str, default='./results', 
                         help='Path to results dir')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                         help='Path to checkpoints dir')
parser.add_argument('--which_epoch', type=str, default='latest', 
                         help='Which epoch to evaluate')
                                 
# Parse args
opt = parser.parse_args()
if opt.config is not None:
    opt = yaml_config(opt, opt.config, ignore_errors=True)

# Get paths
results_dir = Path(opt.results_dir) / opt.name / f'test_{opt.which_epoch}' / 'images'
opt_path = Path(opt.checkpoints_dir) / opt.name / 'opt.txt'
loss_path = Path(opt.checkpoints_dir) / opt.name / 'loss_log.txt'

# If we can't find results_dir
if not results_dir.exists():
    raise Exception(f'Could not find results dir `{results_dir}`! (Did you run test.py with the correct epoch?)')
        
# Get test images
fnames = os.listdir(results_dir)

# Get predicted and target paths
generated_fnames = [f for f in fnames if 'generated' in f]
target_fnames = [f for f in fnames if 'target' in f]

# Sort to get canonical order
generated_fnames = sorted(generated_fnames)
target_fnames = sorted(target_fnames)

def pil2ten(im):
    # Converts PIL to pytorch tensor
    im = np.array(im) / 255.
    im = (im * 2) - 1
    return torch.tensor(im).permute(2,0,1).unsqueeze(0).float().cuda()

# Get metric
if opt.eval_fn == 'lpips':
    eval_fn = lpips.LPIPS(net='alex')
    eval_fn = eval_fn.cuda()
elif opt.eval_fn == 'mse':
    eval_fn = F.mse_loss
elif opt.eval_fn == 'l1':
    eval_fn = F.l1_loss
elif opt.eval_fn == 'ssim':
    eval_fn = ssim
else:
    assert True, "`eval_fn` must be one of [lpips, mse, l1, ssim]"

# Calculate scores for every generated image
scores = []
for f_g, f_t in tqdm(zip(generated_fnames, target_fnames), \
    total=len(target_fnames)):
        
    # Load images
    pred = Image.open(results_dir / f_g)
    target = Image.open(results_dir / f_t)
    
    # Convert to tensor
    pred = pil2ten(pred)
    target = pil2ten(target)

    # Eval metric
    with torch.no_grad():
        if opt.eval_fn == 'ssim':
            pred = pred.cpu().numpy()[0].transpose(1,2,0)
            target = target.cpu().numpy()[0].transpose(1,2,0)
        
            score = eval_fn(pred, target, multichannel=True)
        else:
            score = eval_fn(pred, target)

    scores.append(score.item())
    
score = np.mean(scores)
print(f'Mean {opt.eval_fn} score: {score:.5f}')
