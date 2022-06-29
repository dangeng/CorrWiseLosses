'''
Preprocess KITTI raw data into train/val/test sets according to PredNet by Lotter et al.
preprocessed to 256x512
'''

import os
import requests
import re
from pathlib import Path

from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

import pdb

# Arguments
parser = argparse.ArgumentParser(description='Process KITTI dataset')
parser.add_argument('--source_dir', type=str, default='/n/owens-data1/mnt/big2/data/public/kitti/kitti_raw/raw', help='Directory of the raw KITTI data. Should contain directories with format like "2011_09_26".')
parser.add_argument('--target_dir', type=str, default='/datad/dgeng/kitti/tmp_test', help='Directory to write processed files to.')
args = parser.parse_args()

# Path to the raw dataset
source_dir = Path(args.source_dir)

# Location to save
target_dir = Path(args.target_dir)
target_dir.mkdir(exist_ok=True, parents=True)

# Get train/val/test splits used in Lotter et al.
def get_splits(fname):
    with open(fname, 'r+') as f:
        splits = f.read()
        return splits.strip('\n').split(',')
train_seqs = get_splits('splits_train.txt')
val_seqs = get_splits('splits_val.txt')
test_seqs = get_splits('splits_test.txt')

# Resize and crop to (256, 512) from (375, 1242)
transform = transforms.Compose([transforms.Resize((256, 848)),      
                                transforms.CenterCrop((256, 512))]) 

def process_split(phase, seqs):
    print(f'Preprocessing phase: {phase}')
    for seq in tqdm(seqs, desc=f'Processing {phase} seqs: '):
        day = seq[:10]

        # Make directories
        savedir = target_dir / phase / seq
        savedir.mkdir(parents=True, exist_ok=True)
        
        # Copy and transform images
        imdir = source_dir / day / seq / 'image_03' / 'data'
        tqdm.write(f'Processing: [{phase}] {seq}')
        for im_name in tqdm(os.listdir(imdir), leave=False, desc=f'Seq progress: '):
            im = Image.open(imdir / im_name)
            im = transform(im)
            im.save(savedir / im_name)
    print('\n')

# Process sequences
process_split('test', test_seqs)
process_split('val', val_seqs)
process_split('train', train_seqs)
