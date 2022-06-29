'''
Generate predictions from a trained model 
and save the to `results_dir`
'''
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import numpy as np
from shutil import copyfile

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

# Parse args
opt = TestOptions().parse(save=False)
opt = util.yaml_config(opt, opt.config)

# Force arguments
opt.gpu_ids = [opt.gpu_ids[0]]
opt.nThreads = 0            # test code only supports nThreads = 1
opt.batchSize = 1           # test code only supports batchSize = 1
opt.serial_batches = False  # shuffle, to get good sample of test images
opt.no_flip = True          # no flip

# Init dataset and visualizer
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# Create website viz
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# Copy config file
copyfile(opt.config, os.path.join(web_dir, 'config.yml'))

# Make model
model = create_model(opt)
model.eval()

tested_paths = set()

# Test loop
for i, data in enumerate(dataset):

    if len(tested_paths) >= opt.how_many:
        break

    # Skip ones we've seen before (required because the dataloader is random)
    if data['path'][0] in tested_paths:
        continue
    
    # Make predictions
    preds = []
    context = data['context'].clone().cuda(opt.gpu_ids[0])
    for target_idx in range(opt.target_length):
        generated = model.inference(context)
        
        preds.append(generated)
        context = torch.cat([context[:,:,1:], generated.unsqueeze(2)], dim=2)
        
    # Make visualizations
    visuals = []
    for idx in range(opt.context_length):
        img = util.tensor2im(data['context'][0,:,idx])
        visuals.append([f'context_{idx}', img])
    for idx in range(opt.target_length):
        img = util.tensor2im(preds[idx][0].data)
        target = util.tensor2im(data['targets'][0,:,idx])
        visuals.append([f'generated_{idx}', img])
        visuals.append([f'target_{idx}', target])
    visuals = OrderedDict(visuals)
        
    img_path = data['path']
    print('process image #%s... %s' % (len(tested_paths), img_path))
    visualizer.save_images(webpage, visuals, img_path, gif=True)

    tested_paths.add(img_path[0])

webpage.save()
