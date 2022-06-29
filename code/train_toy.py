'''
Train a toy model on simulated uncertainty. Essentially finds the
"average" image under some loss function
'''
import time
import os
import numpy as np
import torch
from collections import OrderedDict

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

opt = TrainOptions().parse(ignore_errors=True)
opt = util.yaml_config(opt, opt.config)

# Force args
opt.dataloader = 'toy'
opt.netG = 'global'
opt.no_flip = True

# Get base device
base_device = opt.gpu_ids[0] if len(opt.gpu_ids) != 0 else 'cpu'

# Get epoch info
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = max(opt.print_freq // opt.batchSize * opt.batchSize, 1)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.max_dataset_size = 10

# Make dataloader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# Make model/visualizer/optimizer
model = create_model(opt)
visualizer = Visualizer(opt)
optimizer = model.module.optimizer_G

# Step arithmetic
total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        # Step arithmetic
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        ############## Forward Pass ######################
        context, target = data['context'].to(opt.gpu_ids[0]), \
                          data['target'].to(opt.gpu_ids[0])
                          
        loss, generated, info = model(context,
                                      target,
                                      infer=True)

        ############### Backward Pass ####################
        # update generator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {'loss': loss.item()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if total_steps % opt.display_freq == display_delta:
            visuals = OrderedDict()
            visuals['context'] = util.tensor2im(data['context'][0])
            visuals['generated'] = util.tensor2im(generated.detach()[0])
            if opt.corrwise:
                visuals['forward_warp'] = util.tensor2im(info['forward_warped'].detach()[0])
                visuals['backward_warp'] = util.tensor2im(info['backward_warped'][0])
            visuals['target'] = util.tensor2im(data['target'][0])
            target_centered = dataset.dataset.get_centered_image()
            visuals['target_centered'] = util.tensor2im(target_centered)

            visualizer.display_current_results(visuals, epoch, total_steps)

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))
