'''
Train a video prediction model using a 
recursive/autoregressive technique
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

# Parse args
opt = TrainOptions().parse()

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    # Assume we always continue training from latest epoch?
    opt.which_epoch = 'latest'
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

# Force args
opt.print_freq = max(opt.print_freq // opt.batchSize * opt.batchSize, 1)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.max_dataset_size = 10

# Get dataloader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# Make model
model = create_model(opt)
visualizer = Visualizer(opt)
optimizer_G = model.module.optimizer_G

# Step arithmetic
total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

# Training loop
for epoch in range(start_epoch, opt.niter + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ################ Predictions ######################
        # Dict of prediction information
        preds = {'loss': 0,
                 'generated': [],
                 'info': []}
        # Context to feed into model
        context = data['context'].clone().cuda(opt.gpu_ids[0])   

        ############## Forward Pass ######################
        for target_idx in range(opt.target_length):
            target = data['targets'][:,:,target_idx].clone().cuda(opt.gpu_ids[0])
            loss, generated, info = model(context,
                                            target,
                                            infer=True)

            # Keep track of prediction info
            preds['loss'] += loss
            if save_fake:
                preds['generated'].append(generated.clone().detach())
                preds['info'].append(info)

            # Remove earliest context frame and
            # add predicted frame to the end
            context = torch.cat([context[:,:,1:], generated.unsqueeze(2)], dim=2)

        ############### Backward Pass ####################
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        ############## Display results and errors ##########

        loss_dict = {'loss': loss}

        # Modify loss_dict to include information about warp loss
        if opt.corrwise:
            loss_dict['flow_mag_reg']       = info['flow_mag_reg'].mean()
            loss_dict['flow_grad_mag_reg']  = info['flow_grad_mag_reg'].mean()
            loss_dict['edge_penalty']       = info['edge_penalty'].mean()

        # Print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        # Display output images
        if save_fake:
            visuals = OrderedDict()
            for idx in range(opt.context_length):
                visuals[f'context_{idx}'] = util.tensor2im(data['context'][0,:,idx])
            
            for idx in range(opt.target_length):
                visuals[f'pred_{idx}'] = util.tensor2im(preds['generated'][idx].data[0])
                if opt.corrwise:
                    if 'forward_warped' in preds['info'][idx]:
                        warped = util.tensor2im(preds['info'][idx]['forward_warped'].data[0])
                        visuals[f'warped_{idx}'] = warped
                    if 'backward_warped' in preds['info'][idx]:
                        warped = util.tensor2im(preds['info'][idx]['backward_warped'].data[0])
                        visuals[f'back_warped_{idx}'] = warped
                visuals[f'target_{idx}'] = util.tensor2im(data['targets'][0,:,idx])
                    
            visualizer.display_current_results(visuals, epoch, total_steps)

        # Save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # End of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter, time.time() - epoch_start_time))

    # Save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
