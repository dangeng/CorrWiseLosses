# Correspondence-wise Losses

We propose to align images _before_ calculating a loss and optimizing in image generation tasks. Instead of calculating losses pixel-wise we calculate them _correspondence-wise_. Correspondence-wise losses offers greater robustness to spatial uncertainty and a more informative gradient signal.

## Getting Started

### Environment and Dependencies

A conda environment can be set up by running `conda create --name [ENV_NAME] --file requirements.txt`.

TODO: Make a virtualenv requirements.txt

TODO (might have forgotten a few): Requirements are roughly:

- pytorch
- numpy
- scipy
- PIL

### Download Flow Model Checkpoints

First, download the flow model checkpoints by running `sh download_models.sh`. This will download the RAFT checkpoints and copy them to the correct directory, and then do some cleanup.

### Wrapper

Our method is implemented as a wrapper that wraps around a base loss function. As an example, we can wrap an L1 loss as such:

```python
base_loss = nn.L1Loss()
loss = CorrWise(base_loss)
```

In addition, various parameters can be passed to the wrapper to specify different behavior. Details can be found in the docstring of `CorrWise`. A typical use case would be:

```python
base_loss = nn.L1Loss()
loss = CorrWise(base_loss, 
                backward_warp=True, 
                return_warped=False, 
                padding_mode='reflection',
                scale_clip=.1)
```

Please see `example.py` for a short, working example.

## Replicating our Experiments

### Download the data

TODO

### Run experiments

Config files for our experiments are located in `./experiments/` and can be run by calling

```
python train.py --config [path to config file]
```

### Evaluate 

```
python test.py --config [path to config file]
```


## Code credits

Image generation code is modified from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), commit `5a2c87201c5957e2bf51d79b8acddb9cc1920b26`.

We also provide two flow methods:

- PWC-Net, (modified) from [this repo](https://github.com/sniklaus/pytorch-pwc) by Simon Niklaus, commit hash `cf0d2f2cbef4bcb0f6cbf09011960a4899d77dec`
- RAFT, (modified) from [the RAFT repo](https://github.com/princeton-vl/RAFT) by Zachary Teed, commit `13198c355d11c3a0c45f09d1f15ead4b81a5043f`


## TODO

 - [ ] Standardized data loading. Right now there are multiple dataloaders, loading video sequences in a variety of formats.
 - [ ] Write scripts to download data and put them in the correct format and directory structure.
 - [ ] Write script to download raft-things.pth into `flow_models/raft` dir
