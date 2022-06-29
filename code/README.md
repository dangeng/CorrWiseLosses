# Correspondence-wise Losses

We propose correspondence-wise losses, which align images _before_ calculating an arbitrary loss. In our paper we show that this results in improvements to image synthesis tasks and provides robustness to spatial uncertainty.

## Getting Started

### Environment Setup

Create and activate a conda environment by running:

```
conda env create -f environment.yml
conda activate corrwise
```

Download RAFT checkpoints (for correspondence-wise warping) by going to `corrwise/` and running

```
sh download_models.sh
```

### Toy Experiments

The toy experiments can be run by executing:

```
python train_toy.py --config [path to config file]
```

where config files can be found in `configs/toy`. Visualizations are written to `checkpoints/[experiment_name]/web` as an html file.

### KITTI Ablations

To run a video prediction model on KITTI with and without a correspondence-wise loss, first download and process the data (detailed in the next section). Next, change the `dataroot` option in the config files in `configs/kitti` to the appropriate directory. Then run

```
python train.py --config [path to config file]
```

with a config file from `configs/kitti`. Visualizations of the training can be found in `checkpoints/[experiment_name]/web` as an html file.

When training is done, run 

```
python test.py --config [path to config file]
```

to generate predictions on the test set and save them to `results_dir`. Edit `which_epoch` in the config to change the checkpoint used. Generated images can be found in `results/[experiment name]` as html files.

After generating test images, run

```
python eval.py --config [path to config file] --eval_fn [lpips, ssim, mse, l1]
```

to evaluate images under various metrics.

### Downloading KITTI 

Get the [KITTI raw data download script at](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip), unzip it, and then run it to download the KITTI data. Aferwards, process the KITTI data by running the `scripts/make_kitti.py` script. Make sure to supply the desired `--source_dir` and `--target_dir` arguments.

## Wrapper

Our method is implemented as a wrapper that wraps around a base loss function. It can be found in `corrwise/`. As an example, we can wrap an L1 loss as such:

```python
base_loss = nn.L1Loss()
loss = CorrWise(base_loss)
```

## Code credits

The image generation and video prediction code is modified from the [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) repo by Wang et al., commit `5a2c87201c5957e2bf51d79b8acddb9cc1920b26`.

We also provide a subset of the RAFT codebase:

- RAFT, (modified) from [the RAFT repo](https://github.com/princeton-vl/RAFT) by Zachary Teed, commit `13198c355d11c3a0c45f09d1f15ead4b81a5043f`

