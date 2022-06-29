import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np


def CreateDataset(opt):
    dataset = None
    
    if opt.dataloader == 'toy':
        from data.toy_dataset import ToyDataset
        dataset = ToyDataset()
    elif opt.dataloader == 'kitti':
        from data.kitti_dataset import KITTIDataset
        dataset = KITTIDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)

        def worker_init_fn(_):
            seed = int(torch.initial_seed())%(2**32-1)
            np.random.seed(seed)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            worker_init_fn=worker_init_fn)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
