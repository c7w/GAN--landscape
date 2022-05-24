import argparse
from argparse import Namespace
import random
from pathlib import Path

import IPython
import jittor as jt
import jittor.nn as nn
import numpy as np
import yaml

from datasets.datasets_divided import ImageDatasetDivided
from models.Transformer.SegFormer.MixVisionTransformer import SegFormerB3
from models.Transformer.SegFormer.SegFormerHead import SegFormerHead


class SegExperiment:

    def __init__(self, args: Namespace):

        self._parse_args(args)
        self._init_network()

        if args.mode == 'train':
            self._init_dataset_train()
            self.train()
        elif args.mode == 'test':
            self._init_dataset_valid()
            self.test()


    def _init_network(self):
        self.network = nn.Sequential(
            SegFormerB3(),
            SegFormerHead(),
        )

        self.optimizer = jt.optim.Adam(self.network.parameters(), lr=float(self.config['train']['lr']))

        if self.load:
            self._load_model()



    def _init_dataset_train(self):
        self.dataset = ImageDatasetDivided('/home/c7w/landscape/data').train_dataset.set_attrs(
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
        )

    def _init_dataset_valid(self):
        self.dataset = ImageDatasetDivided('/home/c7w/landscape/data').valid_dataset

    def _run_network(self, img, i):
        return self.network(img).transpose(0, 2, 3, 1)

    def train(self):
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.config['train']['max_epochs']):
            for idx, batch in enumerate(self.dataset):
                img, label, photo_id = batch
                result = self._run_network(img, idx).reshape(-1, 29)  # 29 is num_classes
                label = label[:, :, :, 0].reshape(-1)
                loss = loss_fn(result, label)
                # Debug loss
                print(epoch, idx, loss.item())
                self.optimizer.step(loss)

                if idx % 500 == 0:
                    self._save_model(epoch, idx)

    def test(self):
        pass  # TODO: Add test function

    def _parse_args(self, args: Namespace):
        # Config path
        config_path = args.config

        # Open config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.config = config

        # Reset random number generator
        seed = config['meta']['seed']
        jt.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.checkpoint_path = Path(config['meta']['checkpoint_path'])
        self.load = config['meta']['load']
        self.task = config['meta']['task']
    
    def _load_model(self):
        load_path = (self.checkpoint_path / f"{self.load}").resolve().absolute().__str__()

        to_load = jt.load(load_path)
        self.network.load_state_dict(to_load['network'])
        self.optimizer.load_state_dict(to_load['optimizer'])

    def _save_model(self, epochs, iterations):
        # Check if self.checkpoint_path exists
        if not self.checkpoint_path.exists():
            self.checkpoint_path.mkdir()

        save_path = (self.checkpoint_path / f"{self.task}-e{epochs}-i{iterations}.pkl").resolve().absolute().__str__()
        to_save = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        jt.save(to_save, save_path)

if __name__ == "__main__":
    # Use CUDA
    jt.flags.use_cuda = 2  # Force CUDA

    # Add arguments
    parser = argparse.ArgumentParser(description='SegFormer: semantic segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    # Parse arguments
    opts = parser.parse_args()
    experiment = SegExperiment(opts)
