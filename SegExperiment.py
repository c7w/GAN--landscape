import argparse
from argparse import Namespace
from random import random

import jittor as jt
import jittor.nn as nn
import numpy as np
import yaml

from datasets.datasets_divided import ImageDatasetDivided
from models.Transformer.SegFormer.MixVisionTransformer import SegFormerB3



class SegExperiment:

    def __init__(self, args: Namespace):

        self._parse_args(args)
        self._init_network()

        if args.mode == 'train':
            self.train()
        elif args.mode == 'test':
            self.test()


    def _init_network(self):
        self.network = SegFormerB3()

    def _init_dataset_train(self):
        self.dataset = ImageDatasetDivided('/home/c7w/landscape/data').train_dataset.set_attrs(
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
        )

    def _init_dataset_valid(self):
        self.dataset = ImageDatasetDivided('/home/c7w/landscape/data').valid_dataset


    def _run_network(self, batch, i):
        seg = self.network(batch)
        IPython.embed(header=f"{i}")

    def train(self):
        for epoch in range(self.config['train']['max_epochs']):
            for idx, batch in enumerate(self.dataset):
                self._run_network(batch, idx)

    def test(self):
        pass

    def _parse_args(self, args: Namespace):
        # Config path
        config_path = args.config

        # Open config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.config = config

        # Reset random number generator
        seed = config['meta']['seed']
        jt.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser(description='SegFormer: semantic segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    # Parse arguments
    opts = parser.parse_args()
    experiment = SegExperiment(opts)
