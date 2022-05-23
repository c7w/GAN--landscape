import datetime
import time

import yaml
from argparse import Namespace
from pathlib import Path

import numpy as np
import jittor as jt
import logging



class BaseExperiment:
    def __init__(self, args: Namespace):
        self.action = args.action
        self.logger = logging.getLogger(f"experiments.experiment.{self.__class__.__name__}")
        # self.device = None # In Jittor, no such operations ???

        self._load_config(args.config)
        self._init_directories()
        self._init_dataset()
        self._init_network()
        self._init_network_state()

        jt.misc.set_global_seed(self.seed, different_seed_for_mpi=True) # Set seed
        self._save_id = 0

    def _load_config(self, cfg_path: str):
        with open(cfg_path, 'r') as file:
            self.cfg = yaml.safe_load(file)

        # Meta Configurations
        meta = dict(self.cfg.get('meta', {}))
        self.experiment = meta.get('base', 'BaseExperiment')
        self.task = meta.get('task', f'Baseline-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
        self.checkpoint_path = Path(meta.get('checkpoint_path', './checkpoints'))
        self.log_path = Path(meta.get('log_path', './logs'))
        self.load = meta.get('load', None)
        self.seed = int(meta.get('seed', time.time()))
        self.num_loaders = int(meta.get('num_loaders', 1))
        self.log_interval = int(meta.get('log_interval', 20))
        self.shuffle_train = bool(meta.get('shuffle_train', True))

        # Train Configurations
        self.max_epochs = int(self.cfg.get('max_epochs', 100))
        self.max_iterations = int(self.cfg.get('max_iterations', 10000))
        self.batch_size = int(self.cfg.get('batch_size', 1))
        self.save_every_iteration = int(self.cfg.get('save_every_iteration', -1))  # -1 for disabled
        self.save_every_epoch = int(self.cfg.get('save_every_epoch', -1))
        self.test_dev_every_epoch = int(self.cfg.get('test_dev_every_epoch', -1))
        self.valid_every = int(self.cfg.get('valid_every', 1000))

    def _init_directories(self):
        if not self.checkpoint_path.exists():
            self.logger.info(f"Creating checkpoint directory: {self.checkpoint_path.resolve().absolute()}")
            self.checkpoint_path.mkdir(parents=True)
        if not self.log_path.exists():
            self.logger.info(f"Creating log directory: {self.log_path.resolve().absolute()}")
            self.log_path.mkdir(parents=True)

    def _init_network(self):
        raise NotImplementedError("_init_network() is not implemented in BaseExperiment")

    def _init_dataset(self):
        raise NotImplementedError("_init_dataset() is not implemented in BaseExperiment")

    def _init_network_state(self):
        raise NotImplementedError("_init_network_state() is not implemented in BaseExperiment")

    def _train_network(self, batch, i):
        raise NotImplementedError("_train_network() is not implemented in BaseExperiment")

    def _should_stop(self, epoch, iteration):
        return True if 0 < self.max_epochs <= epoch or 0 < self.max_iterations <= iteration else False

    def _should_save_checkpoint(self, epoch, iteration):
        should_save = None
        if self.save_every_epoch > 0 and (epoch % self.save_every_epoch == 0):
            should_save = f"epoch-{epoch}"
        elif self.save_every_iteration > 0 and (iteration % self.save_every_iteration == 0):
            should_save = f"iteration-{iteration}"
        return should_save

    def _save_model(self, epochs, iterations):
        raise NotImplementedError("_save_model() is not implemented in BaseExperiment")

    def _load_model(self):
        raise NotImplementedError("_load_model() is not implemented in BaseExperiment")

    def train(self):
        return NotImplementedError

    def eval(self):
        return NotImplementedError

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        parser.add_argument('--action', type=str, choices=['train', 'test'], help='Action to perform')
        parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                            , default='INFO', help='Logging level')

    def run(self):
        self.__getattribute__(f"{self.action}")()