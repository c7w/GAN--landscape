import os
import random
import zipfile
import argparse
from pathlib import Path

import cv2
import jittor as jt
import jittor.nn as nn

import datetime
from argparse import Namespace
import numpy as np
import yaml
from tqdm import tqdm

from datasets.datasets import ImageDataset
from models import build_generator, build_discriminator
from models.utils.utils import stop_grad, start_grad


class GANExperiment:
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
        # Network
        self.generator, self.optimizer_G = build_generator(self.config['network']['generator'])
        self.discriminator, self.optimizer_D = build_discriminator(self.config['network']['discriminator'])

        self.epoch = 1
        self.iteration = 1

        # Load model
        if self.load is not None:
            self._load_model()

    def _init_dataset_train(self):
        self.dataloader = ImageDataset("/home/c7w/landscape/data/", mode="train").set_attrs(
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=1,
        )

    def _init_dataset_valid(self):
        self.dataloader = ImageDataset("/home/c7w/landscape/data/", mode="test").set_attrs(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12,
        )

    def _should_step(self):
        if self.iteration % 16 == 0:
            return True

    def train(self):

        # TODO: Add more losses...??
        criterion = nn.MSELoss()
        criterion_L1 = nn.L1Loss()
        # Originally a L1 Loss exists here...

        cumulate_loss_D = jt.array(0)
        cumulate_loss_G = jt.array(0)

        for self.epoch in range(self.epoch, self.max_epochs):

            loop = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for batch_idx, batch in loop:
                img, label, photo_id = batch

                # If loss_G is way too high, we should train generator more
                if self.iteration % 2 == 0 and loss_G / loss_D <= 2.0:
                    # Train Discriminator
                    stop_grad(self.generator)
                    start_grad(self.discriminator)

                    fake_img = self.generator(label)

                    loss_fake = criterion(self.discriminator(fake_img, label), -1.0)
                    loss_real = criterion(self.discriminator(img, label), 1.0)

                    loss_D = (loss_fake + loss_real) / 2
                    cumulate_loss_D += loss_D
                    # self.optimizer_D.step(loss_D)

                else:
                    # Train Generator
                    stop_grad(self.discriminator)
                    start_grad(self.generator)

                    fake_img = self.generator(label)

                    loss_G = criterion(self.discriminator(fake_img, label), 1.0)
                    loss_G += 200 * criterion_L1(fake_img, img)  # TODO: Change this 200?
                    cumulate_loss_G += loss_G

                    # self.optimizer_G.step(loss_G)

                if self._should_step():
                    self.optimizer_D.step(cumulate_loss_D)
                    self.optimizer_G.step(cumulate_loss_G)
                    cumulate_loss_D = jt.array(0)
                    cumulate_loss_G = jt.array(0)

                # Logging
                if self.log_interval > 0 and self.iteration % self.log_interval == 1:
                    loop.set_description(f'[{self.task}] [{self.epoch} / {self.iteration}]')
                    loop.set_postfix(
                        lossD=loss_D.item(),
                        lossG=loss_G.item(),
                    )

                # Save for iteration
                if self.save_every_iteration > 0 and self.iteration % self.save_every_iteration == 1:
                    self._save_model(self.epoch, self.iteration)
                    self.test(save_name=f"test-{self.epoch}-{self.iteration}", sample=True)

                self.iteration += 1

            # Save for epoch
            if self.save_every_epoch > 0 and self.epoch % self.save_every_epoch == 1:
                self._save_model(self.epoch, self.iteration)
                self.test(save_name=f"test-{self.epoch}-{self.iteration}", sample=True)

    def test(self, save_name=None, sample=False):
        stop_grad(self.generator)

        save_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + ("" if save_name is None else ("-" + save_name))

        output_dir = self.test_save_path / self.task / save_name

        os.makedirs(output_dir, exist_ok=True)

        if sample:
            for i, (_, real_A, photo_id) in enumerate(self.dataloader):
                fake_B = self.generator(real_A)
                fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')

                for idx in range(fake_B.shape[0]):
                    filename = str(output_dir / f"{photo_id[idx]}.jpg")
                    cv2.imwrite(filename,
                                fake_B[idx].transpose(1, 2, 0)[:, :, ::-1])  # BGR to RGB

                if i == 10:
                    break

        else:

            f = zipfile.ZipFile(str(self.test_save_path / self.task / (save_name + ".zip")), 'w',
                                zipfile.ZIP_DEFLATED)

            # Iterate through val_dataloader
            for i, (_, real_A, photo_id) in enumerate(self.dataloader):
                fake_B = self.generator(real_A)
                fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')

                for idx in range(fake_B.shape[0]):
                    filename = str(output_dir / f"{photo_id[idx]}.jpg")
                    cv2.imwrite(filename,
                                fake_B[idx].transpose(1, 2, 0)[:, :, ::-1])  # BGR to RGB
                    f.write(filename, arcname=f"{photo_id[idx]}.jpg")
            f.close()

    def _load_model(self):
        load_path = (self.checkpoint_path / self.task / f"{self.load}").resolve().absolute().__str__()
        to_load = jt.load(load_path)

        self.generator.load_state_dict(to_load['generator'])
        self.discriminator.load_state_dict(to_load['discriminator'])
        self.optimizer_G.load_state_dict(to_load['optimizer_G'])
        self.optimizer_D.load_state_dict(to_load['optimizer_D'])
        self.epoch = to_load['epoch'] + 1
        self.iteration = to_load['iteration'] + 1

    def _save_model(self, epochs, iterations):

        path = self.checkpoint_path / self.task

        # Check if self.checkpoint_path exists
        if not path.exists():
            path.mkdir()

        save_path = (path / f"{self.task}-e{epochs}-i{iterations}.pkl").resolve().absolute().__str__()
        to_save = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "epoch": self.epoch,
            "iteration": self.iteration
        }
        jt.save(to_save, save_path)

    def _parse_args(self, args: Namespace):
        # Config path
        config_path = args.config

        # Open config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.config = config

        # Meta Data
        self.task = config['meta']['task']
        self.checkpoint_path = Path(config['meta']['checkpoint_path'])
        self.test_save_path = Path(config['meta']['test_save_path'])
        self.load = config['meta']['load']
        self.log_interval = config['meta']['log_interval']

        # Reset random number generator
        self.seed = config['meta']['seed']
        if self.seed is not None:
            jt.misc.set_global_seed(self.seed, different_seed_for_mpi=True)  # Set seed
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Training
        self.max_epochs = config['training']['max_epochs']
        self.max_iterations = config['training']['max_iterations']
        self.batch_size = config['training']['batch_size']
        self.save_every_iteration = config['training']['save_every_iteration']
        self.save_every_epoch = config['training']['save_every_epoch']


if __name__ == "__main__":
    # Use CUDA
    jt.flags.use_cuda = 2  # Force CUDA

    # Add arguments
    parser = argparse.ArgumentParser(description='SegFormer: semantic segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    # Parse arguments
    opts = parser.parse_args()
    experiment = GANExperiment(opts)
