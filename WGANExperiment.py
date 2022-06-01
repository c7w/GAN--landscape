import os
import random
import zipfile
import argparse
from pathlib import Path

from IPython import embed
import cv2
import jittor as jt
import jittor.nn as nn

from argparse import Namespace
import numpy as np
from tqdm import tqdm

from GANExperiment import GANExperiment
from datasets.datasets import ImageDataset
from models import build_generator, build_discriminator
from models.loss import calc_loss, build_loss
from models.utils.utils import stop_grad, start_grad, forgiving_state_restore
import time
import wandb


class WGANExperiment(GANExperiment):
    def __init__(self, args: Namespace):
        super().__init__(args)

    def train(self):

        # TODO: Add more losses...??

        wandb.init(project=self.config["meta"]["task"], entity="jittor-landscape")
        wandb.config = self.config

        loss_fn = build_loss(config=self.config['loss'])
        # Originally a L1 Loss exists here...

        loss_D, loss_G = jt.array(99.0), jt.array(99.0)
        cumulate_loss_D = jt.array(0)
        cumulate_loss_G = jt.array(0)

        for self.epoch in range(self.epoch, self.max_epochs):

            loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            for batch_idx, batch in loop:
                img, label, photo_id = batch

                if self.iteration % self.train_discriminator_every == 0:
                    # Train Discriminator
                    stop_grad(self.generator)
                    start_grad(self.discriminator)

                    fake_img = self.generator(label)

                    label_fake_img = jt.contrib.concat((label, fake_img), dim=1)
                    # Here detach() means ending gradient back-tracing
                    pred_fake = self.discriminator(label_fake_img.detach())

                    label_real_img = jt.contrib.concat((label, img), dim=1)
                    pred_real = self.discriminator(label_real_img)

                    loss_D = calc_loss(loss_fn, type='D', discriminator_result_fake=pred_fake,
                                       discriminator_result_real=pred_real,
                                       fake_img=fake_img, real_img=img,
                                       discriminator=self.discriminator,
                                    )

                    cumulate_loss_D += loss_D

                else:
                    # Train Generator
                    stop_grad(self.discriminator)
                    start_grad(self.generator)

                    fake_img = self.generator(label)

                    label_fake_img = jt.contrib.concat((label, fake_img), dim=1)
                    pred_fake = self.discriminator(label_fake_img)

                    loss_G = calc_loss(loss_fn, type='G', fake_img=fake_img, real_img=img,
                                       discriminator_result_fake=pred_fake)
                    cumulate_loss_G += loss_G

                # optimizer.step()
                if self._should_step():
                    if cumulate_loss_D.item() > 0:
                        wandb.log({"loss_D": cumulate_loss_D.item()})
                        self.optimizer_D.step(cumulate_loss_D)
                        cumulate_loss_D = jt.array(0.0)
                    if cumulate_loss_G.item() > 0:
                        wandb.log({"loss_G": cumulate_loss_G.item()})
                        self.optimizer_G.step(cumulate_loss_G)
                        cumulate_loss_G = jt.array(0.0)


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
                # for p in self.generator.parameters():
                #     if 'noise_weight' in p.name():
                #         print(p.name())
                #         print(p)
                self._save_model(self.epoch, self.iteration)
                self.test(save_name=f"test-{self.epoch}-{self.iteration}", sample=True)



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
