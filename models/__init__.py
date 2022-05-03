from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models.discriminator.CNN import CNNDiscriminator
from models.generator.UNet import UnetGenerator


def get_model(args: Namespace):

    task_name = args.task_name

    if task_name == "baseline":
        generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
        discriminator = CNNDiscriminator()
        return generator, discriminator

    # TODO: Add more models here

    raise NotImplementedError(f"Unknown task name: {task_name}")