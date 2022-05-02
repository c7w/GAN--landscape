from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models.discriminator.CNN import CNNDiscriminator
from models.generator.UNet import UnetGenerator


def start_grad(model):
    for param in model.parameters():
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)

def get_model(args: Namespace):

    task_name = args.task_name

    if task_name == "baseline":
        generator = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)
        discriminator = CNNDiscriminator()
        return generator, discriminator

    raise NotImplementedError