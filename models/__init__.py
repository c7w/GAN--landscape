from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models.discriminator.CNN import CNNDiscriminator
from models.discriminator.CNN_LS import CNNLSDiscriminator
from models.discriminator.ConvGlob import ConvGlobDiscriminator
from models.generator.AttnGen import AttnGen
from models.generator.UNet import UnetGenerator


def build_optimizer(params, config):
    if config['type'] == 'Adam':
        return jt.optim.Adam(
            params,
            lr=config['lr'],
            betas=(config['beta1'], config['beta2']),
            weight_decay=config['weight_decay'],
        )

    else:
        raise NotImplementedError(f"Unknown optimizer type: {config['type']}")


def build_generator(config):
    if config['type'] == 'UNet':
        network = UnetGenerator(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)

    elif config['type'] == 'AttnGen':
        network = AttnGen()

    else:
        raise NotImplementedError(f"Unknown generator type: {config['type']}")

    optimizer = build_optimizer(network.parameters(), config['optimizer'])
    return network, optimizer


def build_discriminator(config):
    if config['type'] == 'CNN':
        network = CNNDiscriminator()

    elif config['type'] == 'CNN-LS':
        network = CNNLSDiscriminator()

    elif config['type'] == 'ConvGlob':
        network = ConvGlobDiscriminator()

    else:
        raise NotImplementedError(f"Unknown discriminator type: {config['type']}")


    optimizer = build_optimizer(network.parameters(), config['optimizer'])
    return network, optimizer

