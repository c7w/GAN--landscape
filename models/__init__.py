from argparse import Namespace

import jittor as jt
import jittor.nn as nn

from models.discriminator.CNN import CNNDiscriminator
from models.discriminator.CNN_LS import CNNLSDiscriminator
from models.discriminator.ConvGlob import ConvGlobDiscriminator
from models.discriminator.WGAN_Discriminator import WGAN_Discriminator
from models.generator.AttnGen import AttnGen
from models.generator.UNet import UnetGenerator
from models.generator.UNetEmbedded import UnetGeneratorEmbedded
from models.generator.UNetNoise import UNetGeneratorNoise
from models.generator.WGAN import W_GAN_GP_Generator


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

    elif config['type'] == 'UNetEmbedded':
        network = UnetGeneratorEmbedded(64, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)

    elif config['type'] == 'AttnGen':
        network = AttnGen()

    elif config['type'] == 'UNNoise':
        network = UNetGeneratorNoise(3, 3, 7, 64, norm_layer=nn.BatchNorm2d, use_dropout=True)

    elif config['type'] == 'W_GAN_GP_Generator':
        network = W_GAN_GP_Generator()

    else:
        raise NotImplementedError(f"Unknown generator type: {config['type']}")

    optimizer = build_optimizer(network.parameters(), config['optimizer'])
    return network, optimizer, config


def build_discriminator(config):
    if config['type'] == 'CNN':
        network = CNNDiscriminator()

    elif config['type'] == 'CNN-LS':
        network = CNNLSDiscriminator()

    elif config['type'] == 'ConvGlob':
        network = ConvGlobDiscriminator()

    elif config['type'] == 'WGAN_Discriminator':
        network = WGAN_Discriminator()

    else:
        raise NotImplementedError(f"Unknown discriminator type: {config['type']}")


    optimizer = build_optimizer(network.parameters(), config['optimizer'])
    return network, optimizer, config
