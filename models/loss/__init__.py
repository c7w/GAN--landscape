import IPython
import jittor.nn

# Least Squares Loss
from jittor import nn
from GradientPenalty import DiscriminatorLossGradientPenalty


def GeneratorLossLS(**kwargs):
    discriminator_result_fake = kwargs['discriminator_result_fake']
    return ((discriminator_result_fake - 1.0) ** 2).mean()


def GeneratorLossL1(**kwargs):
    fake_img = kwargs['fake_img']
    real_img = kwargs['real_img']
    return jittor.nn.l1_loss(fake_img, real_img)


def GeneratorLossBCE(**kwargs):
    discriminator_result_fake = kwargs['discriminator_result_fake']
    return nn.binary_cross_entropy_with_logits(discriminator_result_fake, 1)


def DiscriminatorLossLS(**kwargs):
    discriminator_result_fake = kwargs['discriminator_result_fake']
    discriminator_result_real = kwargs['discriminator_result_real']
    return ((discriminator_result_fake - (-1.0)) ** 2).mean() + ((discriminator_result_real - 1.0) ** 2).mean()


def DiscriminatorLossBCE(**kwargs):
    discriminator_result_fake = kwargs['discriminator_result_fake']
    discriminator_result_real = kwargs['discriminator_result_real']
    return nn.binary_cross_entropy_with_logits(discriminator_result_fake, 0) + \
           nn.binary_cross_entropy_with_logits(discriminator_result_real, 1)


def build_loss(config):
    assert "generator" in config
    assert "discriminator" in config

    losses = {
        "generator": [],
        "discriminator": []
    }

    # Build loss for Generator
    for entry in config['generator']:
        if entry['type'] == 'ls':
            losses['generator'].append([entry['weight'], GeneratorLossLS])
        elif entry['type'] == 'l1':
            losses['generator'].append((entry['weight'], GeneratorLossL1))
        elif entry['type'] == 'bce':
            losses['generator'].append((entry['weight'], GeneratorLossBCE))
        else:
            raise NotImplementedError(f"Unknown loss type: {entry['type']}")

    # Build loss for Discriminator
    for entry in config['discriminator']:
        if entry['type'] == 'ls':
            losses['discriminator'].append((entry['weight'], DiscriminatorLossLS))
        elif entry['type'] == 'bce':
            losses['discriminator'].append((entry['weight'], DiscriminatorLossBCE))
        elif entry['type'] == 'gp':
            losses['discriminator'].append((entry['weight'], DiscriminatorLossGradientPenalty))
        else:
            raise NotImplementedError(f"Unknown loss type: {entry['type']}")

    return losses


def calc_loss(loss_fn, type, **kwargs):
    loss = 0

    if type == "G":
        for (weight, func) in loss_fn['generator']:
            loss += weight * func(**kwargs)
    elif type == "D":
        for (weight, func) in loss_fn['discriminator']:
            loss += weight * func(**kwargs)
    else:
        raise NotImplementedError(f"Unknown loss type: {type}")

    return loss
