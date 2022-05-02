import os
import sys
import time
from tqdm import tqdm
import datetime
import jittor as jt
import jittor.nn as nn


from models.utils.utils import start_grad, stop_grad, weights_init_normal


def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, args):
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixelwise = nn.L1Loss()

    for epoch in range(args.epoch, args.n_epochs):

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (real_B, real_A, _) in loop: # real_A is label, real_B is image
            # Train Discriminator
            start_grad(discriminator)
            stop_grad(generator)

            fake_B = generator(real_A)
            fake_AB = jt.contrib.concat((real_A, fake_B), 1)
            pred_fake = discriminator(fake_AB.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)

            real_AB = jt.contrib.concat((real_A, real_B), 1)
            pred_real = discriminator(real_AB)
            loss_D_real = criterion_GAN(pred_real, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            optimizer_D.step(loss_D)

            # Train Generators
            start_grad(generator)
            stop_grad(discriminator)

            fake_B = generator(real_A)
            fake_AB = jt.contrib.concat((real_A, fake_B), 1)
            pred_fake = discriminator(fake_AB)
            loss_G_GAN = criterion_GAN(pred_fake, True)
            loss_G_L1 = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_G_GAN + args.lambda_pixel * loss_G_L1
            optimizer_G.step(loss_G)

            if i % 5 == 0:
                loop.set_description(f'[{args.task_name}] [Epoch {epoch + 1}/{args.n_epochs}] [Batch {i}/{len(dataloader)}]')
                loop.set_postfix(
                    lossD=loss_D.numpy()[0],
                    lossG=loss_G.numpy()[0],
                    loss_G_L1=loss_G_L1.numpy()[0],
                    loss_G_GAN=loss_G_GAN.numpy()[0]
                )

        # End of epoch
        if (epoch + 1) % args.checkpoint_interval == 0:
            # eval(epoch, writer)
            # Save model checkpoints
            generator.save(os.path.join(f"{args.output_path}/saved_models/generator_{epoch}.pkl"))
            generator.save(os.path.join(f"{args.output_path}/generator.pkl"))
            discriminator.save(os.path.join(f"{args.output_path}/saved_models/discriminator_{epoch}.pkl"))
            discriminator.save(os.path.join(f"{args.output_path}/discriminator.pkl"))
