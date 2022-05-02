import argparse
from pathlib import Path

import jittor as jt

import jittor.transform as transform
from PIL import Image
from datasets import ImageDataset
from models import get_model
from jittor import nn

from models.generator.UNet import UnetGenerator
from tools.predict_tools import predict
from tools.training_tools import train


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='GAN! - Landscape')
    parser.add_argument("--task_name", type=str, default="baseline", help="task name")
    parser.add_argument('--mode', type=str, default="eval", help='Mode: train or eval')
    parser.add_argument("--data_path", type=str, default="/root/landscape/data")
    parser.add_argument("--img_width", type=int, default=512, help="size of image width")
    parser.add_argument("--img_height", type=int, default=384, help="size of image height")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    # Training Arguments
    parser.add_argument('--epoch', type=int, default=0, help='Current epoch ID')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_pixel", type=float, default=10, help="loss: lambda_pixel")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")

    # Evaluation Arguments
    parser.add_argument('--model_path', type=str, default="", help='Path to generator model')
    parser.add_argument('--image_path', type=str, default="", help='Path to image')
    parser.add_argument('--output_path', type=str, default="/root/landscape/results", help='Path to output')

    args = parser.parse_args()
    args.__dict__['output_path'] = Path(args.output_path) / args.task_name

    jt.flags.use_cuda = 1

    # Load Datasets
    transforms = [
        transform.Resize(size=(args.img_height, args.img_width), mode=Image.BICUBIC),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    dataloader = ImageDataset(args.data_path, mode="train", transforms=transforms).set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_cpu,
    )

    val_dataloader = ImageDataset(args.data_path, mode="test", transforms=transforms).set_attrs(
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )

    if args.mode == "train":
        # Load Generator and Discriminator
        generator, discriminator = get_model(args)

        if args.epoch != 0:
            generator.load(f"{args.output_path}/generator.pkl")
            discriminator.load(f"{args.output_path}/discriminator.pkl")

        # Optimizers
        optimizer_G = jt.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        # TODO: Call training function
        train(generator, discriminator, dataloader, optimizer_G, optimizer_D, args)


    elif args.mode == "eval":
        # Load Generator and Discriminator
        generator, discriminator = get_model(args)
        generator.load(f"{args.output_path}/generator.pkl")

        predict(generator, val_dataloader, args)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()