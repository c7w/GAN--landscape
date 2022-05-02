import argparse
from tools.

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='GAN! - Landscape')
    parser.add_argument('--mode', type=str, default="eval", help='Mode: train or eval')

    # Training Arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')


    # Evaluation Arguments
    parser.add_argument('--model_path', type=str, default="", help='Path to generator model')
    parser.add_argument('--image_path', type=str, default="", help='Path to image')
    parser.add_argument('--output_path', type=str, default="", help='Path to output')

    args = parser.parse_args()

    if args.mode == "train":
        raise NotImplementedError
    elif args.mode == "eval":



if __name__ == "__main__":
    main()