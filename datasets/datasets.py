import glob
import random
import os

import IPython
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

# Default transforms
transforms = transform.Compose((
    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
))


class ImageDataset(Dataset):

    def __init__(self, root, mode="train"):
        super().__init__()
        self.transforms = transforms
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
            self.labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
        else:
            self.labels = sorted(glob.glob(os.path.join(root, "test") + "/*.*"))

        self.set_attrs(total_len=len(self.labels))
        print(f"[Dataset] {self.total_len} images loaded from {mode}.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]  # filename remove .png
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))  # 3 channel

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            # Commented out by c7w: what if we do not take data augmentation into account?
            # Answer: VERY BAD GENERATION :( Commented In Back
            if np.random.random() < 0.5:  # random flip
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms(img_A)
        else:
            img_A = np.empty([1])

        img_B = self.transforms(img_B)

        return img_A, img_B, photo_id  # img_A is the original image, img_B is the label


