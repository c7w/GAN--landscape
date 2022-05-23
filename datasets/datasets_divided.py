import glob
import random
import os

import IPython
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

# Default transforms
transforms_img = transform.Compose([
    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transforms_label = transform.Compose([
    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
    transform.ToTensor(),
])

class ImageDatasetDivided:
    def __init__(self, root, shuffle=False):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(root, "train", "imgs") + "/*.*"))
        self.labels = sorted(glob.glob(os.path.join(root, "train", "labels") + "/*.*"))

        self.file2label = {k: v for k, v in zip(self.files, self.labels)}

        self.file_label_pair = list(zip(self.files, self.labels))
        if shuffle:
            random.shuffle(self.file_label_pair)

        divide_idx = int(len(self.file_label_pair) * 0.95)

        train_file_label = dict(self.file_label_pair[:divide_idx])
        valid_file_label = dict(self.file_label_pair[divide_idx:])

        self.train_dataset = ImageDatasetFromList(list(train_file_label.keys()), list(train_file_label.values()))
        self.valid_dataset = ImageDatasetFromList(list(valid_file_label.keys()), list(valid_file_label.values()))


class ImageDatasetFromList(Dataset):

    def __init__(self, file_list, label_list, ):
        super().__init__()
        self.files = file_list
        self.labels = label_list
        self.set_attrs(total_len=len(self.labels))

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]  # filename remove .png

        img_A = Image.open(self.files[index % len(self.files)])
        img_A = transforms_img(img_A)

        img_B = Image.open(label_path)
        img_B = transforms_label(img_B)
        img_B = np.array(img_B)[0][:, :, np.newaxis].repeat(3, 2)  # 3 channel


        if np.random.random() < 0.5:  # random flip
            img_A = np.array(img_A)[:, ::-1, :]
            img_B = np.array(img_B)[:, ::-1, :]

        return img_A, img_B, photo_id  # img_A is the original image, img_B is the label
