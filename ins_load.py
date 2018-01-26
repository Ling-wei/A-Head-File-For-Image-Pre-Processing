"""
load oxford or paris database
"""

import os
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_raw_transformer():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_scale_transformer(size):
    return transforms.Compose([
            transforms.Scale((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    try:
        img = Image.open(filepath).convert('RGB')
        return img
    except Exception as e:
        print(e)
        return None


"""
the class to load images with filenames
"""
class ImageFolderDataSets(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform):
        self.image_dir = image_dir
        self.image_names = [x for x in os.listdir(image_dir)]
        self.input_transform = input_transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.image_dir, image_name)
        image = load_img(path)
        if image is not None and self.input_transform:
            image = self.input_transform(image)
        if image is not None:
            return image_name, image
        else:
            return [], 0

    def __len__(self):
        return len(self.image_names)


"""
define some different loaders
"""
def resize_scale_loader(image_dir, scale_size, batch_size=1, is_shuffle=False, num_workers=4):
    tran = get_scale_transformer(scale_size)
    dataset = ImageFolderDataSets(image_dir, tran)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
        shuffle=is_shuffle, num_workers=num_workers)


def raw_scale_loader(image_dir, batch_size=1, is_shuffle=False, num_workers=4):
    tran = get_raw_transformer()
    dataset = ImageFolderDataSets(image_dir, tran)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
        shuffle=is_shuffle, num_workers=num_workers)




