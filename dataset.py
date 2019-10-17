""" NOTICE: A Custom Dataset SHOULD BE PROVIDED
Created: May 02,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import inception_preprocessing
from torch.utils.data import Dataset

__all__ = ['CustomDataset']


config = {
    # e.g. train/val/test set should be located in os.path.join(config['datapath'], 'train/val/test')
    'datapath': '/data/shaozl/WS-DAN.PyTorch/dataset',
}

class My_transform(object):
    def __call__(self, img):
        return add_and_mul(img)

def add_and_mul(image):
    image = image-0.5
    image = image*2.0
    return image


class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', shape=(512, 512)):
        self.create_lable_map()
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.data_path = os.path.join(config['datapath'], phase)
        self.data_list = os.listdir(self.data_path)

        self.shape = shape
        self.config = config

        if self.phase=='train':
            self.transform = transforms.Compose([
                transforms.Resize(size=(int(self.shape[0]*1.0/0.875), int(self.shape[1]*1.0/0.875))),
                transforms.RandomCrop((self.shape[0], self.shape[1])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.125, contrast=0.5),
                transforms.ToTensor(),
                My_transform()
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.4729, 0.4871, 0.4217], std=[0.1589, 0.1598, 0.1681])
                # tensor([0.4856, 0.4994, 0.4324]) tensor([0.1784, 0.1778, 0.1895]) 没有增强的mean和std
                # tensor([0.4729, 0.4871, 0.4217]) tensor([0.1589, 0.1598, 0.1681]) 有增强的mean和std
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(size=(self.shape[0], self.shape[1])),
                transforms.Resize(size=(int(self.shape[0] * 1.0 / 0.875), int(self.shape[1] * 1.0 / 0.875))),
                transforms.CenterCrop((self.shape[0], self.shape[1])),
                transforms.ToTensor(),
                My_transform()
                # transforms.Normalize(mean=[0.4862, 0.4998, 0.4311], std=[0.1796, 0.1781, 0.1904])
                # tensor([0.4862, 0.4998, 0.4311]) tensor([0.1796, 0.1781, 0.1904]) 没有增强的mean和std
            ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.data_path, self.data_list[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]

        if self.phase != 'test':
            # filename of image should have 'id_label.jpg/png' form
            label = int(self.class_name.index(self.data_list[item].rsplit('_',2)[0].lower()))  # label
            return image, label
        else:
            # filename of image should have 'id.jpg/png' form, and simply return filename in case of 'test'
            return image, self.data_list[item]

    def __len__(self):
        return len(self.data_list)

    def create_lable_map(self):
        with open('/data/shaozl/WS-DAN.PyTorch/CUB_200_2011/classes.txt') as f:
            class_index_and_name = f.readlines()
            class_index_and_name = [i.strip().lower() for i in class_index_and_name]
            self.class_index = [i.split(' ', 1)[0] for i in class_index_and_name]
            self.class_name = [i.split(' ', 1)[1].split('.',1)[1] for i in class_index_and_name]

