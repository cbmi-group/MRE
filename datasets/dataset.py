import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFilter

from data_process.pre_process import *


def img_PreProc_mito(img, pro_type):
    if pro_type == "clahe":
        img = img_clahe(img)
        return img / 65535.

    elif pro_type == "invert":
        img = 65535 - img
        return img / 65535.

    elif pro_type == "edgeEnhance":
        edge = sober_filter(img)
        edge = edge / np.max(edge)
        return ((img / 65535.) + edge) * 0.5

    elif pro_type == "norm":
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img

    elif pro_type == "clahe_norm":
        img = img_clahe(img)
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img


def img_PreProc_er(img, pro_type):
    if pro_type == "clahe":
        img = img_clahe(img)
        return img / 65535.

    elif pro_type == "invert":
        img = 65535 - img
        return img / 65535.

    elif pro_type == "edgeEnhance":
        edge = sober_filter(img)
        edge = edge / np.max(edge)
        return ((img / 65535.) + edge) * 0.5

    elif pro_type == "norm":
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img

    elif pro_type == "clahe_norm":
        img = img_clahe(img)
        img = img / 65535.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img


def img_PreProc_retina(img, pro_type):
    if pro_type == "clahe":
        img = img_clahe(img)
        return img / 255.

    elif pro_type == "invert":
        img = 255 - img
        return img / 255.

    elif pro_type == "edgeEnhance":
        edge = sober_filter(img)
        edge = edge / np.max(edge)
        return ((img / 255.) + edge) * 0.5

    elif pro_type == "norm":
        img = img / 255.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img

    elif pro_type == "clahe_norm":
        img = img_clahe(img)
        img = img / 255.
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        return img


class Seg_Dataset(Dataset):
    def __init__(self, txt, dataset_type, train,
                 img_size=256
                 ):

        self.img_size = img_size
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.train = train

        with open(txt, "r") as fid:
            lines = fid.readlines()

        img_mask_paths = []
        for line in lines:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(" ")
            img_mask_paths.append((words[0], words[1]))

        self.img_mask_paths = img_mask_paths

    def __getitem__(self, index):
        img_path, mask_path = self.img_mask_paths[index]

        # initialize input
        if self.dataset_type == 'er':
            img = cv2.imread(img_path, -1)
            mask = cv2.imread(mask_path, -1)
            img_ = img_PreProc_er(img, pro_type='clahe')

            img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": img_,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}


        elif self.dataset_type == 'retina':
            img = cv2.imread(img_path, -1)
            mask = cv2.imread(mask_path, -1)
            img_ = img_PreProc_retina(img, pro_type='clahe')

            img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": img_,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}


        elif self.dataset_type == 'stare':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img_ = img_PreProc_retina(img, pro_type='clahe')

            img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": img_,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}

        elif self.dataset_type == 'mito':
            img = cv2.imread(img_path, -1)
            mask = cv2.imread(mask_path, -1)
            img_ = img_PreProc_mito(img, pro_type='clahe_norm')

            img_ = torch.from_numpy(img_).unsqueeze(dim=0).float()
            mask_ = torch.from_numpy(mask / 255.).unsqueeze_(dim=0).float()

            sample = {"image": img_,
                      "mask": mask_,
                      "ID": os.path.split(img_path)[1]}

        return sample

    def __len__(self):
        return len(self.img_mask_paths)


def data_loader(data_list, batch_size, dataset_type, is_train=True):
    train_data = Seg_Dataset(txt=data_list, dataset_type=dataset_type, train=is_train)

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=is_train, num_workers=16)

    return data_loader
