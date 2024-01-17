
# https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8/edit#gid=0

import os
import glob
import numpy as np
import torch
from torch.utils import data
import json
import torchvision.transforms as transforms
from PIL import Image
import PIL
import random
from dataloader.custom_transform import *
#from semseg.utils.mask_gen import BoxMaskGenerator
from natsort import natsorted
from einops import rearrange
from pathlib import Path


class ImageFolder(data.Dataset):
    def __init__(self, image_dir,
                 use_crop=False,
                 crop_height=512, crop_width=512,
                 crop_ratio=1,
                 img_height=512, img_width=512,augment_dict=None,
                 *args, **kwargs):

        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width) if crop_width is not None else int(crop_height * crop_ratio)
        self.img_height = img_height
        self.img_width = img_width
        self.random_crop = False
        self.center_crop = False
        self.use_crop = use_crop
        if use_crop:
            if (self.img_width != self.crop_width or self.img_height != self.crop_height):
                self.random_crop = True
            if augment_dict.get('center_crop', False) and \
                    (self.img_width != self.crop_width or self.img_height != self.crop_height):
                self.random_crop = False
                self.center_crop = True

            print(f'---> Resize h x w: {self.img_height} x {self.img_width}')
            print(f'---> Crop h x w: {self.crop_height} x {self.crop_width}')

        # Get image directory
        impths = glob.glob(os.path.join(image_dir, '*.png'))
        self.img_names  = natsorted(impths)

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return  len(self.img_names)

    def __getitem__(self, idx):
        data = {}
        img_name = self.img_names[idx]
        img = Image.open(img_name).convert('RGB')
        img = self.to_tensor(img.copy())
        img = img * 2.0 - 1.0
        data['image'] = img  # Tensor: [3,h,w], (-1,1)
        data['img_pth'] = img_name  # for debugging
        return data


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pylab as plt

    augment_dict = {
        'augment_p': -1,
        'horizontal_flip': False,
    }
    mode = 'val'
    train_mode = True
    shuffle = False
    image_dir = '/fs/scratch/rng_cr_bcai_dl/lyu7rng/0_project_large_models/code_repo/0_ControlNet/z_img_generation/shared_ade_006_33799_epoch13'
    ds = ImageFolder(image_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

    show_num = 5
    for i, data in enumerate(dl):
        if i < show_num:
            img = data['image']
            impth = data['img_pth']
            # print(impth)
            #print(img.shape, label.shape) #[4, 3, 512, 512]
            print(impth[0])
        else:
            break