# COCO-Stuff Dataset


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
# from semseg.utils.mask_gen import BoxMaskGenerator
from natsort import natsorted
from einops import rearrange
from pathlib import Path


class COCOStuffBaseInfo():
    def __init__(self):
        self.num_classes = 171
        self.ignore_label = 255  # origianlly it's 0, replaced by 255 in the dataloader
        self.label_names = self.get_class_name()
        self.colormap = self.create_label_colormap()
        # self.full_label_id = np.arange(len(self.label_names)).reshape(len(self.label_names), 1)
        label_map = [*range(172)]
        self.name_to_id_dict = dict(zip(self.label_names, label_map))
        self.id_to_name_dict = dict(zip(label_map, self.label_names))

    @staticmethod
    def get_class_name():
        # Some words may differ from the class names defined in ADE20K to minimize ambiguity
        return np.array(
            ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building', 'bush',
             'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds',
             'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor', 'floor', 'stone floor',
             'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel', 'ground',
             'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net',
             'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform', 'playingfield', 'railing', 'railroad',
             'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow',
             'solid', 'stairs', 'stone', 'straw', 'structural', 'table', 'tent', 'textile', 'towel', 'tree',
             'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall', 'stone wall', 'tile wall', 'wood wall',
             'water', 'waterdrops', 'blind window', 'window', 'wood'])

    @staticmethod
    def create_label_colormap():
        return np.array([
            [220, 20, 60], [119, 11, 32], [0, 0, 142], [0, 0, 230], [106, 0, 228], [0, 60, 100], [0, 80, 100],
            [0, 0, 70], [0, 0, 192], [250, 170, 30], [100, 170, 30], [220, 220, 0], [175, 116, 175], [250, 0, 30],
            [165, 42, 42], [255, 77, 255], [0, 226, 252], [182, 182, 255], [0, 82, 0], [120, 166, 157], [110, 76, 0],
            [174, 57, 255], [199, 100, 0], [72, 0, 118], [255, 179, 240], [0, 125, 92], [209, 0, 151], [188, 208, 182],
            [0, 220, 176], [255, 99, 164], [92, 0, 73], [133, 129, 255], [78, 180, 255], [0, 228, 0], [174, 255, 243],
            [45, 89, 255], [134, 134, 103], [145, 148, 174], [255, 208, 186], [197, 226, 255], [171, 134, 1],
            [109, 63, 54], [207, 138, 255], [151, 0, 95], [9, 80, 61], [84, 105, 51], [74, 65, 105], [166, 196, 102],
            [208, 195, 210], [255, 109, 65], [0, 143, 149], [179, 0, 194], [209, 99, 106], [5, 121, 0], [227, 255, 205],
            [147, 186, 208], [153, 69, 1], [3, 95, 161], [163, 255, 0], [119, 0, 170], [0, 182, 199], [0, 165, 120],
            [183, 130, 88], [95, 32, 0], [130, 114, 135], [110, 129, 133], [166, 74, 118], [219, 142, 185],
            [79, 210, 114], [178, 90, 62], [65, 70, 15], [127, 167, 115], [59, 105, 106], [142, 108, 45], [196, 172, 0],
            [95, 54, 80], [128, 76, 255], [201, 57, 1], [246, 0, 122], [191, 162, 208], [255, 255, 128],
            [147, 211, 203], [192, 96, 128], [150, 100, 100], [64, 0, 96], [64, 224, 128], [134, 199, 156],
            [192, 0, 224], [168, 171, 172], [128, 192, 128], [146, 139, 141], [192, 224, 128], [128, 192, 64],
            [192, 0, 96], [192, 96, 0], [146, 112, 198], [0, 128, 96], [210, 170, 100], [64, 64, 64], [208, 229, 228],
            [92, 136, 89], [190, 153, 153], [0, 128, 224], [128, 224, 0], [64, 192, 64], [128, 128, 96], [218, 88, 184],
            [241, 129, 0], [0, 64, 96], [0, 160, 128], [217, 17, 255], [128, 64, 224], [152, 251, 152], [124, 74, 181],
            [0, 64, 224], [128, 160, 128], [70, 70, 70], [128, 64, 32], [255, 228, 255], [192, 0, 128], [64, 192, 32],
            [154, 208, 0], [64, 0, 0], [64, 170, 64], [0, 32, 64], [64, 128, 128], [193, 0, 92], [206, 186, 171],
            [96, 96, 96], [76, 91, 113], [128, 96, 192], [64, 0, 128], [255, 180, 195], [106, 154, 176], [192, 0, 0],
            [230, 150, 140], [60, 143, 255], [128, 64, 128], [0, 114, 143], [92, 82, 55], [250, 141, 255],
            [192, 64, 32], [254, 212, 124], [73, 77, 174], [255, 160, 98], [64, 224, 64], [64, 0, 64], [255, 255, 255],
            [64, 96, 64], [104, 84, 109], [0, 192, 160], [192, 224, 64], [64, 128, 64], [209, 226, 140],
            [169, 164, 131], [64, 64, 192], [225, 199, 255], [107, 142, 35], [192, 64, 64], [137, 54, 74],
            [64, 32, 192], [192, 192, 192], [0, 64, 160], [135, 158, 223], [7, 246, 231], [107, 255, 200],
            [58, 41, 149], [192, 64, 128], [183, 121, 142], [255, 73, 97], [64, 64, 0], [0, 0, 0, ]  # void
        ], dtype=np.uint8)


# TODO: just copied from ADE20K
class COCO_Stuff(data.Dataset):
    def __init__(self, mode, train_mode, crop_height=512, crop_width=512, crop_ratio=1,
                 img_height=512, img_width=512, augment_dict=None,
                 mask_encode_mode='color',
                 caption_json=None,
                 drop_caption_ratio=-1.0,
                 *args, **kwargs):
        assert mode in ('train', 'val')
        assert mask_encode_mode in ('color', 'id')

        self.train_mode = train_mode
        self.mask_encode_mode = mask_encode_mode
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width) if crop_width is not None else int(crop_height * crop_ratio)
        self.img_height = img_height
        self.img_width = img_width
        self.random_crop = False
        self.center_crop = False

        if (self.img_width != self.crop_width or self.img_height != self.crop_height) and self.train_mode:
            self.random_crop = True
        if augment_dict.get('center_crop', False) and \
                (self.img_width != self.crop_width or self.img_height != self.crop_height) and \
                self.train_mode:
            self.random_crop = False
            self.center_crop = True

        print(f'---> Resize h x w: {self.img_height} x {self.img_width}')
        print(f'---> Crop h x w: {self.crop_height} x {self.crop_width}')

        self.drop_caption_ratio = drop_caption_ratio
        if self.drop_caption_ratio > 0:
            self.drop_caption = True
        else:
            self.drop_caption = True
        # if mask_encode_mode == 'color':
        self.color_map = COCOStuffBaseInfo.create_label_colormap()

        # Data Augmentation
        self.augment_dict = augment_dict
        if train_mode:
            self.augment_p = augment_dict['augment_p']

        # get image list
        self.datapth = f'/fs/scratch/rng_cr_bcai_dl_students/OpenData/LOCKED/coco/cocostuff/{mode}_label_convert' # Change path here!
        self.rgb_path = f'/fs/scratch/rng_cr_bcai_dl_students/OpenData/LOCKED/coco/cocostuff/{mode}_img' # Change path here!

        # TODO: Read captions
        if caption_json is None: 
            cur_path = os.path.dirname(__file__)
            caption_json = os.path.join(cur_path, f'coco_caption_{mode}.json')
        with open(caption_json, 'r') as json_file:
            self.caption_dict = json.load(json_file)

        # Get image directory
        self.img_name_dir_dict = {}
        impth = self.rgb_path
        impths = glob.glob(os.path.join(impth, '*.jpg'))
        impths = natsorted(impths)

        self.img_names = []
        for img_name in impths:
            temp_name = Path(img_name).stem
            self.img_names.append(temp_name)
        self.img_name_dir_dict.update(dict(zip(self.img_names, impths)))

        # Get gt directory
        self.labels_name_dir_dict = {}
        gtpth = self.datapth
        gtpths = glob.glob(os.path.join(gtpth, '*.png'))
        gtpths = natsorted(gtpths)
        image_names = []
        for lbname in gtpths:
            temp_name = Path(lbname).stem
            image_names.append(temp_name)
        self.labels_name_dir_dict.update(dict(zip(image_names, gtpths)))

        self.len = len(self.img_names)
        print(f'Total COCO-Stuff {mode} images: {self.len}')  # 118287
        # assert set(self.img_names) == set(image_names)
        assert set(self.img_name_dir_dict.keys()) == set(self.labels_name_dir_dict.keys())

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return self.len

    def read_from_dir_return_tensor(self, impth, lbpth):
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        img = img.resize((self.img_width, self.img_height), PIL.Image.LANCZOS)
        label = label.resize((self.img_width, self.img_height), PIL.Image.NEAREST)

        if self.random_crop:
            img, label = random_crop(img, label, self.crop_width, self.crop_height)
        if self.center_crop:
            img, label = center_crop(img, label, self.crop_width, self.crop_height)

        label = np.array(label).astype(np.int64)  # [np.newaxis, :]
        img = self.to_tensor(img.copy())
        img = img * 2.0 - 1.0
        label = torch.LongTensor(label)
        return img, label

    def get_img_label_pth(self, idx):
        img_name = self.img_names[idx]
        lbpth = self.labels_name_dir_dict[img_name]
        impth = self.img_name_dir_dict[img_name]
        return impth, lbpth

    def label_encode_color(self, label):
        # encode the mask using color coding
        # return label: Tensor [3,h,w], (-1,1)
        label_ = np.copy(label)
        label_[label_ == 255] = 171
        label_ = self.color_map[label_]
        label_ = rearrange(label_, 'h w c -> c h w')
        label_ = torch.from_numpy(label_)
        # TODO: tutorial dataset uses Normalization to (0,1) for the condition
        label_ = label_ / 255.0  # * 2 - 1
        return label_

    def label_encode_id(self, label):
        # return label: Tensor [1,h,w]
        label_ = np.copy(label)
        label_[label_ == 255] = 171
        label_ = torch.from_numpy(label_)
        label_ = label_.unsqueeze(0)
        return label_

    def __getitem__(self, idx):
        data = {}
        img_name = self.img_names[idx]
        # print(img_name) # only for debugging! #aachen_000008_000019
        # TODO: to read
        all_captions = self.caption_dict[img_name]
        prompt = random.choice(all_captions)
        if self.drop_caption:
            if np.random.rand(1) < self.drop_caption_ratio:
                prompt = ""

        impth, lbpth = self.get_img_label_pth(idx)
        img, label = self.read_from_dir_return_tensor(impth, lbpth)

        if self.train_mode:
            if self.augment_dict['horizontal_flip']:
                img, label = random_horizontal_flip(img, label, prob=self.augment_p)

        # Encode labels
        # current label: Tensor [512,512], 0-170 + 255 label id
        if self.mask_encode_mode == 'color':
            label_coded = self.label_encode_color(label)  # Tensor [3,h,w], (0,1)
        elif self.mask_encode_mode == 'id':
            label_coded = self.label_encode_id(label)  # Tensor [1,h,w] # map invalid to 171 which is black
        label = label.unsqueeze(0)

        data['image'] = img  # Tensor: [3,h,w], (-1,1)
        data['hint'] = label_coded
        data['txt'] = prompt
        data['label'] = label  # Tensor [1,512,512], 0-149 + 255 label id
        data['img_pth'] = impth  # for debugging
        data['label_pth'] = lbpth  # for generation

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

    ds = COCO_Stuff(mode=mode, train_mode=train_mode, augment_dict=augment_dict,
                    mask_encode_mode='color', drop_caption_ratio=0.)
    dl = DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

    cs_info = COCOStuffBaseInfo()
    class_names = cs_info.label_names
    print(class_names)


    def rearrange_rescale(x, rescale=True):
        x = x.cpu().detach().numpy()
        x = rearrange(x, 'c h w -> h w c')
        if rescale:
            x = (x + 1) * 0.5  # (0,1)
        return x


    def visualization(img_, label_):
        img = rearrange_rescale(img_)
        label = rearrange_rescale(label_, rescale=False)

        fig, axes = plt.subplots(1, 3, dpi=120, figsize=(12, 4))

        # image
        axes[0].imshow(img)
        axes[0].axis('off')

        # label
        axes[1].imshow(label)
        axes[1].axis('off')

        # overlay
        axes[2].imshow(img)
        axes[2].imshow(label, alpha=0.7)
        axes[2].axis('off')

        plt.show()


    show_num = 10
    for i, data in enumerate(dl):
        if i < show_num:
            img = data['image']
            label = data['hint']
            text = data['txt']
            # impth = data['img_pth']
            # print(impth)
            # print(img.shape, label.shape) #[4, 3, 512, 512]
            print(text[0])
            visualization(img[0], label[0])
            unique_ids = data['label'][0].unique()
            label_names = []
            for k in unique_ids:
                if k < 171:
                    label_names.append(class_names[k])
            print(unique_ids)
            print(label_names)
        else:
            break
