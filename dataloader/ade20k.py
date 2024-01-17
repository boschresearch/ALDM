

import os
import glob
import numpy as np
import torch
from torch.utils import data
import json
import torchvision.transforms as transforms
from PIL import Image
import PIL
from dataloader.custom_transform import *
from natsort import natsorted
from einops import rearrange
from pathlib import Path

class ADE20KBaseInfo():
    def __init__(self):
        self.num_classes = 150
        self.ignore_label = 255 #origianlly it's 0, replaced by 255 in the dataloader
        self.label_names = self.get_class_name()
        self.colormap = self.create_label_colormap()
        label_map = [*range(159)]
        self.name_to_id_dict = dict(zip(self.label_names, label_map))
        self.id_to_name_dict = dict(zip(label_map, self.label_names))

    @staticmethod
    def get_class_name():
        return np.array([
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
        'window', 'grass', 'cabinet', 'sidewalk', 'person', 'ground',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'carpet',
        'field', 'armchair', 'seat', 'fence', 'desk', 'stone', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'pedestal', 'box', 'pillar',
        'signboard', 'dresser', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blinds', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television',
        'airplane', 'dirt road', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'pouf', 'bottle', 'sideboard', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'motorbike', 'cradle', 'oven', 'ball', 'food', 'stair', 'tank',
        'brand', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag'
        ])

    @staticmethod
    def create_label_colormap():
        return np.array([
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255], [0,0,0] # fake class
        ], dtype=np.uint8)


class ADE20K(data.Dataset):
    def __init__(self, mode, train_mode, crop_height=512, crop_width=512, crop_ratio=1,
                 img_height=512, img_width=512,augment_dict=None,
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

        self.color_map = ADE20KBaseInfo.create_label_colormap()

        # Data Augmentation
        self.augment_dict = augment_dict
        if train_mode:
            self.augment_p = augment_dict['augment_p']

        # get image list
        self.datapth = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADE20K/ADEChallengeData2016/annotations' # TODO: change path
        self.rgb_path = '/fs/scratch/rng_cr_bcai_dl_students/OpenData/ADE20K/ADEChallengeData2016/images' # TODO: change path

        # Read captions
        if caption_json is None:
            cur_path = os.path.dirname(__file__)
            caption_json = os.path.join(cur_path, f'ade20k_caption_{mode}.json')
        with open(caption_json, 'r') as json_file:
            self.caption_dict = json.load(json_file)

        # Get image directory
        self.img_name_dir_dict = {}
        impth = os.path.join(self.rgb_path, mode)
        impths = glob.glob(os.path.join(impth, '*.jpg'))
        impths = natsorted(impths)

        self.img_names = []
        for img_name in impths:
            temp_name = Path(img_name).stem
            self.img_names.append(temp_name)
        self.img_name_dir_dict.update(dict(zip(self.img_names, impths)))

        # Get gt directory
        self.labels_name_dir_dict = {}
        gtpth = os.path.join(self.datapth, mode)
        gtpths = glob.glob(os.path.join(gtpth, '*.png'))
        gtpths = natsorted(gtpths)
        image_names = []
        for lbname in gtpths:
            temp_name = Path(lbname).stem
            image_names.append(temp_name)
        self.labels_name_dir_dict.update(dict(zip(image_names, gtpths)))

        self.len = len(self.img_names)
        print(f'Total ADE20K {mode} images: {self.len}')
        assert set(self.img_name_dir_dict.keys()) == set(self.labels_name_dir_dict.keys())

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        label = label - 1
        label[label == -1] = 255
        return label

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
        label = self.convert_labels(label)
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
        label_[label_ == 255] = 150
        label_ = self.color_map[label_]
        label_ = rearrange(label_, 'h w c -> c h w')
        label_ = torch.from_numpy(label_)
        # TODO: tutorial dataset uses Normalization to (0,1) for the condition
        label_ = label_ / 255.0  # * 2 - 1
        return label_

    def label_encode_id(self,label):
        # return label: Tensor [1,h,w]
        label_ = np.copy(label)
        label_[label_ == 255] = 150
        label_ = torch.from_numpy(label_)
        label_ = label_.unsqueeze(0)
        return label_

    def __getitem__(self, idx):
        data = {}
        img_name = self.img_names[idx]
        prompt = self.caption_dict[img_name]
        if self.drop_caption:
            if np.random.rand(1) < self.drop_caption_ratio:
                prompt = ""

        impth, lbpth = self.get_img_label_pth(idx)
        img, label = self.read_from_dir_return_tensor(impth, lbpth)

        if self.train_mode:
            if self.augment_dict['horizontal_flip']:
                img, label = random_horizontal_flip(img, label, prob=self.augment_p)

        # Encode labels
        # current label: Tensor [512,512], 0-18 + 255 label id
        if self.mask_encode_mode == 'color':
            label_coded = self.label_encode_color(label)  # Tensor [3,h,w], (0,1)
        elif self.mask_encode_mode == 'id':
            label_coded = self.label_encode_id(label)  # Tensor [1,h,w]
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

    ds = ADE20K(mode=mode, train_mode=train_mode, augment_dict=augment_dict,
                mask_encode_mode='color',drop_caption_ratio=0.99)
    dl = DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=4, drop_last=True)

    cs_info = ADE20KBaseInfo()
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


    show_num = 5
    for i, data in enumerate(dl):
        if i < show_num:
            img = data['image']
            label = data['hint']
            text = data['txt']
            # impth = data['img_pth']
            # print(impth)
            #print(img.shape, label.shape) #[4, 3, 512, 512]
            print(text[0])
            visualization(img[0], label[0])
            unique_ids =data['label'][0].unique()
            label_names = []
            for k in unique_ids:
                if k < 150:
                    label_names.append(class_names[k])
            print(unique_ids)
            print(label_names)
        else:
            break