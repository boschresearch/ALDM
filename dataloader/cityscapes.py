
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


class CityscapesBaseInfo():
    def __init__(self):
        self.num_classes = 19
        self.ignore_label = 255
        self.label_names = self.get_class_name(version='full')
        self.label_names_short = self.get_class_name(version='short')
        self.sorted_name = self.get_class_name(version='sorted')
        self.colormap = self.create_label_colormap()
        label_map = [*range(19)]
        self.name_to_id_dict = dict(zip(self.label_names, label_map))
        self.id_to_name_dict = dict(zip(label_map, self.label_names))

    @staticmethod
    def get_class_name(version='full'):
        if version=='full':
            label_names = np.asarray([
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                'bus', 'train', 'motorcycle', 'bicycle', 'void'
            ])
        elif version =='short':
            label_names = np.asarray([
                'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'tr. light',
                'tr. sign', 'veget.', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
                'bus', 'train', 'motorb.', 'bicycle', 'void'
            ])
        elif version == 'sorted': # sorted by frequency
            label_names = np.asarray([
                'road', 'building', 'vegetation', 'car', 'sidewalk', 'sky', 'pole', 'terrain',
                'person', 'fence', 'wall', 'traffic sign', 'bicycle', 'bus', 'train', 'truck',
                'traffic light', 'rider', 'motorcycle'
            ])
        else:
            raise ValueError('Not supported!')
        return label_names

    @staticmethod
    def create_label_colormap(version='cityscapes'):
        """Creates a label colormap used in Cityscapes segmentation benchmark.
        Returns:
            A Colormap for visualizing segmentation results.
        """
        if version == 'cityscapes':
            colormap = np.array([
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170, 30],
                [220, 220, 0],
                [107, 142, 35],
                [152, 251, 152],
                [70, 130, 180],
                [220, 20, 60],
                [255, 0, 0],
                [0, 0, 142],
                [0, 0, 70],
                [0, 60, 100],
                [0, 80, 100],
                [0, 0, 230],
                [119, 11, 32],
                [0, 0, 0]], dtype=np.uint8)
        elif version =='ade20k':
            colormap = np.array([
                [140, 140, 140],    # road
                [235, 255, 7],      # sidewalk
                [180, 120, 120],    # building
                [120, 120, 120],    # wall
                [255, 184, 6],      # fence
                [51, 0, 255],       # pole
                [41, 0, 255],       # traffic light
                [255, 5, 153],      # traffic sign
                [4, 200, 3],        # vegetation # use tree
                [4, 250, 7],        # terrain # use grass
                [6, 230, 230],      # sky
                [150, 5, 61],       # person
                [255, 225, 0],      # rider # mainly based on bicycle
                [0, 102, 200],      # car
                [255, 0, 20],       # truck
                [255, 0, 245],      # bus
                [255,61,6],         # train # use rail
                [163, 0, 255],      # motorcycle # motorbike
                [255, 245, 0],      # bicycle
                [0, 0, 0]], dtype=np.uint8)
        else:
            raise ValueError('Not supported yet!')

        return colormap


class Cityscapes(data.Dataset):
    def __init__(self, mode, train_mode, crop_height=512, crop_width=512, crop_ratio=1,
                 img_height=512, img_width=1024,augment_dict=None,
                 mask_encode_mode='color',
                 caption_json=None,
                 color_map_version='ade20k',
                 drop_caption_ratio=-1.0,
                 debug=None,
                 *args, **kwargs):
        assert mode in ('train', 'val', 'test')
        assert mask_encode_mode in ('color', 'id')

        self.train_mode = train_mode
        self.mask_encode_mode = mask_encode_mode
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width) if crop_width is not None else int(crop_height * crop_ratio)
        self.img_height = img_height
        self.img_width = img_width
        self.random_crop = False
        self.center_crop = False
        self.drop_caption_ratio = drop_caption_ratio
        if self.drop_caption_ratio >0:
            self.drop_caption = True
        else:
            self.drop_caption = True

        if (self.img_width != self.crop_width or self.img_height != self.crop_height) and self.train_mode:
            self.random_crop = True
        if augment_dict.get('center_crop', False) and\
                (self.img_width != self.crop_width or self.img_height != self.crop_height) and \
                self.train_mode:
            self.random_crop = False
            self.center_crop = True

        print(f'---> Resize h x w: {self.img_height} x {self.img_width}')
        print(f'---> Crop h x w: {self.crop_height} x {self.crop_width}')
        self.color_map = CityscapesBaseInfo.create_label_colormap(version=color_map_version)

        # Data Augmentation
        self.augment_dict = augment_dict
        if train_mode:
            self.augment_p = augment_dict['augment_p']

        self.datapth = '/fs/scratch/rng_cr_bcai_dl/lyu7rng/datasets/cityscapes' # TODO: Change the path
        self.rgb_path = '/fs/scratch/rng_cr_bcai_dl/lyu7rng/datasets/cityscapes' # TODO: Change the path
        cur_abs_dir = os.path.dirname(os.path.abspath(__file__))
        label_map_json = os.path.join(cur_abs_dir, 'cityscapes_info.json')
        with open(label_map_json, 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        # Read captions
        if caption_json is None:
            cur_path = os.path.dirname(__file__)
            caption_json = os.path.join(cur_path,f'cityscapes_caption_{mode}.json')

        with open(caption_json, 'r') as json_file:
            self.caption_dict = json.load(json_file)

        # Get image directory
        self.img_name_dir_dict = {}
        impth = os.path.join(self.rgb_path, 'leftImg8bit', mode)
        impths = glob.glob(os.path.join(impth, '*/*.png'))
        impths = natsorted(impths)

        self.img_names = []
        for img_name in impths:
            temp_name = os.path.basename(img_name)
            temp_name = temp_name.replace('_leftImg8bit.png', '')
            self.img_names.append(temp_name)
        self.img_name_dir_dict.update(dict(zip(self.img_names, impths)))

        # Get gt directory
        self.labels_name_dir_dict = {}
        gtnames = []
        gtpth = os.path.join(self.datapth, 'gtFine', mode)
        folders = [d for d in os.listdir(gtpth) if os.path.isdir(os.path.join(gtpth, d))]
        for fd in folders:
            fdpth = os.path.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames if el.endswith('.png')]
            lbpths = [os.path.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels_name_dir_dict.update(dict(zip(names, lbpths)))

        if debug is not None:
            self.img_names = self.img_names[:100]


        self.len = len(self.img_names)
        print(f'Total Cityscapes {mode} images: {self.len}')
        assert set(self.img_name_dir_dict.keys()) == set(self.labels_name_dir_dict.keys())

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), ])

    def __len__(self):
        return self.len

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def read_from_dir_return_tensor(self, impth, lbpth):
        img = Image.open(impth)
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

    def label_encode_color(self,label):
        # encode the mask using color coding
        # return label: Tensor [3,h,w], (-1,1)
        label_ = np.copy(label)
        label_[label_ == 255] = 19
        label_ = self.color_map[label_]
        label_ = rearrange(label_, 'h w c -> c h w')
        label_ = torch.from_numpy(label_)
        # Tutorial dataset uses Normalization to (0,1) for the condition
        label_ = label_ / 255.0 #* 2 - 1
        return label_

    def label_encode_id(self,label):
        # return label: Tensor [1,h,w]
        label_ = np.copy(label)
        label_[label_ == 255] = 19
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
        if self.mask_encode_mode =='color':
            label_coded = self.label_encode_color(label) # Tensor [3,h,w], (0,1)
        elif self.mask_encode_mode == 'id':
            label_coded = self.label_encode_id(label) # Tensor [1,h,w]
        label = label.unsqueeze(0)

        data['image'] = img # Tensor: [3,h,w], (-1,1)
        data['hint'] = label_coded
        data['txt'] = prompt
        data['label'] = label # Tensor [1,512,512], 0-18 + 255 label id
        data['img_pth'] = impth  # for debugging
        data['label_pth'] = lbpth # for generation
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

    ds = Cityscapes(mode=mode, train_mode=train_mode, augment_dict=augment_dict,mask_encode_mode='color')
    dl = DataLoader(ds,batch_size=4,shuffle=shuffle,num_workers=4,drop_last=True)
    color_map = ds.color_map


    cs_info = CityscapesBaseInfo()
    print(cs_info.label_names)

    def rearrange_rescale(x, rescale=True):
        try:
            x = x.cpu().detach().numpy()
        except:
            pass

        x = rearrange(x, 'c h w -> h w c')
        if rescale:
            x = (x + 1) * 0.5 # (0,1)
        return x

    def visualization(img_, label_):
        img = rearrange_rescale(img_)
        label = rearrange_rescale(label_, rescale=False)

        fig, axes = plt.subplots(1,3, dpi=120,figsize=(12,4))

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

    show_num = 2
    for i, data in enumerate(dl):
        if i < show_num:
            img = data['image']
            label = data['hint']
            text = data['txt']
            # impth = data['img_pth']
            # print(impth)
            print(img.shape, label.shape)
            print(text)
            visualization(img[0], label[0])
        else:
            break
