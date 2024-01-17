import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import matplotlib


# --------------------------------------------------------------------------------------#
# Cityscapes Visualization Tools

class CityscapesBaseInfo():
    def __init__(self):
        self.num_classes = 19
        self.ignore_label = 255
        self.sorted_name = ['road', 'building', 'vegetation', 'car', 'sidewalk', 'sky', 'pole', 'terrain',
                            'person', 'fence', 'wall', 'traffic sign', 'bicycle', 'bus', 'train', 'truck',
                            'traffic light', 'rider', 'motorcycle']
        self.LABEL_NAMES = np.asarray([
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorcycle', 'bicycle', 'void'])
        self.LABEL_NAMES_short = np.asarray([
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'tr. light',
            'tr. sign', 'veget.', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
            'bus', 'train', 'motorb.', 'bicycle', 'void'])

        self.palette = self.create_palette()
        self.colormap = self.create_label_colormap()
        self.FULL_LABEL_MAP = np.arange(len(self.LABEL_NAMES)).reshape(len(self.LABEL_NAMES), 1)
        label_map = [*range(19)]
        self.name_to_id_dict = dict(zip(self.LABEL_NAMES, label_map))
        self.id_to_name_dict = dict(zip(label_map, self.LABEL_NAMES))

    def create_palette(self):
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170,
                   30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                   70, 0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        return palette

    def create_label_colormap(self):
        """Creates a label colormap used in Cityscapes segmentation benchmark.
        Returns:
            A Colormap for visualizing segmentation results.
        """
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
        return colormap


class CityScapesHelper():
    def __init__(self):
        self.cs_basic_info = CityscapesBaseInfo()

    def colorize_mask(self, mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.cs_basic_info.palette)
        return new_mask

    def label_to_color_image(self, label):
        """Adds color defined by the dataset colormap to the label.

        Args:
            label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
                map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        if np.max(label) >= len(self.cs_basic_info.colormap):
            # print(f'Label value too large: replace {np.max(label)} with {len(self.cs_basic_info.colormap)}')
            label[label == np.max(label)] = len(self.cs_basic_info.colormap) - 1

        return self.cs_basic_info.colormap[label]

    def vis_segmentation(self, image, seg_map, gt_map=None, fname=None, dpi=100, plot_res=True):
        """Visualizes input image, segmentation map and overlay view."""
        if gt_map is None:
            plt.figure(figsize=(20, 4), dpi=dpi)
            grid_spec = gridspec.GridSpec(2, 3, height_ratios=[7, 1], wspace=0.03, hspace=0)
        else:
            plt.figure(figsize=(25, 5), dpi=dpi)
            grid_spec = gridspec.GridSpec(2, 4, height_ratios=[7, 1], wspace=0.03, hspace=0)

        plt.subplot(grid_spec[0, 0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')

        plt.subplot(grid_spec[0, 1])
        seg_image = self.label_to_color_image(seg_map).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        plt.subplot(grid_spec[0, 2])
        plt.imshow(image)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')

        if gt_map is not None:
            plt.subplot(grid_spec[0, 3])
            seg_image = self.label_to_color_image(gt_map).astype(np.uint8)
            plt.imshow(seg_image)
            plt.axis('off')
            plt.title('gt map')

        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[-1, :])
        self.FULL_COLOR_MAP = self.label_to_color_image(self.cs_basic_info.FULL_LABEL_MAP)
        # plt.imshow(self.FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        c = self.cs_basic_info.colormap
        cMap = ListedColormap(c / 255.)
        cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cMap)
        if gt_map is None:
            name_list = self.cs_basic_info.LABEL_NAMES_short
        else:
            name_list = self.cs_basic_info.LABEL_NAMES

        for j, lab in enumerate(name_list):
            cb.ax.text((2 * j + 1) / 40.0, 0.5, lab, ha='center', va='center', fontweight="bold", color='white')
        cb.ax.get_xaxis().set_ticks([])
        plt.grid('off')
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        if plot_res:
            plt.show()
