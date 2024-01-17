from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np
import torch.nn.functional as F
import torch

def random_vertical_flip(img, label, prob=0.5):
    # only support numpy array at the moment
    if random.random() < prob:
        #img = np.flip(img, 2)
        #label = np.flip(label, 1)
        img = img.flip(2)
        label = label.flip(1)
        return img, label
    else:
        return img, label

def random_horizontal_flip(img, label, prob=0.5):
    # only support numpy array at the moment
    if random.random() < prob:
        #img = np.flip(img, 2)
        #label = np.flip(label, 1)
        img = img.flip(-1)
        label = label.flip(-1)
        return img, label
    else:
        return img, label


def random_resize(image, label, scale_bounds=(0.5, 2.0)):
    r_image = image.clone().unsqueeze(0)
    r_label = label.clone().unsqueeze(0).unsqueeze(0).float()

    rand_scale = random.random() * (scale_bounds[1] - scale_bounds[0]) + scale_bounds[0]

    r_image = F.interpolate(r_image, scale_factor=rand_scale, mode='bilinear', align_corners=False).squeeze(0)
    r_label = F.interpolate(r_label, scale_factor=rand_scale, mode='nearest').squeeze(0).squeeze(0)

    image = torch.zeros_like(image)
    label = torch.ones_like(label) * 255

    if r_image.shape[-1] <= image.shape[-1]:
        h, w = r_image.shape[-2:]
        image[:, 0:h, 0:w] = r_image
        label[0:h, 0:w] = r_label
    else:
        h, w = image.shape[-2:]
        row = random.randint(0, r_image.shape[-2] - h)
        col = random.randint(0, r_image.shape[-1] - w)
        image = r_image[:, row:row + h, col:col + w]
        label = r_label[row:row + h, col:col + w]

    return image, label.long()


def center_crop(img, label, new_width, new_height, return_corners=False):
    width, height = img.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    if not return_corners:
        return img, label
    else:
        return img, label, (left, top, right, bottom)


def center_crop_single(img, new_width, new_height):
    width, height = img.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))
    return img


def random_crop(img,label, new_width, new_height, return_corners=False):
    width, height = img.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    left = random.randint(0, left)
    top = random.randint(0, top)
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))
    label = label.crop((left, top, right, bottom))
    if not return_corners:
        return img,label
    else:
        return img, label, (left, top, right, bottom)


class PhotoMetricDistortion(torch.nn.Module):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of base_p * multiplier. The position of random contrast is in
    second or second to last.
    - random brightness
    - random contrast (mode 0)
    - luma flip
    - random hue
    - random saturation
    - random contrast (mode 1)
    """
    # TODO: Change lumaflip back to 1 please!
    def __init__(self, brightness=1., contrast=1., lumaflip=1., hue=1., saturation=1.,
                 brightness_std=0.2, contrast_std=0.5, hue_max=1., saturation_std=1.,
                 global_p=0.5, part_p=0.5,mode='fixed'):
        super().__init__()
        #self.p              = augment_p             # overall augmentation probability
        self.mode = mode
        if mode == 'fixed':
            self.global_p = global_p # Overall multiplier for augmentation probability.
            self.register_buffer('p', torch.ones([]) * part_p)  # Augmentation probability for subsequences.
        elif mode == 'ada':
            self.global_p = 0.0
            self.register_buffer('p', torch.ones([])*0.0)  # Overall multiplier for augmentation probability.
        print(f'Setting up color transformation: global p = {self.global_p}')
        print(f'Setting up color transformation: initial partial p = {self.p.item()}')

        self.brightness     = brightness
        self.contrast       = contrast
        self.brightness     = brightness
        self.lumaflip       = lumaflip
        self.hue            = hue
        self.saturation     = saturation
        self.brightness_std = float(brightness_std)  # Standard deviation of brightness.
        self.contrast_std   = float(contrast_std)    # Log2 standard deviation of contrast.
        self.hue_max        = float(hue_max)         # Range of hue rotation, 1 = full circle.
        self.saturation_std = float(saturation_std)  # Log2 standard deviation of saturation.

    def forward(self, images,):
        assert isinstance(images, torch.Tensor) and images.ndim == 4

        temp_mode = np.random.rand(1)  # mode > p -> raw image, o.w. augmented image
        if self.mode=='fixed' and temp_mode > self.global_p:
            #print('----> Noo color remains unchanged!')
            return images
        #print('----> Yeah color is changed!')

        batch_size, num_channels, height, width = images.shape
        device = images.device
        temp_mode = np.random.randint(0,2) # mode == 0 -> do random contrast first, mode == 1 -> do random contrast last

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (contrast * strength) firstly
        if self.contrast > 0 and temp_mode == 0:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (lumaflip * strength).
        v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)  # Luma axis.
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            i = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p, i, torch.zeros_like(i))
            C = (I_4 - 2 * v.ger(v) * i) @ C  # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(torch.rand([batch_size], device=device) < self.hue * self.p, theta, torch.zeros_like(theta))
            C = rotate3d(v, theta) @ C  # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0 and num_channels > 1:
            s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
            s = torch.where(torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p, s, torch.ones_like(s))
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # Apply contrast with probability (contrast * strength) lastly
        if self.contrast > 0 and temp_mode == 1:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            C = scale3d(c, c, c) @ C

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            if num_channels == 3:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
            elif num_channels == 1:
                C = C[:, :3, :].mean(dim=1, keepdims=True)
                images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([batch_size, num_channels, height, width])

        return images

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.

_constant_cache = dict()
def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [x if isinstance(x, torch.Tensor) else constant(x, shape=ref[0].shape, device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = torch.sin(theta); c = torch.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)