import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None,sampling_steps=25,
                 # new!
                 log_split='val', fixed_generator=True, saving_mode='all_in_one',get_label_pred=True,
                 ):
        super().__init__()
        assert saving_mode in ('all_in_one', 'separate')
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.sampling_steps = sampling_steps
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_split = log_split
        self.saving_mode = saving_mode
        self.get_label_pred = get_label_pred
        if fixed_generator:
            self.generator = torch.Generator(device='cuda')
            self.generator.manual_seed(77777)
        else:
            self.generator = None


    @rank_zero_only
    def log_local(self, save_dir, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log")
        # TODO: save all images into one grid
        if self.saving_mode =='all_in_one':
            grid_list = []
            for k in images:
                if 'label' in k:
                    grid = torchvision.utils.make_grid(images[k], nrow=self.max_images) #, value_range=(-1,1)
                else:
                    grid = torchvision.utils.make_grid(images[k], nrow=self.max_images) # 4 columns, tensor [3,h,w]
                    # (-1,1) -> (-1,1)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = (grid.numpy() * 255).astype(np.uint8)
                grid_list.append(grid)
            grid_list = np.stack(grid_list)
            grid_list = rearrange(grid_list, 'n c h w -> (n h) w c')
            # grid_list = grid_list.numpy()
            # grid_list = (grid_list * 255).astype(np.uint8)
            filename = "{:06}-ep{:06}-bs{:06}.png".format(global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid_list).save(path)
        elif self.saving_mode == 'separate':
            # TODO: not up to date
            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=self.max_images)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{:06}-ep{:06}-bs{:06}-{}.png".format(global_step, current_epoch, batch_idx, k)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            if split == 'train':
                use_batch = batch
            elif split == 'val':
                it = iter(pl_module.trainer.val_dataloaders[0])
                use_batch = next(it)
                use_batch = next(it)
                use_batch = next(it)
            else:
                raise ValueError('Given split is not supported!')

            with torch.no_grad():
                images = pl_module.log_images(
                    use_batch,
                    generator=self.generator,
                    N=self.max_images,
                    ddim_steps=self.sampling_steps,
                    split=split,
                    get_fake_label_pred=self.get_label_pred,
                    get_real_label_pred=self.get_label_pred,
                    **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split=self.log_split)


