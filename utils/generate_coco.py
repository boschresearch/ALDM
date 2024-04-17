
import os
import sys
sys.path.append(".")
sys.path.append("..")

from PIL import Image
import torch
import numpy as np
from dataclasses import dataclass
import pyrallis
from tqdm import tqdm

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from dataloader import dataset_factory
from torch.utils.data import DataLoader
from einops import rearrange
import json
from pathlib import Path 

@dataclass
class GenerateConfig:
    checkpoint_dir: str
    model_config: str = "models/cldm_seg_coco_stuff_multi_step_D.yaml"
    output_dir: str = "image_output/coco_generated_ALDM"
    seed: int = 1234
    num_img_per_map: int = 1
    num_img_per_batch: int = 1

    # sampling
    num_timesteps: int = 25
    cfg_scale: float = 7.5

    # dataset
    dataset: str = 'coco'
    dataset_split: str = 'val'
    img_height: int = 512
    img_width: int = 512

    # prompt
    negative_prompt: str = ""

    # split
    data_from: int = 0
    cur_split_id: int = 0
    num_per_split: int = 5000

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


@pyrallis.wrap()
def run(config: GenerateConfig):
    model_config = config.model_config
    segmenter_config = None

    model = create_model(model_config,extra_segmenter_config=segmenter_config).cpu()
    model.load_state_dict(load_state_dict(config.checkpoint_dir, location='cpu'), strict=False)  # , location='cuda'
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    H, W = config.img_height, config.img_width
    if hasattr(model, 'model_attend'):
        model.model_attend.image_ratio = W / H
        model.model_attend.attention_store.image_ratio = W / H
    mask_encode_mode = 'id'
    use_language_enc = True

    # Configure datasets
    augment_dict = {}
    augment_dict['augment_p'] = -1
    augment_dict['horizontal_flip'] = False
    augment_dict['center_crop'] = True
    print('Validate on ', 'ADE20K')

    dataset = dataset_factory(config.dataset)(
        mode=config.dataset_split, train_mode=True, augment_dict=augment_dict,
        crop_height=H, crop_width=W,
        img_height=H, img_width=W,
        mask_encode_mode=mask_encode_mode,
    )
    dataloader = DataLoader(
        dataset, num_workers=0,
        batch_size=1, shuffle=False,
    )
    num_img_per_batch = config.num_img_per_batch
    num_runs = config.num_img_per_map // num_img_per_batch
    n_prompt = config.negative_prompt
    cfg_scale = config.cfg_scale
    num_timesteps = config.num_timesteps

    data_start = config.data_from + config.cur_split_id * config.num_per_split
    data_end = data_start + config.num_per_split
    print(f'---> Start = {data_start}, End = {data_end}')

    name_dict= {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            if i < data_start:
                continue
            if i >= data_end:
                break

            cur_seed = config.seed + i * 177
            seed_everything(cur_seed)
            image_path_stem = Path(data['img_pth'][0]).stem 
            name_dict[image_path_stem] = data['label_pth']
            # img = data['image']
            control = data['hint'].cuda()  # label
            if use_language_enc:
                control = model.class_embedding_manager(control) # [bs, c, h, w]
                control = control.to(model.device)
            prompt = data['txt']

            cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt[0]] * num_img_per_batch)]
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_img_per_batch)]
            }
            shape = (4, H // 8, W // 8)

            for j in range(num_runs):
                samples, intermediates = ddim_sampler.sample(
                    num_timesteps, num_img_per_batch,shape, cond,
                    verbose=False, eta=0.0,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=un_cond
                )
                x_samples = model.decode_first_stage(samples)
                x_samples = (rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
                x_samples = x_samples.clip(0, 255).astype(np.uint8)

                cur_start_id = num_img_per_batch * j
                for k in range(num_img_per_batch):
                    img_id = cur_start_id + k
                    img_name = f'{i}_{image_path_stem}_{cur_seed}.png'
                    img_name = os.path.join(config.output_dir, img_name)
                    Image.fromarray(x_samples[k]).save(img_name)

    json_name = os.path.join(config.output_dir,f'img_name_{data_start}_{data_end}.json')
    with open(json_name, 'w') as fp:
        json.dump(name_dict, fp)
    

if __name__ == '__main__':
    run()