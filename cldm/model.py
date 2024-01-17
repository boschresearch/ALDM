import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path, extra_segmenter_config=None):
    config = OmegaConf.load(config_path)
    if extra_segmenter_config is not None:
        config_new = OmegaConf.merge(config.model.params.segmenter_config.params,extra_segmenter_config)
        config.model.params.segmenter_config.params = config_new

    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def create_model_from_config(config):
    model = instantiate_from_config(config.model).cpu()
    #print(f'Loaded model config from [{config_path}]')
    return model
