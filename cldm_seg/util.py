import os
import torch
import numpy as np
from ldm.modules.diffusionmodules.util import make_ddim_timesteps

import torch
from torch import nn, einsum
from einops import rearrange, repeat


cur_dir = os.path.dirname(__file__)
project_root_dir = os.path.dirname(cur_dir)
class_embedding_dir_dict = {
    'cityscapes': 'models/class_embeddings_cityscapes.pth',
    'ade': 'models/class_embeddings_ade20k.pth',
    'coco': 'models/class_embeddings_coco_stuff.pth',
    'null': 'models/null_embedding.pth',
}


class ClassEmbeddingManager(nn.Module):
    def __init__(self,text_dim, hidden_dim=None, mode='cityscapes', use_time=False,use_mapping=False, resize_h=128):
        super(ClassEmbeddingManager, self).__init__()
        self.use_mapping = use_mapping
        self.use_time = use_time
        self.resize_h = resize_h

        class_embeddings = torch.load(os.path.join(project_root_dir, class_embedding_dir_dict[mode]))
        null_token = torch.load(os.path.join(project_root_dir, class_embedding_dir_dict['null']))
        class_embeddings = torch.cat([class_embeddings, null_token])
        #self.register_buffer('class_embeddings', class_embeddings) # [num_cls, 768]
        self.class_embeddings = class_embeddings.cpu()
        #self.null_token = nn.Parameter(torch.zeros([text_dim]))

    def forward(self, seg_map, timestep=None):
        # given: seg_map (bs, 1, h, w)
        # return: language encoded seg map (bs, c, h, w)
        bs, _, h, w = seg_map.shape
        seg_map_ = rearrange(seg_map, 'b 1 h w ->(b h w)')
        seg_emb = torch.index_select(self.class_embeddings, 0, seg_map_.cpu())
        seg_emb = rearrange(seg_emb, '(bs h w) c -> bs c h w', bs=bs, h=h, w=w)
        scale_factor = h // self.resize_h
        w = w // scale_factor
        h = self.resize_h
        seg_emb = torch.nn.functional.interpolate(seg_emb, (h, w), mode="nearest")

        return seg_emb

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag