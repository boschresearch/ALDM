import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import types
from typing import Dict, Tuple
from einops import rearrange
from omegaconf import OmegaConf
from mit_semseg.models import ModelBuilder, SegmentationModule

segmenter_config_dict = {
    'upernet101': 'ade_upernet101.yaml',
    'upernet101_20cls': 'ade_upernet101_20cls.yaml',
    'upernet101_151cls': 'ade_upernet101_151cls.yaml',
    'upernet101_172cls': 'ade_upernet101_172cls.yaml',
}


class NewSegmentationModule(SegmentationModule):
    def forward(self, image, label, is_inference=None):
        segSize = (label.shape[-2], label.shape[-1])
        pred = self.decoder(self.encoder(image, return_feature_maps=True), segSize=segSize, is_inference=is_inference)
        return pred


def get_segmentation_model(model_config: OmegaConf) -> nn.Module:
    """
    Init a segmentation model given the model config.
    The returned module has an `inference(tensor)` method that takes and returns a NCHW tensor.
    """
    model = None
    if model_config.type == "mit_semseg":
        if os.path.isfile(model_config.encoder_weights):
            encoder_weights = model_config.encoder_weights
        else:
            encoder_weights = ''

        if os.path.isfile(model_config.decoder_weights):
            decoder_weights = model_config.decoder_weights
        else:
            decoder_weights = ''

        net_encoder = ModelBuilder.build_encoder(
            arch=model_config.encoder_arch.lower(),
            fc_dim=model_config.fc_dim,
            weights=encoder_weights,
        )
        net_decoder = ModelBuilder.build_decoder(
            arch=model_config.decoder_arch.lower(),
            fc_dim=model_config.fc_dim,
            num_class=model_config.decoder_classes,
            weights=decoder_weights, # []
            use_softmax=False,
        )
        crit = nn.NLLLoss(reduction='none', ignore_index=255)  # nn.CrossEntropyLoss(ignore_index=255)
        model = NewSegmentationModule(net_encoder, net_decoder, crit)
    else:
        raise NotImplementedError(f"Unknown model type: {model_config.type}")
    return model


class ADESegDiscriminator(nn.Module):
    def __init__(
            self,
            num_classes=150,
            ignore_index=255,
            class_weight=None,
            segmenter_type='upernet101_151cls',
            loss_sampler_version='V1',
            use_r1_reg=False,
            d_rg_every=16,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.loss_sampler_version = loss_sampler_version
        self.use_r1_reg = use_r1_reg
        self.d_rg_every = d_rg_every

        seg_config_file = segmenter_config_dict[segmenter_type]
        cur_dir = os.path.dirname(__file__)
        seg_config_file = os.path.join(cur_dir, seg_config_file)
        self.cfg = OmegaConf.load(seg_config_file)

        # Load segmenter
        self.segmenter = get_segmentation_model(self.cfg.model)

        # no logsofmax is used in models
        # adjust original code in mit_semseg/models/models.py
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=255)

        # pre-processing
        if segmenter_type == 'upernet101_20cls':
            self.rgb_info = {
                'mean': [0.5],
                'std': [0.5]
            }
        else:
            self.rgb_info = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        normalize = transforms.Normalize(mean=self.rgb_info['mean'], std=self.rgb_info['std'])
        # TODO: to be checked
        self.normalize_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2),
            normalize
        ])

    def load_pretrained_segmenter(self):
        self.segmenter.encoder.load_state_dict(torch.load(self.cfg.model.encoder_weights, map_location='cpu'))
        self.segmenter.decoder.load_state_dict(
            torch.load(self.cfg.model.decoder_weights, map_location='cpu'), strict=False)

    def loss_sampler(self, seg_logit, seg_label):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """

        bs, number_classes, H, W = seg_logit.shape
        if self.class_weight is not None:
            indices = seg_label.clone()
            indices = rearrange(indices, 'N 1 h w -> (N h w)')
            if self.ignore_index is not None:
                indices[indices == self.ignore_index] = 0
                seg_weight = torch.index_select(self.class_weight.to(indices.device), 0, indices)
                seg_weight = rearrange(seg_weight, '(N h w) -> N 1 h w', N=bs, h=H)
                seg_weight[seg_label == self.ignore_index] = 0.0
                seg_weight = seg_weight.squeeze(1)
            else:
                seg_weight = torch.index_select(self.class_weight, 0, indices)
                seg_weight = rearrange(seg_weight, '(N h w) -> N h w', N=bs, h=H)
            return seg_weight

        with torch.no_grad():
            class_occurence = torch.bincount(seg_label.view(-1))[:number_classes]
            cur_num_of_classes = (class_occurence > 0).sum()
            if self.ignore_index is not None:
                mask_invalid = seg_label == self.ignore_index
                cur_num_of_pixel = seg_label.numel() - (mask_invalid).sum().item()
            else:
                cur_num_of_pixel = seg_label.numel()

            if self.loss_sampler_version == 'V2':
                coefficients = torch.reciprocal(class_occurence) * cur_num_of_pixel  # (19)
                class_occurence_mask = class_occurence > 0
                coefficients[~class_occurence_mask] = 0
                coefficients = coefficients / coefficients.sum() + 1
                coefficients = coefficients / coefficients[class_occurence_mask].min()
                coefficients[~class_occurence_mask] = 0
            elif self.loss_sampler_version == 'V1':
                coefficients = torch.reciprocal(class_occurence) * cur_num_of_pixel / (
                        cur_num_of_classes * number_classes)  # (20)

            indices = seg_label.clone()
            indices = rearrange(indices, 'N 1 h w -> (N h w)')
            if self.ignore_index is not None:
                indices[indices == self.ignore_index] = 0
                seg_weight = torch.index_select(coefficients, 0, indices)
                seg_weight = rearrange(seg_weight, '(N h w) -> N 1 h w', N=bs, h=H)
                seg_weight[mask_invalid] = 0.0
                seg_weight = seg_weight.squeeze(1)
            else:
                seg_weight = torch.index_select(coefficients, 0, indices)
                seg_weight = rearrange(seg_weight, '(N h w) -> N h w', N=bs, h=H)
        return seg_weight, cur_num_of_pixel

    def cal_valid_num_pixel(self, seg_label):
        if self.ignore_index is not None:
            mask_invalid = seg_label == self.ignore_index
            cur_num_of_pixel = seg_label.numel() - (mask_invalid).sum().item()
        else:
            cur_num_of_pixel = seg_label.numel()
        return cur_num_of_pixel

    def forward(self, image, gt_label, use_sampler=True, batch_idx=None, w_t=None,
                return_pixelwise_loss=False):
        y = self.segmenter(self.normalize_input_for_seg(image), gt_label, is_inference=False)
        loss = self.cross_entropy(y, gt_label.squeeze(1))
        if not return_pixelwise_loss:
            if use_sampler:
                pixel_weight, cur_num_of_pixel = self.loss_sampler(y, gt_label)
                if w_t is None:
                    loss = (loss * pixel_weight).sum() / cur_num_of_pixel
                else:
                    loss = (loss * pixel_weight * w_t).sum() / cur_num_of_pixel
            else:
                cur_num_of_pixel = self.cal_valid_num_pixel(gt_label)
                if w_t is None:
                    loss = loss.sum() / cur_num_of_pixel
                else:
                    loss = (loss * w_t).sum() / cur_num_of_pixel
        else:
            if use_sampler:
                pixel_weight, cur_num_of_pixel = self.loss_sampler(y, gt_label)
                loss = loss * pixel_weight
            else:
                cur_num_of_pixel = self.cal_valid_num_pixel(gt_label)

        loss_dict = {
            'ce_loss': loss,
            'cur_num_of_pixel': cur_num_of_pixel,
        }
        return loss_dict

    def forward_pred_label(self, image, gt_label=None):
        y = self.segmenter(self.normalize_input_for_seg(image), gt_label, is_inference=True)
        prediction = torch.argmax(y, dim=1)
        return prediction

    def normalize_input_for_seg(self, image):
        # image: tensor (-1,1)
        norm_img = self.normalize_transform(image)
        return norm_img
