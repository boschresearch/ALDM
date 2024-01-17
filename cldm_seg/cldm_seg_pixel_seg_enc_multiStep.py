import einops
import torch
import torch as th
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from dataloader import dataset_factory

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    noise_like,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler
from cldm_seg.ddim4train import DDIMSampler
from ldm.util import default, ismap, isimage, mean_flat, count_params
from einops import rearrange
from functools import partial
# from timeit import default_timer as timer

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
            -1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 768, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 768, 512, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 512, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 256, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        # ! new !
        # self.scale_factor = nn.Parameter(torch.randn(1,hint_channels,1,1))

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context,use_hint_condition=True, **kwargs):
        '''
        context: [bs,77,768]
        x: [bs,4,64,64]
        hint: [bs,3,512,512]
        guided_hint: [bs,320,64,64]
        '''
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context) # [bs,320,64,64]

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control,
                 segmenter_config, class_emb_manager_config, mask_encode_mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        self.segmenter = instantiate_from_config(segmenter_config)
        self.mask_encode_mode = mask_encode_mode
        self.class_embedding_manager = instantiate_from_config(class_emb_manager_config)

        # For GAN optimization, backward manually
        self.automatic_optimization = False

    ################################################################
    # Main modifications! #
    ################################################################
    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.batch_size_allGPU


    def set_ddim_sampler(self, num_subset_timesteps, eta=0.0, verbose=False):
        self.ddim_sampler = DDIMSampler(self)
        self.ddim_sampler.make_schedule(ddim_num_steps=num_subset_timesteps, ddim_eta=eta, verbose=verbose)
        self.ddim_sampler.ddim_alphas_prev = torch.from_numpy(self.ddim_sampler.ddim_alphas_prev)
        self.ddim_timesteps = torch.from_numpy(self.ddim_sampler.ddim_timesteps)
        self.num_subset_timesteps = num_subset_timesteps
        alphas = self.ddim_sampler.ddim_alphas

    def train_dataloader(self):
        # Dataset
        augment_dict = {}
        augment_dict['augment_p'] = 0.5
        augment_dict['horizontal_flip'] = True
        print('Train on ', self.dataset)
        dataset = dataset_factory(self.dataset)(
            mode='train', train_mode=True,
            augment_dict=augment_dict,
            drop_caption_ratio=self.drop_caption_ratio,
            mask_encode_mode=self.mask_encode_mode,
        )
        self.label_color_map = dataset.color_map  # used for segmentation map visualization
        dataloader = DataLoader(
            dataset, num_workers=0,
            batch_size=self.batch_size, shuffle=True,
        )  # batch_size per GPU

        return dataloader

    def val_dataloader(self):
        # Dataset
        augment_dict = {}
        augment_dict['augment_p'] = -1
        augment_dict['horizontal_flip'] = False
        augment_dict['center_crop'] = True
        print('Validate on ', self.dataset)
        dataset = dataset_factory(self.dataset)(
            mode='val', train_mode=True,
            augment_dict=augment_dict,
            mask_encode_mode=self.mask_encode_mode,
        )

        dataloader = DataLoader(
            dataset, num_workers=0,
            batch_size=self.val_batch_size, shuffle=False,
        )  # batch_size per GPU
        return dataloader

    def configure_optimizers(self):
        optimizer_dict = {
            'AdamW': torch.optim.AdamW,
            'Adam': torch.optim.Adam,
        }
        optimizer = optimizer_dict[self.optimizer_config['type']]

        # For ControlNet
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        g_opt = optimizer(params, lr=self.optimizer_config['G_lr'])

        # For Discriminator
        params = list(self.segmenter.parameters())
        d_opt = optimizer(params, lr=self.optimizer_config['D_lr'],
                          weight_decay=self.optimizer_config['weight_decay'])

        if self.optimizer_config['D_lr_scheduler'] == 'one_cycle':
            if self.dataset == 'ade' or self.dataset == 'coco':
                total_steps = self.optimizer_config['D_lr_all_it'] // self.n_lazy_guidance
            else:
                total_steps = self.optimizer_config['D_lr_all_it']
            pct_start = self.optimizer_config['pct_start']
            D_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                d_opt, max_lr=self.optimizer_config['D_lr'],
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=10.0,  # initial lr
                final_div_factor=100.0  # final lr
            )
            print(f'---> pct_start = {pct_start}')
            return [g_opt, d_opt], [D_scheduler]
        else:
            return [g_opt, d_opt], []

    @torch.no_grad()
    def get_input(self, batch, k, bs=None,
                  return_first_stage_outputs=False, force_c_encode=False,
                  return_original_cond=False, return_x=False,
                  *args, **kwargs):
        # x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # get image
        x = batch[self.first_stage_key]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()

        # get latent
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        # get prompt
        if self.use_caption_training:
            xc = batch[self.cond_stage_key]
        else:
            xc = [""] * len(batch[self.cond_stage_key])

        if not self.cond_stage_trainable or force_c_encode:
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                c = self.get_learned_conditioning(xc.to(self.device))
        else:
            c = xc
        if bs is not None:
            c = c[:bs]

        return_dict = {'c_crossattn': [c]}

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            return_dict['reconstruction'] = xrec
        if return_x:
            return_dict['image'] = x
        if return_original_cond:
            return_dict['orig_cond'] = xc

        # get control condition
        control = batch[self.control_key]
        control = self.class_embedding_manager(control)  # [bs, c, h, w]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        return_dict['c_concat'] = [control]

        return z, return_dict

    def apply_model(self, x_noisy, t, cond, use_hint_condition=True, *args, **kwargs):
        '''
        Extract features and get noise prediction
        '''
        assert isinstance(cond, dict)
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        diffusion_model = self.model.diffusion_model

        if cond['c_concat'] is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None,
                only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,
                use_hint_condition=use_hint_condition,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control
            )
        return eps

    def p_sample_prev(self, x, e_t, index, temperature=1.0, return_x0=False):
        b = e_t.shape[0]
        alphas = self.ddim_sampler.ddim_alphas
        alphas_prev = self.ddim_sampler.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sampler.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sampler.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = alphas[index][:, None, None, None].to(x.device)
        a_prev = alphas_prev[index][:, None, None, None].to(x.device)

        sigma_t = sigmas[index][:, None, None, None].to(x.device)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index][:, None, None, None].to(x.device)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, self.device, False) * temperature
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev = x_prev.float()
        if return_x0:
            return x_prev, pred_x0
        else:
            return x_prev

    def p_sample_pred_clean(self, x, e_t, index):
        b = e_t.shape[0]
        alphas = self.ddim_sampler.ddim_alphas
        sqrt_one_minus_alphas = self.ddim_sampler.ddim_sqrt_one_minus_alphas

        # select parameters corresponding to the currently considered timestep
        a_t = alphas[index][:, None, None, None].to(x.device)
        sqrt_one_minus_at = sqrt_one_minus_alphas[index][:, None, None, None].to(x.device)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        return pred_x0

    def get_real_noisy(self, x_start, cond):
        index_next = torch.randint(0, self.num_subset_timesteps, (x_start.shape[0],))
        index = index_next - 1
        t = self.ddim_timesteps[index]
        t[index < 0] = 0
        t = t.to(self.device).long()
        cond['index'] = index
        cond['index_next'] = index_next
        cond['timestep_next'] = self.ddim_timesteps[index_next].to(self.device).long()
        noise = torch.randn_like(x_start)
        cond['noise'] = noise
        cond['timestep'] = t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        return x_noisy, cond

    def run_G(self, x_start, cond, noise=None, add_noise=True, use_hint_condition=True,
              return_input=False, given_timestep=None, given_index=None,
              ):
        if add_noise:
            if given_timestep is None:
                # sample t and compute t
                index = torch.randint(0, self.num_subset_timesteps, (x_start.shape[0],))

                t = self.ddim_timesteps[index].to(self.device).long()
                index_prev = index - 1
            else:
                t = given_timestep
                index = given_index

            # print(t)
            noise = default(noise, lambda: torch.randn_like(x_start))
            cond['noise'] = noise
            cond['timestep'] = t
            cond['t_index'] = index

            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        else:
            x_noisy = x_start
            t = torch.tensor([1], device=self.device).long().repeat(x_noisy.shape[0], )

        # main func to get the model prediction for segmentation
        model_output = self.apply_model(
            x_noisy, t, cond,
            use_hint_condition=use_hint_condition,
        )
        x_pred_x0 = self.p_sample_pred_clean(x_noisy, model_output, index)

        if return_input:
            return model_output, x_pred_x0, x_start, cond
        else:
            return model_output, x_pred_x0

    def run_D(self, x, batch, timestep, is_fake=False, loss_weight=None, batch_idx=None):
        # key: decode.loss_ce, decode.acc_seg
        x_samples = self.decode_first_stage_train(x)
        x_samples = torch.clamp(x_samples, -1.0, 1.0)

        gt_semantic_seg = batch['label']
        if is_fake:
            gt_semantic_seg = torch.ones_like(gt_semantic_seg) * self.fake_class_id
            use_sampler = False
        else:
            use_sampler = True
        loss_dict = self.segmenter(
            x_samples, gt_semantic_seg,
            use_sampler=use_sampler,
            batch_idx=None if is_fake else batch_idx,
            w_t=loss_weight[:,None,None] if loss_weight is not None else None,
        )
        loss_dict['ce_loss'] = loss_dict['ce_loss'].mean()
        return loss_dict

    def run_D_for_multistep(self, x, batch, timestep, is_fake=False, loss_weight=None, batch_idx=None):
        x_samples = self.decode_first_stage_train(x)
        x_samples = torch.clamp(x_samples, -1.0, 1.0)

        gt_semantic_seg = batch['label']
        if is_fake:
            gt_semantic_seg = torch.ones_like(gt_semantic_seg) * self.fake_class_id
            use_sampler = False
        else:
            use_sampler = True
        loss_dict = self.segmenter(
            x_samples, gt_semantic_seg,
            use_sampler=use_sampler,
            batch_idx=None if is_fake else batch_idx,
            w_t=loss_weight[:,None,None] if loss_weight is not None else None,
            return_pixelwise_loss=True,
        )
        #loss_dict['ce_loss'] = loss_dict['ce_loss'].mean()
        return loss_dict

    def cal_G_loss(self, model_output, x_start, cond):
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        t = cond['timestep']
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = cond['noise']
        elif self.parameterization == "v":
            target = self.get_v(x_start, cond['noise'], cond['timestep'])
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def cal_dis_loss_coeff(self, cur_idx):
        # 5000-10000
        if cur_idx < self.start_dis_fake:
            return 0.0
        elif cur_idx > self.end_dis_fake:
            return 1.0
        else:
            coef = (cur_idx - self.start_dis_fake) / (self.end_dis_fake - self.start_dis_fake)
            return coef

    def run_G_multi_step(self, x_start, cond, batch, noise=None, use_hint_condition=True,apply_guidance=True):
        # sample t
        if self.random_multi_step:
            multi_step_cur = np.random.randint(self.min_sample_step, self.max_sample_step)
        else:
            multi_step_cur = self.multi_step
        index = torch.randint(multi_step_cur, self.num_subset_timesteps, (x_start.shape[0],))

        t = self.ddim_timesteps[index].to(self.device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        cond['noise'] = noise
        cond['timestep'] = t
        cond['t_index'] = index

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(
            x_noisy, t, cond,
            use_hint_condition=use_hint_condition,
        )
        p_loss, p_loss_dict = self.cal_G_loss(model_output, x_start, cond)
        if apply_guidance:
            x_prev, x_pred_x0 = self.p_sample_prev(x_noisy, model_output, index, return_x0=True)

            loss_G_dict = self.run_D_for_multistep(
                x_pred_x0, batch, t, is_fake=False,
                loss_weight=None,
            )
            cur_num_of_pixel = loss_G_dict['cur_num_of_pixel']
            init_G_loss = loss_G_dict['ce_loss']
            loss_G_list = [loss_G_dict['ce_loss']]
            index_prev = index
            #start = timer()
            for i in range(multi_step_cur):
                with torch.no_grad():
                    index_prev = index_prev - 1
                    prev_t = self.ddim_timesteps[index_prev].to(self.device)
                    model_output_prev = self.apply_model(
                        x_prev, prev_t, cond,
                        use_hint_condition=use_hint_condition,
                    )
                    x_prev, x_pred_x0_prev = self.p_sample_prev(x_prev, model_output_prev, index_prev, return_x0=True)
                    loss_G_prev_dict = self.run_D_for_multistep(
                        x_pred_x0_prev, batch, prev_t, is_fake=False,
                        loss_weight=None,
                    )
                loss_G_list.append(loss_G_prev_dict['ce_loss'])
            #end = timer()
            #print(f'----> Duration for {multi_step_cur} step: ', end - start)
            loss_G_dis = torch.stack(loss_G_list).mean(0)
            avg_init_G_loss = init_G_loss.sum() / cur_num_of_pixel
            valid_mask = init_G_loss != 0
            init_G_loss[valid_mask] = init_G_loss[valid_mask] / init_G_loss[valid_mask].data * loss_G_dis[valid_mask]
            final_G_loss = init_G_loss.sum() / cur_num_of_pixel
            # print('multi_step loss: ', final_G_loss.data, avg_init_G_loss.data)
        else:
            init_G_loss = None
            loss_G_dis = None
            avg_init_G_loss = None
            final_G_loss = None

        return p_loss, p_loss_dict, final_G_loss, avg_init_G_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        '''
        batch: output of the dataloader, e.g., a dict
        '''
        g_opt, d_opt = self.optimizers()

        # Get input
        x_start, cond_batch = self.get_input(batch, self.first_stage_key)
        dis_fake_coef = self.cal_dis_loss_coeff(self.global_step)
        if self.use_lazy_guidance and (self.global_step % self.n_lazy_guidance!=0):
            apply_guidance = False
        else:
            apply_guidance = True
        # ------- Update Discriminator ------- #
        if (self.global_step > self.train_dis_from and apply_guidance) or \
                (self.train_D_before_lazy and self.global_step < self.start_dis_fake):
            train_Dis_now = True
            requires_grad(self.control_model, False)
            requires_grad(self.segmenter, True)
            _, fake_pred_x0, _, cond_temp = self.run_G(
                x_start, cond_batch,
                add_noise=True, use_hint_condition=True,
                return_input=True,
            )
            loss_D_fake = self.run_D(
                fake_pred_x0, batch, cond_temp['timestep'], is_fake=True,
                loss_weight=None
            )

            loss_D_real = self.run_D(
                x_start, batch, torch.ones_like(cond_temp['timestep']), is_fake=False,
                batch_idx=None,
            )
            if self.warm_up_dis:
                loss_D = loss_D_real['ce_loss'] + loss_D_fake['ce_loss'] * dis_fake_coef * self.weight_fake_D
            else:
                loss_D = loss_D_real['ce_loss'] + loss_D_fake['ce_loss'] * self.weight_fake_D

            d_opt.zero_grad(set_to_none=True)
            self.manual_backward(loss_D)
            d_opt.step()
            d_opt.zero_grad(set_to_none=True)
            if self.lr_schedulers() is not None:
                self.lr_schedulers().step()  # TODO: only discriminator has lr schedulers
        else:
            train_Dis_now = False
            d_opt.zero_grad(set_to_none=True)
        requires_grad(self.segmenter, False)

        # ------- Update Generator ------- #
        requires_grad(self.control_model, True)
        p_loss, p_loss_dict, final_G_loss, init_G_loss = self.run_G_multi_step(
            x_start, cond_batch, batch, noise=None, use_hint_condition=True,
            apply_guidance=apply_guidance,
        )
        if final_G_loss is None:
            loss_G = self.lambda_diffusion_loss * p_loss
        else:
            loss_G = self.lambda_diffusion_loss * p_loss + self.lambda_D_loss * final_G_loss * dis_fake_coef

        g_opt.zero_grad(set_to_none=True)
        self.manual_backward(loss_G)
        g_opt.step()
        g_opt.zero_grad(set_to_none=True)

        if self.global_rank == 0:
            if train_Dis_now:
                # D loss
                self.logger.experiment.add_scalar('loss/D_loss', loss_D, self.num_samples)
                self.logger.experiment.add_scalar('loss/D_loss_fake', loss_D_fake['ce_loss'], self.num_samples)
                self.logger.experiment.add_scalar('loss/D_loss_real', loss_D_real['ce_loss'], self.num_samples)
            # G loss
            self.logger.experiment.add_scalar('loss/G_loss', loss_G, self.num_samples)
            if final_G_loss is not None:
                self.logger.experiment.add_scalar('loss/G_loss_dis_seg', final_G_loss, self.num_samples)
                if self.multi_step > 0:
                    self.logger.experiment.add_scalar('loss/G_loss_dis_seg_init', init_G_loss, self.num_samples)
            self.logger.experiment.add_scalar('loss/G_loss_noise_L2', p_loss, self.num_samples)

            # Learning rate
            self.logger.experiment.add_scalar('Lr/D_lr', d_opt.param_groups[0]["lr"], self.num_samples)

        self.log("loss", p_loss, sync_dist=True)  # for checkpoint monitoring
        print()
        requires_grad(self.control_model, False)
        return loss_G
    ################################################################

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
                   use_ema_scope=True, generator=None, get_fake_label_pred=False, get_real_label_pred=False,
                   **kwargs):
        ddim_steps = self.num_subset_timesteps
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16).cpu()
        log["reconstruction"] = self.decode_first_stage(z).cpu()
        log["control"] = self.label_to_seg_map(batch["hint"].squeeze(1)).cpu()

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
        if unconditional_guidance_scale > 1.0:
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             generator=generator,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg.cpu()
            if get_real_label_pred:
                log["real_label_pred"] = self.label_to_seg_map(
                    self.run_segmenter(z, batch, cond).cpu()
                ).cpu()
            if get_fake_label_pred:
                log["fake_label_pred"] = self.label_to_seg_map(
                    self.run_segmenter(samples_cfg, batch, cond).cpu()
                ).cpu()
        return log

    def label_to_seg_map(self, label):
        out = self.label_color_map[label]  # [bs, h, w, 3]
        out = rearrange(out, 'bs h w c -> bs c h w')
        out = (torch.from_numpy(out) / 255.0) * 2.0 - 1.0
        return out  # (-1,1)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, generator=None, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 2, w // 2)
        random_shape = (batch_size, self.channels, h // 2, w // 2)
        x_T = torch.randn(random_shape, generator=generator, device=self.device)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, x_T=x_T,
                                                     **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def run_segmenter(self, z, batch, cond):
        img = self.decode_first_stage(z)
        img = torch.clamp(img, -1.0, 1.0)
        label_pred = self.segmenter.forward_pred_label(img, batch['label'])
        return label_pred  # [bs, h, w]

    # @torch.no_grad()  # TODO: no use!
    # def apply_model_with_seg(self, x_noisy, t, cond, use_text_adapter=True,
    #                          use_hint_condition=True,
    #                          *args, **kwargs):
    #     '''
    #     For hacked DDIM_sampler_seg
    #     '''
    #     # main func to get the model prediction & features for segmentation
    #     model_output, feat_list = self.apply_model(
    #         x_noisy, t, cond,
    #         use_text_adapter=use_text_adapter,
    #         use_hint_condition=use_hint_condition,
    #         return_feats=True,
    #     )
    #     # if self.use_time_segmenter:
    #     #     time_embedding = self.segmenter.extract_time_embedding(t)[:, :, None, None]
    #     #     feat_list = [torch.cat([k, time_embedding.repeat(1, 1, k.shape[-2], k.shape[-1])], dim=1) for k in
    #     #                  feat_list]
    #     h, w = x_noisy.shape[2:]
    #     img_meta = {
    #         'ori_shape': (h * 8, w * 8),
    #     }
    #     label_pred = self.segmenter.simple_test(None, img_meta=[img_meta], rescale=True, features=feat_list)
    #     out = self.label_color_map[label_pred]  # [bs, h, w] -> [bs, h, w, 3]
    #     # out = rearrange(out, 'bs h w c -> bs c h w')
    #     # out = (torch.from_numpy(out) / 255.0) * 2.0 - 1.0
    #     return out  # [bs, h, w, 3] # (0-255) numpy
    #
    #
    # @torch.no_grad()
    # def apply_seg_on_image(self, x, label=None, no_fake_cls=False, *args, **kwargs):
    #     label_pred = self.segmenter.forward_pred_label(x, label) # [bs, h, w]
    #     # out = self.label_to_seg_map(label_pred)  # [bs, h, w] -> [bs, 3, h, w] #  # (-1,1)
    #     out = self.label_color_map[label_pred.cpu()]  # [bs, h, w, 3]
    #     #out = rearrange(out, 'bs h w c -> bs c h w') # [0-255]
    #     # out = rearrange(out, 'bs h w c -> bs c h w')
    #     # out = (torch.from_numpy(out) / 255.0) * 2.0 - 1.0
    #     out = out.clip(0, 255).astype(np.uint8)
    #     return out  # [bs, h, w, 3] # numpy


    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
