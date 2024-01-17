import einops
import torch
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from dataloader import dataset_factory

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import default, ismap, isimage, mean_flat, count_params
from einops import rearrange
from cldm_seg.util import ClassEmbeddingManager


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

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
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


class ControlLDM_Encoded(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, mask_encode_mode,
                 class_emb_manager_config,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.automatic_optimization = False
        self.mask_encode_mode = mask_encode_mode
        self.class_embedding_manager = instantiate_from_config(class_emb_manager_config)

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


    def train_dataloader(self):
        # Dataset
        augment_dict = {}
        augment_dict['augment_p'] = 0.5
        augment_dict['horizontal_flip'] = True
        print('Train on ', self.dataset)
        dataset = dataset_factory(self.dataset)(
            mode='train', train_mode=True, augment_dict=augment_dict,
            mask_encode_mode=self.mask_encode_mode,
            drop_caption_ratio=self.drop_caption_ratio,
        )
        self.label_color_map = dataset.color_map # used for segmentation map visualization
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
            mode='val', train_mode=True, augment_dict=augment_dict,
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

        return [g_opt],[]


    @torch.no_grad()
    def get_input(self, batch, k, bs=None,
                  return_first_stage_outputs=False, force_c_encode=False,
                  return_original_cond=False, return_x=False,
                  *args, **kwargs):
        #x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

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
        xc = batch[self.cond_stage_key]
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
        if bs is not None:
            control = control[:bs]
        control = self.class_embedding_manager(control) # [bs, c, h, w]
        control = control.to(self.device)
        #control = einops.rearrange(control, 'b h w c -> b c h w')

        control = control.to(memory_format=torch.contiguous_format).float()
        return_dict['c_concat'] = [control]

        return z, return_dict


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        '''
        Get noise prediction
        '''

        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        # use prompt
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=None,
                only_mid_control=self.only_mid_control
            )
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control
            )

        return eps

    def run_G(self, batch, noise=None,return_input=False):
        '''
        return_input: True, if requires diffusion loss
        '''

        # reuse the timestep
        x_start, cond = self.get_input(batch, self.first_stage_key)

        # TODO: only sample from 50 steps
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        cond['noise'] = noise
        cond['timestep'] = t
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # main func to get the model prediction
        model_output = self.apply_model(x_noisy, t, cond,)
        if return_input:
            return model_output, x_start, cond
        else:
            return model_output

    def cal_G_loss(self, model_output, x_start, cond):
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        t = cond['timestep']

        #  TODO: use different prediction mode
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

    def training_step(self, batch, batch_idx):
        '''
        batch: output of the dataloader, e.g., a dict
        '''
        g_opt = self.optimizers()
        #torch.distributed.barrier()
        #print('---->',self.global_rank, batch['label'].shape[0], batch_idx)

        # ------- Update Generator ------- #
        model_output_fake, x_start, cond = self.run_G(
            batch, return_input=True,
        )
        loss_G, p_loss_dict = self.cal_G_loss(model_output_fake, x_start, cond)

        # loss_G.backward()
        self.manual_backward(loss_G/self.num_gradient_accumulation)

        if (batch_idx + 1) % self.num_gradient_accumulation == 0:
            g_opt.step()
            g_opt.zero_grad(set_to_none=True)

            if self.global_rank == 0:
                # G loss
                self.logger.experiment.add_scalar('loss/G_loss', loss_G, self.num_samples)
                self.logger.experiment.add_scalar('loss/G_loss_simple', p_loss_dict['train/loss_simple'], self.num_samples)
                self.logger.experiment.add_scalar('loss/G_loss_vlb', p_loss_dict['train/loss_vlb'], self.num_samples)
            self.log("loss", loss_G, sync_dist=True)  # for checkpoint monitoring

        return loss_G
    ################################################################

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)


    # TODO: better logging!
    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=7.5, unconditional_guidance_label=None,
                   use_ema_scope=True,generator=None,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        #c_cat = self.class_embedding_manager(c_cat)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16).cpu()
        log["reconstruction"] = self.decode_first_stage(z).cpu()
        #log["control"] = c_cat * 2.0 - 1.0 # TODO: change here!
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

        return log

    def label_to_seg_map(self, label):
        out = self.label_color_map[label] # [bs, h, w, 3]
        out = rearrange(out, 'bs h w c -> bs c h w')
        out = (torch.from_numpy(out) / 255.0) * 2.0 - 1.0
        return out # (-1,1)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, generator=None, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 2, w // 2) # cond["c_concat"] 128x128
        random_shape = (batch_size, self.channels, h // 2, w // 2)
        x_T = torch.randn(random_shape, generator=generator, device=self.device)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False,x_T=x_T, **kwargs)
        return samples, intermediates

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
