import sys
sys.path.append(".")
sys.path.append("..")
import cv2
from einops import  rearrange
import gradio as gr
import numpy as np
import torch
import random
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import config
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from dataloader import dataset_factory
from gradio_demo.helper import randomize_seed_fn
gr.close_all()


##########################
# Default model settings #
##########################
model_id = 0

cur_dir = os.path.dirname(__file__)
project_root_dir = os.path.dirname(cur_dir)
print(project_root_dir)

checkpoint_dict = {
    0: 'checkpoint/cityscapes_step9.ckpt',
    1: 'checkpoint/cityscapes_step6.ckpt',
}
model_config = 'models/cldm_seg_cityscapes_multi_step_D.yaml'
checkpoint_dir = os.path.join(project_root_dir, checkpoint_dict[model_id])
model_config = os.path.join(project_root_dir, model_config)

model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(checkpoint_dir, location='cpu'), strict=False) # , location='cuda'
model = model.cuda()
ddim_sampler = DDIMSampler(model)
W, H = 1024, 512
if hasattr(model, 'model_attend'):
        model.model_attend.image_ratio = W / H
        model.model_attend.attention_store.image_ratio = W / H

augment_dict = {}
augment_dict['augment_p'] = -1
augment_dict['horizontal_flip'] = False
augment_dict['center_crop'] = True
print('Validate on ', 'cityscapes')

dataset = dataset_factory('cityscapes')(
    mode='val', train_mode=True, augment_dict=augment_dict,
    crop_height=H, crop_width=W,
    img_height=H, img_width=W,
    mask_encode_mode='id',
)
label_color_map = dataset.color_map
use_language_enc = True


def process(image_id, prompt, a_prompt, n_prompt, num_samples, image_width, image_height,
            guess_mode, strength, scale, seed, eta, show_seg_only=False,ddim_steps=25):
    with torch.no_grad():
        #input_image = HWC3(input_image)
        #detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        #img = resize_image(input_image, image_resolution)
        #H, W, C = img.shape

        data = dataset[image_id]
        #img = data['image']
        control = data['hint'] # label
        label_id = control.clone()
        detected_map = label_color_map[label_id[0].cpu().detach().numpy()]

        if not show_seg_only and use_language_enc:
            print(control.shape)
            control = model.class_embedding_manager(control.unsqueeze(0))  # [bs, c, h, w]
            control = control.to(model.device)
            control = control.squeeze(0)
        text = data['txt']
        print('---> text: ', text)

        #prompt = text + ',' + prompt
        prompt = prompt

        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        #control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        if use_language_enc:
            detected_map = detected_map.astype(np.uint8)
        else:
            detected_map = (control * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
            detected_map = rearrange(detected_map, 'c h w -> h w c')

        if show_seg_only:
            return [detected_map]

        control = torch.stack([control for _ in range(num_samples)], dim=0)
        # print(control.shape)
        control = control.clone().cuda()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


demo = gr.Blocks().queue()
with demo:
    with gr.Row():
        gr.Markdown("##  🧨 ALDM: Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive (ICLR 2024) 🧨")
    with gr.Row():
        with gr.Column():
            #input_image = gr.Image(source='upload', type="numpy")
            image_id = gr.Slider(label="Cityscapes Validation Set", minimum=0, maximum=499, value=0, step=1)
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        randomize_seed = gr.Checkbox(label='Randomize seed',value=True)
                    with gr.Column(scale=1):
                        show_seg_only = gr.Checkbox(label='Show segmentation only', value=False)

                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=7.0, step=0.1)
                # ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=25, step=1)

                #  vvv Stable Choices vvvvvvvv #
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=2, step=1)
                image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=1024, step=64)
                image_height = gr.Slider(label="Image Height", minimum=256, maximum=768, value=512, step=64)
                #detect_resolution = gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [image_id, prompt, a_prompt, n_prompt, num_samples, image_width, image_height,
           guess_mode, strength, scale, seed, eta, show_seg_only]
    run_button.click(
        fn=randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed,
    ).then(
        fn=process, inputs=ips, outputs=[result_gallery]
    )


#demo.launch(server_name='0.0.0.0')
print('launch demo')
demo.launch(server_port=9774)


# Examples
# 393062 'snowy street'