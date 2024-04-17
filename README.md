


# Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive (ALDM)   

:fire:  Official implementation of "Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive" (ICLR 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2401.08815-red)](https://arxiv.org/pdf/2401.08815.pdf) [![Static Badge](https://img.shields.io/badge/Project_Page-ALDM-blue)](https://yumengli007.github.io/ALDM) [![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-Green)
](https://huggingface.co/Yumeng/ALDM/tree/main)

![overview](docs/overview.jpg)
![result](docs/result.png)


<br />


## Getting Started

Our environment is built on top of [ControlNet](https://github.com/CompVis/stable-diffusion):
```
conda env create -f environment.yaml  
conda activate aldm
pip install mit-semseg # for segmentation network UperNet
```

## Pretrained Models
Pretrained models [ade20k_step9.ckpt](https://huggingface.co/Yumeng/ALDM/blob/main/ade20k_step9.ckpt) and [cityscapes_step9.ckpt](https://huggingface.co/Yumeng/ALDM/blob/main/cityscapes_step9.ckpt)  can be downloaded from [here](https://huggingface.co/Yumeng/ALDM/tree/main). They should be stored in the [checkpoint](checkpoint) folder.


## Dataset Preparation

Datasets should be structured as follows to enable ALDM training. Dataset path should be adjusted accordingly in [dataloader/cityscapes.py](https://github.com/boschresearch/ALDM/blob/3edbad80eaf208eacd0eb4a161a4998a0c75fb50/dataloader/cityscapes.py#L151-L152),  [dataloader/ade20k.py](https://github.com/boschresearch/ALDM/blob/3edbad80eaf208eacd0eb4a161a4998a0c75fb50/dataloader/ade20k.py#L144-L145) and [dataloader/coco_stuff.py](https://github.com/boschresearch/ALDM/blob/dc9ed8cdd4787485db2ab07c3b104361af04d0d3/dataloader/coco_stuff.py#L132-L133). Check [convert_coco_stuff_id.ipynb](https://github.com/boschresearch/ALDM/blob/main/convert_coco_stuff_id.ipynb) for converting coco stuff labels. 

<details>
  <summary>Click to expand</summary>
  
```
datasets
├── cityscapes
│   ├── gtFine
│       ├── train 
│       └── val 
│   └── leftImg8bit
│       ├── train 
│       └── val 
├── ADE20K
│   ├── annotations
│       ├── train 
│       └── val 
│   └── images
│       ├── train 
│       └── val 
├── COCOStuff
│   ├── train_img
│   ├── val_img
│   ├── train_label
│   ├── val_label
│   ├── train_label_convert # New: after converting
│   └── val_label_convert # New: after converting
└── ...
```
</details>

## Inference
We provide three ways for testing: (1) [JupyterNotebook](demo_generation.ipynb), (2) [Gradio Demo](gradio_demo/gradio_seg2image_cityscapes.py), (3) [Bash scripts](bash_script).

1. [JupyterNotebook](demo_generation.ipynb): we provided one sample layout for quick test without requiring dataset setup.
2. [Gradio Demo](gradio_demo/gradio_seg2image_cityscapes.py):

	> Run the command after the dataset preparation.    

	```
	gradio gradio_demo/gradio_seg2image_cityscapes.py
	```
	![demo](docs/gradio_demo.png)

<br />

3. [Bash scripts](bash_script): we provide some bash scripts to enable large scale generation for the whole dataset. The synthesized data can be further used for training downstream models, e.g., semantic segmentation networks.


## Citation
If you find our work useful, please star  this repo and cite: 

```
@inproceedings{li2024aldm,
  title={Adversarial Supervision Makes Layout-to-Image Diffusion Models Thrive},
  author={Li, Yumeng and Keuper, Margret and Zhang, Dan and Khoreva, Anna},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).


## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. 


## Contact     

Please feel free to open an issue or contact personally if you have questions, need help, or need explanations. Don't hesitate to write an email to the following email address:
liyumeng07@outlook.com

