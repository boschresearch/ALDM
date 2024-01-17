from .cityscapes import Cityscapes
from .ade20k import  ADE20K
from .coco_stuff import COCO_Stuff

__all__ = ["dataset_factory"]

__DATA_DICT__ = {
  "cityscapes": Cityscapes,
  "ade": ADE20K,
  "coco": COCO_Stuff,
}

def dataset_factory(model_type):
  return __DATA_DICT__[model_type]