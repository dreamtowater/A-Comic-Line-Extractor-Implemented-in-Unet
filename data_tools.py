import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader, Dataset
from torchvision import transforms as tf

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageSet(Dataset):
    def __init__(self, folderPath, img_Size=[512, 512], num_classes=128):
        super().__init__()
        self.folderPath = folderPath
        self.img_skts = os.listdir(folderPath)
        self.img_Size = img_Size

        self.transforms = build_transform(img_Size, False)
        self.target_transforms = build_target_transform(img_Size, num_classes)

    def __len__(self):
        return len(self.img_skts)

    @torch.no_grad
    def __getitem__(self, index):
        img_skt_path = os.path.join(self.folderPath, self.img_skts[index])
        image_sketch = Image.open(img_skt_path)

        image_sketch = tf.ToTensor()(image_sketch)
        image, sketch = image_sketch.chunk(2, dim=-1)

        image = self.transforms(image)
        sketch = self.target_transforms(sketch)

        return image, sketch
    

def build_transform(img_Size, need_toTensor, is_image_sketch=False):
    t = []
    if need_toTensor:
        t.append(tf.ToTensor())
    if is_image_sketch:
        t.append(tf.Lambda(lambda x: x.chunk(2, dim=-1)[0]))
    t.append(tf.Lambda(
        lambda x: F.interpolate(x.unsqueeze_(0), img_Size, mode='bicubic').squeeze_(0)))
    t.append(tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return tf.Compose(t)


def build_target_transform(img_Size, num_classes):
    t = []
    t.append(tf.Lambda(
        lambda x: F.interpolate(x.unsqueeze_(0), img_Size, mode='bicubic').squeeze_(0)))
    t.append(tf.Lambda( # 量化
        lambda x: (x[0] * num_classes).floor_().to(torch.int64).clamp_(0, num_classes-1)))
    return tf.Compose(t)