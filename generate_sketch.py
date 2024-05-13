import os
from PIL import Image
import argparse
import json

import torch
from torchvision import transforms as tf

from data_tools import build_transform
from modules import Unet
from hps_utils import HParams


@torch.no_grad
def generate_one_sketch(model, img_path, is_image_sketch, img_Size, save_name, **model_kwargs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if type(model) is str:
        state_dict = torch.load(model, map_location=device)
        model = Unet(**model_kwargs)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = Image.open(img_path)
    image = build_transform(img_Size, True, is_image_sketch)(image).to(device)

    sketch = model(image.unsqueeze_(0))
    num_classes = sketch.shape[1]
    sketch = sketch.argmax(dim=1) / (num_classes-1)
    sketch = tf.ToPILImage()(sketch.clamp_(0, 1))
    sketch.save(save_name)
    


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='State_dict file path')
    parser.add_argument('-i', '--image_path', type=str, required=True,
                        help='Image file path')
    parser.add_argument('-c', '--config_path', type=str, default='./config.json',
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config_path, "r") as f: 
        hps = HParams(**json.loads(f.read()))

    model_kwargs = dict(
        in_channels = hps.image_channels, 
        start_channels = hps.start_channels, 
        out_channels = hps.num_classes, 
        n_steps = hps.n_steps
    )

    save_name = os.path.basename(args.image_path)[:-4] + "-sketch.png"

    generate_one_sketch(args.model_path, args.image_path, False, hps.img_Size, save_name, **model_kwargs)