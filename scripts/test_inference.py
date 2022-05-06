
from pathlib import Path

import cv2
import yaml
import torch
import albumentations as A


def load_model(model_dir):

    model_dir = Path(model_dir)
    model_path = model_dir.joinpath('model.pt')
    basic_transforms_path = model_dir.joinpath('basic_transforms.yaml')
    config_path = model_dir.joinpath('hparams.yaml')

    model = torch.jit.load(model_path)
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    basic_transforms = A.load(basic_transforms_path, 'yaml')

    return model, basic_transforms, config


model, basic_transforms, _ = load_model('saved_models/test')

image = cv2.imread('data/train/case2/case2_day1/scans/slice_0010_266_266_1.50_1.50.png')[..., ::-1]

X = basic_transforms(image=image)['image'][None, ...]

print((model(X)['large_bowel'].cpu().sigmoid() >= .5).unique())
