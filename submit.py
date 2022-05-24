from pathlib import Path

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A


IMAGES_DIR = Path('data/train')
MODEL_DIR = Path('saved_models/submission-11')
OUTPUT_DIR = Path('trash')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_SUBMISSION = pd.read_csv('data/train.csv')
SAMPLE_SUBMISSION.rename(columns={'segmentation': 'predicted'}, inplace=True)


def load_image(path):
    # int16 -> float32
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')

    image = np.tile(image[..., None], [1, 1, 3])

    # Scale to [0, 255]
    image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)

    return image


def rle_encode(mask):

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def load_model(model_dir):

    model_dir = Path(model_dir)
    model_path = model_dir.joinpath('model.pt')
    basic_transforms_path = model_dir.joinpath('basic_transforms.yaml')
    config_path = model_dir.joinpath('hparams.yaml')

    model = torch.jit.load(model_path)
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    basic_transforms = A.load(basic_transforms_path, 'yaml')

    return model, basic_transforms, config


submission = {'id': [], 'class': [], 'predicted': []}

model, basic_transforms, _ = load_model(MODEL_DIR)
model.to(DEVICE)

for image_path in tqdm(list(IMAGES_DIR.rglob('*.png'))):
    image_id = f'{image_path.parent.parent.name}_slice_{image_path.name.split("_")[1]}'

    image = load_image(str(image_path))
    X = basic_transforms(image=image)['image'][None, ...]

    Y = model(X.to(DEVICE))

    for cs, y in Y.items():
        submission['id'].append(image_id)
        submission['class'].append(cs)
        mask = (y[0].sigmoid().cpu().numpy() >= .5).astype('uint8')
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        assert mask.shape == image.shape[:2]
        submission['predicted'].append(rle_encode(mask))


if len(submission['id']) != 0:
    submission = pd.DataFrame(submission)
    del SAMPLE_SUBMISSION['predicted']
    submission = SAMPLE_SUBMISSION.merge(submission, on=['id', 'class'])
    submission.to_csv(OUTPUT_DIR.joinpath('submission.csv'), index=False)

else:
    submission = pd.DataFrame(submission)
    submission.to_csv(OUTPUT_DIR.joinpath('submission.csv'), index=False)
