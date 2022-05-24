import random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt

from src.data_loading import DataModule


def get_parser():
    parser = ArgumentParser()

    parser.add_argument('--images-dir', default='data/train', type=Path)
    parser.add_argument('--output-dir', default='trash/data_aug', type=Path)

    return parser


def load_image(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype('float32')
    return image


if __name__ == '__main__':
    args = get_parser().parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    transforms = DataModule.get_aug_transforms(input_size=256, debug=True)

    images_paths = list(args.images_dir.rglob('*.png'))

    while True:
        image_path = random.choice(images_paths)
        image = load_image(str(image_path))
        plt.imshow(image)
        plt.savefig(args.output_dir.joinpath('original_image.png'))
        transformed = transforms(image=image)['image']
        plt.imshow(transformed)
        plt.savefig(args.output_dir.joinpath('transformed_image.png'))
        input('Press for another image...')
