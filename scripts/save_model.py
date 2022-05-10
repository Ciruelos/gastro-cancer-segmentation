import re
import pprint
from pathlib import Path
from argparse import ArgumentParser

import yaml
import torch
import albumentations as A

from src.model import Model
from src.data_loading import DataModule


def get_parser():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint-dir', type=str, help='Directory of the model')
    parser.add_argument('--output-dir', type=str, help='Directory to save the model')

    return parser


def save_model(checkpoint_dir: str, output_dir: str):
    """ "Saves PyTorch Lightning weights and hparams and Albumentations transforms into a deployable directory."""

    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    checkpoint_path = [str(p) for p in Path(checkpoint_dir).rglob('*.ckpt')][0]
    config_path = [str(p) for p in Path(checkpoint_dir).rglob('*.yaml')][0]

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    metric, value = re.findall(r'epoch=[0-9]+_(.+)=(.+).ckpt', checkpoint_path)[0]
    config[metric] = value

    model = Model.load_from_checkpoint(checkpoint_path, **config).cpu()

    pprint.pprint(config)

    model.freeze()

    output_model_path = output_dir.joinpath('model.pt')
    output_basic_transforms_path = output_dir.joinpath('basic_transforms.yaml')
    output_config_path = output_dir.joinpath('hparams.yaml')

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.to_torchscript(output_model_path)
    except Exception:
        print('Unscriptable: fallback to trace method.')
        x = torch.randn(2, 3, *(config['input_size'],) * 2)
        # passing strict=False to be allowed to return dict in the model
        model.to_torchscript(output_model_path, method='trace', example_inputs=x, strict=False)

    A.save(DataModule.get_basic_transforms(**config), output_basic_transforms_path, 'yaml')

    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    save_msg = 'Saved:' + (f'\n\t{output_model_path}' f'\n\t{output_basic_transforms_path}' f'\n\t{output_config_path}')
    print(save_msg)


if __name__ == '__main__':
    args = get_parser().parse_args()

    save_model(**vars(args))
