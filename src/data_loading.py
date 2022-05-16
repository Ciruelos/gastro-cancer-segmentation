from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


from src.constants import CLASSES


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, df, transforms
    ):
        super().__init__()

        self.df = df
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        sample = self.df.loc[index]
        image = self.load_image(sample.image_path)

        masks = []
        for cs in CLASSES:
            if not pd.isna(sample[cs]):
                decoded_mask = rle_decode(sample[cs], shape=image.shape)[..., 0]
            else:
                decoded_mask = np.zeros((sample.height, sample.width), dtype='float32')

            masks.append(decoded_mask)

        mask = np.stack(masks, axis=2).astype('float32')

        transformed = self.transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        return image, {name: m.int() for name, m in zip(CLASSES, mask)}

    @staticmethod
    def load_image(image_path: str):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype('float32')
        image = np.tile(image[..., None], [1, 1, 3])
        image /= image.max()
        return image


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_path: str = 'data/train.csv',
        images_dir: str = 'data/train/',
        input_size: int = 224,
        batch_size: int = 2,
        num_workers: int = 4,
        val_size: float = 0.2,
        **kwargs
    ):
        super().__init__()
        self.df = pd.read_csv(df_path)
        self.images_dir = images_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size

    def prepare_data(self) -> None:
        df_new = pd.DataFrame({'id': self.df['id'][::3]})

        df_new['large_bowel'] = self.df['segmentation'][::3].values
        df_new['small_bowel'] = self.df['segmentation'][1::3].values
        df_new['stomach'] = self.df['segmentation'][2::3].values

        df_new['case'] = df_new['id'].apply(lambda x: x.split('_')[0])
        df_new['day'] = df_new['id'].apply(lambda x: x.split('_')[1])
        df_new['slice'] = df_new['id'].apply(lambda x: x.split('_')[3])

        paths = []
        heights = []
        widths = []
        for case, day, slice in zip(df_new['case'], df_new['day'], df_new['slice']):
            scans_dir = Path(self.images_dir).joinpath(case, case + '_' + day, 'scans')
            for image_path in scans_dir.glob('*'):
                if 'slice_' + slice in image_path.name:
                    paths.append(str(image_path))
                    # NOTE: widths and heigths are wrong in competition description
                    heights.append(int(image_path.name.split('_')[3]))
                    widths.append(int(image_path.name.split('_')[2]))
                    break

        df_new['image_path'] = paths
        df_new['width'] = widths
        df_new['height'] = heights

        df_new = df_new.reset_index(drop=True)
        self.df = df_new

    def setup(self, stage=None):

        train_df, val_df = train_test_split(self.df, test_size=self.val_size, shuffle=True, random_state=42)
        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)

        self.train_dataset = Dataset(train_df, transforms=self.get_aug_transforms(self.input_size))

        self.val_dataset = Dataset(val_df, transforms=self.get_basic_transforms(self.input_size))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @staticmethod
    def get_aug_transforms(input_size, **kwargs):
        return A.Compose(
            [
                A.Rotate(limit=30, p=1.0),
                A.RandomResizedCrop(input_size, input_size, (0.8, 1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.ImageCompression()], p=1 / 3),
                ToTensorV2(transpose_mask=True),
            ]
        )

    @staticmethod
    def get_basic_transforms(input_size, **kwargs):
        return A.Compose(
            [
                A.Resize(input_size, input_size),
                ToTensorV2(transpose_mask=True),
            ]
        )


def rle_decode(mask_rle, shape, color=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = color
    return img.reshape(shape)
