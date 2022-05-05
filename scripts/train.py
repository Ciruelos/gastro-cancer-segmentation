import datetime
from argparse import ArgumentParser

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.model import Model
from src.data_loading import DataModule


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    # Data
    parser.add_argument('--df-path', default='data/train.csv', type=str, help=h)
    parser.add_argument('--images-dir', default='data/train', type=str, help=h)
    parser.add_argument('--input-size', default=224, type=int, help=h)
    parser.add_argument('--batch-size', default=2, type=int, help=h)
    parser.add_argument('--val-size', default=0.2, type=float, help=h)
    parser.add_argument('--num-workers', default=4, type=int, help=h)

    # Model
    parser.add_argument('--arch', default='deeplabv3plus', type=str, help=h)
    parser.add_argument(
        '--encoder-name', default='resnext50_32x4d', choices=smp.encoders.get_encoder_names(), type=str, help=h
    )
    parser.add_argument('--ecoder-weights', default='swsl', type=str, help=h)

    # Train
    parser.add_argument('--learning-rate', default=1e-3, type=float, help=h)
    parser.add_argument('--earlystopping-patience', default=3, type=int, help=h)
    parser.add_argument('--accumulate-grad-batches', default=32, type=int, help=h)

    # Other
    parser.add_argument('--log-name', default='default', type=str, help=h)
    parser.add_argument('--comments', default='', type=str, help=h)

    return parser


def CHECKPOINT_FILENAME(monitor):
    return '{epoch}_{' + monitor + ':.5f}'


if __name__ == '__main__':
    args = get_parser().parse_args()

    data_module = DataModule(**vars(args))

    model = Model(
        date=str(datetime.datetime.now().date()),
        **vars(args)
    )

    callbacks = [
        pl.callbacks.ProgressBar(),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(monitor='macro_iou', patience=args.earlystopping_patience, mode='max'),
        pl.callbacks.ModelCheckpoint(monitor='macro_iou', mode='max', filename=CHECKPOINT_FILENAME('macro_iou')),
    ]

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        benchmark=True,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=pl.loggers.TensorBoardLogger(save_dir='lightning_logs/', name=args.log_name),
    )

    trainer.fit(model, data_module)
