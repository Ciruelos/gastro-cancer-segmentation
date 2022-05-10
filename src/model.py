import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.constants import CLASSES
from src.metrics import Dice, HausdorffDistance


class Model(pl.LightningModule):
    LOSSES = {
        'bce': torch.nn.BCEWithLogitsLoss(),
        'dice': smp.losses.DiceLoss(mode='binary', from_logits=True, smooth=1),
    }

    def __init__(
        self,
        arch: str = 'deeplabv3plus',
        encoder_name: str = 'resnext50_32x4d',
        loss_name: str = 'bce',
        use_scse: bool = False,
        encoder_weights: str = 'swsl',
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        learning_rate: float = 2e-4,
        **kwargs
    ):
        super().__init__()

        self.decoder_attention_type = 'scse' if use_scse else None

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=len(CLASSES),
            decoder_attention_type=self.decoder_attention_type
        )
        self.model.encoder.eval()

        self.learning_rate = learning_rate
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        self.loss_name = loss_name
        self.loss = self.LOSSES[self.loss_name]

        self.metrics = torch.nn.ModuleDict({
            'dice': torch.nn.ModuleDict({cs: Dice(compute_on_step=False) for cs in CLASSES}),
            'haus_d': torch.nn.ModuleDict({cs: HausdorffDistance(device=self.device) for cs in CLASSES})
        })

        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return dict(zip(CLASSES, y.unbind(dim=1)))

    def training_step(self, batch, batch_idx):
        x, target = batch

        pred = self.forward(x)

        loss = 0
        for cs in CLASSES:
            loss += self.loss(pred[cs], target[cs].float())
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch

        pred = self.forward(x)
        loss = 0.
        for cs in CLASSES:
            loss += self.loss(pred[cs], target[cs].float())
            for metric in self.metrics:
                self.metrics[metric][cs].update(pred[cs].sigmoid(), target[cs])

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = {name: [] for name in self.metrics}

        for cs in CLASSES:
            for metric in self.metrics:
                metric_cs = self.metrics[metric][cs].compute()
                self.metrics[metric][cs].reset()

                metrics[metric].append(metric_cs)

                self.log(f'{cs}_{metric}', metric_cs, prog_bar=True, sync_dist=True)

        metrics = {name: torch.tensor(m).mean() for name, m in metrics.items()}

        for metric in metrics:
            self.log(f'gobal_{metric}', metrics[metric], prog_bar=True, sync_dist=True)

        kaggle_metric = .4 * metrics['dice'] + .6 * metrics['haus_d']

        self.log('kaggle_metric', kaggle_metric, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.plateau_factor, patience=self.plateau_patience, mode='max', verbose=True
            ),
            'monitor': 'kaggle_metric',
        }
