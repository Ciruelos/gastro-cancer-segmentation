import torch
import torchmetrics
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from src.constants import CLASSES


class Model(pl.LightningModule):
    def __init__(
        self,
        arch: str = 'deeplabv3plus',
        encoder_name: str = 'resnext50_32x4d',
        loss_name: str = 'bce',
        encoder_weights: str = 'swsl',
        plateau_factor: float = 0.5,
        plateau_patience: int = 2,
        learning_rate: float = 2e-5,
        **kwargs
    ):
        super().__init__()

        self.model = smp.create_model(
            arch=arch, encoder_name=encoder_name, encoder_weights=encoder_weights, classes=len(CLASSES)
        )
        self.model.encoder.eval()

        self.learning_rate = learning_rate
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        self.loss_name = loss_name

        if self.loss_name == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss()

        elif self.loss_name == 'dice':
            self.loss = smp.losses.DiceLoss(mode='binary', from_logits=True, smooth=1)

        self.metrics = torch.nn.ModuleDict(
            {cs: torchmetrics.classification.JaccardIndex(2) for cs in CLASSES}
        )

        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return dict(zip(CLASSES, y.unbind(dim=1)))

    def training_step(self, batch, batch_idx):
        x, target = batch

        pred = self.forward(x)

        loss = 0.
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
            self.metrics[cs].update(pred[cs].sigmoid(), target[cs])

        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        macro_metric = 0.0
        for cs in CLASSES:
            metric = self.metrics[cs].compute()
            self.log(f'{cs}_iou', metric, prog_bar=True, sync_dist=True)
            self.metrics[cs].reset()
            macro_metric += metric
        macro_metric = macro_metric / len(CLASSES)
        self.log('macro_iou', macro_metric, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.plateau_factor, patience=self.plateau_patience, mode='max', verbose=True
            ),
            'monitor': 'macro_iou',
        }
