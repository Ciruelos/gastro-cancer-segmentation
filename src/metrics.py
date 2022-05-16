from typing import Any, Dict, Optional

import torch
import numpy as np
import torchmetrics
from torchmetrics.utilities.distributed import reduce
from monai.metrics.utils import get_mask_edges, get_surface_distance


class Dice(torchmetrics.ConfusionMatrix):

    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = 'elementwise_mean',
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ):
        super().__init__(
            num_classes=2,
            normalize=None,
            threshold=threshold,
            multilabel=False,
            compute_on_step=compute_on_step,
            **kwargs,
        )

        self.reduction = reduction

    def compute(self):
        intersection = torch.diag(self.confmat)
        union = self.confmat.sum(0) + self.confmat.sum(1)

        scores = 2 * intersection.float() / union.float()
        return reduce(scores, reduction=self.reduction)


class HausdorffDistance(torchmetrics.Metric):
    def __init__(
        self,
        threshold: float = .5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs: Dict[str, Any]
    ):
        super().__init__(device=device, **kwargs)
        self.threshold = threshold

        self.add_state('distances', default=[], dist_reduce_fx='sum')

    def update(self, preds, targets):
        preds = (preds >= self.threshold).float().detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        hausdorf_dist = 1 - np.mean([self.hausdorff_dist(pred, target) for pred, target in zip(preds, targets)])
        hausdorf_dist = torch.tensor(hausdorf_dist).to(self.device)

        self.distances.append(hausdorf_dist)

        return hausdorf_dist

    def compute(self):
        return torch.tensor(self.distances).mean()

    @staticmethod
    def hausdorff_dist(pred: np.ndarray, target: np.ndarray) -> float:

        if np.all(pred == target):
            return 0.0

        edges_pred, edges_gt = get_mask_edges(pred, target)
        surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric='euclidean')

        if surface_distance.shape == (0,):
            return 0.0

        dist = surface_distance.max()
        max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2))

        if dist > max_dist:
            return 1.0

        return dist / max_dist
