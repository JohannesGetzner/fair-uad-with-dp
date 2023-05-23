"""
Bootstrapping method from https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.data import _flatten_dict


class MyMetricCollection(MetricCollection):
    def __init__(
            self,
            metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
    ) -> None:
        super().__init__(metrics)

    def compute(self, **kwargs) -> Dict[str, Any]:
        """Compute the result for each metric in the collection."""
        res = {k: m.compute(**kwargs) for k, m in self.items(keep_base=True, copy_state=False)}
        res = _flatten_dict(res)
        return {self._set_name(k): v for k, v in res.items()}


def build_metrics(subgroup_names: List[str]) -> MyMetricCollection:
    classification_metrics = MyMetricCollection({
        'AUROC': AUROC(subgroup_names),
        'AveragePrecision': AveragePrecision(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@95tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
        'avg_anomaly_score': AvgAnomalyScore(subgroup_names),
    })
    return classification_metrics


class AvgAnomalyScore(Metric):
    """
    Just a wrapper to bootstrap the anomaly score if necessary
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Optional[Tensor] = None):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(subgroups)
        self.preds.append(preds)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        anomaly_score = preds[subgroups == subgroup].mean()
        if anomaly_score.isnan():
            anomaly_score = torch.tensor(0.)
        return anomaly_score

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        fake_targets = torch.zeros_like(preds)
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, fake_targets, subgroups, subgroup)
            res[f'{subgroup_name}_anomaly_score'] = result
        return res


class AUROC(Metric):
    """
    Computes the AUROC naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the area under the ROC curve
        auroc = roc_auc_score(subgroup_targets, subgroup_preds)

        return torch.tensor(auroc)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AUROC'] = result
        return res


class AveragePrecision(Metric):
    """
    Computes the Average Precision (AP) naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the Average Precision
        ap = average_precision_score(subgroup_targets, subgroup_preds)

        return torch.tensor(ap)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AP'] = result
        return res


class TPR_at_FPR(Metric):
    """True positive rate at x% FPR."""
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str], xfpr: float = 0.05):
        super().__init__()
        assert 0 <= xfpr <= 1
        self.xfpr = xfpr
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor,
                         subgroup: int, xfpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        # Compute FPR threshold for total dataset
        fpr, _, thresholds = roc_curve(targets, preds, pos_label=1)
        threshold_idx = np.argwhere(fpr < xfpr)
        if len(threshold_idx) == 0:
            threshold_idx = -1
        else:
            threshold_idx = threshold_idx[-1, 0]
        threshold = thresholds[threshold_idx]
        # Compute TPR for subgroup
        targets = targets[subgroups == subgroup]  # [N_s]
        preds = preds[subgroups == subgroup]  # [N_s]
        preds_bin = (preds > threshold).long()  # [N_s]
        tpr = (preds_bin * targets).sum() / (targets.sum() + 1e-8)
        return tpr

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup, self.xfpr)
            res[f'{subgroup_name}_tpr@{self.xfpr}'] = result
        return res


class FPR_at_TPR(Metric):
    """False positive rate at x% TPR."""
    is_differentiable: bool = False
    higher_is_better: bool = False

    def __init__(self, subgroup_names: List[str], xtpr: float = 0.95):
        super().__init__()
        assert 0 <= xtpr <= 1
        self.xtpr = xtpr
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b] (dummy)
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor,
                         subgroup: int, xtpr: float):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(1.)
        # Compute TPR threshold for total dataset
        _, tpr, thresholds = roc_curve(targets, preds, pos_label=1)
        threshold = thresholds[np.argwhere(tpr > xtpr)[0, 0]]
        # Compute FPR for subgroup
        targets = targets[subgroups == subgroup]  # [N_s]
        preds = preds[subgroups == subgroup]  # [N_s]
        preds_bin = (preds > threshold).long()  # [N_s]
        fpr = (preds_bin * (1 - targets)).sum() / ((1 - targets).sum() + 1e-8)
        return fpr

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup, self.xtpr)
            res[f'{subgroup_name}_fpr@{self.xtpr}'] = result
        return res


class AvgMeter:
    def __init__(self):
        self.reset()
        self.value: float
        self.n: int

    def reset(self):
        self.value = 0.0
        self.n = 0

    def add(self, value):
        self.value += value
        self.n += 1

    def compute(self):
        return self.value / self.n


class AvgDictMeter:
    def __init__(self):
        self.reset()
        self.values: dict
        self.n: int

    def reset(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}


if __name__ == '__main__':
    subgroup_names = ['subgroup1', 'subgroup2']
    metrics = MyMetricCollection({
        'avg_anomaly_score': AvgAnomalyScore(subgroup_names),
        'AUROC': AUROC(subgroup_names),
        'AveragePrecision': AveragePrecision(subgroup_names),
        'tpr@5fpr': TPR_at_FPR(subgroup_names, xfpr=0.05),
        'fpr@5tpr': FPR_at_TPR(subgroup_names, xtpr=0.95),
    })

    scores = torch.tensor([0.8, 0.6, 0.2, 0.9, 0.5, 0.7, 0.3])
    labels = torch.tensor([1, 0, 0, 1, 1, 0, 1])
    subgroups = torch.tensor([0, 1, 0, 0, 1, 1, 0])
    metrics.update(subgroups, scores, labels)
    print(metrics.compute(do_bootstrap=False))
