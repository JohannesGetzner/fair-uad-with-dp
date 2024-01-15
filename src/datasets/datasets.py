from typing import Callable, Dict, List, Tuple, Any, Optional
from torch import Tensor
import torch
from torch.utils.data import Dataset
from src.datasets.data_utils import load_dicom_img


class NormalDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            load_fn: Callable = load_dicom_img
    ):
        self.data = data
        self.filenames = data.copy()
        for i, d in enumerate(self.data):
            if isinstance(d, str):
                self.data[i] = transform(load_fn(d))
        self.labels = labels
        self.meta = meta
        self.load_fn = load_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, int, int]:
        img = self.data[idx]
        label = self.labels[idx]
        meta = self.meta[idx]
        return img, label, meta


class AnomalFairnessDataset(Dataset):
    def __init__(
            self,
            data: Dict[str, List[str]],
            labels: Dict[str, List[int]],
            meta: Dict[str, List[int]],
            transform=None,
            load_fn: Callable = load_dicom_img):
        super().__init__()
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn

    def __len__(self) -> int:
        return min([len(v) for v in self.data.values()])

    def __getitem__(self, idx: int) -> tuple[dict, dict, dict]:
        img = {k: self.transform(self.load_fn(v[idx])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta


class MEMMAP_NormalDataset(Dataset):
    """
    Anomaly detection training dataset.
    Receives a list of filenames
    """

    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            index_mapping: Optional[List[int]] = None,
            load_fn: Callable = lambda x: x,
            filenames = None
    ):
        """
        :param filenames: Paths to training images
        :param labels: Class labels (0 == normal, other == anomaly)
        :param meta: Metadata (such as age or sex labels)
        :param transform: Transformations to apply to images
        :param index_mapping: Mapping from indices to data
        """
        self.data = data
        self.labels = labels
        self.meta = meta
        self.load_fn = load_fn
        self.transform = transform
        self.index_mapping = index_mapping
        self.filenames = filenames

        if self.index_mapping is None:
            self.index_mapping = torch.arange(len(self.data))

        for i, d in enumerate(self.data):
            if isinstance(d, str):
                self.data[i] = transform(load_fn(d))
                self.load_fn = lambda x: x
                self.transform = lambda x: x

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        data_idx = self.index_mapping[idx]
        img = self.transform(self.load_fn(self.data[data_idx]))
        label = self.labels[idx]
        meta = self.meta[idx]
        return img, label, meta


class MEMMAP_AnomalFairnessDataset(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(
            self,
            data: Dict[str, List[str]],
            labels: Dict[str, List[int]],
            meta: Dict[str, List[int]],
            transform=None,
            index_mapping: Optional[Dict[str, List[int]]] = None,
            load_fn: Callable = lambda x: x):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn
        self.index_mapping = index_mapping

        if self.index_mapping is None:
            self.index_mapping = {mode: torch.arange(len(self.data)) for mode in self.data.keys()}

    def __len__(self) -> int:
        return min([len(v) for v in self.labels.values()])

    def __getitem__(self, idx: int) -> tuple[dict, dict, dict]:
        img = {k: self.transform(self.load_fn(v[self.index_mapping[k][idx]])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta
