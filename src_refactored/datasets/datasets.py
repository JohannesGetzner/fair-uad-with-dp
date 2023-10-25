from typing import Callable, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from src_refactored.datasets.data_utils import load_dicom_img


class NormalDataset(Dataset):
    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            load_fn: Callable = load_dicom_img):

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

    def __getitem__(self, idx: int) -> Tensor:
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

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(self.load_fn(v[idx])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta
