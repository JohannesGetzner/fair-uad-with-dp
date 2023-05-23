import pydicom as dicom
import torch

from PIL import Image
from torch import Tensor
from torchvision import transforms


def load_dicom_img(filename: str) -> Tensor:
    """Loads a DICOM image."""
    ds = dicom.dcmread(filename)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32) / 255.0
    return img[None]  # (1, H, W)


def load_png_img_grayscale(filename: str) -> Tensor:
    """Loads a PNG image."""
    img = Image.open(filename).convert('L')
    img = transforms.ToTensor()(img)
    return img


if __name__ == '__main__':
    img = load_dicom_img('/datasets/RSNA/stage_2_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm')
    print(img)
