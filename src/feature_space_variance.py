import os
os.environ["WANDB_SERVICE_WAIT"] = "300"
import sys
sys.path.append('..')
import torch
from datetime import datetime
from argparse import ArgumentParser
from utils.utils import seed_everything
from utils.train_utils import load_data
import torchvision.models as models
import torch.nn as nn
from src import RSNA_DIR
from src.data.rsna_pneumonia_detection import (load_rsna_age_two_split,
                                               load_rsna_gender_split)
from torchvision import transforms
from torch.utils.data import DataLoader
import pydicom as dicom
from src.data.datasets import NormalDataset
from torch import Generator


parser = ArgumentParser()
parser.add_argument('--run_config', default="dp", type=str)
parser.add_argument('--run_version', default="v1", type=str)
parser.add_argument('--protected_attr_percent', default=None, type=float)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")))
DYNAMIC_PARAMS = parser.parse_args()


def init_inception_net():
    model = models.inception_v3(pretrained=True)
    # remove last layer
    model = nn.Sequential(*list(model.children())[:-1])
    # print(model)
    return model


def load_dicom_img_custom(filename: str):
    ds = dicom.dcmread(filename)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32) / 255.0
    # repeat dimension two times
    img = img.repeat(3, 1, 1)
    return img  # (1, H, W)


def load_data(protected_attr, protected_attr_percent):
    if protected_attr == 'age':
        data, labels, meta = load_rsna_age_two_split(
            RSNA_DIR,
            old_percent=protected_attr_percent,
            upsampling_strategy=None,
            effective_dataset_size=1.0
        )
    else:
        data, labels, meta = load_rsna_gender_split(
            RSNA_DIR,
            male_percent=protected_attr_percent,
            upsampling_strategy=None,
            effective_dataset_size=1.0
        )
    data_complete = []
    labels_complete = []
    meta_complete = []
    for k, v in data.items():
        data_complete += v
        labels_complete += labels[k]
        meta_complete += list(meta[k])
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = NormalDataset(
        data_complete,
        labels_complete,
        meta_complete,
        transform=transform,
        load_fn=load_dicom_img_custom
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        generator=Generator().manual_seed(2147483647)
    )
    return dataloader


def run():
    dataloader = load_data("age", 0.5)
    print(f"Setting seed to {1}...")
    seed_everything(1)
    model = init_inception_net()
    model = model.to("cuda")
    model.eval()
    # run inference on all images
    for i, (img, label, meta) in enumerate(dataloader):
        print(f"Batch {i}")
        img = img.to("cuda")
        with torch.no_grad():
            output = model(img)
            print(output.shape)
    pass


if __name__ == '__main__':
    run()