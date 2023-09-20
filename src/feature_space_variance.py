import sys
import json
sys.path.append('..')
import torch
from datetime import datetime
from argparse import ArgumentParser
from utils.utils import seed_everything
import torchvision.models as models
import torch.nn as nn
from src import RSNA_DIR
from src.data.rsna_pneumonia_detection import (load_rsna_age_two_split,load_rsna_gender_split)
from torchvision import transforms
from torch.utils.data import DataLoader
import pydicom as dicom
from src.data.datasets import NormalDataset
from torch import Generator


parser = ArgumentParser()
parser.add_argument('--protected_attr', default="age", type=str)
parser.add_argument('--protected_attr_percent', default=None, type=float)
parser.add_argument('--d', type=str, default=str(datetime.strftime(datetime.now(), format="%Y-%m-%d %H:%M:%S")))
DYNAMIC_PARAMS = parser.parse_args()


def init_inception_net():
    model = models.inception_v3(pretrained=True)
    # remove last layer or the model
    model.fc = nn.Identity()
    # print(model)
    return model


def load_dicom_img_custom(filename: str):
    ds = dicom.dcmread(filename)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32) / 255.0
    # repeat dimension two times
    img = img.repeat(3, 1, 1)
    return img  # (3, H, W)


def load_data():
    if DYNAMIC_PARAMS.protected_attr == "age":
        print("Loading age split...", "with percent", DYNAMIC_PARAMS.protected_attr_percent)
        data, labels, meta = load_rsna_age_two_split(
            RSNA_DIR,
            old_percent=DYNAMIC_PARAMS.protected_attr_percent,
            upsampling_strategy=None,
            effective_dataset_size=1.0
        )
    else:
        print("Loading sex split...", "with percent", DYNAMIC_PARAMS.protected_attr_percent)
        data, labels, meta = load_rsna_gender_split(
            RSNA_DIR,
            male_percent=DYNAMIC_PARAMS.protected_attr_percent,
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
        batch_size=32,
        shuffle=True,
        num_workers=4,
        generator=Generator().manual_seed(2147483647)
    )
    return dataloader


def run():
    dataloader = load_data()
    print(f"Setting seed to {1}...")
    seed_everything(1)
    model = init_inception_net()
    model.to("cuda")
    model.eval()
    results = []
    # run inference on all images
    for (img, label, meta) in dataloader:
        print(f"Batch", img.shape)
        img = img.to("cuda")
        with torch.no_grad():
            output = model(img)
            batch_results = [
                {"label": label[i].item(), "meta": meta[i].item(), "output": output[i].tolist()}
                for i in range(len(label))
            ]
            results += batch_results
    # json dump results
    with open("results.json", "w") as f:
        json.dump(results, f)
    pass


if __name__ == '__main__':
    run()