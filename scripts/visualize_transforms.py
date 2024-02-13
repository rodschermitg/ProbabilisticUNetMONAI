"""run from project root: python3 -m scripts.visualize_transforms"""
import json
import os

import matplotlib
import monai
import torch

from src import config
from src import utils


matplotlib.use("TkAgg")
monai.utils.set_determinism(config.RANDOM_STATE)

# set prob = 1.0 for all non-deterministic transforms to visualize
for transform in config.train_transforms.transforms:
    if hasattr(transform, "prob"):
        transform.prob = 1.0
transforms = monai.transforms.Compose([
    config.base_transforms,
    config.train_transforms
])

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(data=data["train"], transform=transforms)
dataloader = monai.data.DataLoader(dataset, batch_size=1)

for batch in dataloader:
    all_images = torch.cat((
        batch["images"].squeeze(0),
        torch.argmax(batch["label"], dim=1)  # decode one-hot labels
    ))

    patient_name = utils.get_patient_name(
        batch["label_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        all_images,
        title=patient_name,
        slice_dim=0,
        labels=config.modality_keys + ["label"]
    )
