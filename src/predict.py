import json
import os

import matplotlib
import monai
import torch

import config
import models
import utils


matplotlib.use("TkAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

checkpoint_list = [
    torch.load(
        os.path.join(config.model_dir, f"{config.MODEL_NAME}_fold{fold}.tar"),
        map_location=device
    )
    for fold in range(config.FOLDS)
]
model_list = [
    models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS).to(device)
    for _ in range(config.FOLDS)
]
for model, checkpoint in zip(model_list, checkpoint_list):
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
dataset = monai.data.Dataset(
    data=data["test"],
    transform=monai.transforms.Compose([
        config.base_transforms,
        config.eval_transforms
    ])
)
print(f"Using {len(dataset)} test samples")

dataloader = monai.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=pin_memory
)

for batch in dataloader:
    images = batch["images"].to(device)
    label = batch["label"].to(device)
    label = torch.argmax(label, dim=1)  # decode one-hot labels

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            preds = [
                monai.inferers.sliding_window_inference(
                    inputs=images,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=config.BATCH_SIZE,
                    predictor=model
                )
                for model in model_list
            ]
    preds = torch.cat(preds, dim=0)
    pred = torch.mean(preds, dim=0, keepdim=True)
    pred = torch.argmax(pred, dim=1)

    all_images = torch.cat((
        images.squeeze(0),
        label,
        pred
    )).cpu()

    patient_name = utils.get_patient_name(
        batch["label_meta_dict"]["filename_or_obj"][0]
    )

    utils.create_slice_plots(
        all_images,
        title=patient_name,
        labels=config.modality_keys + ["label", "pred"]
    )
