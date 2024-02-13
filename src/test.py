import json
import os

import monai
import torch

import config
import models
import utils


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
discretize = monai.transforms.AsDiscrete(argmax=True, to_onehot=2)

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

dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    metric_name=("precision", "recall"),
    include_background=False
)
precision_list = []
recall_list = []

for batch in dataloader:
    images = batch["images"].to(device)
    label = batch["label"].to(device)

    patient_name = utils.get_patient_name(
        batch["label_meta_dict"]["filename_or_obj"][0]
    )

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

    # store discretized batches in lists for metric functions
    pred = [discretize(p) for p in monai.data.decollate_batch(pred)]
    label = monai.data.decollate_batch(label)

    # metric results are stored internally
    dice_fn(pred, label)
    confusion_matrix_fn(pred, label)

    # store precision and recall in separate lists for later calculations
    precision_list.append(confusion_matrix_fn.aggregate()[0].item())
    recall_list.append(confusion_matrix_fn.aggregate()[1].item())
    confusion_matrix_fn.reset()

    print(f"{patient_name}:")
    print(f"\tdice: {dice_fn.get_buffer()[-1].item():.4f}")
    print(f"\tprecision: {precision_list[-1]:.4f}")
    print(f"\trecall: {recall_list[-1]:.4f}")

mean_dice = torch.mean(dice_fn.get_buffer()).item()
std_dice = torch.std(dice_fn.get_buffer(), correction=0).item()
mean_precision = torch.mean(torch.tensor(precision_list))
std_precision = torch.std(torch.tensor(precision_list), correction=0)
mean_recall = torch.mean(torch.tensor(recall_list))
std_recall = torch.std(torch.tensor(recall_list), correction=0)

print(f"\nmean dice: {mean_dice:.4f}, std dice: {std_dice:.4f}")
print(
    f"mean precision: {mean_precision:.4f}, std precision: {std_precision:.4f}"
)
print(f"mean recall: {mean_recall:.4f}, std recall: {std_recall:.4f}")
