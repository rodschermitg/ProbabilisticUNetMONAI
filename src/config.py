import os

import monai
import torch


# input path
data_dir = os.path.join("data", "processed", "patients")
train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")
raw_data_dir = os.path.join("data", "raw", "patients")
DATA_FILENAME = "dataset.json"

# random state
RANDOM_STATE = 42

# data
TEST_SIZE = 0.1
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 2
TIMESTEPS = ["1", "2", "3"]
num_timesteps = len(TIMESTEPS)
MODALITIES = ["CT1", "FLAIR", "T1", "T2"]
num_modalities = len(MODALITIES)
modality_keys = (
    [f"{modality}_{TIMESTEPS[0]}" for modality in MODALITIES] +
    [f"{modality}_{TIMESTEPS[1]}" for modality in MODALITIES]
)

# transforms
base_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(
        keys=modality_keys + ["label"],
        image_only=False,
        ensure_channel_first=True
    ),
    monai.transforms.ConcatItemsd(
        keys=modality_keys,
        name="images",
        dim=0
    ),
    monai.transforms.DeleteItemsd(keys=modality_keys),
    monai.transforms.CropForegroundd(
        keys=["images", "label"],
        source_key="images",
    ),
    monai.transforms.ThresholdIntensityd(
        keys="label",
        threshold=1,
        above=False,
        cval=1
    ),
    monai.transforms.AsDiscreted(keys="label", to_onehot=NUM_CLASSES),
    # monai.transforms.Orientationd(
    #     keys=["images", "label"],
    #     axcodes="SPL",
    # ),
])
train_transforms = monai.transforms.Compose([
    monai.transforms.RandAffined(
        keys=["images", "label"],
        prob=0.1,
        rotate_range=0.1,
        scale_range=0.1,
        mode=("bilinear", "nearest")
    ),
    monai.transforms.RandCropByPosNegLabeld(
        keys=["images", "label"],
        label_key="label",
        spatial_size=PATCH_SIZE,
        pos=1,
        neg=1,
        num_samples=1,
    ),
    monai.transforms.RandGaussianNoised(
        keys=["images"],
        prob=0.1,
        mean=0.0,
        std=0.1
    ),
    monai.transforms.NormalizeIntensityd(keys=["images"], channel_wise=True)
])
eval_transforms = monai.transforms.Compose([
    monai.transforms.NormalizeIntensityd(keys=["images"], channel_wise=True)
])

# model
MODEL_KWARGS = {
    "in_channels": 2 * num_modalities,
    "out_channels": NUM_CLASSES,
    "task_kwargs": {
        "output_activation_op": torch.nn.LogSoftmax,
        "output_activation_kwargs": {"dim": 1},
        "activation_kwargs": {"inplace": True}
    },
    "prior_kwargs": {
        "norm_depth": "full"
    },
    "posterior_kwargs": {
        "norm_depth": "full"
    },
}

# train
LR = 1e-4
FOLDS = 5
EPOCHS = 250
BATCH_SIZE = 2
VAL_INTERVAL = 5
DISPLAY_INTERVAL = 5
KL_WEIGHT = 1
SAVE_MODEL_EACH_FOLD = True

# output path
model_dir = os.path.join("models")
MODEL_NAME = "probunet_patients"
train_logs_path = os.path.join("logs", "train_logs.json")
nll_loss_plot_path = os.path.join("logs", "nll_loss_plot.png")
kl_loss_plot_path = os.path.join("logs", "kl_loss_plot.png")
val_metric_plot_path = os.path.join("logs", "val_metric_plot.png")
