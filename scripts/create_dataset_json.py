"""run from project root: python3 -m scripts.create_dataset_json"""
import json
import os

from src import config


base_keys = config.MODALITIES + ["label"]
data_dict = {"train": [], "test": []}

for dataset_key in data_dict.keys():
    if dataset_key == "train":
        dataset_dir = config.train_data_dir
    else:
        dataset_dir = config.test_data_dir

    for patient_dir in sorted(os.listdir(dataset_dir)):
        patient_dict = {}
        week_dirs = sorted(os.listdir(os.path.join(dataset_dir, patient_dir)))

        for timestep_idx, timestep in enumerate(config.TIMESTEPS):
            image_dir = os.path.join(
                dataset_dir, patient_dir, week_dirs[timestep_idx]
            )
            for base_key in base_keys:
                if base_key == "label" and timestep == config.TIMESTEPS[-1]:
                    image_path = os.path.join(image_dir, "seg_mask.nii.gz")
                    patient_dict["label"] = image_path
                elif (
                    base_key in config.MODALITIES and
                    timestep != config.TIMESTEPS[-1]
                ):
                    image_path = os.path.join(
                        image_dir, f"{base_key.lower()}_skull_strip.nii.gz"
                    )
                    patient_dict[f"{base_key}_{timestep}"] = image_path

        data_dict[dataset_key].append(patient_dict)

with open(os.path.join(config.data_dir, config.DATA_FILENAME), "w") as file:
    json.dump(data_dict, file, indent=4)
