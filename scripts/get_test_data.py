"""run from project root: python3 -m scripts.get_test_data"""
import os
import shutil

from sklearn.model_selection import train_test_split

from src import config


def reset():
    for patient_dir in os.listdir(config.test_data_dir):
        src_path = os.path.join(config.test_data_dir, patient_dir)
        dst_path = os.path.join(config.train_data_dir, patient_dir)
        shutil.move(src_path, dst_path)


def perform_train_test_split():
    patient_dirs = os.listdir(config.train_data_dir)

    _, test_patient_dirs = train_test_split(
        patient_dirs,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    for test_patient_dir in test_patient_dirs:
        src_path = os.path.join(config.train_data_dir, test_patient_dir)
        dst_path = os.path.join(config.test_data_dir, test_patient_dir)
        shutil.move(src_path, dst_path)

    print(f"randomly selected {len(test_patient_dirs)} test samples")


if __name__ == "__main__":
    reset()
    if config.TEST_SIZE > 0:
        perform_train_test_split()
