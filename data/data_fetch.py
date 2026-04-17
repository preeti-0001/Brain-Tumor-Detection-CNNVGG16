import os
import zipfile
import subprocess
from utilities.constants import DATA_DIR, ZIP_PATH, EXTRACTED_DATA_DIR, DATASET


ZIP_PATH = os.path.join(DATA_DIR, "brain-mri-dataset.zip")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def dataset_exists():
    return os.path.exists(EXTRACTED_DATA_DIR)


def zip_exists():
    return os.path.exists(ZIP_PATH)


def download_dataset():
    print("Downloading dataset from Kaggle...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", DATA_DIR,
        "--force"
    ]
    subprocess.run(cmd, check=True)
    print("Download complete.")


def unzip_dataset():
    print("Unzipping dataset...")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(DATA_DIR, file)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
    print("Unzip complete.")


def fetch_data():
    ensure_data_dir()

    if dataset_exists():
        print("Dataset already exists. Skipping download & unzip.")
        return

    if not zip_exists():
        download_dataset()
    else:
        print("Zip file already exists. Skipping download.")

    unzip_dataset()
    print("Dataset ready to use.")

