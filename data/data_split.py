import os
from sklearn.model_selection import train_test_split
from utilities.constants import SPLIT_DIR
import shutil
from utilities.constants import DATA_DIR, EXTRACTED_DATA_DIR


def copy_split(img_list, src_dir, cls_name):
    """Split 70% train, 15% val, 15% test and copy files."""
    train_imgs, temp = train_test_split(img_list, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=0.50, random_state=42)
    for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        for f in imgs:
            shutil.copy(
                os.path.join(src_dir, f), os.path.join(SPLIT_DIR, split, cls_name, f)
            )
    return len(train_imgs), len(val_imgs), len(test_imgs)


def split_data():

    YES_DIR, NO_DIR = None, None
    for root, dirs, _ in os.walk(EXTRACTED_DATA_DIR):
        for d in dirs:
            if d.lower() == "yes":
                YES_DIR = os.path.join(root, d)
            if d.lower() == "no":
                NO_DIR = os.path.join(root, d)

    # Count images
    yes_imgs = [
        f for f in os.listdir(YES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    no_imgs = [
        f for f in os.listdir(NO_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Create train/val/test folder structure
    for split in ["train", "val", "test"]:
        for cls in ["yes", "no"]:
            os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

    y_tr, y_v, y_te = copy_split(yes_imgs, YES_DIR, "yes")
    n_tr, n_v, n_te = copy_split(no_imgs, NO_DIR, "no")

    print("✅ Split Complete!")
    print(f"   Train → Tumor: {y_tr}  | No Tumor: {n_tr}  | Total: {y_tr+n_tr}")
    print(f"   Val   → Tumor: {y_v}   | No Tumor: {n_v}   | Total: {y_v+n_v}")
    print(f"   Test  → Tumor: {y_te}  | No Tumor: {n_te}  | Total: {y_te+n_te}")
