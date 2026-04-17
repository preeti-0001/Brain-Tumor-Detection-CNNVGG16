import os
from utilities.constants import SPLIT_DIR, IMG_SIZE, BATCH_SIZE
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    print(f"✅ Generators ready!")
    print(f"   Class mapping: {train_gen.class_indices}")
    print("   0 = no (No Tumor)  |  1 = yes (Tumor)")
    return train_gen, val_gen, test_gen
