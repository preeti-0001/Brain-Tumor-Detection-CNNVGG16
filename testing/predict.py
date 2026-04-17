import os

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from utilities.constants import IMG_SIZE, SPLIT_DIR
import matplotlib.pyplot as plt


def predict_image(model, img_path, model_name="VGG16"):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = model.predict(arr, verbose=0)[0][0]
    label = "TUMOR DETECTED 🔴" if prob > 0.5 else "NO TUMOR 🟢"
    conf = prob if prob > 0.5 else 1 - prob
    color = "red" if prob > 0.5 else "green"

    plt.figure(figsize=(4, 4))
    plt.imshow(load_img(img_path))
    plt.title(
        f"{model_name}\n{label}\nConfidence: {conf*100:.1f}%",
        fontsize=12,
        fontweight="bold",
        color=color,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def predict_sample(vgg_model, cnn_model):

    tumor_sample = os.path.join(
        SPLIT_DIR, "test", "yes", os.listdir(os.path.join(SPLIT_DIR, "test", "yes"))[0]
    )
    no_tumor_sample = os.path.join(
        SPLIT_DIR, "test", "no", os.listdir(os.path.join(SPLIT_DIR, "test", "no"))[0]
    )

    predict_image(vgg_model, tumor_sample, "VGG16")
    predict_image(vgg_model, no_tumor_sample, "VGG16")
    predict_image(cnn_model, tumor_sample, "Custom CNN")
    predict_image(cnn_model, no_tumor_sample, "Custom CNN")
