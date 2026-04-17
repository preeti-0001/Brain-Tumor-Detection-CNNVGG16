from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from utilities.constants import TIMESTAMP


def get_callbacks(name):
    return [
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        ModelCheckpoint(
            f"output/{TIMESTAMP}/{name}_best.keras", monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]


def train_cnn_model(cnn_model, train_gen, val_gen):
    # Train Custom CNN
    print("=" * 45)
    print("  🔵 Training Custom CNN...")
    print("=" * 45)
    cnn_history = cnn_model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=get_callbacks("cnn"),
        verbose=1,
    )
    print("\n✅ Custom CNN training complete!")
    return cnn_history


def train_vgg16_model(vgg_model, vgg_base, train_gen, val_gen):
    # Train VGG16 Phase 1 (frozen base)
    print("=" * 45)
    print("  🟠 Training VGG16 Phase 1...")
    print("=" * 45)

    vgg_h1 = vgg_model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=get_callbacks("vgg_p1"),
        verbose=1,
    )
    print("\n✅ VGG16 Phase 1 complete!")

    # VGG16 Phase 2 — Fine-tune last 4 layers
    print("=" * 45)
    print("  🔴 VGG16 Phase 2 — Fine-tuning...")
    print("=" * 45)

    vgg_base.trainable = True
    for layer in vgg_base.layers[:-4]:
        layer.trainable = False

    # Use much lower learning rate for fine-tuning
    vgg_model.compile(
        optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"]
    )

    vgg_h2 = vgg_model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=get_callbacks("vgg_p2"),
        verbose=1,
    )
    print("\n✅ Fine-tuning complete!")
    return vgg_h1, vgg_h2

