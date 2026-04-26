"""
CIFAR-10 Model Training Script
Run this ONCE to train and save the model before starting the server.
Usage: python train_model.py
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱",
    "deer": "🦌", "dog": "🐶", "frog": "🐸", "horse": "🐴",
    "ship": "🚢", "truck": "🚚"
}

def build_model():
    """CNN model with BatchNorm and Dropout for CIFAR-10."""
    model = models.Sequential([

        layers.Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),


        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),


        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),


        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    return model


def train():
    print("=" * 60)
    print("  CIFAR-10 Training Script")
    print("=" * 60)

    print("\n📦 Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0


    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test,  10)
    print(f"   Train: {x_train.shape}  |  Test: {x_test.shape}")


    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    datagen.fit(x_train)

  
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

  
    os.makedirs("model", exist_ok=True)
    callbacks = [
        ModelCheckpoint("model/cifar10_best.keras", save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

 
    print("\n🚀 Starting training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )


    print("\n📊 Final evaluation on test set:")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Accuracy : {acc * 100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")


    meta = {
        "class_names": CLASS_NAMES,
        "class_emoji": CLASS_EMOJI,
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "epochs_trained": len(history.history["accuracy"]),
    }
    with open("model/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Model saved to  model/cifar10_best.keras")
    print("✅ Metadata saved to model/metadata.json")
    print("\nNow run:  python app.py")


if __name__ == "__main__":
    train()
