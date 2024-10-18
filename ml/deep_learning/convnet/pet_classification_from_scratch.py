import os, shutil, pathlib, random

original_dir = pathlib.Path("pet_images/train")
new_base_dir = pathlib.Path("pet_images/cat_vs_dogs_small")

# Create a smaller dataset from available pet images
def make_subsets():
    try:
        shutil.rmtree(new_base_dir)
    except FileNotFoundError:
        pass
    indices = [i for i in range(12500)]
    random.shuffle(indices)
    for subset_name, index_range in {"train": (0, 1000), "validation": (1000, 1500), "test": (1500, 2500)}.items():
        for category in ("cat", "dog"):
            dir = new_base_dir / subset_name / category
            os.makedirs(dir)
            fnames = [f"{category}.{i}.jpg" for i in indices[index_range[0] : index_range[1]]]

            for fname in fnames:
                shutil.copyfile(src=original_dir / fname, dst=dir / fname)

# Uncomment the following line if you want to regenerate the subset to train, validate and test on
# make_subsets()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs) # Converts uint8 values to float32 values between 0 and 1
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Data preprocessing

from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180,180),
    batch_size=32
)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180,180),
    batch_size=32
)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180,180),
    batch_size=32
)

# Check the shapes of data and labels

for data_batch, labels_batch in train_dataset:
    print(data_batch.shape)
    print(labels_batch.shape)
    print(labels_batch)
    break

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch2.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(train_dataset, epochs=30, validation_data=validation_dataset, callbacks=callbacks)

import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "b", label="Training accuracy")
plt.plot(epochs, val_accuracy, "g", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.show()

test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") # Test accuracy is 70.6%