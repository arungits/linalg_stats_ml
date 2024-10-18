import numpy as np
from tensorflow import keras

# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets or weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize the convolutional base of VGG16 pretrained model
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(180,180,3))

# Visualize the model
print(conv_base.summary())

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers
import pathlib

base_dir = pathlib.Path("pet_images/cat_vs_dogs_small")

# Load datasets

train_dataset = image_dataset_from_directory(
    base_dir / "train",
    image_size=(180,180),
    batch_size=512
)
validation_dataset = image_dataset_from_directory(
    base_dir / "validation",
    image_size=(180,180),
    batch_size=512
)
test_dataset = image_dataset_from_directory(
    base_dir / "test",
    image_size=(180,180),
    batch_size=512
)

# Extract pretrained features for the train, validation and test datasets using conv_base model

def get_features_and_labels(dataset):
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images, verbose=False)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = get_features_and_labels(train_dataset)
print("Shape of training features and labels from conv_base")
print(train_features.shape)
print(train_labels.shape)
validation_features, validation_labels = get_features_and_labels(validation_dataset)
print("Shape of validation features and labels from conv_base")
print(validation_features.shape)
print(validation_labels.shape)
test_features, test_labels = get_features_and_labels(test_dataset)
print("Shape of test features and labels from conv_base")
print(test_features.shape)
print(test_labels.shape)

# Initialize the new classifier model

inputs = keras.Input(shape=(5,5,512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Visualize the new classifier model
print(model.summary())

# Training the model

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="vgg16_feature_extraction.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_features, train_labels,
    epochs=20,
    validation_data=(validation_features, validation_labels),
    callbacks=callbacks
)

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

# Test dataset evaluation

test_model = keras.models.load_model("vgg16_feature_extraction.keras")
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
print(f"Test accuracy: {test_acc:.3f}")

