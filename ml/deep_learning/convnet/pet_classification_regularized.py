# Data preprocessing

import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

base_dir = pathlib.Path("pet_images/cat_vs_dogs_small")
train_dataset = image_dataset_from_directory(
    base_dir / "train",
    image_size=(180,180),
    batch_size=32
)
validation_dataset = image_dataset_from_directory(
    base_dir / "validation",
    image_size=(180,180),
    batch_size=32
)
test_dataset = image_dataset_from_directory(
    base_dir / "test",
    image_size=(180,180),
    batch_size=32
)

# Model initialization
# Model with augmentation for regularization

data_augmentation_layer = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

# Test data augmentation layer

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for images,_ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation_layer(images)
        ax = plt.subplot(3,3,i+1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

plt.show()

inputs = keras.Input(shape=(180,180,3))
x = data_augmentation_layer(inputs)
x = layers.Rescaling(1./255)(x) # Converts uint8 values to float32 values between 0 and 1
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
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
print(model.summary())

# Training

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="pet_classification_regularized2.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]
# history = model.fit(train_dataset, epochs=30, callbacks=callbacks, validation_data=validation_dataset)
#
# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "b", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "g", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.show()

test_model = keras.models.load_model("pet_classification_regularized.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

# Testset evaluation
