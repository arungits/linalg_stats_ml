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

conv_base.trainable = False

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

# Initialize the new model

data_augmentation_layer = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)
inputs = keras.Input(shape=(180,180,3))
x = data_augmentation_layer(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])

# Visualize the new classifier model
print(model.summary())

# OUTPUT:
# Total params: 17,992,001 (68.63 MB)
# Trainable params: 3,277,313 (12.50 MB)
# Non-trainable params: 14,714,688 (56.13 MB)

# Training the model

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="feature_extraction_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# Test dataset evaluation

test_model = keras.models.load_model("feature_extraction_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") # Test accuracy is 97.4%

# Make the last 4 layers of the base model trainable to finetune the model

for layer in conv_base.layers[-4:]:
    layer.trainable = True

# Use a low learning rate while finetuning
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())

# OUTPUT:

# Total params: 17,992,001 (68.63 MB)
# Trainable params: 10,356,737 (39.51 MB)
# Non-trainable params: 7,635,264 (29.13 MB)

# Training the finetuned model

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="finetuned_model.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# Test dataset evaluation for the finedtuned model

test_model = keras.models.load_model("finetuned_model.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") # Test accuracy is 97.6%