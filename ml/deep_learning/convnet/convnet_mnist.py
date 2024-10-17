from tensorflow import keras
from tensorflow.keras import layers

# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load data
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_shape = train_images.shape
test_shape = test_images.shape
train_images = train_images.reshape((train_shape[0], train_shape[1], train_shape[2], 1)).astype("float32") / 255
test_images = test_images.reshape((test_shape[0], test_shape[1], test_shape[2], 1)).astype("float32") / 255

inputs = keras.Input(shape=(28,28,1))
# x = layers.Conv2D(filters=32, kernel_size=3, padding="same", strides=2, activation="relu")(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
# x = layers.AveragePooling2D(pool_size=2)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=64, kernel_size=3, padding="same", strides=2, activation="relu")(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
# x = layers.AveragePooling2D(pool_size=2)(x)
x = layers.MaxPooling2D(pool_size=2)(x)
# x = layers.Conv2D(filters=128, kernel_size=3, padding="same", strides=2, activation="relu")(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)