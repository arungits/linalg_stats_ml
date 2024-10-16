# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load data
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("### Printing train data info ###")
print(type(train_images))
print(train_images.shape)
print(train_labels.shape)
print(train_images.dtype)

print('### Printing test data info ###')
print(test_images.shape)
print(test_labels.shape)

# Tranforming train & test data

train_images = train_images.reshape(60000, 28*28).astype('float32') / 255
test_images = test_images.reshape(10000, 28*28).astype('float32') / 255

# Neural net

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from custom_metric import RootMeanSquaredError
from custom_callback import  LossHistory

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="best_mnist_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    keras.callbacks.TensorBoard(
        log_dir="tensorlog",
    ),
    LossHistory(),
]

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", RootMeanSquaredError()]) # Custom metric
model.fit(train_images, train_labels, epochs=20, batch_size=128, callbacks=callbacks_list, validation_data=(test_images, test_labels))
results = model.evaluate(test_images, test_labels)
print(results)

# Load the best model saved by ModelCheckpoint and evaluate it using test data

model = keras.models.load_model("best_mnist_model.keras")
results = model.evaluate(test_images, test_labels)
print(results)

# Linear Classifier (shallow learning)

from linear_classifier_tf import LinearClassifier

model = LinearClassifier(28 * 28, 10)
model.fit(train_images, train_labels, epochs=500)
predictions = tf.math.argmax(model.predict(test_images), 1)
print(len(predictions))
print(predictions.shape)
total_matches = tf.math.reduce_sum(tf.cast(predictions == test_labels, tf.int32)).numpy()
print(total_matches)
accuracy = total_matches / len(test_images)
print("Accuracy of Linear Classifier is ", accuracy)