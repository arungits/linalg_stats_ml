import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(type(train_images))
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
