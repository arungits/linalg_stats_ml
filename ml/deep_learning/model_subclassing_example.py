import tensorflow as tf
from tensorflow import keras
from keras import layers

# Customer ticket model using Model subclassing
class CustomerTicketModel(keras.Model):
    def __init__(self, num_depts):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_layer = layers.Dense(1, activation="sigmoid")
        self.dept_classifier = layers.Dense(num_depts, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        description = inputs["description"]
        tags = inputs["tags"]

        features = self.concat_layer([title, description, tags])
        features = self.mixing_layer(features)
        priority = self.priority_layer(features)
        department = self.dept_classifier(features)
        return priority, department

vocab_size = 10000
num_tags = 100
num_depts = 4

# Fake data prep

num_samples = 2560

title_data = keras.random.randint(shape=(num_samples, vocab_size), minval=0, maxval=2)
description_data = keras.random.randint(shape=(num_samples, vocab_size), minval=0, maxval=2)
tags_data = keras.random.randint(shape=(num_samples, num_tags), minval=0, maxval=2)

priority_targets = tf.random.uniform(shape=(num_samples,1), minval=0., maxval=1.)
department_targets = keras.random.randint(shape=(num_samples,), minval=0, maxval=num_depts)

# Model initialization
model = CustomerTicketModel(num_depts)
inputs = {"title": title_data, "description": description_data, "tags": tags_data}
model.compile(optimizer="rmsprop", loss=["mean_squared_error", "sparse_categorical_crossentropy"], metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit(inputs, [priority_targets, department_targets], epochs=1)
results = model.evaluate(inputs, [priority_targets, department_targets])
print(results)

