import tensorflow as tf
from tensorflow import keras
from keras import layers

vocab_size = 10000
num_tags = 100
num_depts = 4

# Model initialization using functional APIs

title = keras.Input(shape=(vocab_size,), name="title")
description = keras.Input(shape=(vocab_size,), name="description")
tags = keras.Input(shape=(num_tags,), name="tags")

features = layers.Concatenate(name="concatenation-layer")([title, description, tags])
features = layers.Dense(64, activation="relu", name="extraction-layer")(features)

priority = layers.Dense(1, activation="sigmoid", name="priority-layer")(features)
department = layers.Dense(num_depts, activation="softmax", name="department-layer")(features)

model = keras.Model(inputs=[title, description, tags], outputs=[priority, department])

# Fake data prep

num_samples = 2560

title_data = keras.random.randint(shape=(num_samples, vocab_size), minval=0, maxval=2)
description_data = keras.random.randint(shape=(num_samples, vocab_size), minval=0, maxval=2)
tags_data = keras.random.randint(shape=(num_samples, num_tags), minval=0, maxval=2)

priority_targets = tf.random.uniform(shape=(num_samples,1), minval=0., maxval=1.)
department_targets = keras.random.randint(shape=(num_samples,), minval=0, maxval=num_depts)

# Model training and evaluation

model.compile(optimizer="rmsprop", loss=["mean_squared_error", "sparse_categorical_crossentropy"], metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, description_data, tags_data], [priority_targets, department_targets], epochs=1)
results = model.evaluate([title_data, description_data, tags_data], [priority_targets, department_targets])
print(results)

# Plot the layer graph of the model

keras.utils.plot_model(model, show_shapes=True)
