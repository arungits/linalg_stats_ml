# Regression model to predict house prices using neural nets

# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets
import ssl

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print("### Training data info ###")

print(train_data.shape)
print(train_targets.shape)

print("### Test data info ###")

print(test_data.shape)
print(test_targets.shape)

# Feature normalization

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

from tensorflow import keras
from tensorflow.keras import layers

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

# Evaluate using k-fold validation as there are only 404 training samples

num_epochs = 500
k = 4
num_val_samples = len(train_data) // k
all_mae_histories = []

for i in range(k):
    val_data = train_data[i * num_val_samples:num_val_samples * (i+1)]
    val_targets = train_targets[i * num_val_samples:num_val_samples * (i+1)]
    partial_train_data = np.concatenate([
        train_data[:i * num_val_samples],
        train_data[num_val_samples * (i+1):]
    ], axis=0)
    partial_train_targets = np.concatenate([
        train_targets[:i * num_val_samples],
        train_targets[num_val_samples * (i + 1):]
    ], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0, validation_data=(val_data, val_targets))
    all_mae_histories.append(history.history['val_mae'])

average_mae_histories = [np.mean([mae_history[i] for mae_history in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, num_epochs + 1), average_mae_histories)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

plt.clf()
plt.plot(range(11, num_epochs+1), average_mae_histories[10:])
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# Retraining the model with all of training data for 130 epochs as the validation loss is at the minimum around 130 epochs

model = build_model()
model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print("Test MAE: ", test_mae_score)