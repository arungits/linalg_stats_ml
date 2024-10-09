# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load data
import numpy as np
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print("### Training data stats ###")
print(len(train_data))
print(train_data.shape)
print(train_labels.shape)

print("### Test data stats ###")
print(len(test_data))
print(test_data.shape)
print(test_labels.shape)

print("Printing sample input and label")
print(train_data[0])
print(train_labels[0])

print("Printing all labels")
all_labels = sorted(list(set(train_label for train_label in train_labels)))
print(len(all_labels))
print(all_labels)

def vectorize_sequences(sequences, num_dimension=10000):
    # Multi hot encoding
    results = np.zeros((len(sequences), num_dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("### Shape of vectorized train and test data ###")
print(x_train.shape)
print(x_test.shape)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(all_labels), activation="softmax")
])
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

def plot_metrics(history_dict):
    import matplotlib.pyplot as plt
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "g", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "g", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.show()

# plot_metrics(history.history)

# Model overfits after 9 epochs so retraining the model for only 9 epochs with full train data

model1 = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(len(all_labels), activation="softmax")
])
model1.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model1.fit(x_train, train_labels, epochs=9, batch_size=512)

# Prediction

def encode(review):
    tokens = review.split(" ")
    indices = [reuters.get_word_index().get(token, -1) for token in tokens]
    indices = [index + 3 for index in indices if index != -1]
    return indices

test_articles = ["markets were volatile yesterday", "President blamed opposition for not passing bill"]
predictions = model1.predict(vectorize_sequences([encode(article) for article in test_articles]))

print(predictions[0].argmax(), predictions[1].argmax())
print(predictions[0][predictions[0].argmax()], predictions[1][predictions[1].argmax()])