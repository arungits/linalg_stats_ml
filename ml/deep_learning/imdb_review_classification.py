# IMDB movie reviews classification / sentiment analysis
# Binary classification problem

# the below lines are to get over the SSL verification error that sometimes comes when downloading datasets
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load data
import numpy as np
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) # keeps only 10000 most frequently used words in training data

print("### Training data ###")
print(train_data[0])
print(train_data.shape)
print(train_labels[0])
print(train_labels.shape)

print("### Test data ###")
print(test_data.shape)
print(test_labels.shape)

# Make sure that the maximum word index across training samples is 9999 as we have restricted num_words to 10000
print(max(max(sequence) for sequence in train_data))

# Decode the sequence of word index to words
print("### Decoded sample reviews ###")
word_index = imdb.get_word_index()
reverse_word_index = { value: key for key, value in word_index.items() }
for i in range(5):
    decoded_review = ' '.join(reverse_word_index.get(idx - 3, "?") for idx in train_data[i])
    print("Review:", decoded_review)
    print("Target: ", 'Positive' if train_labels[i] == 1 else 'Negative')

# Vecotrize sequences using multi hot encoding

def vectorize_sequences(sequences, num_dimension=10000):
    result = np.zeros((len(sequences), num_dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            result[i, j] = 1
    return result

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print("### Shape of train and test data after vectorization as multi hot encodings ###")
print(x_train.shape)
print(x_test.shape)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

print("### Shape of train and test labels after vectorization ###")
print(y_train.shape)
print(y_test.shape)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# set aside validation set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
history_dict = history.history
print(history_dict.keys())

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

# plot_metrics(history_dict)

# Retraining model with all of the training data and stopping training after 4th epoch as the model starts to overfit after 4th epoch
model1 = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model1.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
model1.fit(x_train, y_train, epochs=4, batch_size=512)
results = model1.evaluate(x_test, y_test)
print(results)

#Prediction

def encode(review):
    tokens = review.split(" ")
    indices = [word_index.get(token, -1) for token in tokens]
    indices = [index + 3 for index in indices if index != -1]
    return indices

# test encode()
test_review = "this movie was amazing"
assert(' '.join(reverse_word_index.get(idx - 3, "?") for idx in encode(test_review)) == test_review)


test_reviews = ["this movie was amazing", "the movie was total crap it's amazing that they even made it"]
predictions = model1.predict(vectorize_sequences([encode(test_review) for test_review in test_reviews]))
for i, prediction in enumerate(predictions):
    result = 'Positive' if prediction >= 0.5 else 'Negative'
    print(test_reviews[i], ' ----> ', result)