import os, pathlib, shutil, random

base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

def create_val_dataset():
    for category in ("pos", "neg"):
        os.makedirs(val_dir / category)
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files)
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname, val_dir / category / fname)
# create_val_dataset()

# Load train, val and test datasets

from tensorflow import keras
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)

# Check the datasets

for inputs, targets in train_ds:
    print(inputs.shape)
    print(inputs.dtype)
    print(targets.shape)
    print(targets.dtype)
    print(inputs[0])
    print(targets[0])
    break

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

from tensorflow.keras.layers import TextVectorization
def generate_datasets(ngrams=1, output_mode="multi_hot"):
    text_vectorization = TextVectorization(
        ngrams=ngrams,
        max_tokens=20000,
        output_mode=output_mode
    )

    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)
    binary_ngram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    binary_ngram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    binary_ngram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y), num_parallel_calls=4)
    return (binary_ngram_train_ds, binary_ngram_val_ds, binary_ngram_test_ds)

def evaluate_model(model_file_name, test_ds):
    model = keras.models.load_model(model_file_name)
    result = model.evaluate(test_ds)
    print("Test accuracy ", result[1])

def train_and_evaluate(ngrams=1, output_mode="multi_hot"):
    binary_ngram_train_ds, binary_ngram_val_ds, binary_ngram_test_ds = generate_datasets(ngrams, output_mode)

    for inputs, targets in binary_ngram_train_ds:
        print(inputs.shape)
        print(inputs.dtype)
        print(targets.shape)
        print(targets.dtype)
        print(inputs[0])
        print(targets[0])
        break

    # Train the model
    model = get_model()
    print(model.summary())
    model_file_name = f"binary_ngram_{ngrams}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_file_name, save_best_only=True)
    ]

    model.fit(binary_ngram_train_ds.cache(), validation_data=binary_ngram_val_ds.cache(), epochs=10, callbacks=callbacks)

    evaluate_model(model_file_name, binary_ngram_test_ds)

# Train and evaluate Unigram model with Bag of words encoding

print("### Unigram model ###")
train_and_evaluate(1, "multi_hot")

# Train and evaluate Bigram model with Bag of words encoding

print("### Bigram model (multi hot) ###")
train_and_evaluate(2, "multi_hot")

# Train and evaluate Bigram model with TF-IDF (word frequency) encoding

print("### Bigram model with term frequency ###")
train_and_evaluate(2, "tf_idf")

