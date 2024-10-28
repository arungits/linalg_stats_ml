import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(output_mode="multi_hot", max_tokens=10)
dataset = [
    "I write, erase, rewrite",
    "Erase again and then.",
    "A poppy blooms.",
    "I write, rewrite, and still rewrite again",
    "This is a very long string to check if the length matters at all in this process"
]

text_vectorization.adapt(dataset)
# Get and print the corpus of the given dataset
vocabulary = text_vectorization.get_vocabulary()
print(vocabulary)

test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print(encoded_sentence)
inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print(decoded_sentence)

int_sequence_dataset = text_vectorization(dataset)
print(int_sequence_dataset)