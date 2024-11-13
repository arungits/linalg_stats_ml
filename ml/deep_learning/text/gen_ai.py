import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.supports_masking = True
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        tril_mask = tf.cast(i >= j, dtype="int32")
        tril_mask_with_batch = tf.zeros(shape=(batch_size, sequence_length, sequence_length), dtype="int32")
        tril_mask = tril_mask_with_batch + tril_mask
        return tril_mask
    
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask = None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attention_output_1 = self.layernorm1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query=attention_output_1, key=encoder_outputs, value=encoder_outputs, attention_mask=padding_mask)
        attention_output_2 = self.layernorm2(attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm2(attention_output_2 + proj_output)
    
dataset = keras.utils.text_dataset_from_directory(
    directory="aclImdb", label_mode=None, batch_size=256)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br/>", ""))

from tensorflow.keras.layers import TextVectorization
sequence_length = 100
vocab_size = 10000

text_vectorization = TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length = sequence_length + 1,
)
text_vectorization.adapt(dataset)

def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    x = vectorized_sequences[:, :-1]
    y = vectorized_sequences[:, 1:]
    return x, y

lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

embed_dim = 256
latent_dim = 2048
num_heads = 2


inputs = keras.Input(shape=(None,), dtype="int64")
embedding_layer = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
x = embedding_layer(inputs)
pos_embed_layer = layers.Embedding(sequence_length, embed_dim)
x = x + pos_embed_layer(tf.range(sequence_length))
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, x, mask=embedding_layer.compute_mask(inputs))
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop")
print(model.summary())

model.fit(lm_dataset, epochs=200)

class TextGenerator:
    def __init__(self, prompt, generate_length, model_input_length, temperature=1.0):
        self.prompt = prompt.split()
        assert(len(self.prompt) <= model_input_length)
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperature = temperature
    
    def sample_next(self, predictions, temperature=None):
        if temperature is None:
            temperature = self.temperature
        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        return np.argmax(probas)
    
    def __call__(self):
        sentence = self.prompt
        for i in range(self.generate_length):
            sentence = sentence[-self.model_input_length:]
            tokenized_sentence = text_vectorization([" ".join(sentence)])[:,:-1]
            next_token = 1
            count = 1
            while next_token == 1 and count <= 10:
                predictions = model(tokenized_sentence)
                next_token = self.sample_next(predictions[0, len(sentence) - 1, :])
                count += 1
            if next_token == 0:
                break
            sampled_token = tokens_index[next_token]
            sentence.append(sampled_token)
        return sentence

def generate_text(prompt):
    for length in (50, 100, 200):
        for temperature in (0.5, 0.75, 1., 1.5):
            text_generator = TextGenerator(prompt, length, sequence_length, temperature)
            text = text_generator()
            print(" ".join(text))
            
prompts = ["This movie", "I", "The", "Yesterday"]
for prompt in prompts:
    generate_text(prompt)