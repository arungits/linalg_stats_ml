# A multi-class linear classifier in pure tensorflow

import tensorflow as tf

class LinearClassifier:
    def __init__(self, input_dimensions, num_classes):
        self.input_dimensions = input_dimensions
        self.num_classes = num_classes
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dimensions, num_classes)))
        self.b = tf.Variable(initial_value=tf.zeros(shape=(num_classes,)))

    def forward(self, x):
        B, C = x.shape
        assert(C == self.input_dimensions)
        self.out = tf.nn.softmax(tf.matmul(x, self.W) + self.b)
        return self.out

    def fit(self, x, y, epochs=5, learning_rate = 1e-2):
        B, C = x.shape
        assert(C == self.input_dimensions)
        assert(len(y.shape) == 1)
        assert(B == len(y))
        sparse_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy()

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self.forward(x)
                self.loss = sparse_crossentropy(y, y_pred)
            grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(self.loss, [self.W, self.b])
            self.W.assign_sub(grad_loss_wrt_W * learning_rate)
            self.b.assign_sub(grad_loss_wrt_b * learning_rate)
            print(f"Epoch {epoch}, Loss {self.loss}")

    def predict(self, x):
        B, C = x.shape
        assert (C == self.input_dimensions)
        y = self.forward(x)
        return y




