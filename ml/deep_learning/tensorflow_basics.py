import tensorflow as tf
x = tf.ones(shape=(3,2))
assert(x.shape == (3,2))
assert(tf.reduce_all(x == 1.0))
x = tf.zeros(shape=(4,3))
assert(x.shape == (4,3))
assert(tf.reduce_all(x == 0.0))

x = tf.random.normal(shape=(100,100), mean=0., stddev=1.)
assert(x.shape == (100,100))
mean = tf.reduce_mean(x)
assert(abs(mean - 0) <= 0.05)
std = tf.sqrt(tf.reduce_sum(tf.square(x - mean)))/100.0
assert(abs(std - 1.0) <= 0.05)

x = tf.random.uniform(shape=(2,2), minval=0., maxval=1.)
assert(x.shape == (2,2))
assert(tf.reduce_all(x >= 0.) and tf.reduce_all(x <= 1.))

# Variable

v = tf.Variable(initial_value=tf.random.normal(shape=(3,2)))
assert(v.shape == (3,2))
assert(not tf.reduce_all(v == 1.0))

v = v.assign(tf.ones(shape=(3,2)))
assert(v.shape == (3,2))
assert(tf.reduce_all(v == 1.0))
assert(tf.reduce_sum(v) == 6.)

v[0,0].assign(3.)
assert(tf.reduce_sum(v) == 8.)

v.assign_sub(tf.ones(shape=(3,2)))
assert(v.shape == (3,2))
assert(v[0,0].numpy() == 2)
assert(tf.reduce_sum(v) == 2.)

v.assign_add(tf.ones(shape=(3,2)))
assert(v.shape == (3,2))
assert(tf.reduce_sum(v) == 8.)

A = tf.Variable(initial_value=tf.random.normal(shape=(2,2)))
I = tf.Variable(initial_value=tf.zeros(shape=(2,2)))
I[0,0].assign(1.)
I[1,1].assign(1.)
assert(tf.reduce_all(tf.matmul(A, I) == A))
assert(tf.reduce_all(tf.matmul(A, I) == tf.matmul(I, A)))

# First order gradient
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = tf.square(x)
grad = tape.gradient(y, [x])
assert(len(grad) == 1)
assert(grad[0].numpy() == 6.)

# Second order gradient
t = tf.Variable(3.0)
with tf.GradientTape() as second_order_tape:
    with tf.GradientTape() as first_order_tape:
        disp = 4.9 * (t**2)
    vel = first_order_tape.gradient(disp, [t])
acc = second_order_tape.gradient(vel, [t])
assert(abs(vel[0].numpy() - 29.4) <= 0.01)
assert(abs(acc[0].numpy() - 9.8) <= 0.01)
