__author__ = 'Mohammad'

import tensorflow as tf

sess = tf.Session()

W1 = tf.Variable([.3], tf.float32)
b1 = tf.Variable([-.3], tf.float32)
W2 = tf.Variable([.3], tf.float32)
b2 = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model1 = W1 * x + b1
linear_model2 = W2 * x + b2

x_train = [1,2,3,4]
y_train = [0,-1,4,5]
z_train = [0,0,1,1]

init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)
squared_deltas1 = tf.square((linear_model1 - y)*z)
loss1 = tf.reduce_sum(squared_deltas1)
squared_deltas2 = tf.square((linear_model2 - y)*(1-z))
loss2 = tf.reduce_sum(squared_deltas2)
loss = loss1 + loss2

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(10000):
    sess.run(train, {x: x_train, y: y_train, z: z_train})

curr_W1, curr_b1, curr_W2, curr_b2, curr_loss = sess.run([W1, b1, W2, b2, loss], {x: x_train, y: y_train, z: z_train})
print("W1: %s b1: %s W2: %s b2: %s loss: %s"%(curr_W1, curr_b1, curr_W2, curr_b2, curr_loss))

