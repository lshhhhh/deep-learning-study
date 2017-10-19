import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

N = x_data.shape[1]
X = tf.placeholder(tf.float32, [None, N])
Y = tf.placeholder(tf.int32, [None, 1])

nb_classes = len(np.unique(y_data))
target = tf.one_hot(Y, nb_classes)
target = tf.reshape(target, [-1, nb_classes])
target = tf.cast(target, tf.float32)

W = tf.Variable(tf.random_normal([N, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

def sigmoid(x):
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

def sigmoid_prime(x):
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigmoid(x) * (1. - sigmoid(x))

layer1 = tf.matmul(X, W) + b
y_pred = sigmoid(layer1)

loss_i = -target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

assert y_pred.shape.as_list() == target.shape.as_list()

d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7)
d_sigmoid = sigmoid_prime(layer1)
d_layer = d_loss * d_sigmoid
d_b = d_layer
d_W = tf.matmul(tf.transpose(X), d_layer)

learning_rate = 0.01
train_step = [
    tf.assign(W, W - learning_rate * d_W), 
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b))
]

prediction = tf.argmax(y_pred, 1)
acc_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
acc_res = tf.reduce_mean(tf.cast(acc_mat, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        sess.run(train_step, feed_dict={X: x_data, Y: y_data})

        if step % 10 == 0:
            step_loss, acc = sess.run([loss, acc_res], feed_dict={X: x_data, Y: y_data})
            print('Step: {:5}\t Loss: {:10.5f}\t Acc: {:2%}'.format(step, step_loss, acc))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data):
        msg = '[{}]\t Prediction: {:d}\t True y: {:d}'
        print(msg.format(p == int(y[0]), p, int(y[0])))


