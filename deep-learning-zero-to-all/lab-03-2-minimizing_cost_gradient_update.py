import tensorflow as tf
tf.set_random_seed(777)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
#b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W
#hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
#gradient = tf.reduce_mean((W * X + b - Y) * X)
descent_w = W - learning_rate * gradient
#descent_b = b - learning_rate * gradient
update_w = W.assign(descent_w)
#update_b = b.assign(descent_b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update_w, feed_dict={X: x_data, Y: y_data})
    #sess.run([update_w, update_b], feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), 'W: ', sess.run(W))
    #print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), 'W: ', sess.run(W), 'b: ', sess.run(b))
