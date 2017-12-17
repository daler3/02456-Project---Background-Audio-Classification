import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from preprocessor import preprocessor


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def apply_convolution(x, kernel_size, num_channels, depth):
    weights = weight_variable(
        [kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights), biases))


def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride_size, stride_size, 1],
                          padding='SAME')

if __name__ == '__main__':
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
               'jackhammer', 'siren', 'street_music']
    train_dirs = []

    # logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)

    n_folders = 9
    for i in range(1, n_folders + 1):
        train_dirs.append('fold{0}'.format(i))

    pp = preprocessor()
    pp.load_extracted_fts_lbs(load_path='extracted_long_60', train_dirs=train_dirs, test_fold='fold10')
    # pp.load_extracted_fts_lbs(load_path=load_path, train_dirs=train_dirs, test_fold=test_fold, val_fold=val_fold)

    tr_features = pp.train_x
    tr_labels = pp.train_y
    ts_features = pp.test_x
    ts_labels = pp.test_y

    frames = 101
    bands = 60

    feature_size = 6060  # 60x101
    num_labels = 10
    num_channels = 2

    batch_size = 50
    kernel_size = 30
    depth = 20
    num_hidden = 200

    learning_rate = 0.01
    total_iterations = 2000

    X = tf.placeholder(tf.float32, shape=[None, bands, frames, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    cov = apply_convolution(X, kernel_size, num_channels, depth)

    shape = cov.get_shape().as_list()
    cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

    f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
    f_biases = bias_variable([num_hidden])
    f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights), f_biases))

    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    cost_history = np.empty(shape=[1], dtype=float)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for itr in range(total_iterations):
            offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
            batch_x = tr_features[offset:(offset + batch_size), :, :, :]
            batch_y = tr_labels[offset:(offset + batch_size), :]

            _, c = session.run([optimizer, loss], feed_dict={
                X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, c)

        print('Test accuracy: ', round(session.run(
            accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
        fig = plt.figure(figsize=(15, 10))
        plt.plot(cost_history)
        plt.axis([0, total_iterations, 0, np.max(cost_history)])
        plt.show()
