import numpy as np
import tensorflow as tf
import pandas as pd
import os.path

learning_rate = 0.01
batch_size = 100
classes = 10
epochs = 20
dropout = 0.5
keep_rate = 0.8

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32)

train_x = pd.read_csv('train.csv')
train_y = train_x['label']
train_x = train_x.drop(['label'], 1)
train_x = np.array(train_x)
train_y = np.array(train_y)
valid_x = train_x[41900:,:]
valid_y = train_y[41900:]
train_x = train_x[:-100,:]
train_y = train_y[:-100]

test_x = pd.read_csv('test.csv')
test_x = np.array(test_x)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

train_y = dense_to_one_hot(train_y, classes)
valid_y = dense_to_one_hot(valid_y, classes)

print('Train X : ', train_x.shape)
print('Train Y : ', train_y.shape)
print('Validation X : ', valid_x.shape)
print('Validation Y : ', valid_y.shape)
print('Test X : ', test_x.shape)

def conv2d(input, W):
    return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn(input):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([classes]))}

    input = tf.reshape(input, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(input, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

num_examples = train_x.shape[0]
index_in_epoch = 0
epochs_completed = 0

def next_batch(batch_size):
    global train_x
    global train_y
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_x = train_x[perm]
        train_y = train_y[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_x[start:end], train_y[start:end]


def train_network(input):
    prediction = cnn(input)
    out = tf.nn.softmax_cross_entropy_with_logits(prediction, y)
    cost = tf.reduce_mean(out)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        predict = tf.argmax(prediction,1)
        if os.path.exists('digit_model.ckpt'):
            saver.restore(sess, 'digit_model.ckpt')
            print('Model restored...')
            print(sess.run(predict, feed_dict={x: valid_x}))
        else:
            print('Training...')
            for epoch in range(epochs):
                loss = 0
                for i in range(int(num_examples / batch_size)):
                    input_x, input_y = next_batch(batch_size)
                    v, c = sess.run([optimizer, cost], feed_dict={x: input_x, y: input_y})
                    loss += c
                print('Epoch:', epoch + 1, 'Loss:', loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x: valid_x, y: valid_y}))
            path = saver.save(sess, 'digit_model.ckpt')
            print('Model save in', path)


train_network(x)
