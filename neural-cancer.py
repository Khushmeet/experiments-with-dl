import tensorflow as tf
import pandas as pd
import numpy as np

'''
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
'''


# breast cancer data
cancer_df = pd.read_csv('data/breast-cancer-wisconsin.data',
                        header=None,
                        names=['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                               'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
                               'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
                        )
cancer_df.replace('?', 0, inplace=True)
cancer_df['Class'].replace(2, 0, inplace=True)
cancer_df['Class'].replace(4, 1, inplace=True)
X_cancer_train = cancer_df.drop(['Sample code number','Class'], 1)
Y_cancer_train = cancer_df['Class']

print(X_cancer_train.head())
print(Y_cancer_train.head())

X_cancer_train = np.array(X_cancer_train)
Y_cancer_train = np.array(Y_cancer_train)

X_cancer_test = X_cancer_train[679:, :]
Y_cancer_test = Y_cancer_train[679:]
X_cancer_train = X_cancer_train[:-20, :]
Y_cancer_train = Y_cancer_train[:-20]

print(X_cancer_train.shape)
print(Y_cancer_train.shape)
print(X_cancer_test.shape)
print(Y_cancer_test.shape)


# neural network
h1_nodes = 300
h2_nodes = 300
output_nodes = 2
classes = 2
batch_size = 7

x = tf.placeholder('float')
y = tf.placeholder('float')


def model(x):
    h1_layer = {'weight': tf.Variable(tf.random_normal(9, h1_nodes)),
                'bias': tf.Variable(tf.random_normal(h1_nodes))}
    h2_layer = {'weight': tf.Variable(tf.random_normal(h1_nodes, h2_nodes)),
                'bias': tf.Variable(tf.random_normal(h2_nodes))}
    out_layer = {'weight': tf.Variable(tf.random_normal(h2_nodes, output_nodes)),
                 'bias': tf.Variable(tf.random_normal(output_nodes))}
    l1_value = tf.add(tf.matmul(x, h1_layer['weight']), h1_layer['bias'])
    l1_value = tf.nn.relu(l1_value)
    l2_value = tf.add(tf.matmul(l1_value, h2_layer['weight']), h2_layer['bias'])
    l2_value = tf.nn.relu(l2_value)
    out_value = tf.add(tf.matmul(l2_value, out_layer['weight']), out_layer['bias'])
    return out_value


def train(x_train, y_train, x_test, y_test):
    predicted = model(x_train)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predicted, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        c = sess.run([optimizer, cost], feed_dict={x:x, y:y_train})
        print('Cost', c)
        correct = tf.equal(tf.argmax(predicted, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: x_test, y: y_test}))

train(X_cancer_train, Y_cancer_train, X_cancer_test, Y_cancer_test)






