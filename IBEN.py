#coding=utf-8
from math import sin,cos
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report
os.environ['CUDA_VISIBLE_DEVICES']='1, 3'

def split(dataset, labels, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(dataset.T, labels,
                                                        test_size=test_size,
                                                        random_state=1)

    return x_train, x_test, y_train, y_test

# Data Normalization
def Normalize(data1):
    m = np.mean(data1)
    mx = np.max(data1)
    mn = np.min(data1)
    return (data1-m)/(mx-mn)

def Gen_coder(X, H, v_num, reuse=False):
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data = min_max_scaler.fit_transform(np.array(X[v_num]))
    with tf.variable_scope('gencoder'+str(v_num), reuse=reuse):
        data = Normalize(X[v_num])
        feature_dim = len(data)
        gen_dense = tf.layers.dense(inputs=H,
                                    units=500,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        gen_dense = tf.layers.dense(inputs=gen_dense,
                                    units=1000,
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        gen_out = tf.layers.dense(inputs=gen_dense,
                                  units=feature_dim,
                                  activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return gen_out

def Discriminator(latent_input, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        dis_dense = tf.layers.dense(inputs=latent_input,
                                    units=1000,
                                    activation=tf.nn.leaky_relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        dis_dense = tf.layers.dense(inputs=dis_dense,
                                    units=100,
                                    activation=tf.nn.leaky_relu,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        dis_out = tf.layers.dense(inputs=dis_dense,
                                  units=1,
                                  activation=None,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return dis_out

def gaussian_mixture(batch_size, class_num, n_dim=100, x_var=0.5, y_var=0.1, y_batch=None):

    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label):
        shift = 1.4
        if label >= class_num:
            label = np.random.randint(0, class_num)
        r = 2.0 * np.pi / float(class_num) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if y_batch is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], y_batch[batch])
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, class_num))

    return z

def model_loss(X, Y, H, view_num, class_num):
    gen_loss = 0
    LAMBDA = 100
    BETA = 1e-4
    for v_num in range(view_num):
        gen_out = Gen_coder(X, H, v_num)
        data = Normalize(np.transpose(X[v_num]))
        view_loss = tf.norm(gen_out - data, ord='euclidean')
        gen_loss += view_loss
    latent_fake = H
    latent_real = gaussian_mixture(len(Y), class_num=class_num, y_batch=Y-1)
    labels = np.zeros((len(Y), class_num), dtype=np.float32)
    labels[range(len(Y)), np.array(Y - 1).T.astype(int)] = 1
    fake_concated = tf.concat([latent_fake, labels], 1)
    real_concated = tf.concat([latent_real, labels], 1)
    d_out_real = Discriminator(fake_concated)
    d_out_fake = Discriminator(real_concated, reuse=True)
    d_loss_real = tf.reduce_mean(d_out_real)
    d_loss_fake = tf.reduce_mean(d_out_fake)
    H_loss = d_loss_fake+LAMBDA*gen_loss
    d_loss = -d_loss_fake+d_loss_real
    alpha = tf.random_uniform(shape=[len(Y), 1], minval=0., maxval=1.)
    differences = fake_concated - real_concated
    interpolates = real_concated + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    d_loss += BETA * gradient_penalty
    return gen_loss, d_loss, H_loss

def model_op(H, gen_loss, d_loss, H_loss):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('gencoder')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    gen_train_opt = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_loss, var_list=g_vars)
    d_train_opt = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(d_loss, var_list=d_vars)
    H_train_opt = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(H_loss, var_list=H)
    return gen_train_opt, d_train_opt, H_train_opt

def test(H, H_test, training_labels, testing_data, testing_labels, view_num):
    test_loss = 0
    for v_num in range(view_num):
        gte_out = Gen_coder(testing_data, H_test, v_num, reuse=True)
        tdata = Normalize(np.transpose(testing_data[v_num]))
        v_loss = tf.norm(gte_out - tdata, ord='euclidean')
        test_loss += v_loss
    H_test_op = tf.train.AdamOptimizer(learning_rate=1, beta1=0.5, beta2=0.9).minimize(test_loss, var_list=H_test)

    with tf.Session() as tsess:
        # Fit H_test
        initt = tf.global_variables_initializer()
        tsess.run(initt)
        for epoch in range(1500):
            tsess.run(H_test_op)
            t_loss = test_loss.eval()
            print(epoch, "epoch\ntest_loss=%f" %t_loss)
            plt.plot(epoch, t_loss, 'ro')
        plt.show()
        Ht_latent = H_test.eval()
        Ht_latent_ = pd.DataFrame(Ht_latent)
        Ht_latent_.to_csv('H_test.csv', header=False, index=False)
        clf = (knn())
        clf.fit(np.array(H.eval()), training_labels)
        pred_y = clf.predict(np.array(H_test.eval()))

    return classification_report(testing_labels,pred_y)

if __name__ == '__main__':
    # Obtain data and process
    data_mat = loadmat(r"/home/gengyu/pyproject/IBENM/IBEN/handwritten1.mat")
    view_num = len(data_mat['X'][0])
    X = np.split(data_mat['X'], view_num, axis=1)
    Y = data_mat['gt']
    class_num = len(np.unique(Y))
    N = len(Y)               # Get sample size
    datasets = []
    x_train = []
    x_test = []
    # H = tf.Variable(tf.zeros([N, 100]), name='H')
    for v_num in range(view_num):
        datasets.append(X[v_num][0][0])              # Simplize data frame
    for v_num in range(view_num):
        x_tr, x_te, y_train, y_test = split(datasets[v_num], Y, test_size=0.3)
        x_train.append(x_tr.T), x_test.append(x_te.T)
    tr_size = len(y_train)
    te_size = len(y_test)
    # Latent space variable
    H = tf.Variable(tf.random.normal([tr_size, 100], mean=0.0, stddev=1e-6),name='H')
    H_test = tf.Variable(tf.random.normal([te_size, 100], mean=0.0, stddev=1e-6), name='H_test')
    gen_loss, d_loss, H_loss = model_loss(x_train, y_train, H, view_num, class_num)
    gen_op, d_op, H_op = model_op(H, gen_loss, d_loss, H_loss)


    with tf.Session() as sess:
        # Training
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(1500):
            sess.run([gen_op, d_op])
            g_loss = gen_loss.eval()
            print(epoch, "epoch\ngen_loss=%f,dis_loss=" %g_loss, d_loss.eval())
            # plt.plot(epoch, g_loss, 'ro')
            if epoch > 100:
                sess.run(H_op)
        # plt.show()
        H_latent = H.eval()
        H_latent_ = pd.DataFrame(H_latent)
        H_latent_.to_csv('H.csv', header=False, index=False)
        # Testing
        train_ac = test(H, H, y_train, x_train, y_train, view_num)
        test_ac = test(H, H_test, y_train, x_test, y_test, view_num)
        print(test_ac)




