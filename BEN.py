#coding=utf-8
from math import sin,cos,sqrt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd

# Data Normalization
def Normalize(data1):
    m = np.mean(data1)
    mx = np.max(data1)
    mn = np.min(data1)
    return (data1-m)/(mx-mn)

def Gen_coder(X, H, v_num):
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data = min_max_scaler.fit_transform(np.array(X[v_num]))
    with tf.variable_scope('gencoder'+str(v_num)):
        data = Normalize(X[v_num])
        feature_dim = len(data)
        gen_dense = tf.layers.dense(inputs=H,
                                    units=500,
                                    activation=tf.nn.sigmoid,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        gen_dense = tf.layers.dense(inputs=gen_dense,
                                    units=1000,
                                    activation=tf.nn.sigmoid,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        gen_out = tf.layers.dense(inputs=gen_dense,
                                  units=feature_dim,
                                  activation=tf.nn.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return gen_out

def Discriminator(latent_input, labels, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        concated = tf.concat([latent_input, labels], 1)
        dis_dense = tf.layers.dense(inputs=concated,
                                    units=1000,
                                    activation=tf.nn.sigmoid,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        dis_dense = tf.layers.dense(inputs=dis_dense,
                                    units=1000,
                                    activation=tf.nn.sigmoid,
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
    for v_num in range(view_num):
        gen_out = Gen_coder(X, H, v_num)
        data = Normalize(np.transpose(X[v_num]))
        view_loss = tf.norm(gen_out - data, ord='euclidean')
        gen_loss += view_loss
    latent_fake = H
    latent_real = gaussian_mixture(len(Y), class_num=class_num, y_batch=Y-1)
    labels = np.zeros((len(Y), class_num), dtype=np.float32)
    labels[range(len(Y)), np.array(Y - 1).T.astype(int)] = 1
    valid = np.ones((len(Y), 1), dtype=np.float32)
    fake = np.zeros((len(Y), 1), dtype=np.float32)
    d_out_real = Discriminator(latent_real, labels)
    d_out_fake = Discriminator(latent_fake, labels, reuse=True)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_real,
                                                                         labels=valid))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_out_fake,
                                                                         labels=fake))
    H_loss = d_loss_fake+gen_loss
    d_loss = d_loss_fake+d_loss_real
    return gen_loss, d_loss, H_loss

def model_op(H, gen_loss, d_loss, H_loss):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('gencoder')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    gen_train_opt = tf.train.RMSPropOptimizer(0.01).minimize(gen_loss, var_list=g_vars)
    d_train_opt = tf.train.RMSPropOptimizer(0.001).minimize(d_loss, var_list=d_vars)
    H_train_opt = tf.train.RMSPropOptimizer(0.001).minimize(H_loss, var_list=H)
    return gen_train_opt, d_train_opt, H_train_opt

if __name__ == '__main__':
    # Obtain data and process
    data_mat = loadmat(r"C:\Users\Young Geng\Desktop\gLMSC\handwritten1.mat")
    view_num = len(data_mat['X'][0])
    X = np.split(data_mat['X'], view_num, axis=1)
    Y = data_mat['gt']
    class_num = len(np.unique(Y))
    N = len(Y)               # Get sample size
    D_data = np.zeros([view_num], dtype=int)        # Get feature dimension for every view
    x_train = []
    H = tf.Variable(tf.random.normal([N, 100], mean=0.0, stddev=1e-6),name='H')             # Latent space variable
    # H = tf.Variable(tf.zeros([N, 100]), name='H')
    for v_num in range(view_num):
        x_train.append(X[v_num][0][0])              # Simplize data frame
        D_data[v_num] = len(X[v_num][0][0])
    gen_loss, d_loss, H_loss = model_loss(x_train, Y, H, view_num, class_num)
    gen_op, d_op, H_op = model_op(H, gen_loss, d_loss, H_loss)
    tvars = tf.trainable_variables()
    # Training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(30):
            sess.run([gen_op, d_op])
            g_loss = gen_loss.eval()
            print(epoch, "epoch\ngen_loss=%f,des_loss=" %g_loss, d_loss.eval())
            plt.plot(epoch, g_loss, 'ro')
            if epoch > 10:
                sess.run(H_op)
        plt.show()
        H_latent = H.eval()
        H_latent_ = pd.DataFrame(H_latent)
        H_latent_.to_csv('H.csv', header=False, index=False)

