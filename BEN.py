#coding=utf-8
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

def Gen_coder(X, H, view_num):
    data = Normalize(np.transpose(X[v_num]))
    feature_dim = len(data[v_num])
    Layers['Gen_view' + str(v_num) + 'dense1'] = tf.layers.dense(inputs=H,
                                                                 units=500,
                                                                 activation=tf.nn.sigmoid,
                                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    Layers['Gen_view' + str(v_num) + 'dense2'] = tf.layers.dense(inputs=Layers['Gen_view' + str(v_num) + 'dense1'],
                                                                 units=5000,
                                                                 activation=tf.nn.sigmoid,
                                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    Layers['Gen_data_view' + str(v_num)] = tf.layers.dense(inputs=Layers['Gen_view' + str(v_num) + 'dense2'],
                                                           units=feature_dim,
                                                           activation=None,
                                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    Layers['Gen_view' + str(v_num) + 'loss'] = tf.norm(Layers['Gen_data_view' + str(v_num)] - data, ord='euclidean')

# Obtain data and process
data_mat = loadmat(r"C:\Users\Young Geng\Desktop\gLMSC\ORL_mtv.mat")
view_num = len(data_mat['X'][0])
X = np.split(data_mat['X'], view_num, axis=1)
Y = data_mat['gt']
N = len(np.transpose(X[0][0][0]))               # Get sample size
D_data = np.zeros([view_num], dtype=int)        # Get feature dimension for every view
class_num = len(np.unique(Y))
x_train = []
H = tf.Variable(tf.zeros([N, 200]))             # Latent space variable
loss_sum = 0
Layers = {}
for v_num in range(view_num):
    x_train.append(X[v_num][0][0])              # Simplize data frame
    D_data[v_num] = len(X[v_num][0][0])
    Gen_coder(x_train, H, view_num)
    loss_sum += Layers['Gen_view' + str(v_num) + 'loss']
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_sum)
# Training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(30):
        _, model_loss = sess.run([train_step, loss_sum])
        print(i, "epoch,loss:", model_loss)
        plt.plot(i, model_loss, 'ro')
    plt.show()
    H_latent = H.eval()
    H_latent_ = pd.DataFrame(H_latent)
    H_latent_.to_csv('H.csv', header=False, index=False)

