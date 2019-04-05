#coding=utf-8
from math import sin,cos,sqrt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os

os.environ['CUDA_VISIBLE_DEVICES']='1, 3'

def split(dataset, labels, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(dataset.T, labels, test_size=test_size,
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
    with tf.variable_scope('gencoder' + str(v_num), reuse=reuse):
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
                                  activation=tf.nn.tanh,
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

def model_op(H, num_epoch, gen_lr, gen_loss, d_loss, H_loss):
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    gen_train_opt = tf.train.RMSPropOptimizer(gen_lr).minimize(gen_loss, global_step=num_epoch)
    d_train_opt = tf.train.RMSPropOptimizer(0.001).minimize(d_loss, var_list=d_vars)
    H_train_opt = tf.train.RMSPropOptimizer(0.001).minimize(H_loss, var_list=H)
    return gen_train_opt, d_train_opt, H_train_opt

def test(H, H_test, training_labels, testing_data, testing_labels, view_num, train=True):
    with tf.name_scope('test'):
        test_loss = 0
        for v_num in range(view_num):
            gte_out = Gen_coder(testing_data, H_test, v_num, reuse=True)
            tdata = Normalize(np.transpose(testing_data[v_num]))
            v_loss = tf.norm(gte_out - tdata, ord='euclidean')
            test_loss += v_loss
        test_epoch = tf.Variable(0, trainable=False)
        test_lr = tf.train.natural_exp_decay(learning_rate=1e-4, global_step=test_epoch,
                                            decay_steps=1500, decay_rate=0.9, staircase=False)
        H_test_op = tf.train.RMSPropOptimizer(learning_rate=test_lr).minimize(test_loss, var_list=H_test)
        # Fit H_test
        uninitial_vars = [var for var in tf.all_variables() if 'test' in var.name]
        initt = tf.variables_initializer(uninitial_vars)
        sess.run(initt)
        if train == True:
            for epoch in range(10000):
                sess.run(test_lr)
                sess.run(H_test_op)
                t_loss = test_loss.eval()
                if epoch%100 == 0:
                    print(epoch, "epoch\ntest_loss=%f" %t_loss)
            Ht_latent = H_test.eval()
            Ht_latent_ = pd.DataFrame(Ht_latent)
            Ht_latent_.to_csv('H_test.csv', header=False, index=False)
        clf = (knn())
        clf.fit(np.array(H.eval()), training_labels.ravel())
        pred_y = clf.predict(np.array(H_test.eval()))
        print(accuracy_score(testing_labels, pred_y, normalize=True) )
    return classification_report(testing_labels,pred_y)

if __name__ == '__main__':
    # Obtain data and process
    data_mat = loadmat(r"/home/gengyu/datasets/handwritten1.mat")
    view_num = len(data_mat['X'][0])
    X = np.split(data_mat['X'], view_num, axis=1)
    Y = data_mat['gt']
    class_num = len(np.unique(Y))
    N = len(Y)  # Get sample size
    datasets = []
    x_train = []
    x_test = []
    for v_num in range(view_num):
        datasets.append(X[v_num][0][0])  # Simplize data frame
        x_tr, x_te, y_train, y_test = split(datasets[v_num], Y, test_size=0.3)
        x_train.append(x_tr.T), x_test.append(x_te.T)
    tr_size = len(y_train)
    te_size = len(y_test)
    ytr = pd.DataFrame(y_train)
    ytr.to_csv('ytr.csv', header=False, index=False)
    yte = pd.DataFrame(y_test)
    yte.to_csv('yte.csv', header=False, index=False)
    # Latent space variable
    H = tf.Variable(tf.random.normal([tr_size, 100], mean=0.0, stddev=1e-6),name='H')
    H_test = tf.Variable(tf.random.normal([te_size, 100], mean=0.0, stddev=1e-6), name='test_H')
    gen_loss, d_loss, H_loss = model_loss(x_train, y_train, H, view_num, class_num)
    # Learning rate decay
    num_epoch = tf.Variable(0, trainable=False)
    gen_lr = tf.train.natural_exp_decay(learning_rate=0.001, global_step=num_epoch,
                                decay_steps=1500, decay_rate=0.9, staircase=True)
    gen_op, d_op, H_op = model_op(H, num_epoch, gen_lr, gen_loss, d_loss, H_loss)
    # Save the value
    saver = tf.train.Saver()

    # Training
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Restore saved variables
        try:
            saver.restore(sess, "/home/gengyu/pyproject/tmp/vars.ckpt")
            print("Checkpoint restored successfully!")
            epochs = 0
        except ValueError:
            print("Not find a checkpoint file.\nReset all variables.")
            epochs = 10000
        except tf.errors.NotFoundError:
            print("Directory doesn't exist.\nReset all variables.")
            epochs = 10000
        else:
            print("Checkpoint restored successfully!")
        for epoch in range(epochs):
            sess.run([gen_op, d_op])
            g_loss = gen_loss.eval()
            if epoch%100 ==0:
                print(gen_lr.eval())
                print(epoch, "epoch\ngen_loss=%f,dis_loss=%f,H_loss=%f" %(g_loss, d_loss.eval(), H_loss.eval()))
            if epoch > 100:
                sess.run(H_op)
        H_latent = H.eval()
        H_latent_ = pd.DataFrame(H_latent)
        H_latent_.to_csv('H.csv', header=False, index=False)
        # Save
        save_path = saver.save(sess, "/home/gengyu/pyproject/tmp/vars.ckpt")
        print("Model saved in path: %s" % save_path)
        # Testing
        train_ac = test(H, H, y_train, x_train, y_train, view_num, train=False)
        print(train_ac)
        test_ac = test(H, H_test, y_train, x_test, y_test, view_num)
        print(test_ac)