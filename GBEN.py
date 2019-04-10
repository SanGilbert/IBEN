# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import sin, cos, sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import classification_report

LAMBDA = 0.1
BETA = 10


def split(dataset, labels, test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=test_size, random_state=1)
    return x_train, x_test, y_train, y_test


def Descriminator(Dis_Input, reuse=False):
    # concated = tf.concat([Dis_Input[0], Dis_Input[1]], axis=1)
    with tf.variable_scope('discriminator', reuse=reuse):
        D_dense1 = tf.layers.dense(inputs=Dis_Input,
                                   units=1000,
                                   activation=tf.nn.sigmoid,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        D_dense2 = tf.layers.dense(inputs=D_dense1,
                                   units=100,
                                   activation=tf.nn.sigmoid,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        D_dense3 = tf.layers.dense(inputs=D_dense2,
                                   units=1,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return D_dense3


def gaussian_mixture(class_num, batch_size, n_dim=100, x_var=0.5, y_var=0.1, y_batch=None):
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
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], y_batch[batch])
            else:
                z[batch, zi * 2:zi * 2 + 2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, self.class_num))

    return z


def Normalize(data1):
    m = np.mean(data1)
    mx = np.max(data1)
    mn = np.min(data1)
    return (data1 - m) / (mx - mn)


def BEN(view_num, X, H_1, reuse=False):
    with tf.variable_scope('BEN', reuse=reuse):
        loss_sum = 0
        names = locals()
        for v_num in range(view_num):
            data = X[v_num]
            D_data = len(data[1])
            names['view' + str(v_num) + 'dense1'] = tf.layers.dense(inputs=H_1,
                                                                    units=500,
                                                                    activation=tf.nn.sigmoid,
                                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                        0.1))
            names['view_' + str(v_num) + 'dense2'] = tf.layers.dense(inputs=names['view' + str(v_num) + 'dense1'],
                                                                     units=1000,
                                                                     activation=tf.nn.sigmoid,
                                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                         0.1))
            names['view_' + str(v_num) + 'data_'] = tf.layers.dense(inputs=names['view_' + str(v_num) + 'dense2'],
                                                                    units=D_data,
                                                                    activation=None,
                                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                                        0.1))

            names['view_' + str(v_num) + 'loss'] = tf.norm(names['view_' + str(v_num) + 'data_'] - data,
                                                           ord='euclidean')
            loss_sum += names['view_' + str(v_num) + 'loss']
    return loss_sum


def test(H, H_test, training_labels, testing_data, testing_labels, view_num, train=True):
    with tf.name_scope('test'):
        # for v_num in range(view_num):
        #     gte_out = Gen_coder(testing_data, H_test, v_num, reuse=True)
        #     tdata = Normalize(testing_data[v_num])
        #     v_loss = tf.norm(gte_out - tdata, ord='euclidean')
        #     test_loss += v_loss
        test_loss = BEN(view_num, testing_data, H_test, reuse=True)
        H_test_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.9).minimize(test_loss,
                                                                                               var_list=H_test)

        # testvars = tf.trainable_variables()
        # print([v.name for v in testvars])
        # Fit H_test
        uninitial_vars = [var for var in tf.all_variables() if 'test' in var.name]
        initt = tf.variables_initializer(uninitial_vars)
        sess.run(initt)
        if train == True:
            for epoch in range(2000):
                sess.run(H_test_op)
                t_loss = test_loss.eval()
                if epoch % 100 == 0:
                    print(epoch, "epoch\ntest_loss=%f" % t_loss)
                    plt.plot(epoch, t_loss, 'ro')
            plt.show()
            Ht_latent = H_test.eval()
            Ht_latent_ = pd.DataFrame(Ht_latent)
            Ht_latent_.to_csv('H_test.csv', header=False, index=False)
        clf = (knn(3))
        clf.fit(np.array(H.eval()), training_labels.ravel())
        pred_y = clf.predict(np.array(H_test.eval()))
    return classification_report(testing_labels, pred_y)


if __name__ == '__main__':
    m = loadmat(r"/home/gengyu/pyproject/2view-caltech101-8677sample.mat")
    view_num = len(m['X'][0])
    X = np.split(m['X'], view_num, axis=1)
    data = Normalize(np.transpose(X[0][0][0]))
    N = len(data)

    Y = m['gt']
    class_num = len(np.unique(Y))
    datasets = []
    for v_num in range(view_num):
        datasets.append(Normalize(np.transpose(X[v_num][0][0])))
    x_train = []
    x_test = []
    for v_num in range(view_num):
        # datasets.append(X[v_num][0][0])  # Simplize data frame
        x_tr, x_te, y_train, y_test = split(datasets[v_num], Y, test_size=0.3)
        x_train.append(x_tr), x_test.append(x_te)
    tr_size = len(y_train)
    te_size = len(y_test)
    Y = y_train
    X = x_train
    batch_size = tr_size

    ytr = pd.DataFrame(y_train)
    ytr.to_csv('ytr.csv', header=False, index=False)
    yte = pd.DataFrame(y_test)
    yte.to_csv('yte.csv', header=False, index=False)

    H = tf.Variable(tf.random_normal([tr_size, 100], stddev=0.0001), name='H')
    H_test = tf.Variable(tf.random_normal([te_size, 100], stddev=0.0001), name='test_')

    latent_real = gaussian_mixture(class_num, batch_size, y_batch=Y - 1)
    latent_real_tensor = tf.convert_to_tensor(latent_real)
    # H_latent_ = pd.DataFrame(latent_real)
    # H_latent_.to_csv('latent.csv', header=False, index=False)

    labels = np.zeros((len(y_train), class_num))
    labels[range(len(y_train)), np.array(y_train - 1).T.astype(int)] = 1

    H_concated = tf.concat([H, tf.cast(labels, tf.float32)], axis=1)
    real_concated = tf.concat([latent_real_tensor, tf.cast(labels, tf.float32)], axis=1)

    sigmoid_real = Descriminator(real_concated, reuse=False)
    sigmoid_fake = Descriminator(H_concated, reuse=True)

    BEN_loss = BEN(view_num, X, H)
    D_loss = -tf.reduce_mean(sigmoid_fake) + tf.reduce_mean(sigmoid_real)
    H_g_loss = tf.reduce_mean(sigmoid_fake) + LAMBDA * BEN_loss
    # H_g_loss = BEN_loss

    alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
    differences = H_concated - real_concated
    interpolates = real_concated + (alpha * differences)
    gradients = tf.gradients(Descriminator(interpolates, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    D_loss += BETA * gradient_penalty

    tvars = tf.trainable_variables()
    #
    d_vars = [var for var in tvars if 'discriminator' in var.name]
    g_vars = [var for var in tvars if 'H' in var.name]
    BEN_vars = [var for var in tvars if 'BEN' in var.name]
    gen_vars = g_vars + BEN_vars

    print("d_vars", [v.name for v in d_vars])
    print([v.name for v in g_vars])

    # BEN_solver = tf.train.RMSPropOptimizer(0.00005).minimize(BEN_loss, var_list=BEN_vars)

    D_solver = tf.train.RMSPropOptimizer(5e-6).minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.RMSPropOptimizer(5e-4).minimize(H_g_loss, var_list=gen_vars)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(100):
            _, d_Loss = sess.run([D_solver, D_loss])
        for i in range(10000):
            # _, BEN_Loss = sess.run([BEN_solver, BEN_loss])

            _, d_Loss = sess.run([D_solver, D_loss])

            _, g_Loss = sess.run([G_solver, H_g_loss])

            if i % 200 == 0:
                print('Iter: {}'.format(i))
                print('D loss: {:.4}'.format(d_Loss))
                print('G_loss: {:.4}'.format(g_Loss))
                # print('BEN_loss: {:.4}'.format(BEN_Loss))
                plt.subplot(221)
                plt.plot(i, d_Loss, 'ro')
                plt.subplot(222)
                plt.plot(i, g_Loss, 'ro')
                plt.subplot(223)
                # plt.plot(i, BEN_Loss, 'ro')
        plt.show()
        H_latent = H.eval()
        # H_latent = pd.Series(H)
        H_latent_ = pd.DataFrame(H_latent)
        H_latent_.to_csv('H.csv', header=False, index=False)
        # Testing
        train_ac = test(H, H, y_train, x_train, y_train, view_num, train=False)
        print(train_ac)
        test_ac = test(H, H_test, y_train, x_test, y_test, view_num)
        print(test_ac)