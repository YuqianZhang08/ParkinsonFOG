# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:41:58 2018

@author: zyq
"""

import tensorflow as tf
# import scipy.io as sc
import numpy as np
import time
from scipy import stats
import pandas as pd


# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(int(n_values))[np.array(y_, dtype=np.int32)]


"""""
######Parkinson's disease, 3 features ,4th col is PD label{0,1}, 5th col is person label{0-4}
feature = sc.loadmat("PD_5sub.mat")
all = feature['dataset'][1:]

#  select the single person
idx = np.where(all[:, -1]==1)
all = all[idx]
print (all.shape)
"""""


def compute_accuracy(prediction, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})  
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, keep_prob: 1})
    return result


def readcsv(path):
    df = pd.read_csv(path)
    acc = np.array(df)
    return acc


def file2matrix(path):
    fr = open(path)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 4))
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:4]
        index += 1
    return returnMat


path = u'E:/zyq/FOG/dataset/sumup.csv'
all = file2matrix(path)[1:]
# select the first 4th columns, dropout the person ID.
all = all[:, 0:4]
n_classes = 2

# Cliping
print('all', all.shape)
len_sample = 100
data_size_or = all.shape[0]
# clip the data, make sure it can be divided into four parts: 3parts for training and 1 part for testing
all = all[:4 * len_sample * int(data_size_or / (4 * len_sample))]
data_size = all.shape[0]
print('all', all.shape)

no_fea = all.shape[1] - 1
F_ = all[:, 0:no_fea]
L_ = all[:, no_fea:no_fea + 1]

# segmentation
# Sliding window
len_seg = 100
overlap = 50
_overlap = 100 - overlap  # the non-overlap part
seg = F_[0:len_seg]
print(seg.shape)
seg = seg[np.newaxis, :]
print(seg.shape)
label_seg = np.transpose(L_[0:len_seg])  # the label vector of this segment
print('label', label_seg.shape)

for i in range(1, int(data_size_or / _overlap - 5)):
    new = F_[_overlap * i:_overlap * i + len_seg]
    new = new[np.newaxis, :]
    label_new = np.transpose(L_[_overlap * i:_overlap * i + len_seg])
    modes, _ = stats.mode(label_new, axis=1)
    # if the mean = modes, are the samples in this segment are from the same label, stack it.
    if np.mean(label_new) == modes:
        seg = np.vstack((seg, new))
        label_seg = np.vstack((label_seg, label_new))

# stacked the last segment doublely, make the datasize even
label_seg = label_seg[:, 0:1]
print(seg.shape, label_seg.shape, sum(label_seg))

# zip
zipped = zip(seg, label_seg)
# np.random.shuffle(list(zipped))
seg, label_seg = zip(*zipped)
seg = np.array(seg)
label_seg = np.array(label_seg)

data_size = seg.shape[0]
seg = seg[:4 * int(data_size / 4)]
label_seg = label_seg[:4 * int(data_size / 4)]

data_size = seg.shape[0]
middle = int(data_size * 0.75)

for i in range(len(label_seg)):
    if label_seg[i] == -1:
        label_seg[i] = 0

feature_training = seg[0: middle]
label_training = one_hot(label_seg[0:middle])
feature_testing = seg[middle: data_size]
label_testing = one_hot(label_seg[middle: data_size])

# batch split
a = feature_training
b = feature_testing
nodes = 164
lameda = 0.001
lr = 0.001
fg = 0.3
batch_size=tf.placeholder(tf.int32,[])
keep_prob=tf.placeholder(tf.float32,[])
n_group = 3

hidden_size=256
layer_num=4

n_inputs = no_fea
n_steps = len_seg  # time steps

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="x")
y = tf.placeholder(tf.float32, [None, n_classes])
X = tf.reshape(x, [-1, n_inputs])
# Define weights
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)

init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
W = tf.Variable(tf.truncated_normal([hidden_size, n_classes], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[n_classes]), dtype=tf.float32)
y_pre = tf.nn.sigmoid(tf.matmul(outputs[-1], W) + bias)
#y_pre = tf.argmax(tf.matmul(outputs[-1], W) + bias)
l2 = lameda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pre, labels=y)) + l2  # Softmax loss
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

label_true = y
correct_pred = tf.equal(y_pre, label_true)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
config = tf.ConfigProto()

batch_size = 200 # 0.25
train_fea = []
n_group = data_size/batch_size
for i in range(n_group):
    f = a[(0 + batch_size * i):(batch_size + batch_size * i)]
    train_fea.append(f)
print(train_fea[0].shape)

train_label = []
for i in range(n_group):
    f = label_training[(0 + batch_size * i):(batch_size + batch_size * i), :]
    train_label.append(f)
print(train_label[0].shape)

with tf.Session(config=config) as sess:
   sess.run(tf.global_variables_initializer())
   step=0
   acc_his = []
   start = time.clock()
   while step < 2000:
       for j in range(n_group):  
            sess.run(train_op, feed_dict={x: train_fea[j], y: train_label[j],keep_prob: 0.5})
            if (i+1)%200 == 0:
               train_accuracy = sess.run(accuracy, feed_dict={x: train_fea[j], y: train_label[j], keep_prob: 1.0})
       if step % 50 == 0:
            pp = sess.run(y_pre, feed_dict={x: b, y: label_testing})
            hh = sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing
            })
            h2 = sess.run(accuracy, feed_dict={x: train_fea[i],
                                               y: train_label[i]})
            print("training acc", h2)
            print("The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is:", hh)

            print("The cost is :", sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
            acc_his.append(hh)
       step += 1
       endtime = time.clock()
   print("run time:, max acc", endtime - start, max(acc_his))           


