import tensorflow as tf
#import scipy.io as sc
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
def compute_accuracy(prediction,v_xs,v_ys):   
    y_pre=sess.run(prediction,feed_dict={x:v_xs,keep_prob:1}) #这里的keep_prob是保留概率，即我们要保留的RELU的结果所占比例  
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))  
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))  
    result=sess.run(accuracy,feed_dict={x:v_xs,y:v_ys,keep_prob:1})  
    return result

def readcsv(path):
    df = pd.read_csv(path)
    acc = np.array(df)
    return acc

def file2matrix(path):
    fr=open(path)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,4))
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:4]
        index+=1
    return returnMat

path = u'F:/学习/Parkinson/FOG/dataset/PD9ACCLabel.csv'
f=open(path)
#all = file2matrix(path)[1:]
all=readcsv(f)
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

##segmentation
## Sliding window
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

## stacked the last segment doublely, make the datasize even
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
    if label_seg[i]==-1:
        label_seg[i] = 0
        
feature_training = seg[0: middle]
label_training = one_hot(label_seg[0:middle])
feature_testing = seg[middle: data_size]
label_testing = one_hot(label_seg[middle: data_size])

# batch split
a = feature_training
b = feature_testing
nodes = 164
lameda = 0.0015
lr = 0.0004
fg = 0.3

batch_size = int(data_size * 0.25)    #0.25
train_fea = []
n_group = 3
for i in range(n_group):
    f = a[(0 + batch_size * i):(batch_size + batch_size * i)]
    train_fea.append(f)
print(train_fea[0].shape)

train_label = []
for i in range(n_group):
    f = label_training[(0 + batch_size * i):(batch_size + batch_size * i), :]
    train_label.append(f)
print(train_label[0].shape)
print(train_label[-1].shape, label_testing.shape)

# hyperparameters
n_inputs = no_fea
n_steps = len_seg  # time steps
n_hidden1_units = nodes  # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units = nodes
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="x")
y = tf.placeholder(tf.float32, [None, n_classes])
# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
    'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
    'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True)
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd2 = tf.sigmoid(tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2'])
    X_hidd3 = tf.sigmoid(tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3'])
    X_in = tf.reshape(X_hidd3, [-1, n_steps, n_hidden4_units])
    # cell
    ##########################################
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=fg, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=fg, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results, outputs[-1]


pred, Feature = RNN(x, weights, biases)
l2 = lameda * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + l2  # Softmax loss
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result = tf.argmax(pred, 1, name="pred_result")
label_true = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 

keep_prob=tf.placeholder(tf.float32)
init = tf.global_variables_initializer()
config = tf.ConfigProto()
saver = tf.train.Saver()  
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    step = 0
    start = time.clock()
    acc_his = []
    predd=[]
    while step <800:             #1000
    #    train_cost=0
        for i in range(n_group):    
            sess.run(train_op, feed_dict={x: train_fea[i],y: train_label[i]})
         #   batch_cost=sess.run(cost,feed_dict={x: train_fea[i],y: train_label[i]})
        #    train_cost+=batch_cost/n_group
        
  #      if (step+1)%1==0:  
          
        
      
#            print("Epoch :","%04d"%(step+1),"Train_cost","{:9f}".format(train_cost))  
        
    
        if step % 50 == 0:
            pp = sess.run(pred_result, feed_dict={x: b, y: label_testing})
            gt = np.argmax(label_testing, 1)
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
            predd.append(pp)
        step += 1
   
    """
   
    print("Train_accuracy :",sess.run(accuracy,feed_dict={x:train_fea[i],y: train_label[i]}))  
    print("Test_accuracy :",sess.run(accuracy,feed_dict={x: b,y: label_testing})) 
    """
    endtime = time.clock()
    print("run time:, max acc", endtime - start, max(acc_his))
    predd=pd.DataFrame(np.array(predd))
    predd.to_csv(u'F:/学习/Parkinson/FOG/dataset/predd.csv')
    testreal=pd.DataFrame(np.array(label_testing))
    testreal.to_csv(u'F:/学习/Parkinson/FOG/dataset/real.csv')
    
path1=u'F:/学习/Parkinson/FOG/dataset/predd.csv'
path2=u'F:/学习/Parkinson/FOG/dataset/real.csv'
def sensitivity_spec(path1,path2,num):
    f1=open(path1)
    pred=readcsv(f1)[num,1:]
    f2=open(path2)
    truelabel=readcsv(f2)[:,1:3].argmax(1)
    lenlabel=len(pred)
    TT=FF=TF=FT=0
    for i in range(lenlabel):
        if pred[i]==truelabel[i]==1:
            TT+=1
        elif pred[i]==truelabel[i]==0:
            FF+=1
        elif pred[i]==1 and truelabel[i]==0:
            FT+=1
        elif pred[i]==0 and truelabel[i]==1:
            TF+=1
    sens=TT/(TT+TF)
    spec=FF/(FF+FT)
    return sens, spec
    
