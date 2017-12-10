#coding:utf-8
'''
Created on 2017年12月4日

@author: qiujiahao

@email:997018209@qq.com
基于卷积神经网络的中文文本分类
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from data import data
learn = tf.contrib.learn

#最小词频数
MIN_WORD_FREQUENCE = 1
#词嵌入的维度
EMBEDDING_SIZE = 50
#filter个数
N_FILTERS = 128
#感知野大小
WINDOW_SIZE = 10
#filter的形状
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
#池化
POOLING_WINDOW = 4
POOLING_STRIDE = 2


#处理本地数据
my_data=data()

#设置调试信息显示级别
tf.logging.set_verbosity(tf.logging.INFO)

#features是数据，target是标签
def cnn_model(features, target,params):
    """
    2层的卷积神经网络，用于短文本分类
    """
    # 先把词转成词嵌入
    # 我们得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵
    # 接着我们可以把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的矩阵形式
    embed_dim = params['embed_dim']
    label_num = params['label_num']
    vocab_size = params['vocab_size']
    target = tf.one_hot(target, label_num, 1, 0)
    word_vectors = tf.contrib.layers.embed_sequence(
            features, vocab_size=vocab_size, embed_dim=embed_dim, scope='words')
    #增加一个维度，变为:[128,32,50,1]
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # 添加卷积层做滤波,VALID如果窗口不够，会被丢弃，same则是补0
        #[128,7,1,10]
        conv1 = tf.contrib.layers.convolution2d(
                word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # 添加RELU非线性
        conv1 = tf.nn.relu(conv1)
        # 最大池化
        pool1 = tf.nn.max_pool(
                conv1,
                ksize=[1, POOLING_WINDOW, 1, 1],
                strides=[1, POOLING_STRIDE, 1, 1],
                padding='SAME')
        # 将tensor的对应的维数进行置换
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # 第2个卷积层
        conv2 = tf.contrib.layers.convolution2d(
                pool1, N_FILTERS, FILTER_SHAPE2, padding='VALID')
        # 抽取特征
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # 全连接层
    logits = tf.contrib.layers.fully_connected(pool2, label_num, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=0.01)

    return ({
            'class': tf.argmax(logits, 1),
            'prob': tf.nn.softmax(logits)
    }, loss, train_op)

# Train and predict
x_train,x_test,y_train,y_test=my_data.get_train_test2()

model_fn = cnn_model  
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn,model_dir='data/runs',params={'embed_dim':50,'label_num':len(my_data.labels),'vocab_size':len(my_data.word_dic)}))
classifier.fit(x_train, y_train, batch_size=128,steps=12000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))