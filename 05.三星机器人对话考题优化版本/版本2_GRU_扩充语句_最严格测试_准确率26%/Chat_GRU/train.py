#coding:utf-8
'''
Created on 2017年12月4日

@author: qiujiahao

@email:997018209@qq.com

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from data import data
import numpy as np
import pandas
import tensorflow as tf
from sklearn import metrics
from tensorflow.contrib.layers.python.layers import encoders

#设置调试信息显示级别
tf.logging.set_verbosity(tf.logging.INFO)

#处理本地数据
my_data=data()

learn = tf.contrib.learn


#features是训练数据，target是标签
def rnn_model(features, target,params):
    """用RNN模型(这里用的是GRU)完成文本分类"""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    label_num=params['label_num']
    embed_dim=params['embed_dim']
    vocab_size=params['vocab_size']
    #将[128,32]维度的数据，变为[128,32,50]
    word_vectors = tf.contrib.layers.embed_sequence(
            features, vocab_size=vocab_size, embed_dim=embed_dim, scope='words')

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    #tf.unstack矩阵分解，按照维度1的方向进行分解,得到32个[128,1,50]
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    #创建一个GRU的cell,state_size是50
    cell = tf.contrib.rnn.GRUCell(embed_dim)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    #一共有32个输出结果,每一个结果是[128,50]
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    #对label_num个类别进行ont-hot化
    target = tf.one_hot(target, label_num, 1, 0)
    #全连接层，将输出转化为[1,label_num],fully_connected内部应该有将[128,32,50]维度的数据扁平化为[1,xx]格式，然后进行全连接层的计算
    logits = tf.contrib.layers.fully_connected(encoding, label_num, activation_fn=None)
    #损失函数:交叉熵
    loss = tf.losses.softmax_cross_entropy(target,logits)

    # Create a training op.
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

model_fn = rnn_model  
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn,model_dir='data/runs',params={'embed_dim':50,'label_num':len(my_data.labels),'vocab_size':len(my_data.word_dic)}))
classifier.fit(x_train, y_train, batch_size=128,steps=4000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))