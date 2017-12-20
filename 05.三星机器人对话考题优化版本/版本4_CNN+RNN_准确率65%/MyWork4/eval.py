#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import sys
import os
import code
import time
from sklearn import metrics
from datetime import timedelta
from cnn_module import TextCNN
from rnn_module import TextRNN
from data import data
import conf as conf
import tensorflow as tf
import numpy as np
from numpy import dtype


base_dir = 'data/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'test.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

def feed_data(x_batch, y_batch, keep_prob,model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(save_path, x_, y_,type):
    """评估在某一数据上的准确率和损失""" 
    with tf.Graph().as_default() as g:
        if(type=='cnn'):
            model = TextCNN() 
        else:
            model = TextRNN() 
        session = tf.Session(graph=g)
        with session.as_default():
            session.run(tf.global_variables_initializer()) 
            saver = tf.train.Saver()
            saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
            feed_dict = feed_data(x_, y_, 1.0,model)
            scores, acc = session.run([model.logits,model.acc], feed_dict=feed_dict)
    return scores, acc

def test_cnn():
    print("Loading test data...")
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径
    print('CNN Testing...')
    scores_test, acc_test = evaluate(save_path, x_test, y_test,'cnn')
    print('cnn accuracy:',acc_test)
    return scores_test, acc_test
    
def test_rnn():
    print("Loading test data...")
    save_dir = 'checkpoints/textrnn'
    save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径
    print('RNN Testing...')
    scores_test, acc_test = evaluate(save_path, x_test, y_test,'rnn')
    print('rnn accuracy:',acc_test)
    return scores_test, acc_test
 
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)
        
def cal_accuracy(scores,predict):
    pred=np.argmax(scores,1)
    predict=np.argmax(predict,1)
    #code.interact(local=locals())
    return np.mean(np.equal(pred,predict))

def final_accuracy():
    deep_scores=scores_cnn+scores_rnn     
    acc=cal_accuracy(deep_scores,y_test)
    print('rnn + cnn accuracy:',acc)

if __name__=='__main__':
    data=data()
    x_test,y_test=data.get_test_data() 
    model = TextCNN()   
    scores_cnn,acc_cnn=test_cnn()
    model = TextRNN()
    scores_rnn,acc_rnn=test_rnn()
    final_accuracy()