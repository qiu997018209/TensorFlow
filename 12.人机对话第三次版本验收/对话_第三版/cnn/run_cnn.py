#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')

import os
import time
from sklearn import metrics
from datetime import timedelta
from cnn.cnn_module import TextCNN
from cnn.data import data
from conf import *
from log import log
import tensorflow as tf
import numpy as np


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob,model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_,model):
    """评估在某一数据上的准确率和损失"""
    feed_dict = feed_data(x_, y_, 1.0,model)
    loss, acc,scores,correct_predictions = sess.run([model.loss, model.acc,model.scores,model.correct_predictions], feed_dict=feed_dict)

    return loss, acc,scores,correct_predictions

def train(args,data):
    with tf.Graph().as_default() as g:
        model=TextCNN(args)
        session = tf.Session(graph=g)
        with session.as_default():
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if not os.path.exists(args.module_path):
                os.makedirs(args.module_path)
            start_time = time.time()
            log('Training and evaluating...')
            start_time = time.time()
            total_batch = 0              # 总批次
            best_acc_val = 0.0           # 最佳验证集准确率
            last_improved = 0            # 记录上一次提升批次
            require_improvement = 3000   # 如果超过1000轮未提升，提前结束训练
        
            flag = False
            batches=data.get_batch_data()
            for index,batch in enumerate(batches):
                x_batch, y_batch,x_val,y_val=batch
                feed_dict = feed_data(x_batch, y_batch, args.dropout_keep_prob,model)
                if len(data.quest_label)<1000:
                    args.print_per_batch=1
                if total_batch % args.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train= session.run([model.loss, model.acc], feed_dict=feed_dict)
                    loss_val, acc_val,scores,correct_predictions = evaluate(session, x_val, y_val,model)   # todo
        
                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=args.module_path)
                        improved_str = '*'
                        data.get_accuracy_rate(correct_predictions,scores)#计算准确率阈值
                    else:
                        improved_str = ''
        
                    time_dif = get_time_dif(start_time)
                    msg = 'total_batch: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                        + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    log(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
        
                session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1
        
                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    log("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                log('最佳准确率:{}'.format(best_acc_val))


if __name__ == '__main__': 
    args=get_args()
    data=data(args)  
    train(args,data)