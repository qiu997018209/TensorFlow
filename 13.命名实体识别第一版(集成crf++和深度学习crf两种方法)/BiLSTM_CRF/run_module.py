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
from datetime import timedelta
from BiLSTM_CRF.module import BiLSTM_CRF
from BiLSTM_CRF.data import data
from tensorflow.contrib.crf import viterbi_decode
from conf import *
import tensorflow as tf
import numpy as np


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch,x_batch_lenth,keep_prob,model):
    feed_dict = {
        model.batch_x: x_batch,
        model.batch_y: y_batch,
        model.sequence_lengths:x_batch_lenth,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_,x_lenth,model):
    """评估在某一数据上的准确率和损失"""
    feed_dict=feed_data(x_,y_,x_lenth,1,model)
    loss,logits,transition_params = sess.run([model.loss,model.logits, model.transition_params],feed_dict=feed_dict)
    label_list = []
    acc,num=0,-1
    for logit, seq_len in zip(logits, x_lenth):
        num+=1
        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)#基于当前状态的得分和之前状态转移过来的得分，然后进行viterbi的解码
        label_list.append(viterbi_seq)
        if(viterbi_seq == (list(y_[num]))[:seq_len]):
            acc+=1
            #print('l:{} v:{} y:{}'.format(seq_len, viterbi_seq,y_[num])) 
    return acc/num,loss,label_list

def train(args,data):
    with tf.Graph().as_default() as g:
        model=BiLSTM_CRF(data,args)
        session = tf.Session(graph=g)
        with session.as_default():
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if not os.path.exists(args.module_path):
                os.makedirs(args.module_path)
            print('Training and evaluating...')
            start_time = time.time()
            total_batch = 0              # 总批次
            best_lost_val = 100           # 最佳验证集准确率
            last_improved = 0            # 记录上一次提升批次
            require_improvement = 3000   # 如果超过1000轮未提升，提前结束训练
        
            flag = False
            batches=data.get_batch_data()
            for _,batch in enumerate(batches):
                x_batch, y_batch,x_batch_lenth,x_val,y_val,x_val_lenth=batch
                feed_dict = feed_data(x_batch, y_batch,x_batch_lenth,args.dropout,model)
                #print('x_batch:{},y_batch:{},x_batch_lenth:{}'.format(x_batch.shape,y_batch.shape,x_batch_lenth.shape))
                if total_batch % args.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[model.keep_prob] = 1.0
                    loss_train= session.run(model.loss, feed_dict=feed_dict)
                    acc,loss_val,_= evaluate(session, x_val, y_val,x_val_lenth,model)  
                    if loss_val < best_lost_val:
                        # 保存最好结果
                        best_lost_val = loss_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=args.module_path)
                        improved_str = '*'
                    else:
                        improved_str = ''
        
                    time_dif = get_time_dif(start_time)
                    msg = 'total_batch: {0:>6}, Train Loss: {1:>6.2},'\
                        + ' Val Loss: {2:>6.2}, acc:{3:>3.5},Time: {4} {5} '
                    print(msg.format(total_batch, loss_train, loss_val, acc,time_dif, improved_str))
        
                session.run(model.train_op, feed_dict=feed_dict)  # 运行优化
                total_batch += 1
        
                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                print('最佳准确率:',best_lost_val)
    
if __name__ == '__main__': 
    args=get_args()
    data=data(args)  
    train(args,data)