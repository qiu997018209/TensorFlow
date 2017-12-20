#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import time
from sklearn import metrics
from datetime import timedelta
from cnn_module import TextCNN
from data import data
from data2 import data2

import conf 
import tensorflow as tf
import numpy as np

base_dir = 'data/'
train_dir = os.path.join(base_dir, 'train.txt')
test_dir = os.path.join(base_dir, 'test.txt')
val_dir = os.path.join(base_dir, 'test.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    feed_dict = feed_data(x_, y_, 1.0)
    loss, acc,scores = sess.run([model.loss, model.acc,model.logits], feed_dict=feed_dict)

    return loss, acc,scores

def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_acc_val = 0.0           # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 100000   # 如果超过1000轮未提升，提前结束训练

    flag = False
    batches=data.get_batch_data()
    for x_batch, y_batch,x_val,y_val in batches:
        feed_dict = feed_data(x_batch, y_batch, conf.dropout_keep_prob)
        if total_batch % conf.save_per_batch == 0:
            # 每多少轮次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, total_batch)

        if total_batch % conf.print_per_batch == 0:
            # 每多少轮次输出在训练集和验证集上的性能
            feed_dict[model.keep_prob] = 1.0
            loss_train, acc_train= session.run([model.loss, model.acc], feed_dict=feed_dict)
            loss_val, acc_val,scores = evaluate(session, x_val, y_val)   # todo

            if acc_val > best_acc_val:
                # 保存最好结果
                best_acc_val = acc_val
                last_improved = total_batch
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''

            time_dif = get_time_dif(start_time)
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
            print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

        session.run(model.optim, feed_dict=feed_dict)  # 运行优化
        total_batch += 1

        if total_batch - last_improved > require_improvement:
            # 验证集正确率长期不提升，提前结束训练
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break  # 跳出循环
    if flag:  # 同上
        print('最佳准确率:',best_acc_val)
        
'''
def test():
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
    for i in range(num_batch):   # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
'''

if __name__ == '__main__': 
    data=data()  
    model = TextCNN()
    train()