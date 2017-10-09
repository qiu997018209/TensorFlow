#coding:utf-8
'''
Created on 2017年9月21日

@author: qiujiahao

@email:997018209@qq.com

'''
import time
import os
import tensorflow as tf
import data_help as dh
import numpy as np

my_data = dh.data_help()
#读入测试集合
my_data.read_test_file()

session_conf = tf.ConfigProto(allow_soft_placement=my_data.Flags.allow_soft_placement,log_device_placement=my_data.Flags.log_device_placement)
        
def test(sess):
    input_x_1 = tf.get_default_graph().get_tensor_by_name("input_x_1:0")
    input_x_2 = tf.get_default_graph().get_tensor_by_name("input_x_2:0")
    input_x_3 = tf.get_default_graph().get_tensor_by_name("input_x_3:0")
    dropout_keep_prob = tf.get_default_graph().get_tensor_by_name("dropout_keep_prob:0")
    run_score = tf.get_default_graph().get_tensor_by_name("output/scores:0")
    for quest in my_data.test_quests:
        my_data.record_data['cost_time'] = time.clock()
        x_test_1, x_test_2, x_test_3 = my_data.get_test_data(quest)
        feed_dict = {
            input_x_1: x_test_1,
            input_x_2: x_test_2,
            input_x_3: x_test_3,
            dropout_keep_prob: 1.0
        }        
        scores = list(sess.run(run_score,feed_dict))
        my_data.show_test_result(scores,quest,"记录模型训练参数对结果的影响","测试标签对应语料规模对结果的影响")

with tf.Session(config=session_conf) as sess:
    # 通过检查点文件锁定最新的模型
    ckpt = tf.train.get_checkpoint_state(os.path.join(my_data.Flags.file_path,'runs','checkpoints'))
    # 在下面的代码中，默认加载了TensorFlow计算图上定义的全部变量
    # 直接加载持久化的图
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
    # 载入参数，参数保存在两个文件中，不过restore会自己寻找
    saver.restore(sess,ckpt.model_checkpoint_path)   
    test(sess)

     