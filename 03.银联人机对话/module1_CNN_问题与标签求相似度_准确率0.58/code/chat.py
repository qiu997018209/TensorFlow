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

session_conf = tf.ConfigProto(allow_soft_placement=my_data.Flags.allow_soft_placement,log_device_placement=my_data.Flags.log_device_placement)
# 通过检查点文件锁定最新的模型
ckpt = tf.train.get_checkpoint_state(os.path.join(my_data.Flags.file_path,'runs','checkpoints'))
# 在下面的代码中，默认加载了TensorFlow计算图上定义的全部变量
# 直接加载持久化的图
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
 
input_x_1 = tf.get_default_graph().get_tensor_by_name("input_x_1:0")
input_x_2 = tf.get_default_graph().get_tensor_by_name("input_x_2:0")
input_x_3 = tf.get_default_graph().get_tensor_by_name("input_x_3:0")
dropout_keep_prob = tf.get_default_graph().get_tensor_by_name("dropout_keep_prob:0")
run_score = tf.get_default_graph().get_tensor_by_name("output/scores:0")

def quick_chat(sess,usrs_input,ans_num=1):  
    max_scoreresult = []
    x_test_1, x_test_2, x_test_3 = my_data.get_test_data(usrs_input)
    feed_dict = {
        input_x_1: x_test_1,
        input_x_2: x_test_2,
        input_x_3: x_test_3,
        dropout_keep_prob: 1.0
    }        
    scores = list(sess.run(run_score, feed_dict))        
    max_score = max(scores)
    label = my_data.labels[scores.index(max_score)]    
    ans = my_data.label_ans[label]
    print("匹配的语义标签为:%s,答案为:%s,得分为:%s\n"%(label,ans,max_score))
 
def start_chat(sess,usrs_input,ans_num=3):
   
    result = []
    x_test_1, x_test_2, x_test_3 = my_data.get_test_data(usrs_input)
    feed_dict = {
        input_x_1: x_test_1,
        input_x_2: x_test_2,
        input_x_3: x_test_3,
        dropout_keep_prob: 1.0
    }        
    scores = list(sess.run(run_score, feed_dict))        
    for key,value in enumerate(scores):
        result.append((key,value))                      
    result = sorted(result,key=lambda x:x[1],reverse=True)
    #找到前3个答案
    for i in range(ans_num):
        label = my_data.labels[result[i][0]]
        score = result[i][1]     
        ans = my_data.label_ans[label]
        print("匹配的语义标签为:%s,答案为:%s,得分为:%s\n"%(label,ans,score))
        
with tf.Session(config=session_conf) as sess:
    # 载入参数，参数保存在两个文件中，不过restore会自己寻找
    saver.restore(sess,ckpt.model_checkpoint_path)  
    while True:
        usrs_input = input("请输入您的问题:")
        start = time.clock()   
        quick_chat(sess,usrs_input)
        end = time.clock()
        print("本次对话耗时:%s"%(end-start)) 
     