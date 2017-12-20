#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
#通用模型参数
#词汇表大小
vocab_size=5000

num_class=1000

batch_size=64

num_epochs=100

#句子最大长度
max_document_lenth=0
# 每多少轮次将训练结果写入tensorboard scalar
save_per_batch=100
print_per_batch=100

learning_rate=1e-3
dropout_keep_prob=0.5
l2_reg_lambda=0
embedding_size=128

#CNN模型
filter_sizes="1,2,3,4,5"
num_filters=128
#RNN模型参数
#隐藏层层数
num_layers=2
#隐藏层神经元
hidden_dim=128
# lstm 或 gru
rnn = 'lstm' 

        
