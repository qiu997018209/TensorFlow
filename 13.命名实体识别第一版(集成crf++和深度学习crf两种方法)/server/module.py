# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import os
import time
from BiLSTM_CRF.module import BiLSTM_CRF
from BiLSTM_CRF.run_module import train as lstm_train
from threading import Thread
from BiLSTM_CRF.data import data as lstm_data
from tensorflow.contrib.crf import viterbi_decode
from collections import defaultdict

class lstm_module():
    def __init__(self,args,lstm_data):
        self.args=args
        self.data=lstm_data
        #self.load_module()
        
    def load_module(self):     
        with tf.Graph().as_default() as g:
            self.module=BiLSTM_CRF(self.data,self.args)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer()) 
                saver = tf.train.Saver()
                saver.restore(sess=self.session, save_path=self.args.module_path)  # 读取保存的模型
        print('NER模型加载完毕,对话功能已开启')
        
    def predict(self,client_params,server_param):
        params={}
        quest=client_params['params']['quest']
        try: 
            x_batch,x_batch_lenth,y_batch=self.data.build_vector([quest],[[0]*len(quest)])#看大神的代码备注上都说此处的y用不上
            feed_dict = {self.module.batch_x: x_batch,self.module.batch_y:y_batch,self.module.sequence_lengths:x_batch_lenth,self.module.keep_prob: 1}       
            logits,transition_params= self.session.run([self.module.logits,self.module.transition_params], feed_dict=feed_dict)
            params['answer']=self.viterbi_decode(logits,transition_params,x_batch_lenth)#viterbi解码    
            params['success']="true"
            server_param['result']=params
        except Exception as e:
            print(e)
            params['answer']='使用对话功能前请先训练模型'
            params['success']="false"
            print("深度学习有数据，但是没有训练就开始对话")
            server_param['result']=params
            return            
              
    def train(self,client_params,server_param):
        params={}   
        params['success']="true"
        params['message']="retrain start"
        server_param['result']=params
        #启动一个线程开始重新训练
        t=Thread(target=self.retrain)
        t.start()
        #重新训练
    def retrain(self):
        print('开始重新训练')
        try:
            #重新训练的时候数据通常都发生了变化
            self.data=lstm_data(self.args)
            lstm_train(self.args,self.data)
            print('开始重新载入模型')
            self.load_module()       
        except Exception as e:
            print('训练失败',e)
        #viterbi解码
    def viterbi_decode(self,logits,transition_params,x_lenth):
        label_list = []
        for logit, seq_len in zip(logits, x_lenth):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)#基于当前状态的得分和之前状态转移过来的得分，然后进行viterbi的解码
            label_list.append(viterbi_seq)
        label=''.join([self.data.label2tag[w]+' ' for w in label_list[0]])     
        return label.strip()
    