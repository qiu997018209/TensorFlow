# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import os
import time
from cnn.cnn_module import TextCNN
from cnn.run_cnn import train as cnn_train
from threading import Thread
from cnn.data import data as cnn_data

class cnn_module():
    def __init__(self,args,cnn_data):
        self.args=args
        self.data=cnn_data
       
    def load_module(self):     
        with tf.Graph().as_default() as g:
            self.module=TextCNN(self.args)
            self.session = tf.Session(graph=g)
            with self.session.as_default():
                self.session.run(tf.global_variables_initializer()) 
                saver = tf.train.Saver()
                saver.restore(sess=self.session, save_path=self.args.module_path)  # 读取保存的模型
        
    def predict(self,client_params,server_param):
        params={}
        quest=client_params['params']['quest']
        #深度学习没有数据的时候进行对话功能
        if len(self.data.quest_label)==0:
            params['answer']='深度学习没有导入数据，无法提供对话功能'
            params['success']="false"
            print("深度学习没有导入数据，无法提供对话功能")
            server_param['result']=params
            return
        try:            
            feed_dict = {self.module.input_x: [self.data.build_one_vector(quest)],self.module.keep_prob: 1.0}
            predict,probs = self.session.run([self.module.predictions,self.module.scores], feed_dict=feed_dict)
            if client_params['params']['rate'] !=0:
                if probs[0][predict]>=client_params['params']['rate']:
                    print("实际准确率为:%f,要求准确率为:%f"%(probs[0][predict],client_params['params']['rate']))
                    params['success']="true"
                    params['answer']=self.data.id_to_label[str(predict[0])]
                else:
                    params['answer']='准确率低于要求，结果不可用'
                    params['success']="false"
                    print("实际准确率为:%f,要求准确率为:%f"%(probs[0][predict],client_params['params']['rate']))
            else:
                if probs[0][predict]>=self.accuracy:
                    print("实际准确率为:%f,要求准确率为:%f"%(probs[0][predict],self.accuracy))
                    params['success']="true"
                    params['answer']=self.data.id_to_label[str(predict[0])]
                else:
                    params['answer']='准确率低于要求，结果不可用'
                    params['success']="false"
                    print("实际准确率为:%f,要求准确率为:%f"%(probs[0][predict],self.accuracy))                       
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
        if len(self.data.quest_label)==0:
            params['success']="false"
            params['message']="深度学习没有数据，不能训练"
            server_param['result']=params
            return       
        else:    
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
            self.data=cnn_data(self.args)
            cnn_train(self.args,self.data)
            print('开始重新载入模型')
            self.load_module()
            self.get_accuracy_rate() 
            print('cnn模型加载完毕,对话功能已开启')       
            #考虑到模型载入期间有人对话，知道模型载入后再将进度条设为1
            self.args.rate=1.0
            self.args.time=0
        except Exception as e:
            print('训练失败',e)
            self.args.rate=1.0
            self.args.time=0
        #查询进度
    def lookup(self,client_params,server_param):
        params={} 
        params['progress']=("%d%%")%(self.args.rate*100)
        #以分钟计算
        params['need_time']=self.args.time
        params['success']='true'
        server_param['result']=params
        #计算准确率阈值
    def get_accuracy_rate(self):
        result=[]
        with open('../data/xianliao.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split('\t')
                feed_dict = {self.module.input_x: [self.data.build_one_vector(line[0],False)],self.module.keep_prob: 1.0}
                predict,probs = self.session.run([self.module.predictions,self.module.scores], feed_dict=feed_dict)
                result.append(probs[0][predict])
        #允许90%的闲聊数据通过
        acc_1=list(sorted(result)[int(len(result)*0.9)])[0]#result里是array格式
        acc_2=self.data.min_accuracy
        if acc_2>acc_1:
            self.accuracy=(acc_1+acc_2)/2
        else:
            self.accuracy=acc_2
        print('最终的准确率阈值为:{},训练数据最低准确率要求:{},过滤掉90%的闲聊数据的准确率要求:{}'.format(self.accuracy,acc_2,acc_1))
        return 
        
        
        
        
        
        
        
        
             