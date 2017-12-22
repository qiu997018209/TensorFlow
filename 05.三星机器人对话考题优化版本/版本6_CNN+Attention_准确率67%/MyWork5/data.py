#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import math
import xlrd
import numpy as np
import tensorflow.contrib.keras as kr
from data_lib import data_lib
import conf
from collections import Counter
from collections import defaultdict

np.random.seed(0) 

class data(data_lib):
    def __init__(self):
        #初始化，数据预处理
        self.data_process()
        
    def data_process(self):
        print('开始处理数据')
        #建立词汇表
        self.build_vocab_size()
        
        #读取标签
        self.get_labels()
            
        #读取标签    
    def get_labels(self):
        self.label_quest=defaultdict(set)
        with open('data/train.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t')
                self.label_quest[line[1]].add(line[0])
                if len(line[0]) > conf.max_document_lenth:
                    conf.max_document_lenth=len(line[0])
        print('max_document_lenth',conf.max_document_lenth)
        print('标签数量:%d'%(len(self.label_quest)))
        self.label_to_id=dict(zip(self.label_quest.keys(),range(len(self.label_quest))))
        self.id_to_label={str(v):k for k,v in self.label_to_id.items()}
        conf.num_class = len(self.label_quest)
        #建立词汇表
    def build_vocab_size(self):
        """根据训练集构建词汇表，存储"""
        all_data=''
        with open('data/train.txt','r',encoding='utf-8') as f:
            all_data=f.read()
        counter = Counter(all_data)
        count_pairs = counter.most_common(conf.vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<UNK>'] + list(words)
    
        self.word_to_id=dict(zip(words,range(len(words))))
        with open('data/vocab.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(words) + '\n')
        print('词汇表数量:%d'%(len(words))) 
        conf.vocab_size=len(words)
        #获取批量测试数据
    def get_batch_data(self):
        """生成批次数据"""
        train_data,test_data=[],[]
        with open('data/train.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t')        
                train_data.append((line[0],line[1]))
                                                       
        np.random.shuffle(train_data)
        train_x,train_y=zip(*train_data)
        train_x,train_y=self.build_vector(train_x,train_y)
             
        with open('data/test.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t')        
                test_data.append((line[0],line[1])) 
                   
        np.random.shuffle(test_data)    
        test_x,test_y=zip(*test_data)
        test_x,test_y=self.build_vector(test_x,test_y)
        print('train_x,train_y,test_x,test_y',train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        print('x数据举例',train_x[0])
        print('y数据举例',train_y[0])
        num_batches_per_epoch = int((len(train_x)-1)/conf.batch_size) + 1
        print('num_batches_per_epoch:',num_batches_per_epoch)
        for epoch in range(conf.num_epochs): 
            print('Epoch:', epoch + 1)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * conf.batch_size  
                end_index = min((batch_num + 1) * conf.batch_size, len(train_x))
                batch_x=train_x[start_index:end_index]
                batch_y=train_y[start_index:end_index]
                yield batch_x,batch_y,test_x,test_y
                           
    #向量化           
    def build_vector(self,data,label):
        """将文件转换为id表示"""  
        data_id, label_id = [], []
        for i in range(len(data)):
            if label[i] not in self.label_to_id:
                print('build_vector错误:',label[i])
                continue
            vector=[]
            for x in data[i]:
                if x in self.word_to_id:
                    vector.append(self.word_to_id[x])
                else:
                    vector.append(self.word_to_id['<UNK>'])
                    print('未知词汇',x)
            data_id.append(vector)
            label_id.append(self.label_to_id[label[i]])
    
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, conf.max_document_lenth,value=self.word_to_id['<UNK>'])
        y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示
    
        return x_pad, y_pad 
    def get_test_data(self):
        test_data=[]
        with open('data/test.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t')        
                test_data.append((line[0],line[1])) 
                   
        np.random.shuffle(test_data)    
        test_x,test_y=zip(*test_data)
        test_x,test_y=self.build_vector(test_x,test_y)        
        return test_x,test_y
if __name__=='__main__':
    data=data()
    #batchs = data.get_batch_data()

               