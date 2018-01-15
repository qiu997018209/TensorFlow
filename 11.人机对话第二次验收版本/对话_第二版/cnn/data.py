#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import pymysql
import math
import xlrd
import time
import numpy as np
import tensorflow.contrib.keras as kr
from collections import Counter
from collections import defaultdict

np.random.seed(0) 

class data():
    def __init__(self,args):
        #初始化，数据预处理
        self.args=args
        self.data_process()
         
    def data_process(self):
            
        self.process_mysql_data()
        
        print('开始本地处理数据')
        #建立词汇表
        self.build_vocab_size()
        
        #读取标签
        self.get_labels()
    
    #处理mysql数据
    def process_mysql_data(self):
        self.label_quest=defaultdict(set)#每个标签下对应的所有quest
        self.quest_label={}
        self.quests=set()
        self.db = pymysql.connect(host="192.168.1.245",port=3306,user="robosay",password="robosay",db="platform",charset="utf8")
        self.cursor = self.db.cursor() 
        self.cursor.execute('''
        select kqq.qa_id,kq.id,kq.content
        from kb_question kq,kb_qa_question kqq
        where kq.id = kqq.question_id 
        and kq.is_delete = '0'
        and kqq.is_delete = '0'
        and kqq.qa_id in(select id from kb_qa where parent_id in(select id from kb_scml where type='0' and user_id = %d and is_delete = '0'));
        '''%(self.args.user_id))       
        datas = self.cursor.fetchall()
        for data in datas:
            label_id=data[0]#id值
            quest=self.clean_str(data[2])
            self.quests.add(quest)
            self.label_quest[label_id].add(quest)
            self.quest_label[quest]=label_id
            if len(quest) > self.args.max_document_lenth:
                self.args.max_document_lenth=len(quest)    
        #读取标签    
    def get_labels(self):
        print('max_document_lenth',self.args.max_document_lenth)
        print('标签数量:%d'%(len(self.label_quest)))
        self.label_to_id=dict(zip(self.label_quest.keys(),range(len(self.label_quest))))
        self.id_to_label={str(v):k for k,v in self.label_to_id.items()}
        self.args.num_class = len(self.label_quest)
        #建立词汇表
    def build_vocab_size(self):
        """根据训练集构建词汇表，存储"""
        all=''
        for quest in self.quests:
            all+=quest
        #兼容数据库里没有数据的情况
        if len(all)!=0:
            self.args.vocab_size=5000#第一次没有数据,第二次加入一条数据重新训练，此时self.args.vocab_size为0
            counter = Counter(all)
            count_pairs = counter.most_common(self.args.vocab_size - 1)
            words, _ = list(zip(*count_pairs))
            # 添加一个 <PAD> 来将所有文本pad为同一长度
            words = ['<UNK>'] + list(words)
        else:
            words = ['<UNK>']
        self.word_to_id=dict(zip(words,range(len(words))))
        print('词汇表数量:%d'%(len(words))) 
        self.args.vocab_size=len(words)
        #获取批量测试数据
    def get_batch_data(self):
        """生成批次数据"""
        all=[]
        for quest,label in self.quest_label.items():
            all.append((quest,label))
        np.random.shuffle(all)         
        train_data,test_data=all,all                                            

        train_x,train_y=zip(*train_data)
        train_x,train_y=self.build_vector(train_x,train_y)
                                
        np.random.shuffle(test_data)    
        test_x,test_y=zip(*test_data)
        test_x,test_y=self.build_vector(test_x,test_y)
        print('train_x,train_y,test_x,test_y',train_x.shape,train_y.shape,test_x.shape,test_y.shape)
        print('x数据举例',train_x[0])
        print('y数据举例',train_y[0])
        num_batches_per_epoch = int((len(train_x)-1)/self.args.batch_size) + 1
        print('num_batches_per_epoch:',num_batches_per_epoch)
        for epoch in range(self.args.num_epochs): 
            print('Epoch:', epoch + 1)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.args.batch_size  
                end_index = min((batch_num + 1) * self.args.batch_size, len(train_x))
                batch_x=train_x[start_index:end_index]
                batch_y=train_y[start_index:end_index]
                one_time=time.time()
                yield batch_x,batch_y,test_x,test_y
                one_time=time.time()-one_time
                #正确时间+1
                self.args.time=round((one_time*((self.args.num_epochs-epoch)*(num_batches_per_epoch)-batch_num))/60)+1
                #等模型载入后再将self.args.rate设为1,规避时间误差
                self.args.rate=(epoch)/self.args.num_epochs
    def build_one_vector(self,raw_quest):
        quest=[self.word_to_id.get(word,self.word_to_id['<UNK>']) for word in raw_quest]
        if len(quest)>=self.args.max_document_lenth:
            quest=quest[:self.args.max_document_lenth]
        else:
            #pad_sequences补0是往前面补
            quest=(self.args.max_document_lenth-len(quest))*[self.word_to_id['<UNK>']]+quest
        print('问题:{}\n向量:{}'.format(raw_quest,quest))
        return np.array(quest)                  
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
                vector.append(self.word_to_id.get(x,self.word_to_id['<UNK>']))
            data_id.append(vector)
            label_id.append(self.label_to_id[label[i]])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.args.max_document_lenth,value=self.word_to_id['<UNK>'])
        y_pad = kr.utils.to_categorical(label_id,num_classes=len(self.label_to_id))  # 将标签转换为one-hot表示
    
        return x_pad, y_pad 
    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string 
                 