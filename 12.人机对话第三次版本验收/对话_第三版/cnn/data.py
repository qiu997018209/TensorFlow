#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import pymysql
import math
import code
import xlrd
import time
import jieba
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
from collections import Counter
from collections import defaultdict
from cnn.tool import word_parser
from log import *

class data():
    def __init__(self,args):
        np.random.seed(0)
        #初始化，数据预处理
        self.args=args        
        self.data_process()
         
    def data_process(self):
        #资源初始化
        self.data_init()
        
        #获取mysql数据    
        self.process_mysql_data()
        
        #建立词汇表
        self.build_vocab_size()
        
        #读取标签
        self.get_labels()
        
        #资源回收
        self.data_fini()
    
    #处理mysql数据
    def process_mysql_data(self):
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
        log('原始问题数量:{num}'.format(num=len(datas)))
        for data in datas:
            label_id=str(data[0])#id值
            try:
                ori_quest=self.clean_str(data[2])
            except Exception as e:
                print(e)
                log('mysql中label_id为{}的问题为空'.format(label_id),'error')
            for quest in self.extend_by_sameword(ori_quest):
                self.quests.add(quest)
                self.label_quest[label_id].add(quest)
                self.quest_label[quest]=label_id
                #基于同义词的扩展
                if len(quest) > self.args.max_document_lenth:
                    self.args.max_document_lenth=len(quest) 
        log('扩充后问题数量:{num}'.format(num=len(self.quest_label)))
        self.db.close()
        
    def data_init(self):
        if self.args.same == 1:
            #动态词典
            self.word_parser=word_parser(self)
            self.same_words=defaultdict(set)#同义词词组
            file_names=self.get_industry_name()
            for name in file_names:
                log('当前加载的同义词词表文件为:{name}'.format(name=name))
                with open(name,'r',encoding='utf-8') as f: 
                    for line in f.readlines():
                        line=line.strip().split('=')
                        if len(line)<=1:
                            continue
                        words=line[1].strip().split(";")
                        if len(words)<=1:
                            continue
                        self.same_words[line[0]].update(words)
            log('同义词词表的数量为:{num}'.format(num=len(self.same_words)))
                   
        if self.args.log==1:
            self.log=open('../data/log.txt','w',encoding='utf-8')
            log('同义词扩充log日志功能已开启')
            
        self.label_quest=defaultdict(set)#每个标签下对应的所有quest
        self.quest_label={}
        self.quests=set()         
    #获取指定行业的同义词词表                  
    def get_industry_name(self):
        file_names=[]
        file_names.append('../data/common.txt')
        if self.args.industry == 1:
            file_names.append('../data/car.txt')
        elif self.args.industry == 2:
            file_names.append('../data/bank.txt')
        return file_names
        
    def data_fini(self):
        if self.args.log==1:
            self.log.close()        
        #基于同义词的扩展
    def extend_by_sameword(self,quest):         
        if self.args.same == 0:
            return [quest]
        local_sentences=[]
        #查找同义词词表
        for word in self.word_parser.cut(quest):
            temp=word
            for _,item in self.same_words.items():
                if word in item:
                    temp=list(item)     
            local_sentences.append(temp) 
        #扩充语句
        quests=self.extend_quests(quest,local_sentences)
        if len(quests)>1 and self.args.log==1:
            for q in quests:
                self.log.write(q+'\t')
            self.log.write('\n')       
        return quests
    
    def extend_quests(self,quest,local_sentences):
        quests=set([quest])
        if len(local_sentences)==0:
            return quests
        #遍历  
        self.get_next_word(0,'',quests,local_sentences)
        return quests
        
        #遍历                         
    def get_next_word(self,index,quest,quests,local_sentences):
        if index==len(local_sentences):
            quests.add(quest)
            return              
        word=local_sentences[index] 
        if isinstance(word, list):
            for w in word:
                self.get_next_word(index+1,quest+w,quests,local_sentences) 
        else:
            self.get_next_word(index+1,quest+word,quests,local_sentences)
            
        #读取标签    
    def get_labels(self):
        log('max_document_lenth{}'.format(self.args.max_document_lenth))
        log('标签数量:%d'%(len(self.label_quest)))
        self.label_to_id=dict(zip(self.label_quest.keys(),range(len(self.label_quest))))
        self.id_to_label={str(v):k for k,v in self.label_to_id.items()}
        self.args.num_class = len(self.label_quest)
        #建立词汇表
    def build_vocab_size(self):
        """根据训练集构建词汇表，存储"""
        all=''
        for quest in self.quests:
            all+=quest
        #加载闲聊知识库里的词汇,防止很少数量的问题的时候，闲聊数据的输入向量都是一致的
        with open('../data/xianliao.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                all+=line.strip().split('\t')[0]
        
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
        log('词汇表数量:%d'%(len(words))) 
        self.args.vocab_size=len(words)
        #获取批量测试数据
    def get_batch_data(self):
        """生成批次数据"""
        quests,labels=np.array(list(self.quest_label.keys())),np.array(list(self.quest_label.values()))
        shuffle_indices=np.random.permutation(np.arange(len(self.quest_label)))  
        #生成训练数据                                                
        train_x,train_y=quests[shuffle_indices],labels[shuffle_indices]
        #再次随机取测试集合
        shuffle_indices=np.random.permutation(np.arange(len(self.quest_label)))[0:min(len(self.quest_label),5000)]
        test_x,test_y=quests[shuffle_indices],labels[shuffle_indices]
        test_x,test_y=self.build_vector(test_x,test_y)        
        log('train_x:{},train_y:{},test_x:{},test_y:{}'.format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
        num_batches_per_epoch = int((len(train_x)-1)/self.args.batch_size) + 1
        log('num_batches_per_epoch:{}'.format(num_batches_per_epoch))
        for epoch in range(self.args.num_epochs): 
            log('Epoch:{}'.format(epoch + 1))
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * self.args.batch_size  
                end_index = min((batch_num + 1) * self.args.batch_size, len(train_x))
                batch_x=train_x[start_index:end_index]
                batch_y=train_y[start_index:end_index]
                batch_x,batch_y=self.build_vector(batch_x,batch_y)
                one_time=time.time()
                yield batch_x,batch_y,test_x,test_y
                one_time=time.time()-one_time
                #正确时间+1
                self.args.time=round((one_time*((self.args.num_epochs-epoch)*(num_batches_per_epoch)-batch_num))/60)+1
                #等模型载入后再将self.args.rate设为1,规避时间误差
                self.args.rate=(epoch)/self.args.num_epochs
    def build_one_vector(self,raw_quest,bShow=True):
        quest=[self.word_to_id.get(word,self.word_to_id['<UNK>']) for word in raw_quest]
        if len(quest)>=self.args.max_document_lenth:
            quest=quest[:self.args.max_document_lenth]
        else:
            #pad_sequences补0是往前面补
            quest=(self.args.max_document_lenth-len(quest))*[self.word_to_id['<UNK>']]+quest
        if (bShow==True):
            log('问题:{}\n向量:{}'.format(raw_quest,quest))
        return np.array(quest)                  
    #向量化           
    def build_vector(self,data,label):
        """将文件转换为id表示"""  
        data_id, label_id = [], []
        for i in range(len(data)):
            if label[i] not in self.label_to_id:
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
    def get_accuracy_rate(self,correct_predictions,scores):
        correct_predictions=list(correct_predictions.astype(np.int32))
        #找到答对的索引
        indexs=np.array([index for index,i in enumerate(correct_predictions) if i == 1])
        #找到所有答对的里面的最小值
        temp=[max(s) for s in scores[indexs]]
        self.min_accuracy=min(temp)
        log('准确率最低要求:{},平均准确率:{},最高准确率:{}'.format(min(temp),np.mean(temp),max(temp)))
