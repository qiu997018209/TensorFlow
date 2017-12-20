#coding:utf-8
'''
Created on 2017年12月4日

@author: qiujiahao

@email:997018209@qq.com

'''
import re
import os
import jieba
import time
import shutil
import math
import random
import jieba.analyse
import jieba.posseg as pseg
import numpy as np
import csv
import conf
from collections import defaultdict
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class data2(object):
    def __init__(self,user_id=None):
        self.stop_words=[]#停用词
        self.word_dic={}#词典
        self.labels=[]
        self.quest_label=defaultdict(str)#问题和标签的对应关系
        self.label_quest=defaultdict(set)#标签和问题的对应关系
        self.quest_quests=defaultdict(set)#问题和对应的扩充后的句子
        self.quest_rules=defaultdict(list)#问题和对应的句式
        self.samewords=defaultdict(list)#同义词词表
        self.train_test_data=[]#元祖(x,y)
        self.test_quests={}#记录每个测试问题的顺序
        self.word_idf=defaultdict(lambda:0)#词频的idf值
        #开始处理数据
        self.data_process(user_id)

        
    def data_process(self,user_id):
        np.random.seed(0)    
        #载入停用词词表
        with open('data/stop_words.utf8','r',encoding='utf-8') as f:
            for line in f.readlines():
                self.stop_words.append(line.strip())
  
        #建立词汇表
        self.build_word_dic()
        
        #处理本地数据
        self.process_local_data()
        
        #删除上一次的训练数据
        out_dir = os.path.abspath(os.path.join('data', "runs"))
        if(os.path.exists(out_dir)):
            #shutil.rmtree(out_dir)
            pass

        #抽取关键词信息
        self.jieba_extract_idf()
        
        #训练word2vec
        #self.init_word_embedding()
        #抽取词汇表的idf值
    def jieba_extract_idf(self):
        #没有停用词
        print('开始抽取词汇表的idf值')
        for word in self.word_dic:
            count=0
            for label in self.label_quest:
                quests=','.join(self.label_quest[label])
                if word in quests:
                    count+=1
            if count!=0:
                self.word_idf[word]=math.log(len(self.label_quest)/count)
        with open('data/words_idf.txt','w',encoding='utf-8') as f:
            for word,score in self.word_idf.items():
                f.write('%s %d\n'%(word,score))
    #清理字符串
    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string
    
    #建立词汇表,先使用旧的表，并跟新词汇表文件
    def build_word_dic(self): 
        print('开始处理词汇表')
        self.word_dic['<UNK>']=len(self.word_dic)
        with open('data/train.txt','r',encoding='utf-8') as f:
            for line in f:
                line=self.clean_str(line).split('\t')
                words=jieba.lcut(line[0])
                for word in words:
                    if word not in self.word_dic and word not in self.stop_words:
                        self.word_dic[word]=len(self.word_dic)
                if line[1] not in self.labels:
                    self.labels.append(line[1])
                if len(words) > conf.max_document_lenth:
                    conf.max_document_lenth=len(words)
        conf.vocab_size=len(self.word_dic) 
        conf.num_class=len(self.labels)          
        print('词汇表数量',conf.vocab_size)
        print('标签数量',conf.num_class)
        print('最长的问题长度',conf.max_document_lenth)
    #对每句话向量化
    def build_vector(self,sentence):
        vector=[]
        for word in jieba.lcut(sentence):
            if word in self.word_dic:
                vector.append(self.word_dic[word])
            elif word not in self.stop_words:
                vector.append(self.word_dic['<UNK>'])
                #print("未知词汇:%s,%s"%(word,sentence))
                
        if(len(vector)>=conf.max_document_lenth):
            vector=vector[0:conf.max_document_lenth]
        else:
            num = conf.max_document_lenth-len(vector)
            vector += num*[self.word_dic['<UNK>']]
        return vector         
    def one_hot(self,label):
        vectors = [0]*len(self.labels)
        index = self.labels.index(label)
        vectors[index] = 1
        return vectors 
        
    def process_local_data(self):
        #获取标签数量
        print('开始处理本地数据')
        with open('data/train.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t')
                self.quest_label[line[0]]=line[1] 
                self.label_quest[line[1]].add(line[0])       
                try:
                    #将标签转为从0~xx的大小
                    self.train_test_data.append((self.build_vector(line[0]),self.one_hot(line[1])))
                except Exception as e:
                    print(e)
                    return               
    #获取批次数据
    def get_batch_data(self):
        #打乱顺序
        print('开始处理批量数据')
        random.shuffle(self.train_test_data)
        train_data, train_target = zip(*self.train_test_data)
        num_batches_per_epoch = int((len(train_data)-1)/conf.batch_size) + 1
        self.train_test_data=[]
        with open('data/test.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=self.clean_str(line).split('\t') 
                label=line[1]
                if label not in self.labels:
                    print('过滤错误的标签:%s'%(line[0]))
                    continue             
                self.train_test_data.append((self.build_vector(line[0]),self.one_hot(label)))
                self.test_quests[len(self.test_quests)]=(line[0],label)
                
        test_data, test_target = zip(*self.train_test_data)
        train_data,train_target,test_data,test_target=np.array(train_data),np.array(train_target),np.array(test_data),np.array(test_target)
        print("训练数据维度:%s,训练标签维度:%s,测试数据维度:%s,测试标签维度:%s,标签数量:%s"%(train_data.shape,train_target.shape,test_data.shape,test_target.shape,len(self.labels)))
        
        for epoch in range(conf.num_epochs): 
            print('Epoch:', epoch + 1)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * conf.batch_size  
                end_index = min((batch_num + 1) * conf.batch_size, len(train_data))
                batch_x=train_data[start_index:end_index]
                batch_y=train_target[start_index:end_index]
                yield batch_x,batch_y,test_data,test_target
                      
        #得分
    def get_top_k(self,score,k=10):
        result=[]
        score=list(score)
        num=-1
        for s in score:
            num+=1
            result.append((s,self.labels[num]))
        result=sorted(result,key=lambda x:x[0],reverse=True)
        #print("深度学习得分前10为:",result[0:k])
        return result[0:k]

        #计算tf-idf结果
    def tfidf_calculate_score(self,quest,labels):
        label_tfidf={}
        words=jieba.lcut(quest.upper())
        for label in labels:
            value=0
            quests=','.join(self.label_quest[label])
            for word in words:
                if word in self.stop_words:
                    continue
                rule=re.compile('(%s)'%(word))
                num=len(rule.findall(quests))
                #得到每个词的tf-idf值
                value+=num/len(words) * self.word_idf[word]
            label_tfidf[label]=value

        return label_tfidf
    #对深度学习的结果进行再加工，得出最后结果
    def get_result_tfidf(self,scores): 
        right=0
        count=-1
        scores=list(scores)
        for score in scores:
            #每一次问法的对话结果
            count+=1
            #(得分,类别)
            result=self.get_top_k(score,10)
            test_quest,test_label=self.test_quests[count]
            fina_result=[]
            #深度学习分类的类别
            _,classs=zip(*result)
            #对每个类别进行再分类
            tf_scores=self.tfidf_calculate_score(test_quest,classs)
            for num,label in result:              
                fina_result.append((num+tf_scores[label],label,tf_scores[label]))
                         
            #对fina_result排序
            fina_result=sorted(fina_result,key=lambda x:x[0],reverse=True)
            if(count<3):
                print('综合得分:深度学习得分:tf-idf分类值')
                print(fina_result[0][0],fina_result[0][0]-fina_result[0][2],fina_result[0][2])
                
            if(fina_result[0][1]==test_label):
                right+=1
            #print(fina_result[0][1],test_label)
            
        print('TF-idf Acuracy:%f%%\n'%((right/len(self.test_quests))*100)) 
            
    def init_word_embedding(self):      
        #每个词的word2vec向量矩阵
        self.word_weights = np.zeros((len(self.word_dic), 128), dtype='float32')        
        print('开始训练word2vec模型')
        model = Word2Vec(
            sg=1, sentences=LineSentence('data/data.txt'),
            size=128, window=5, min_count=3, workers=4, iter=40)

        for word in self.word_dic:
            index=self.word_dic[word]
            #得到每个词的向量
            if word not in model:
                print('%s not in Word2Vec'%(word))
                random_vec = np.random.uniform(
                    -0.25, 0.25, size=(128,)).astype('float32')
                self.word_weights[index, :] = random_vec
            else:
                self.word_weights[index,:]=model[word]
        
    def init_tag_embedding(self):
        pass   
if __name__=='__main__':
    my_data=data2()

