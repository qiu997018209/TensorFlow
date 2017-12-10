#coding:utf-8
'''
Created on 2017年12月4日

@author: qiujiahao

@email:997018209@qq.com

'''
"""
使用RNN完成文本分类
"""
import re
import os
import jieba
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

class data(object):
    def __init__(self,user_id=None):
        self.stop_words=[]#停用词
        self.word_dic={}#词典
        self.labels=[]
        self.quest_label=defaultdict(str)#问题和标签的对应关系
        self.quest_quests=defaultdict(set)#问题和对应的扩充后的句子
        self.quest_rules=defaultdict(list)#问题和对应的句式
        self.samewords=defaultdict(list)#同义词词表
        self.train_test_data=[]#元祖(x,y)
        self.max_document_lenth=32
        
        #开始处理数据
        self.data_process(user_id)
        
    def data_process(self,user_id):
        #载入停用词词表
        with open('data/stop_words.utf8','r',encoding='utf-8') as f:
            for line in f.readlines():
                self.stop_words.append(line.strip())
        
        #扩充数据               
        self.extend_data()
        
        #建立词汇表
        self.build_word_dic(add=True)
        
        #处理本地数据
        self.process_local_data()
           
    #清理字符串
    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string
    
    #扩充数据
    def extend_data(self):
        #载入同义词词表
        with open('data/sameword.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line == '' or line.startswith('#'):
                    continue
                line=line.split('\t')
                for word in line[1:]:
                    self.samewords[line[0]].append(word)        
        
        with open('data/sentences.txt','r',encoding='utf-8') as f:
            key=''
            for line in f.readlines():
                line=line.strip()
                if line =='' or line.startswith('####'):
                    continue
                if line.startswith('#'):
                    key=line[1:]
                    line=key
                else:
                    quests=self.extend_algorithm_rule(line)
                    if (len(quests) != 0):
                        #set会去重
                        self.quest_quests[key].update(quests)
        
        with open('data/quest_label.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split(':')
                self.quest_label[line[1]]=line[0]
                if line[0] not in self.labels:                    
                    self.labels.append(line[0])                
                
        with open('data/data.txt','w',encoding='utf-8') as f:
            for quest,values in self.quest_quests.items():
                label = self.quest_label[quest]
                if label =='':
                    continue
                for value in values:
                    f.write(label+':'+value+'\n')
                
    def get_words(self,line):
        quests=[]
        words=''
        i=0
        bFlag=False
            
        while i<len(line):
            if line[i] == '{':
                bFlag=True
            elif line[i] == '}':
                if(bFlag==False):
                    print("错误的{}",line[i])
                bFlag=False
                if words!='':
                    if('/' not in words):
                        if(words in self.samewords):
                            quests.append(self.samewords[words])
                            words=''
                        else:
                            print('错误的同义词标签:',line) 
                    else:
                        words=words.split('/')
                        quests.append(words)
                        words=''
            elif(bFlag==False):
                quests.append(line[i])                    
            else:
                words+=line[i]
            i+=1
        return quests
                    
    def extend_algorithm_rule(self,line):
        all_quests=set()
        quest_words=self.get_words(line) 
        self.get_next_words(0,'',quest_words,all_quests)
        return all_quests      
    
    #递归调用
    def get_next_words(self,cur_index,quest,quest_words,all_quests):
        if cur_index >= len(quest_words):
            all_quests.add(quest)
            return
        if(isinstance(quest_words[cur_index], str)):
            quest+=quest_words[cur_index]
            cur_index+=1
            self.get_next_words(cur_index,quest,quest_words,all_quests)
        else:
            #不存在多级嵌套
            for word in quest_words[cur_index]:
                temp=quest+word
                self.get_next_words(cur_index+1,temp,quest_words,all_quests)    
         
    #建立词汇表,先使用旧的表，并跟新词汇表文件
    def build_word_dic(self,add=False): 
        self.word_dic['<UNK>']=0
        
        with open('data/vocab.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split(' ')
                self.word_dic[line[0].upper()]=int(line[1])
        if add==True:
            with open('data/data.txt','r',encoding='utf-8') as f:
                for line in f.readlines():
                    line=line.strip()
                    if line =='':
                        continue
                    line=self.clean_str(line.split(':')[1])
                    for word in jieba.lcut(line):
                        if word not in self.word_dic :
                            self.word_dic[word]=len(self.word_dic)
            with open('data/vocab.txt','w',encoding='utf-8') as f:
                for key,value in self.word_dic.items():
                    f.write('%s %d\n'%(key,value))
                               
        print("生成词汇表:%s,size is:%d"%('data/vocab.txt',len(self.word_dic))) 
    
    #对每句话向量化
    def build_vector(self,sentence):
        vector=[]
        for word in jieba.lcut(self.clean_str(sentence)):
            if word in self.word_dic:
                if word not in self.stop_words:
                    vector.append(self.word_dic[word])
            else:
                self.word_dic[word]=len(self.word_dic)
                print("新增词汇:%s,%s"%(word,sentence))
                
        if(len(vector)>=self.max_document_lenth):
            vector=vector[0:self.max_document_lenth]
        else:
            num = self.max_document_lenth-len(vector)
            vector += num*[self.word_dic['<UNK>']]
        return vector         
    
        
    def process_local_data(self):
        #获取标签数量
        with open('data/data.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line =='':
                    continue
                line=line.split(':')
                try:
                    #将标签转为从0~xx的大小
                    self.train_test_data.append((self.build_vector(line[1]),self.labels.index(line[0])))
                except:
                    print(line)                
            print('标签数量为:',len(self.labels))
            print('数据总量为:',len(self.train_test_data)) 
    #随机抽取四分之一来测试    
    def get_train_test(self):
        x, y = zip(*self.train_test_data)
        #释放内存
        self.train_test_data=[]
        train_data, test_data, train_target, test_target = train_test_split(x,y,random_state=1234)
        train_data,test_data,train_target,test_target=np.array(train_data),np.array(test_data),np.array(train_target),np.array(test_target)
        print('train_data',train_data[0])
        print('test_data',test_data[0])
        print('train_target',train_target[0])
        print('test_target',test_target[0])
                
        print("测试数据维度:%s,训练数据维度为:%s,训练标签维度为:%s"%(test_data.shape,train_data.shape,train_target.shape))
        return train_data, test_data, train_target, test_target 
    
    #使用指定的测试集合来测试
    def get_train_test2(self):
        train_data, train_target = zip(*self.train_test_data)
        #释放内存
        self.train_test_data=[]
        with open('data/origin_test.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split('\t') 
                label=self.quest_label[line[0]]
                if label=='':
                    print('无法找到标签:%s'%(line[0]))
                    continue
                #print(line[1],line[2],label,len(self.labels))
                self.train_test_data.append((self.build_vector(line[1]),self.labels.index(label)))
                self.train_test_data.append((self.build_vector(line[2]),self.labels.index(label)))

                    
        test_data,test_target=zip(*self.train_test_data)              
        train_data,test_data,train_target,test_target=np.array(train_data),np.array(test_data),np.array(train_target),np.array(test_target)
        print('train_data',train_data[0])
        print('test_data',test_data[0])
        print('train_target',train_target[0])
        print('test_target',test_target[0])
                
        print("测试数据维度:%s,训练数据维度为:%s,训练标签维度为:%s,标签数量为:%d"%(test_data.shape,train_data.shape,train_target.shape,len(self.labels)))
        return train_data, test_data, train_target, test_target                              