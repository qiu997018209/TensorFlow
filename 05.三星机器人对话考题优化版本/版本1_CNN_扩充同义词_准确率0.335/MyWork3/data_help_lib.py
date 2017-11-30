#coding:utf-8
'''
Created on 2017年10月19日

@author: qiujiahao

@email:997018209@qq.com

'''
import time
import os
import six
import struct
import sys
import re
import xlrd
import os
import jieba
import pickle
import requests
import numpy as np
import random 
import pymysql
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.core.example import example_pb2
from gensim.models.word2vec import Word2Vec


class my_lib(object):
    def __init__(self,file_path='data/'):
        self.max_lenth_quest=0#最长问题长度
        self.file_path=file_path#文件根目录
        self.samewords={}#同义词,用于扩充句式时的
        self.quest_to_quests={}#原始quest:扩充后的quests
        self.rule_quests={}#句式:扩充后的quests
        self.quest_rules={}#问题:句式
        self.quests=[]#所有的quest
        self.labels = []#所有的label
        self.quest_label={}#quest:label
        self.quest_rules={}#quest:句式
        self.label_quests={}#quest:labels
        self.words_id={}#存储所有的字符串,用来建立索引 
        self.id_words={}#根据索引找到对应的ID
        self.x_vocab=[]#所有quest向量化后的矩阵
        self.y_vocab=[]#所有标签向量化后的矩阵 
        self.jieba=jieba#jieba分词 
        self.temp={}#临时
        self.word_dic={}#词典
        self.max_document_length=64 #一句话的最大长度
        self.stop_words=[]#停用词
    #清理字符串
    def clean_str(self,string):
        #处理数字
        #string = re.sub(r"[0-9]+",'<数字>', string)
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        #去除停用词
        new_string=''
        for word in self.jieba.cut(string):
            if(word not in self.stop_words):
                new_string += word
        string=new_string
        if(string=="\ufeffIC卡闪付是什么"):
            string="IC卡闪付是什么" 
        return string
    #是否是中文
    def isChinese(self,word):
        if u'\u4e00' <= word <= u'\u9fff':
            return True
        return False
    #得到一行数据的单词列表
    def get_words(self,sentence):
        words  = ''
        result = []
        for word in sentence:
            #连续的字母和数字视为一个词
            if(self.isChinese(word)):
                if(words != ''):
                    result.append(words)
                    words = '' 
                result.append(word)
            elif word in '@{<“"[(（': 
                if(words != ''):
                    result.append(words)
                    words = '' 
                result.append(word) 
                words=''
            elif word in '}，,.。 >]？?？，）)、/、*"”':
                if(words != ''): 
                    result.append(words)
                    words = ''
                result.append(word) 
                words=''                                             
            else:
                words +=word
        #最后一个字符
        if(words != ''):
            result.append(words)     
        return result
    
    def init_same_word(self,user_id):
        if user_id!=None:
            #获取在线填充的同义词词表
            self.cursor.execute('''
            select  name,ext from kb_dictionary where is_delete=0 and belong=2 and catagory=2 and type=4;
            ''')
            datas = self.cursor.fetchall()
            for data in datas:
                #为了兼容系统，此处制造一个标签
                label='{'+str(datas.index(data))+'}'
                self.samewords[label]=self.clean_str(data[1]).split(';')
                self.samewords[label].append(self.clean_str(data[0]))
        '''
        #处理本地积累的数据        
        with open(os.path.join(self.file_path,'private_sameword.txt'),'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#') or line=='':
                    continue
                datas=line.split('=')
                self.samewords[datas[0]]=self.clean_str(datas[1]).split(';')
        '''        
        with open(os.path.join(self.file_path,'jieba.txt'),'w',encoding='utf-8') as f:
            for key,words in self.samewords.items():
                for word in words: 
                    f.write("%s %d\n"%(word,1000))
                    
        self.jieba.load_userdict(os.path.join(self.file_path,'jieba.txt'))
        '''
        #处理哈工大同义词词表       
        with open(os.path.join(self.file_path,'common_sameword.txt'),'r',encoding='utf-8') as f:
            for line in f.readlines():
                if line.startswith('#') or line=='' or '@' in line or '=' not in line:
                    continue            
                datas=line.split('=')
                words = set(datas[1].upper().strip().split(' '))
                words.add(datas[0].upper().strip())
                self.samewords2.append(words)
        '''                   
    def Init(self,user_id):
        if user_id!=None:
            self.db = pymysql.connect(host="192.168.1.245",port=3306,user="robosay",password="robosay",db="platform",charset="utf8")
            self.cursor = self.db.cursor()
        #处理同义词
        self.init_same_word(user_id)
        
        #载入停用词词表
        with open(os.path.join(self.file_path,'stop_words.utf8'),'r',encoding='utf-8') as f:
            for line in f.readlines():
                self.stop_words.append(line.strip())
        
    def Fini(self,user_id):
        if user_id!=None:
            self.cursor.close()
            self.db.close()
    #处理单个entry数据集   
    def process_entry(self,entry):
        quest_id = 0
        ans_id = 0
        label_id = 0
        subject_id = 0
        quest = ''
        ans = ''
        label = ''
        subject = ''
        for line in entry:
            if line.startswith('    question ID'):
                quest_id = int(line.split(':')[-1].strip('\n '))
            elif line.startswith('    question :'):  
                quest = line.split(':')[-1].strip('\n ').replace(' ','').upper()
            elif line.startswith('    answer ID:'): 
                ans_id =  int(line.split(':')[-1].strip('\n '))
            elif line.startswith('    answer :'): 
                ans = line.split(':')[-1].strip('\n ').replace(' ','').upper()       
            elif line.startswith('    label ID:'): 
                label_id =  int(line.split(':')[-1].strip('\n '))           
            elif line.startswith('    label :'): 
                label = line.split(':')[-1].strip('\n ').replace(' ','').upper()
            elif line.startswith('    subject ID:'):
                subject_id = line.split(':')[-1].strip('\n ').replace(' ','').upper()
            elif line.startswith('    subject:'):
                subject = line.split(':')[-1].strip('\n ').replace(' ','').upper()            
            else:
                print('parse entry error:',line)  
        return quest_id,quest,ans_id,ans,label_id,label,subject_id,subject  
       
    #扩充句式
    def sentences_algorithm(self,quest):
        all_quests=[]
        quest_words=self.get_sentence_words(quest)
        self.get_next_words(0,'',quest_words,all_quests)
        if quest in self.quest_to_quests: 
            print("重复扩充相同问题句式:%s"%(quest))
        return all_quests 
    #递归调用
    def get_next_words(self,cur_index,quest,quest_words,all_quests):
        if cur_index >= len(quest_words):
            all_quests.append(self.words_to_id(quest))
            return
        
        if(isinstance(quest_words[cur_index], str)):
            quest+=quest_words[cur_index]
            cur_index+=1
            self.get_next_words(cur_index,quest,quest_words,all_quests)
        else:
            #不存在多级嵌套
            #quest_words[cur_index]是list
            for word in quest_words[cur_index]:
                temp=quest+word
                self.get_next_words(cur_index+1,temp,quest_words,all_quests)            
    #获取sentence里的每一个word基础单元
    def get_sentence_words(self,quest):
        words=[]
        i=0
        label_num=0        
        while(i<len(quest)):
            word=''
            if(quest[i]=='{'):
                i+=1
                while(quest[i]!='}'):
                    word+=quest[i]
                    i+=1
                    if(i>=len(quest)):
                        print("缺少}:%s"%(quest))
                        return                    
                if(i<len(quest) and quest[i]!='}'):
                    print("缺少后括号:%s"%(quest))
                if(i<len(quest) and "@" in word):
                    print("%s:{}中不支持同义词标签"%(quest))
                words.append(self.samewords["{"+word+"}"])
                #到"}"的下一个字符
                i+=1              
            elif(self.isChinese(quest[i])==False):                                 
                while(i<len(quest) and self.isChinese(quest[i])==False):
                    word+=quest[i]
                    i+=1
                    if(i>=len(quest) or quest[i] in "@{<"):
                        break
                words.append(word)
            elif(self.isChinese(quest[i])):
                words.append(quest[i])
                i+=1 
            else:
                print("get_sentence_words,Error:%s"%(quest[i]))
                i+=1
        return words
     
    def build_vector(self,sentence):
        vector=[]
        words=self.get_words(self.id_to_words(sentence))
        for word in words:
            if word in self.word_dic:
                vector.append(self.word_dic[word])
            else:
                vector.append(self.word_dic['<UNK>'])
                
        if(len(vector)>=self.max_document_length):
            vector=vector[0:self.max_document_length]
        else:
            num = self.max_document_length-len(vector)
            vector += num*[self.word_dic['<UNK>']]
        return vector             
    #将quest转化为ID值
    def words_to_id(self,words):     
        if(words not in self.words_id):
            num=len(self.words_id)
            self.words_id[words]=num
            self.id_words[num]=words      
        return self.words_id[words]
    #将ID值转化为quest
    def id_to_words(self,id):
        return self.id_words[id]
                
        #获取词袋
    def build_word_dic(self):
        '增加一些特殊单词'
        self.word_dic['<p>']=1
        self.word_dic['</p>']=2 
        self.word_dic['<s>']=3
        self.word_dic['</s>']=4 
        self.word_dic['<UNK>']=5
        self.word_dic['<PAD>']=6 
        self.word_dic['<d>']=7
        self.word_dic['</d>']=8
        
        for quest in self.quests:
            words=self.get_words(self.id_to_words(quest))
            for word in words:
                if word in self.word_dic:
                    continue
                else:
                    self.word_dic[word]=len(self.word_dic)        
        #写入文件
        path=os.path.join(self.file_path,'my_vocab')
        with open(path,'w',encoding='utf-8') as f:
            for word in self.word_dic:
                line = word+' '+str(self.word_dic[word])+'\n' 
                f.write(line)
        print("生成词汇表:%s,size is:%d"%('data/my_vocab',len(self.word_dic)))          
                
    def process_local_data(self):
        output = {}
        with open(os.path.join(self.file_path,'data.txt'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            entry_info = []
            for line in lines:                 
                #获取一个entry的数据
                if line.startswith('entry {'):
                    entry_info = []
                    pass
                elif line.startswith('}'):
                    #处理一个entry的数据
                    quest_id,quest,ans_id,ans,label_id,label,subject_id,subject=self.process_entry(entry_info)
                    quest=self.clean_str(quest)
                    label=self.clean_str(label)                    
                    if quest is None or label is None:
                        continue 
                    #建立索引
                    quest=self.words_to_id(quest)
                    label=self.words_to_id(label)                             
                    if quest not in self.quests:
                        self.quests.append(quest)                        
                    if label not in self.labels:
                        self.labels.append(label)
                    self.quest_label[quest]=label
                    if label not in self.label_quests:
                        self.label_quests[label]=[quest]
                    else:
                        self.label_quests[label]+=[quest]  
                elif line.startswith('#'):
                    #统计数据暂时处理
                    pass
                else:
                    entry_info.append(line)                    
        with open(os.path.join(self.file_path,'all_quests.txt') ,'w',encoding='utf-8') as f:
            for quest in self.quests:
                f.write('%s\t%s\n'%(self.id_to_words(quest),self.id_to_words(self.quest_label[quest]))) 
                
    def process_local_data2(self):
        with open(os.path.join(self.file_path,'data2.txt'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                quest,label=line.strip().split('\t')
                quest=self.clean_str(quest)
                label=self.clean_str(label)                    
                if quest is None or label is None:
                    continue 
                #建立索引
                quest=self.words_to_id(quest)
                label=self.words_to_id(label)                             
                if quest not in self.quests:
                    self.quests.append(quest)                        
                if label not in self.labels:
                    self.labels.append(label)
                self.quest_label[quest]=label
                if label not in self.label_quests:
                    self.label_quests[label]=[quest]
                else:
                    self.label_quests[label]+=[quest] 
        #处理excel表中的数据
    def process_local_data3(self):
        workbook = xlrd.open_workbook('data/三星机器人考题.xlsx')
        '''
        booksheet = workbook.sheet_by_name('ask')
        for row in range(booksheet.nrows):
            quest_id=0
            label_id=0
            if row == 0:
                continue
            for col in range(booksheet.ncols):
                cell=booksheet.cell(row,col)
                val = self.clean_str(str(cell.value))
                if(col==1):
                    quest_id=self.words_to_id(val)
                    self.quests.append(quest_id)
                elif(col==2):
                    label_id=self.words_to_id(val)
                    self.labels.append(label_id)
                    self.quest_label[quest_id]=label_id
                else:
                    continue
        
        booksheet = workbook.sheet_by_name('faq')
        for row in range(booksheet.nrows):
            quest_id=0
            label_id=0
            if row == 0:
                continue
            for col in range(booksheet.ncols):
                cell=booksheet.cell(row,col)
                if(col==2):
                    val = self.clean_str(str(cell.value))
                    quest_id = self.words_to_id(val)
                elif(col==3):
                    #地址，不处理
                    val = cell.value
                    label_id=self.words_to_id(val)
                    self.quest_label[quest_id]=label_id
                else:
                    continue            
        '''              
        booksheet = workbook.sheet_by_name('Test')
        quests_test={} 
        quest=''
        for row in range(booksheet.nrows):
            if(row==0 or row==1 or row==2):
                continue
            for col in range(booksheet.ncols): 
                if(col!=1 and col!=2):
                    continue
                cell=booksheet.cell(row,col)
                val = cell.value
                if(val==''):
                    continue
                val=self.clean_str(val)
                if(col==1):
                    quest=val
                    self.temp[quest]=[quest]
                elif (col==2):
                    self.temp[quest].append(val)
               
    #从mysql里拉取数据
    def process_remote_data(self,user_id,bExtend=False):
        self.cursor.execute('''
        select kqq.qa_id,kq.id,kq.content
        from kb_question kq,kb_qa_question kqq
        where kq.id = kqq.question_id 
        and kq.is_delete = '0'
        and kqq.is_delete = '0'
        and kqq.qa_id in(select id from kb_qa where parent_id in(select id from kb_scml where type='0' and user_id = %d and is_delete = '0'));
        '''%(user_id))
        datas = self.cursor.fetchall()
        f=open('data/extendends.txt','w',encoding='utf-8')
        for data in datas:
            print("扩充数据进度:%d/%d"%(datas.index(data),len(datas)))
            label_id=data[0]#id值
            quest=self.clean_str(data[2])
            if(quest is None):
                print("问题不符合要求:%s"%(data[2]))
                continue
            quest_id = self.words_to_id(quest)
            if(quest_id in self.quests):
                print("重复的问题:%s"%(data[2]))
                continue            
            self.labels.append(label_id)
            self.quests.append(quest_id)
            self.quest_label[quest_id]=label_id 
            
            if(bExtend==True):                
                #用同义词扩充句子
                quests_ex=self.extend_sentences_by_sameword(quest)
                f.write("%s\n"%(quest))
                if quests_ex is None:
                    continue 
                if(len(quests_ex)>500):
                    quests_ex=random.sample(quests_ex, 500)                 
                for quest_id in quests_ex:
                    self.quests.append(quest_id)
                    self.quest_label[quest_id]=label_id 
                    f.write("%s\n"%(self.id_to_words(quest_id)))        
        f.close()                
        return             
    #根据同义词来扩充句子
    def extend_sentences_by_sameword(self,quest):
        #限制同义词数量
        num=0
        origin=quest
        jie_words=list(self.jieba.cut(quest))
        for key,words in self.samewords.items():
            for w in words:
                #分完词后的词汇如果有同义词词表中的同义词，那么将原句子中的同义词替换为标签
                if w in jie_words and w in quest:
                    quest=quest.replace(w,key)
                    num+=1
                    break
            #最多只能出现2个同义词标签
            if(num==2):
                break
        all_quests=self.sentences_algorithm(quest)
        return all_quests
        
    def Reload_vocab(self,quest):
        quests=[]
        quest=self.words_to_id(self.clean_str(quest))
        quests.append(self.build_vector(quest))         
        return np.array(quests)         
    def one_hot(self,label):
        vectors = [0]*len(self.labels)
        index = self.labels.index(label)
        vectors[index] = 1
        return vectors 
     
    def data_process(self,user_id):
        if(user_id==None):
            self.process_local_data3()
        else:
            self.process_remote_data(user_id)
        #建立词汇表
        self.build_word_dic()
        
        self.process_local_data3()
                            
    def batch_iter(self, batch_size, num_epochs,rate=0.01):
        """
        Generates a batch iterator for a dataset.
        """
        #打乱顺序
        quests=np.random.permutation(self.quests)
        test_x=quests[0:int(len(quests)*rate)]
        test_y=[self.quest_label[test] for test in test_x]
        dev_x = np.array([self.build_vector(quest) for quest in test_x])
        dev_y = np.array([self.one_hot(label) for label in test_y])
        train_x=quests[int(len(quests)*rate):]
        train_y=[self.quest_label[test] for test in train_x]
        num_batches_per_epoch = int((len(train_x)-1)/batch_size) + 1
        
        #先写入一次，防止数据量大的时候，进度没有更新
        with open(os.path.join(self.file_path,'process_rate.txt'),'w',encoding='utf-8') as f:
            f.write("%d:%d:%d\n"%(0,num_epochs-1,6000))
        
        for epoch in range(num_epochs):
            begin=time.clock()
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size  
                end_index = min((batch_num + 1) * batch_size, len(quests))
                batch_x=np.array([self.build_vector(quest) for quest in train_x[start_index:end_index]])
                batch_y=np.array([self.one_hot(label) for label in train_y[start_index:end_index]])
                yield batch_x,batch_y,dev_x,dev_y
                
            with open(os.path.join(self.file_path,'process_rate.txt'),'w',encoding='utf-8') as f:
                f.write("%d:%d:%d\n"%(epoch,num_epochs-1,time.clock()-begin)) 
                
    @classmethod                
    def write_to_pickle(cls,file_name,x):
        print("Write data to pickle file:%s"%(file_name))
        output=open(file_name,'wb')
        pickle.dump(x,output)       
        output.close() 
    @classmethod  
    def read_from_pickle(cls,file_name):
        print("Read data from pickle file:%s"%(file_name))        
        output=open(file_name,'rb')
        x=pickle.load(output)
        output.close() 
        return x
    def test(self,sess,cnn):
        right=0
        total=0
        for key,values in self.temp.items():
            label=0
            result=[]
            for value in values:
                value_id=self.words_to_id(value)
                x_batch=np.array([self.build_vector(value_id)])
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.dropout_keep_prob: 1.0
                }            
                predict = list(sess.run(cnn.predictions, feed_dict))[0]
                result.append(predict)
            if(result[0]==result[1]):
                right+=1
            if(result[0]==result[2]):
                right+=1            
            total+=2
            
        print("%d/%d:%f"%(right,total,right/total))
                
                  