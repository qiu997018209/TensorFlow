#coding:utf-8
'''
Created on 2017年9月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import re
import numpy as np
import time
import random
import os
import six
import struct
import sys
import pickle
import tensorflow as tf
import data_convert_example as dc
from tensorflow.contrib import learn
from tensorflow.core.example import example_pb2
'''
本文件处理数据
entry {
    question ID:0
    question :﻿IC卡闪付是什么
    answer ID:0
    answer :银联金融IC卡——IC卡是集成电路卡（IntegratedCircuitCard）的英文简称，也称之为智能卡、芯片卡等。银联金融IC卡符合国际EMV统一标准及安全要求，具有安全、快捷、多应用的特性。“闪付”——“闪付”是银联金融IC卡的一种快捷交易方式。使用“闪付”功能时，只需将卡片靠近POS机的“闪付”感应区（即“挥卡”），即可快速完成交易。
    label ID:0
    label :什么是IC卡闪付？
    subject ID:0
    subject:银联二期_基本业务_IC卡业务_FAQ
}
'''
class data_help(object):
    def __init__(self,file_path='data/',bProcessData=True):
        self.quest_key_value={}#问题:问题ID
        self.ans_key_value = {}#答案:答案ID
        self.label_key_value = {}#标签:标签ID
        self.quest_info = {}#问题ID:[答案ID,标签ID]
        self.label_ans = {}#label:答案
        self.labels = []#所有的label
        self.quests = []#所有的quest
        self.quest_label = {}#quest:label
        self.quest_ans = {}#quest:ans
        self.label_id_count = {}#标签ID:次数
        self.test_quests = []#测试集中的所有quest
        self.train_quests = []#训练集中的所有quest
        self.subject_key_value = {}#主题ID:主题
        self.label_quests = {}#标签:quest 
        self.words_dic={}#字典,单词:id
        self.labels_ans_subject={}#label:ans,subject
        self.file_path = file_path
        self.Vocab_Size=0
        self.quests_limit=4
        #开始处理数据
        if(bProcessData==True):
            self.data_process2() 
          
    #处理数据文件
    def data_process(self):
        start = time.clock()
        #self.show_params()
        #处理文件数据     
        with open(os.path.join(self.file_path,'data.txt'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            entry_info = []
            for line in lines:
                #去除http地址
                line = self.clean_str(line)                  
                #获取一个entry的数据
                if line.startswith('entry {'):
                    entry_info = []
                    pass
                elif line.startswith('}'):
                    #处理一个entry的数据
                    quest_id,quest,ans_id,ans,label_id,label,subject_id,subject=self.process_entry(entry_info)
                    if(self.filt_words(quest)==True or self.filt_words(label)==True):
                        continue                    
                    self.ans_key_value[ans_id] = ans
                    self.label_key_value[label_id] = label
                    self.quest_info[quest_id]=[ans_id,label_id,subject_id]
                    self.label_ans[label] = ans
                    self.quest_label[quest]=label
                    self.quest_key_value[quest_id]=quest
                    self.labels_ans_subject[label]=[ans,subject]
                    self.quest_ans[quest]=ans
                    if label_id in self.label_id_count: 
                        self.label_id_count[label_id] += 1
                    else:
                        self.label_id_count[label_id] = 1
                    if(label not in self.labels):
                        self.labels.append(label)
                    if(quest not in self.quests):
                        self.quests.append(quest)
                    if label not in self.label_quests:
                        self.label_quests[label]=[quest]
                    elif (quest not in self.label_quests[label]):
                        self.label_quests[label].append(quest)
                elif line.startswith('#'):
                    #统计数据暂时处理
                    pass
                else:
                    entry_info.append(line)
        
        #self.get_max_lenth_quest()
        #生成词汇表
        #self.build_words_dic()
        #self.get_word2vec_file()
        end = time.clock()  
        print("数据预处理完成,总共耗时%s"%(end-start)) 
    def data_process2(self):
        file=open('data/all_quests.txt','w',encoding='utf-8')
        output = {}
        with open(os.path.join(self.file_path,'data.txt'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            entry_info = []
            quests=[]
            labels=[]
            for line in lines:                 
                #获取一个entry的数据
                if line.startswith('entry {'):
                    entry_info = []
                    pass
                elif line.startswith('}'):
                    #处理一个entry的数据
                    quest_id,quest,ans_id,ans,label_id,label,subject_id,subject=self.process_entry(entry_info)
                    if label_id not in output:
                        output[label_id]=[quest,label]
                    else:
                        output[label_id].append(quest)
                        
                elif line.startswith('#'):
                    #统计数据暂时处理
                    pass
                else:
                    entry_info.append(line)
        for key,value in output.items():
            file.write('####%s####\n') 
        file.close()          
    #过滤不符合要求的
    def filt_words(self,sentence):
        #纯数字不要
        if(re.search(r'^[0-9]+$',sentence)!=None):
            return True
        #少于三个字符不要
        if(len(sentence)<=3):
            return True
    def clean_str(self,string):
        string = re.sub(r"(http|www|WWW|HTTP)[:.#+=_A-Za-z0-9(),/!?\'\`]*","", string)
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
            else:
                words +=word
        #最后一个字符
        if(words != ''):
            result.append(words)     
        return result                                                            
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
                           
        #return quest_id,self.clean_str(quest),ans_id,self.clean_str(ans),label_id,self.clean_str(label),subject_id,self.clean_str(subject)
        return quest_id,quest,ans_id,ans,label_id,label,subject_id,subject 
       
    #获取词袋
    def build_word_dic(self):
        x = [' '.join(self.get_words(quest)) for quest in self.quests]
        
        ans =[' '.join(self.get_words(self.labels_ans_subject[self.quest_label[quest]][0])) for quest in self.quests]
        sub =[' '.join(self.get_words(self.labels_ans_subject[self.quest_label[quest]][1])) for quest in self.quests]        
        #向量化
        x_all=x + ans + sub
        word_dic={}
        '增加一些特殊单词'
        word_dic['<p>']=1
        word_dic['</p>']=1 
        word_dic['<s>']=1
        word_dic['</s>']=1 
        word_dic['<UNK>']=1
        word_dic['<PAD>']=1 
        word_dic['<d>']=1
        word_dic['</d>']=1                   
        for sent in x_all:
            words = sent.split()
            for word in words:
                if word in word_dic:
                   word_dic[word]+=1
                else:
                   word_dic[word]=1
        #写入文件
        with open('data/my_vocab','w',encoding='utf-8') as f:
            for word in word_dic:
                line = word+' '+str(word_dic[word])+'\n' 
                f.write(line)
        print("write words dic to file:%s,size is:%d"%('data/my_vocab',len(word_dic)))
    
    def build_binary_file(self):
        out_path_train = 'data/my_data_train'
        out_path_test = 'data/my_data_test'        
        inputs=[]
        inputs_train = []
        inputs_test = []
        for quest,label in self.quest_label.items():
            if(quest==label):
                #去除一样的
                continue
            quest = ' '.join(self.get_words(quest))
            label = ' '.join(self.get_words(label))
            article='<d> <p> <s> %s </s> </p> </d>'%(quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(label)
            line = 'article='+article+'\t'+'abstract='+abstract+'\t'+'publisher='+'qiu'+'\n'
            inputs.append(line)
        inputs = np.array(inputs)
        shuffle_indices = np.random.permutation(np.arange(len(inputs)))
        shuffled_data = inputs[shuffle_indices]
        inputs_test = shuffled_data[:100]
        inputs_train = shuffled_data[100:]
        print('sentence_to_sentence num:%d,example:%s'%(len(self.quest_label),inputs[0]))
        print('train num:%d,test num:%d'%(len(inputs_train),len(inputs_test)))
               
        writer = open(out_path_train,'wb')
        for inp in inputs_train:
            tf_example = example_pb2.Example()
            for feature in inp.strip().split('\t'):
                (k, v) = feature.split('=')
                #tensorflow默认的源码都是基于python2环境下的,此处需要将其转化为字节            
                tf_example.features.feature[k].bytes_list.value.extend([v.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()
        print('writing binary file to:%s\n'%(out_path_train))
        
        writer = open(out_path_test,'wb')
        for inp in inputs_test:
            tf_example = example_pb2.Example()
            for feature in inp.strip().split('\t'):
                (k, v) = feature.split('=')
                #tensorflow默认的源码都是基于python2环境下的,此处需要将其转化为字节            
                tf_example.features.feature[k].bytes_list.value.extend([v.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()        
        print('writing binary file to:%s\n'%(out_path_test))
        
    def write_to_pickle(self,file_name,test):
        path = os.path.join(self.file_path,file_name)
        print("Write data to pickle file:%s"%(path))
        output=open(path,'wb')
        pickle.dump(test,output)       
        output.close()
        
    def read_from_pickle(self,file_name):
        path = os.path.join(self.file_path,file_name)
        print("Read data from pickle file:%s"%(path))        
        output=open(path,'rb')
        test=pickle.load(output)
        output.close() 
        return test
                
    def binary_to_text(self,input,output):
        reader = open(input, 'rb')
        writer = open(output, 'w',encoding='utf-8')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                sys.stderr.write('Done reading\n')
                break
            str_len = struct.unpack('q', len_bytes)[0]
            tf_example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            tf_example = example_pb2.Example.FromString(tf_example_str)
            examples = []
            for key in tf_example.features.feature:
                examples.append('%s=%s' % (key, tf_example.features.feature[key].bytes_list.value[0]))
            writer.write('%s\n' % '\t'.join(examples))
        reader.close()
        writer.close()
                      
         
    def build_binary_file2(self):
        out_path_train = 'data/my_data_train'
        out_path_test = 'data/my_data_test'        
        test_list=[]
        train_list=[]
        inputs_train = {}
        inputs_test = {}
        for label,quests in self.label_quests.items():
            #quest与label之间互相组合
            if label not in quests:
                quests.append(label)
            #数量太少的不要
            if len(quests)<=3:
                continue
            index = random.randint(0,len(quests)-1)
            #用来做测试
            inputs_test[quests[index]]=label
            #互相之间用来做训练
            quests.remove(quests[index])
            for quest in quests:
                for label in quests:
                    #排除彼此相同的情况
                    if(quest==label):
                        continue
                    if(quest not in inputs_train):
                        inputs_train[quest]=[label]
                    else:
                        inputs_train[quest].append(label)
        for quest,label in inputs_test.items():
            quest = ' '.join(self.get_words(quest))
            label = ' '.join(self.get_words(label))
            article='<d> <p> <s> %s </s> </p> </d>'%(quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(label)
            #最终的ref是abstract
            line = 'article='+article+'\t'+'abstract='+article+'\t'+'publisher='+'qiu'+'\n'
            test_list.append(line)            
        for quest,labels in inputs_train.items():
            for label in labels:
                quest = ' '.join(self.get_words(quest))
                label = ' '.join(self.get_words(label))
                article='<d> <p> <s> %s </s> </p> </d>'%(quest)
                abstract='<d> <p> <s> %s </s> </p> </d>'%(label)
                line = 'article='+article+'\t'+'abstract='+abstract+'\t'+'publisher='+'qiu'+'\n'
                train_list.append(line)
        #打乱数据
        train_list=np.random.permutation(train_list)
        test_list=np.random.permutation(test_list)
        print('train example:%s'%(train_list[0]))
        print('test example:%s'%(test_list[0]))
        print('train num:%d,test num:%d'%(len(train_list),len(test_list)))
               
        writer = open(out_path_train,'wb')
        for inp in train_list:
            tf_example = example_pb2.Example()
            for feature in inp.strip().split('\t'):
                (k, v) = feature.split('=')
                #tensorflow默认的源码都是基于python2环境下的,此处需要将其转化为字节            
                tf_example.features.feature[k].bytes_list.value.extend([v.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()
        print('writing binary file to:%s\n'%(out_path_train))
        
        writer = open(out_path_test,'wb')
        self.write_to_pickle('test.pickle',test_list)
        
        for inp in test_list:
            tf_example = example_pb2.Example()
            for feature in inp.strip().split('\t'):
                (k, v) = feature.split('=')
                #tensorflow默认的源码都是基于python2环境下的,此处需要将其转化为字节            
                tf_example.features.feature[k].bytes_list.value.extend([v.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()        
        print('writing binary file to:%s\n'%(out_path_test))
         
if __name__=="__main__":
    my_data = data_help()
    #my_data.build_word_dic()
    #my_data.build_binary_file2()
    #可用来观察原始的数据格式
    #my_data.binary_to_binary('data/my_data_train','data/my_data_train_txt')
    #my_data.binary_to_text('data/my_data_test','data/my_data_test_txt')   

    