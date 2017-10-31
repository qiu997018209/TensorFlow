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
import os
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
        self.samewords={}#同义词
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
    #清理字符串
    def clean_str(self,string):
        #长度太短的过滤掉
        if(len(string)<=4):
            return None
        #处理数字
        string = re.sub(r"[0-9]+",'<数字>', string)
        if(string=="\ufeffIC卡闪付是什么"):
            string="IC卡闪付是什么" 
        return string
    #是否是中文
    def isChinese(self,word):
        if u'\u4e00' <= word <= u'\u9fff':
            return True
        return False
    #过滤不符合要求的
    def filt_words(self,sentence):
        #纯数字不要
        if(re.search(r'^[0-9]+$',sentence)!=None):
            return True
        #少于三个字符不要
        if(len(sentence)<=3):
            return True
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
        return quest_id,quest,ans_id,ans,label_id,label,subject_id,subject  
 
    def send_post(self,quest,address='http://127.0.0.1:8000/deep_chat'):
        data={"quest":quest}
        r = requests.post(address, data)
        
        print (r.status_code)
        print (r.headers['content-type'])
        r.encoding = 'utf-8'
        print (r.text)

    #处理同义词词表
    def process_same_words(self):
        #获取同义词
        path = os.path.join(self.file_path,'01_sameword.txt')
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line=line.strip()
                words = line.split('\t')
                if(words[0] not in self.samewords):
                    self.samewords[words[0]]=words[1:]
                else:
                    print("同义词标签%s定义重复"%(words[0]))
        print("一共存在%d个同义词标签"%(len(self.samewords)))
    #处理句式
    def process_sentences(self):
        path = os.path.join(self.file_path,"01_sentence.txt")
        recent_quest=''
        quest_nums=0
        line_num=0
        with open(path,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                line=line.strip().replace(' ','')
                line_num+=1
                #空行
                if(len(line)==0):
                    continue
                #print("解析句式文件的的行号为:%d/%d"%(line_num,len(lines)))
                if line.startswith('####'):
                    continue
                elif line.startswith('#'):
                    recent_quest=self.words_to_id(line[1:])
                    if recent_quest not in self.quests:
                        print("%s:错误的问题，请确保与原始quest一致"%(self.id_to_words(recent_quest)))
                        return                    
                    if recent_quest not in self.quest_rules:
                        self.quest_rules[recent_quest]=[]
                        quest_nums+=1
                    else:
                        print('重复扩充quest:%s'%(self.id_to_words(recent_quest)))
                        return
                else:
                    #扩充句式
                    all_quests=self.sentences_algorithm(line)
                    if(len(all_quests)==0):
                        print("%s:句式扩充失败"%(line))
                        return
                    if(recent_quest==''or recent_quest not in self.quest_rules):
                        print("%s格式错误,请先使用#注明原始问题，然后开始扩充句式"%(line)) 
                        return                   
                    line = self.words_to_id(line)
                    self.quest_rules[recent_quest].append(line)
                    if line not in self.rule_quests:
                        self.rule_quests[line]=all_quests
                    else:
                        print("重复的句式:%s"%(self.id_to_words(line)))
                        return
                        
        print("总共%d个原始问题"%(quest_nums))
        self.write_all_quests()
        self.combina_quests()
        
    def combina_quests(self):
        path=os.path.join(self.file_path,"need_extend_by_label.txt")
        f_e=open(path,'w',encoding='utf-8')
        path=os.path.join(self.file_path,"need_cut_by_by_label.txt")
        f_c=open(path,'w',encoding='utf-8')
        f_e.write("#以下标签的问题数量太少，需要进一步扩充\n")
        f_c.write("#以下标签的问题数量太多，需要删除多余的\n")
        label_num=0
        print('开始合并标签问题集合')        
        for label,quests in self.label_quests.items():
            for quest in quests.copy():                    
                if quest not in self.quest_rules:
                    #print('问题%s没有扩充句式'%(quest))
                    continue
                #将所有扩充后的句子与原来的句子进行合并
                for rule in self.quest_rules[quest]:
                    #此处去重
                    for quest in self.rule_quests[rule]:
                        if(quest not in self.label_quests[label]):
                            self.label_quests[label].append(quest) 
            label_num+=len(self.label_quests[label])
            #print("当前label%s的quest数量:%d"%(self.id_to_words(label),len(self.label_quests[label])))
            if(len(self.label_quests[label])<=20):
                f_e.write("%s:%d\n"%(self.id_to_words(label),len(self.label_quests[label])))
            if(len(self.label_quests[label])>=1000):
                f_c.write("%s:%d\n"%(self.id_to_words(label),len(self.label_quests[label])))           
        print("扩充后所有标签总的quest数量为:%d"%(label_num))     
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
            for word in quest_words[cur_index]:
                temp=quest+word
                self.get_next_words(cur_index+1,temp,quest_words,all_quests)   
    #将扩充后的全部写到文件里
    def write_all_quests(self):
        path=os.path.join(self.file_path,"need_cut_by_by_quest.txt")
        f_c=open(path,'w',encoding='utf-8')
        path=os.path.join(self.file_path,"need_extend_by_by_quest.txt")
        f_e=open(path,'w',encoding='utf-8')
        f_c.write("#以下问题的数量太多，需要删减\n")
        f_e.write("#以下问题的数量太少，需要进一步扩充\n")        
        print('开始将扩充后的句式写入到文件中')
        path = os.path.join(self.file_path,"extend_quests.txt")
        with open(path,'w',encoding='utf-8') as f: 
            for quest,rules in self.quest_rules.items():
                num=0 
                for rule in rules:
                    for ex_quest in self.rule_quests[rule]:
                        num+=1
                        f.write("%d\t%s\t%s\t%s\n"%(num,self.id_to_words(ex_quest),self.id_to_words(rule),self.id_to_words(quest)))
                if(num>=300):
                    f_c.write("%d\t%s\n"%(num,self.id_to_words(quest)))
                elif(num<=10):
                    f_e.write("%d\t%s\n"%(num,self.id_to_words(quest)))
        f_c.close()
        f_e.close()           
    #获取sentence里的每一个word基础单元
    def get_sentence_words(self,quest):
        words=[]
        bEntry=False
        i=0
        while(i<len(quest)):
            word=''
            if quest[i] == '@':
                word+=quest[i]
                i+=1
                while(i<len(quest) and self.isChinese(quest[i])==False):
                    if(quest[i] in ',{，。<?？@）*、“'):
                        break
                    word+=quest[i]
                    i+=1                   
                if word not in self.samewords:
                    print("错误的同义词标签:%s,%s"%(word,quest))
                words.append(self.samewords[word])
            elif(quest[i]=='{'):
                i+=1
                while(i<len(quest) and quest[i]!='}'):
                    word+=quest[i]
                    i+=1
                if(i<len(quest) and quest[i]!='}'):
                    print("缺少后括号:%s"%(quest))
                if(i<len(quest) and "@" in word):
                    print("%s:{}中不支持同义词标签"%(quest))
                words.append(word.split('/'))
                #到"}"的下一个字符
                i+=1
            #特殊标签
            elif(quest[i]=='<'):
                word+=quest[i]
                i+=1
                while(i<len(quest) and quest[i]!='>'):
                    word+=quest[i]
                    i+=1                  
                word+=quest[i]
                words.append(word)
                i+=1                
            elif(self.isChinese(quest[i])==False):
                if(quest[i] in "}>/"):
                    print("%s:格式错误"%(quest))                                  
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
    #将同一个句式之间的quest作为label
    def write_train_test3(self):
        print("开始生成二进制的训练与测试数据文件") 
        test_list=[]
        train_list=[]
        inputs_train = {}
        inputs_test = {}
        num_index=0
        #先删除文件
        test_path = os.path.join(self.file_path,"my_data_test")
        train_path = os.path.join(self.file_path,"my_data_train")
        if(os.path.exists(test_path) and os.path.exists(train_path)):
            print("删除上一次训练的测试和训练数据")
            os.remove(test_path)
            os.remove(train_path)     
                 
        for rule,quest in self.rule_quests.items():
            quests=quest.copy()          
            #num_index += 1
            #print("当前生成训练与测试文件进度:%d/%d"%(num_index,len(self.label_quests))) 
            if(len(quests)<2):
                continue
            if(len(quests)>100):
                num=random.randint(0,len(quests)-1)
                inputs_test[quests[num]]=quests[num] 
                quests.remove(quests[num])      
            for quest in quests:
                label=quests[random.randint(0,len(quests)-1)]
                if(quest==label):
                    label=quests[random.randint(0,len(quests)-1)]
                inputs_train[quest]=label
        
        test_index=0                                                                                                                                    
        for quest,label in inputs_test.items():
            test_index+=1
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            #最终的ref是abstract
            line = 'article='+article+'\t'+'abstract='+article+'\t'+'publisher='+'qiu'+'\n'
            test_list.append(line)                          
        self.build_binary_file(test_list,test_path)
        total_index=0           
        for quest,label in inputs_train.items():         
            total_index+=1              
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            line = 'article='+article+'\t'+'abstract='+abstract+'\t'+'publisher='+'qiu'+'\n'
            train_list.append(line)
        self.build_binary_file(train_list,train_path)
        print("测试数据数量为:%d"%(test_index)) 
        print("训练数据数量为:%d"%(total_index))   
    
    #quest采用同一个标签里的quest作为标签
    def write_train_test2(self):
        print("开始生成二进制的训练与测试数据文件") 
        test_list=[]
        train_list=[]
        inputs_train = {}
        inputs_test = {}
        num_index=0
        #先删除文件
        test_path = os.path.join(self.file_path,"my_data_test")
        train_path = os.path.join(self.file_path,"my_data_train")
        if(os.path.exists(test_path) and os.path.exists(train_path)):
            print("删除上一次训练的测试和训练数据")
            os.remove(test_path)
            os.remove(train_path)     
                 
        for label,quest in self.label_quests.items():
            quests=quest.copy()
            if label not in quests:
                quests.append(label)            
            #num_index += 1
            #print("当前生成训练与测试文件进度:%d/%d"%(num_index,len(self.label_quests))) 
            if(len(quests)<=2):
                continue
            if(len(quests)>100):
                num=random.randint(0,len(quests)-1)
                inputs_test[quests[num]]=quests[num] 
                quests.remove(quests[num])      
            for quest in quests:
                label=quests[random.randint(0,len(quests)-1)]
                if(quest==label):
                    label=quests[random.randint(0,len(quests)-1)]
                inputs_train[quest]=label
  
        test_index=0                                                                                                                                    
        for quest,label in inputs_test.items():
            test_index+=1
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            #最终的ref是abstract
            line = 'article='+article+'\t'+'abstract='+article+'\t'+'publisher='+'qiu'+'\n'
            test_list.append(line)                          
        self.build_binary_file(test_list,test_path)
        total_index=0           
        for quest,label in inputs_train.items():         
            total_index+=1              
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            line = 'article='+article+'\t'+'abstract='+abstract+'\t'+'publisher='+'qiu'+'\n'
            train_list.append(line)
        self.build_binary_file(train_list,train_path)
        print("测试数据数量为:%d"%(test_index)) 
        print("训练数据数量为:%d"%(total_index))
 
    #相同quest之间两两组合，生成二进制文件                
    def write_train_test(self):
        print("开始生成二进制的训练与测试数据文件") 
        test_list=[]
        train_list=[]
        inputs_train = {}
        inputs_test = {}
        num_index=0
        #先删除文件
        test_path = os.path.join(self.file_path,"my_data_test")
        train_path = os.path.join(self.file_path,"my_data_train")
        if(os.path.exists(test_path) and os.path.exists(train_path)):
            print("删除上一次训练的测试和训练数据")
            os.remove(test_path)
            os.remove(train_path)     
                 
        for label,quest in self.label_quests.items():
            quests=quest.copy()
            num_index += 1
            #print("当前生成训练与测试文件进度:%d/%d"%(num_index,len(self.label_quests)))            
            #quest与label之间互相组合
            if label not in quests:
                quests.append(label)
            #数量太少的不要
            if len(quests)<=100:
                continue
            index = random.randint(0,len(quests)-1)
            #用来做测试
            inputs_test[quests[index]]=label
            #互相之间用来做训练
            quests.remove(quests[index])
            quests=np.random.permutation(quests)
            #高于500的随机组合500*len(quests)次
            if(len(quests)>500):
                for _ in range(500):
                    for quest in quests:
                        label = quests[random.randint(0,len(quests)-1)]
                        if(quest not in inputs_train):
                            inputs_train[quest]=[label]
                        else:
                            inputs_train[quest].append(label)
            else:        
                #低于500的互相之间用来做训练    
                for quest in quests:
                    for label in quests:
                        #排除彼此相同的情况
                        if(quest==label):
                            continue
                        if(quest not in inputs_train):
                            inputs_train[quest]=[label]
                        else:
                            inputs_train[quest].append(label)
        test_index=0                                                                                                                                    
        for quest,label in inputs_test.items():
            test_index+=1
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            #最终的ref是abstract
            line = 'article='+article+'\t'+'abstract='+article+'\t'+'publisher='+'qiu'+'\n'
            test_list.append(line)                          
        #释放部分内存
        self.build_binary_file(test_list,test_path)
        num_index=0
        total_index=0
        self.label_quests={}             
        for quest,labels in inputs_train.items():
            num_index+=1
            print('生成训练数据:%d/%d'%(num_index,len(inputs_train)))              
            for label in labels:
                total_index+=1 
                #print("quest:%s,label:%s"%(self.id_to_words(quest),self.id_to_words(label)))             
                out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
                out_label = ' '.join(self.get_words(self.id_to_words(label)))
                article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
                abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
                line = 'article='+article+'\t'+'abstract='+abstract+'\t'+'publisher='+'qiu'+'\n'
                train_list.append(line)
                if(len(train_list)>=500000):
                    self.build_binary_file(train_list,train_path)
                    train_list=[]
        if(len(train_list)!=0):
            self.build_binary_file(train_list,train_path)
        print("测试数据数量为:%d"%(test_index)) 
        print("训练数据数量为:%d"%(total_index))
                                  
    def build_binary_file(self,data_list,file_path):     
        #打乱数据
        data_list=np.random.permutation(data_list)               
        writer = open(file_path,'ab')
        for inp in data_list:
            tf_example = example_pb2.Example()
            for feature in inp.strip().split('\t'):
                #print(feature)
                (k, v) = feature.split('=')
                #tensorflow默认的源码都是基于python2环境下的,此处需要将其转化为字节            
                tf_example.features.feature[k].bytes_list.value.extend([v.encode('utf-8')])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        writer.close()
        print('writing binary file to:%s\n'%(file_path))
             
    #将quest转化为ID值
    def words_to_id(self,words,id=None):
        #指定ID号时
        if id is not None:
            if(id in self.id_words):
                print("错误:重复的ID号或者同一个问题存在不同的ID号")
            self.id_words[id]=words
            self.words_id[words]=id
            return id 
        #没有指定ID号时按照顺序建立      
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
        x = [' '.join(self.get_words(quest)) for quest in self.words_id]
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
        for sent in x:
            words = sent.split()
            for word in words:
                if word in word_dic:
                   word_dic[word]+=1
                else:
                   word_dic[word]=1
        #写入文件
        path=os.path.join(self.file_path,'my_vocab')
        with open(path,'w',encoding='utf-8') as f:
            for word in word_dic:
                line = word+' '+str(word_dic[word])+'\n' 
                f.write(line)
        print("生成词汇表:%s,size is:%d"%('data/my_vocab',len(word_dic)))          
                
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
    #从mysql里拉取数据
    def process_remote_data(self,user_id):
        self.db = pymysql.connect(host="192.168.1.62",port=3306,user="root",password="chen123",db="platformdev",charset="utf8")
        self.cursor = self.db.cursor()
        self.cursor.execute('''
        select kqq.qa_id,kq.id,kq.content
        from kb_question kq,kb_qa_question kqq
        where kq.id = kqq.question_id 
        and kq.is_delete = '0'
        and kqq.is_delete = '0'
        and kqq.qa_id in(select id from kb_qa where parent_id in(select id from kb_scml where type='0' and user_id = %d and is_delete = '0'));
        '''%(user_id))
        datas = self.cursor.fetchall()
        for data in datas:
            label_id=data[0]#id值
            quest_id=data[1]#id值
            quest=self.clean_str(data[2])#id值
            if(quest is None or len(quest)<=2):
                print("问题不符合要求，已过滤:%s"%(data[2]))
                continue
            self.labels.append(label_id)
            self.quests.append(quest_id)
            self.words_to_id(quest,quest_id)
        return             
                
    def build_y_vocab(self):    
        labels = [self.one_hot(label) for label in self.labels]
        print('获取语料与对应的标签完成,问题:%d 标签:%d'%(len(self.quests),len(self.labels)))    
        self.y_vocab=np.array(labels)
    #对quest进行向量化
    def build_x_vocab(self):
        #以空格来切分每一个词
        max_document_length = max([len(self.id_to_words(quest)) for quest in self.quests])
        x = [' '.join(self.get_words(self.id_to_words(quest))) for quest in self.quests]
        #向量化
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x)))
        self.Vocab_Size = len(vocab_processor.vocabulary_)
        print("Vocabulary Size: {:d},quests Size:{:d}".format(self.Vocab_Size,len(x)))     
        vocab_processor.save(os.path.join(self.file_path, "vocab"))
        print("quest:%s"%(x[0]))
        self.x_vocab=x 
    def Reload_vocab(self,x=''):
        if(x==''):           
            print("Reload vocab,quests size:%d"%(len(self.quests)))
            x = [' '.join(self.get_words(quest)) for quest in self.quests]
        else:
            print("Reload vocab,quest is:%s"%(x))
            x = [' '.join(self.get_words(x))]
                   
        vocab_path = os.path.join(self.file_path, "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x)))
        return x_test         
    def one_hot(self,label):
        vectors = [0]*len(self.labels)
        index = self.labels.index(label)
        vectors[index] = 1
        return vectors 

    def get_test_train(self,rate=0.1):
        all_quests=[]
        test=[]
        train=[]
        for quest in self.quests:
            all_quests.append(self.quests.index(quest))
        all_quests=np.random.permutation(all_quests)
         
        index=int(len(all_quests)*rate)
        test=all_quests[0:index]
        train=all_quests[index:]
        test_x=self.x_vocab[test]
        test_y=self.y_vocab[test]
        train_x=self.x_vocab[train]
        train_y=self.y_vocab[train]
        print("测试数据的维度:%s,训练数据的维度:%s"%(test_x.shape,train_x.shape))
        return train_x,train_y,test_x,test_y 
       
    def data_process(self,user_id):
        if(user_id==None):
            self.process_local_data2()
        else:
            self.process_remote_data(user_id)
    def write_new_data(self):
        path=os.path.join(self.file_path,'data2.txt')
        with open(path,'w',encoding='utf-8') as f:
            for label,quests in self.label_quests.items():
                for quest in quests:
                    f.write("%s\t%s\n"%(self.id_to_words(quest),self.id_to_words(label)))
                    
    def batch_iter(self,data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            #np.arange(5)生成一个[0,1,2,3,4]的矩阵，permutation是打乱顺序
            begin=time.clock()
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index] 
            with open(os.path.join(self.file_path,'process_rate.txt'),'w',encoding='utf-8') as f:
                f.write("%d:%d:%d\n"%(epoch+1,num_epochs,time.clock()-begin))   
                
    def write_to_pickle(self,file_name,x,y):
        path = os.path.join(self.file_path,file_name)
        print("Write data to pickle file:%s"%(path))
        output=open(path,'wb')
        pickle.dump(x,output)
        pickle.dump(y,output)       
        output.close()
    #以二进制读取文件   
    def read_from_pickle(self,file_name):
        path = os.path.join(self.file_path,file_name)
        print("Read data from pickle file:%s"%(path))        
        output=open(path,'rb')
        x=pickle.load(output)
        y=pickle.load(output)
        output.close() 
        return x,y 