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
import pickle
import numpy as np
import random 
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.core.example import example_pb2
from gensim.models.word2vec import Word2Vec

class my_lib2(object):
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
            if(self.(word)):
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
    #以二进制保存文件
    def write_to_pickle(self,file_name,test):
        path = os.path.join(self.file_path,file_name)
        print("Write data to pickle file:%s"%(path))
        output=open(path,'wb')
        pickle.dump(test,output)       
        output.close()
    #以二进制读取文件   
    def read_from_pickle(self,file_name):
        path = os.path.join(self.file_path,file_name)
        print("Read data from pickle file:%s"%(path))        
        output=open(path,'rb')
        test=pickle.load(output)
        output.close() 
        return test 
        #填充一句话            
    def fill_element(self,sTemp,sequence_length):
        sTemp = self.get_words(sTemp)
        lenth = sequence_length-len(sTemp)
        if(lenth>0):
            for _ in range(lenth):
                sTemp.append('<b>')
        else:
            sTemp = sTemp[:sequence_length]
        #用_做分隔符避免后面还需要分词
        sTemp = '<a>'.join(sTemp)
        return sTemp
    #获取指定集合里的最长字符串长度
    def get_max_lenth_quest(self,quests):
        max_lenth = 0
        max_quest = ''
        for quest in quests:
            if len(quest) > max_lenth:
                max_lenth = len(quest)
                max_quest = quest
        self.max_lenth_quest=max_quest
        print("\n本数据集里最长问题为:%s长度为:%d"%(max_quest,max_lenth))    
        return 
    #生成word2vec文件               
    def get_word2vec_file(self):
        corpus = []            
        for word in self.words_dic:
            corpus.append(word)                   
        model = Word2Vec(corpus, size=self.Flags.embedding_dim, min_count=1, workers=4)
        model.wv.save_word2vec_format(os.path.join(self.file_path,'vectors.nobin'), binary=False)
        print('生成word2vec文件，文件路径为:%s,词嵌入维度为:%d'%(os.path.join(self.file_path,'vectors.nobin'),self.Flags.embedding_dim))
    #生成词汇表
    def build_words_dic(self):
        
        self.words_dic['UNKNOWN'] = len(self.words_dic)
        
        for index in self.quest_key_value:
            words = self.get_words(self.quest_key_value[index])
            for word in words:
                if(word not in self.words_dic):
                    self.words_dic[word]=len(self.words_dic)
                                        
        for index in self.ans_key_value:
            words = self.get_words(self.ans_key_value[index])
            for word in words:
                if(word not in self.words_dic):
                    self.words_dic[word]=len(self.words_dic)
                                                            
        for index in self.label_key_value:
            words = self.get_words(self.label_key_value[index])
            for word in words:
                if(word not in self.words_dic):
                    self.words_dic[word]=len(self.words_dic)

        for index in self.subject_key_value:
            words = self.get_words(self.subject_key_value[index])
            for word in words:
                if(word not in self.words_dic):
                    self.words_dic[word]=len(self.words_dic)
                    
        print('生成词汇表完成，总共:%d个词汇'%(len(self.words_dic))) 
           
    def next_device(self,bUseCPU=True):
        device=''
        if(bUseCPU):
            device = '/cpu:%d'
        else:
            device = '/gpu:%d'
        if(self.record_data['cur_device_id']+1<=self.Flags.num_cpu_core):
            self.record_data['cur_device_id'] += 1
        return device%(self.record_data['cur_device_id'])
    data={"quest":"银联卡闪付功能，是不是只要卡片上带有闪付标志，就可以直接辉卡交易"}
    
    def send_post(self,quest,address='http://127.0.0.1:8000/deep_chat'):
        data={"quest":quest}
        r = requests.post(address, data)
        
        print (r.status_code)
        print (r.headers['content-type'])
        r.encoding = 'utf-8'
        print (r.text)
    #对quest进行向量化
    def build_vocab(self,quests):
        #以空格来切分每一个词
        max_document_length = max([len(quest) for quest in quests])
        x = [' '.join(self.get_words(quest)) for quest in quests]      
        #向量化
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x)))
        self.Vocab_Size = len(vocab_processor.vocabulary_)
        print("Vocabulary Size: {:d},X Size:{:d}".format(self.Vocab_Size,len(x)))
        
        vocab_processor.save(os.path.join(self.file_path, "vocab"))
        print("quest:%s"%(x[0]))
        return x 
    #载入向量文件进行向量化  
    def Reload_vocab(self,quests):
        x = [' '.join(self.get_words(quest)) for quest in self.quests]             
        vocab_path = os.path.join(self.file_path, "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x = np.array(list(vocab_processor.transform(x)))
        return x       
    #对数据进行洗牌并产生批量数据
    def batch_iter(self,data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            #np.arange(5)生成一个[0,1,2,3,4]的矩阵，permutation是打乱顺序
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]  
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
                print("当前处理句式数量为:%d/%d"%(line_num,len(lines)))
                if line.startswith('####'):
                    continue
                elif line.startswith('#'):
                    recent_quest=self.words_to_id(line[1:])
                    if recent_quest not in self.quest_rules:
                        self.quest_rules[recent_quest]=[]
                        quest_nums+=1
                    else:
                        print('重复扩充quest:%s'%(recent_quest))
                else:
                    #扩充句式
                    all_quests=self.sentences_algorithm(line)
                    line = self.words_to_id(line)
                    if(recent_quest==''or recent_quest not in self.quest_rules):
                        print(recent_quest)
                        print("process_sentences error")
                    self.quest_rules[recent_quest].append(line)
                    if line not in self.rule_quests:
                        self.rule_quests[line]=all_quests
                    else:
                        print("重复的句式:%s"%(self.id_to_words(line)))
                        
        print("Total %d sentences"%(quest_nums))
        self.write_all_quests()
        self.combina_quests()
        
    def combina_quests(self):
        label_num=0
        print('开始合并标签问题集合')        
        for label,quests in self.label_quests.items():
            label_num+=1
            for quest in quests.copy():
                if quest not in self.quest_rules:
                    #print('问题%s没有扩充句式'%(quest))
                    continue
                #将所有扩充后的句子与原来的句子进行合并
                for rule in self.quest_rules[quest]:
                    self.label_quests[label]+=self.rule_quests[rule] 
            print("当前label%s的quest数量:%d"%(self.id_to_words(label),len(self.label_quests[label])))     
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
        print('开始将扩充后的句式写入到文件中')
        path = os.path.join(self.file_path,"extend_quests.txt")
        with open(path,'w',encoding='utf-8') as f: 
            for quest,rules in self.quest_rules.items():
                num=0 
                for rule in rules:
                    for ex_quest in self.rule_quests[rule]:
                        num+=1
                        f.write("%d\t%s\t%s\t%s\n"%(num,self.id_to_words(ex_quest),self.id_to_words(rule),self.id_to_words(quest)))
                              
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
                    if(quest[i] in ',{，。<?？@'):
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
                while(i<len(quest) and self.isChinese(quest[i])==False):
                    word+=quest[i]
                    i+=1
                    if(i>=len(quest) or quest[i]=="@"):
                        break
                words.append(word)
            elif(self.isChinese(quest[i])):
                words.append(quest[i])
                i+=1 
            else:
                print("get_sentence_words,Error:%s"%(quest[i]))
                i+=1
        return words

    def write_train_test(self):
        #相同quest之间两两组合，生成二进制文件  
        print("开始生成二进制的训练与测试数据文件")                                                             
        self.build_binary_file()

    def build_binary_file(self):
        out_path_train = os.path.join(self.file_path,'my_data_train')
        out_path_test = os.path.join(self.file_path,'my_data_test')       
        test_list=[]
        train_list=[]
        inputs_train = {}
        inputs_test = {}
        num_index=0
        for label,quest in self.label_quests.items():
            quests=quest.copy()
            num_index += 1
            print("当前生成训练与测试文件进度:%d/%d"%(num_index,len(self.label_quests)))
            
            #quest与label之间互相组合
            if label not in quests:
                quests.append(label)
            #数量太少的不要
            if len(quests)<=10:
                continue
            index = random.randint(0,len(quests)-1)
            #用来做测试
            inputs_test[quests[index]]=label
            #互相之间用来做训练
            quests.remove(quests[index])
            
            if(len(quests)>3000):
                quests=quests[0:3000]
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
            out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
            out_label = ' '.join(self.get_words(self.id_to_words(label)))
            article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
            abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
            #最终的ref是abstract
            line = 'article='+article+'\t'+'abstract='+article+'\t'+'publisher='+'qiu'+'\n'
            test_list.append(line)            
        for quest,labels in inputs_train.items():              
            for label in labels:              
                out_quest = ' '.join(self.get_words(self.id_to_words(quest)))
                out_label = ' '.join(self.get_words(self.id_to_words(label)))
                article='<d> <p> <s> %s </s> </p> </d>'%(out_quest)
                abstract='<d> <p> <s> %s </s> </p> </d>'%(out_label)
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
     
    #将quest转化为ID值
    def words_to_id(self,words):
        if(words not in self.words_id):
            num=len(self.words_id)
            self.words_id[words]=num
            self.id_words[num]=words      
        return self.words_id[words]
    #将ID值转化为quest
    def id_to_words(self,id):
        if(int(id)>=len(self.id_words)):
            print("错误的id:%d"%(id))
        return self.id_words[id]
                
               
                
            
                
        
        