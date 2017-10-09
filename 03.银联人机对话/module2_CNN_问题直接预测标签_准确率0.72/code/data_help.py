#coding:utf-8
'''
Created on 2017年9月15日

@author: qiujiahao

@email:997018209@qq.com

'''
import numpy as np
import time
import random
import os
import tensorflow as tf
import pickle
from tensorflow.contrib import learn
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
    def __init__(self,file_path='../data/',bProcessData=True):
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
        self.label_id_quests_id = {}#标签ID:quest ID
        self.words_dic={}#字典,单词:id
        self.labels_ans_subject={}#label:ans,subject
        self.file_path = file_path
        self.Vocab_Size=0
        self.quests_limit=4
        #开始处理数据
        if(bProcessData==True):
            self.data_process() 
          
    #处理数据文件
    def data_process(self):
        start = time.clock()
        #self.show_params()
        #处理文件数据     
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
                    if label_id not in self.label_id_quests_id:
                        self.label_id_quests_id[label_id]=[quest_id]
                    elif (quest_id not in self.label_id_quests_id[label_id]):
                        self.label_id_quests_id[label_id].append(quest_id)
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
    #获取测试数据集合
    def read_test_file(self):
        with open(os.path.join(self.file_path,'test'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                label,quest,ans,subject=self.split_line_data(line)
                quest = ''.join(quest.strip().split('<a>')).replace('<b>','')
                self.test_quests.append(quest)
        print("测试问题示例:%s,长度为:%d"%(quest,len(quest)))
    #生成测试和训练集    
    def write_test_train_file(self):
        test = open(os.path.join(self.file_path,'test'),'w',encoding='UTF-8')
        train = open(os.path.join(self.file_path,'train'),'w',encoding='UTF-8')
        all = []
        test_data = []
        write_labels = [] 
        index = 0
        train_index = 0
        test_index = 0
        pass_index = 0
        for label_id in self.label_id_quests_id:            
            quest_id = -1
            quests_id = self.label_id_quests_id[label_id]
            for q_id in quests_id:
                index+=1
            if  label_id not in write_labels and self.label_id_count[label_id] >= self.Flags.quests_limit:
                #随机选一个quest写入
                test_index += 1
                write_labels.append(label_id) 
                rand_num = random.randint(0,len(quests_id)-1)
                quest_id = quests_id[rand_num]
                ans_id = self.quest_info[quest_id][0]
                label_id = self.quest_info[quest_id][1]
                subject_id = self.quest_info[quest_id][2]
                self.test_quests.append(self.quest_key_value[quest_id])
                quest = self.fill_element(self.quest_key_value[quest_id])
                ans = self.fill_element(self.ans_key_value[ans_id])
                label = self.fill_element(self.label_key_value[label_id])
                subject = self.fill_element(self.subject_key_value[subject_id])  
                test.write("0 qid:%d label:%s quest:%s ans:%s subject:%s\n"%(quest_id,label,quest,ans,subject))           
            for q_id in quests_id:
                if(q_id == quest_id):
                    continue 
                train_index += 1              
                ans_id = self.quest_info[q_id][0]
                label_id = self.quest_info[q_id][1]
                subject_id = self.quest_info[q_id][2]
                self.train_quests.append(self.quest_key_value[q_id])
                quest = self.fill_element(self.quest_key_value[q_id])
                ans = self.fill_element(self.ans_key_value[ans_id])
                label = self.fill_element(self.label_key_value[label_id])
                subject = self.fill_element(self.subject_key_value[subject_id])                    
                train.write("1 qid:%d label:%s quest:%s ans:%s subject:%s\n"%(q_id,label,quest,ans,subject))             
        test.close()
        train.close()
        print("训练数据集文件完成,路径为:%s,总共%d个数据集"%(os.path.join(self.file_path,'train'),len(self.train_quests)))
        print("测试数据集文件完成,路径为:%s,总共%d个数据集"%(os.path.join(self.file_path,'test'),len(self.test_quests)))                
    #填充一句话            
    def fill_element(self,sTemp):
        sTemp = self.get_words(sTemp)
        lenth = self.Flags.sequence_length-len(sTemp)
        if(lenth>0):
            for _ in range(lenth):
                sTemp.append('<b>')
        else:
            sTemp = sTemp[:self.Flags.sequence_length]
        #用_做分隔符避免后面还需要分词
        sTemp = '<a>'.join(sTemp)
        return sTemp
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
    #统计最长的quest长度
    def get_max_lenth_quest(self):
        max_lenth = 0
        max_quest = ''
        for quest in self.quests:
            if len(quest) > max_lenth:
                max_lenth = len(quest)
                max_quest = quest
        print("\n本数据集里最长问题为:%s长度为:%d"%(max_quest,max_lenth))
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
   
    #产生一个除自己以外的随机数
    def rand_qa_except(self,quest):
        nega = self.train_quests[random.randint(0, len(self.train_quests) - 1)]
        while(nega == quest):
            nega = self.train_quests[random.randint(0, len(self.train_quests) - 1)]
        return nega

    #对一句话进行向量化
    def encode_sent(self,sentence):
        x = []
        words=sentence.strip().split('<a>')
        for word in words:
            if word in self.words_dic:
                x.append(self.words_dic[word])
            else:
                x.append(self.words_dic['UNKNOWN'])
        if(len(x)!=self.Flags.sequence_length):
            print("Error:%d %s"%(len(x),x))
        return x
    
    #切分一行数据,例如:"1 qid:%d label:%s quest:%s ans:%s subject:%s\n"
    def split_line_data(self,line):
        items = line.strip().split(' ')
        label = items[2].split(':')[1]
        quest = items[3].split(':')[1]
        ans = items[4].split(':')[1]
        subject = items[5].split(':')[1]  
        return label,quest,ans,subject    

    def one_hot(self,label):
        vectors = [0]*len(self.labels)
        index = self.labels.index(label)
        vectors[index] = 1
        return vectors
        
    def build_labels(self):    
        labels = [self.quest_label[quest] for quest in self.quests]
        labels = [self.one_hot(label) for label in labels]
        print('获取语料与对应的标签完成,问题:%d 标签:%d'%(len(self.quests),len(self.labels)))
        
        return np.array(labels)
    #对quest进行向量化
    def build_vocab(self):
        #以空格来切分每一个词
        max_document_length = max([len(quest) for quest in self.quests])
        x = [' '.join(self.get_words(quest)) for quest in self.quests]
        
        ans =[' '.join(self.get_words(self.labels_ans_subject[self.quest_label[quest]][0])) for quest in self.quests]
        sub =[' '.join(self.get_words(self.labels_ans_subject[self.quest_label[quest]][1])) for quest in self.quests]        
        #向量化
        x_all=x + ans + sub
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x_all = np.array(list(vocab_processor.fit_transform(x_all)))
        self.Vocab_Size = len(vocab_processor.vocabulary_)
        print("Vocabulary Size: {:d},X Size:{:d}".format(self.Vocab_Size,len(x_all)))
        
        vocab_processor.save(os.path.join(self.file_path, "vocab"))
        print("quest:%s ans:%s label:%s"%(x[0],ans[0],sub[0]))
        return x_all[0:len(x)],x_all[len(x):2*len(x)],x_all[2*len(x):3*len(x)]
    
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

    #在标签对应的语料数目如果大于4，则随机选一个这个语料
    def get_test_train(self,x,y):
        write_labels = []
        for label_id in self.label_id_quests_id:
            quest_id = -1
            quests_id = self.label_id_quests_id[label_id]            
            if  label_id not in write_labels and self.label_id_count[label_id] >= self.quests_limit:
                #随机选一个quest写入
                rand_num = random.randint(0,len(quests_id)-1)
                quest_id = quests_id[rand_num]                
                self.test_quests.append(self.quest_key_value[quest_id])           
            for q_id in quests_id:
                if(q_id == quest_id):
                    continue
                self.train_quests.append(self.quest_key_value[q_id])
        
        test_index=[]
        train_index=[]    
        for quest in self.test_quests:
            test_index.append(self.quests.index(quest)) 
        for quest in self.train_quests:
            train_index.append(self.quests.index(quest))
            
        test_x = x[test_index] 
        test_y = y[test_index]
        train_x = x[train_index]
        train_y = y[train_index]
                
        return train_x,train_y,test_x,test_y
    
    def record_test_data(self,accuracy,params,log='----'):
        result ='####开始记录本次实验参数####\n'
        result +='####'+log+'####\n'
        result += str(accuracy)+' '+'quests_limit:%d'%(self.quests_limit)+'\n'        
        result += params+'\n'
        path = os.path.join(self.file_path,'模型参数对准确率影响记录.txt')
        with open(path,'a+',encoding='utf-8') as f:
            f.write(result)
 
    def write_to_pickle(self,file_name,test_x,test_y):
        path = os.path.join(self.file_path,file_name)
        print("Write data to pickle file:%s"%(path))
        output=open(path,'wb')
        pickle.dump(test_x,output)
        pickle.dump(test_y,output)        
        output.close()
        
    def read_from_pickle(self,file_name):
        path = os.path.join(self.file_path,file_name)
        print("Read data from pickle file:%s"%(path))        
        output=open(path,'rb')
        test_x=pickle.load(output)
        test_y=pickle.load(output)
        output.close() 
        return test_x,test_y           
        
        
if __name__=="__main__":
    my_data = data_help()

    