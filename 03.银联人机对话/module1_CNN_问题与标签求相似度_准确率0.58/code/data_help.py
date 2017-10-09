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
import config
import tensorflow as tf
from numpy import dtype
from gensim.models.word2vec import Word2Vec
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
    def __init__(self,bProcessData=True):
        self.quest_key_value={}#问题:问题ID
        self.ans_key_value = {}#答案:答案ID
        self.label_key_value = {}#标签:标签ID
        self.quest_info = {}#问题ID:[答案ID,标签ID]
        self.label_ans = {}#label:答案
        self.labels = []#所有的label
        self.quests = []#所有的quest
        self.quest_label = {}#quest:label
        self.label_id_count = {}#标签ID:次数
        self.test_quests = []#测试集中的所有quest
        self.train_quests = []#训练集中的所有quest
        self.subject_key_value = {}#主题ID:主题
        self.label_id_quests_id = {}#标签ID:quest ID
        self.words_dic={}#字典,单词:id
        self.Flags = config.create_hparams()
        self.wrong_quests = {}#测试中匹配错误的quest
        self.record_data = {'cost_time':0,'right_num':0,'wrong_num':0,'cur_device_id':-1}
        #开始处理数据
        if(bProcessData==True):
            self.data_process() 
          
    #处理数据文件
    def data_process(self):
        start = time.clock()
        self.show_params()
        #处理文件数据     
        with open(os.path.join(self.Flags.file_path,'data.txt'),'r',encoding='UTF-8') as f:
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
                    self.subject_key_value[subject_id]=subject
                    self.quest_key_value[quest_id]=quest
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
        
        self.get_max_lenth_quest()
        #生成词汇表
        self.build_words_dic()
        #self.get_word2vec_file()
        end = time.clock()  
        print("数据预处理完成,总共耗时%s"%(end-start))
    #获取测试数据集合
    def read_test_file(self):
        with open(os.path.join(self.Flags.file_path,'test'),'r',encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                label,quest,ans,subject=self.split_line_data(line)
                quest = ''.join(quest.strip().split('<a>')).replace('<b>','')
                self.test_quests.append(quest)
        print("测试问题示例:%s,长度为:%d"%(quest,len(quest)))
    #生成测试和训练集    
    def write_test_train_file(self):
        test = open(os.path.join(self.Flags.file_path,'test'),'w',encoding='UTF-8')
        train = open(os.path.join(self.Flags.file_path,'train'),'w',encoding='UTF-8')
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
        print("训练数据集文件完成,路径为:%s,总共%d个数据集"%(os.path.join(self.Flags.file_path,'train'),len(self.train_quests)))
        print("测试数据集文件完成,路径为:%s,总共%d个数据集"%(os.path.join(self.Flags.file_path,'test'),len(self.test_quests)))                
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
    #生成word2vec文件               
    def get_word2vec_file(self):
        corpus = []            
        for word in self.words_dic:
            corpus.append(word)                   
        model = Word2Vec(corpus, size=self.Flags.embedding_dim, window=5, min_count=1, workers=4)
        model.wv.save_word2vec_format(os.path.join(self.Flags.file_path,'vectors.nobin'), binary=False)
        print('生成word2vec文件，文件路径为:%s,词嵌入维度为:%d'%(os.path.join(self.Flags.file_path,'vectors.nobin'),self.Flags.embedding_dim))
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
    #获取一个批次的测试数据
    def get_batch_train_data(self):
        x_train_1 = []
        x_train_2 = []
        x_train_3 = []
        with open(os.path.join(self.Flags.file_path,'train'),encoding='utf-8') as f:
            lines=f.readlines()        
            for _ in range(self.Flags.batch_size):
                #随机选择一行进行训练
                line = lines[random.randint(0, len(lines) - 1)]
                label,quest,ans,subject=self.split_line_data(line)
                #噪音数据
                nega = self.fill_element(self.rand_qa_except(quest))
                x_train_1.append(self.encode_sent(label))
                x_train_2.append(self.encode_sent(quest))
                x_train_3.append(self.encode_sent(nega))            
        x_train_1 = np.array(x_train_1)
        x_train_2 = np.array(x_train_2)
        x_train_3 = np.array(x_train_3)
        return x_train_1, x_train_2, x_train_3
    #获取所有的测试数据get_batch_train_data
    def get_test_data(self,test_quest):
        x_test_1 = []
        x_test_2 = []
        x_test_3 = []
        for label in self.labels:
            label = self.fill_element(label)            
            quest = self.fill_element(test_quest)
            x_test_1.append(self.encode_sent(label))
            x_test_2.append(self.encode_sent(quest))
            x_test_3.append(self.encode_sent(quest))
                
        x_test_1 = np.array(x_test_1)
        x_test_2 = np.array(x_test_2)
        x_test_3 = np.array(x_test_2)
               
        return x_test_1, x_test_2, x_test_3 
    
    def show_params(self):
        print("Config Parameters:")
        for item in self.Flags._asdict().items():
            print('%s=%s'%(item[0],item[1]),end=' ')     

    def record_test_data(self,file_name,accuracy,log):
        if(file_name==''):
            return
        path = os.path.join(self.Flags.file_path,file_name+'.txt')
        max_score = 0
        with open(path, 'a+', encoding='utf-8') as f:
            f.write('####开始记录本轮测试数据####\n')
            f.write('测试log:%s\n'%(log))
            for item in self.Flags._asdict().items():
                line = ('%s=%s\t'%(item[0],item[1]))
                f.write(line)
            f.write('\n')
            for quest in self.wrong_quests:
                score = self.wrong_quests[quest][1] - self.wrong_quests[quest][3]
                if(score > max_score):
                    max_score = score
            f.write('accuracy=%f\ttime cost=%f\t正确与错误回答的准确率误差:max_score=%f\n'%(accuracy,self.record_data['cost_time'],max_score))
            f.write('####本轮测试数据记录完毕####\n')
            
    def show_test_result(self,scores,quest,file_name='',log=''):
        quests_num = len(self.test_quests)
        max_score = max(scores)
        predict_label = self.labels[scores.index(max_score)]
        right_label = self.quest_label[quest]
        
        if (predict_label == right_label):
            self.record_data['right_num'] +=1
        else:
            self.record_data['wrong_num'] +=1            
            self.wrong_quests[quest]=[predict_label,max_score,right_label,scores[self.labels.index(right_label)]] 
        total = self.record_data['right_num']+self.record_data['wrong_num']
        
        if(0 != self.record_data['cost_time']):        
            self.record_data['cost_time'] = time.clock() - self.record_data['cost_time'] 
        
        accuracy = self.record_data['right_num'] / total     
        print("quest_num:%d/%d,right_num:%d/%d"%(total,quests_num,self.record_data['right_num'],total))
        print("time cost %fs,test function accuracy:%f"%(self.record_data['cost_time'],accuracy)) 
        
        if(total == quests_num):
            #显示错误匹配:
            print("错误匹配的问题与标签内容如下:\n")   
            for index,quest in enumerate(self.wrong_quests): 
                print("%d quest:%s-->wrong label:%s scores:%f right label:%s scores:%f"%(index,quest,self.wrong_quests[quest][0],
                                                                                        self.wrong_quests[quest][1],self.wrong_quests[quest][2],self.wrong_quests[quest][3]))    
            self.record_test_data(file_name,accuracy,log)
            
    def next_device(self,bUseCPU=True):
        device=''
        if(bUseCPU):
            device = '/cpu:%d'
        else:
            device = '/gpu:%d'
        if(self.record_data['cur_device_id']+1<=self.Flags.num_cpu_core):
            self.record_data['cur_device_id'] += 1
        return device%(self.record_data['cur_device_id'])

         
if __name__=="__main__":
    my_data = data_help()
    my_data.write_test_train_file()

    