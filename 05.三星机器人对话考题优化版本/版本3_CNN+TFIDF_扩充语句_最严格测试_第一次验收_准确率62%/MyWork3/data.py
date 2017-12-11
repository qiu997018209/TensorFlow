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
import time
import shutil
import math
import random
import jieba.analyse
import jieba.posseg as pseg
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class data(object):
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
        self.max_document_lenth=32
        self.word_idf=defaultdict(lambda:0)#词频的idf值
        #开始处理数据
        self.data_process(user_id)

        
    def data_process(self,user_id):
        np.random.seed(0)    
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
        
        #删除上一次的训练数据
        out_dir = os.path.abspath(os.path.join('data', "runs"))
        if(os.path.exists(out_dir)):
            #shutil.rmtree(out_dir)
            pass
        #生成新的词汇表文件
        with open('data/vocab.txt','w',encoding='utf-8') as f:
            for key,value in self.word_dic.items():
                f.write('%s %d\n'%(key,value))
        print("生成词汇表:%s,size is:%d"%('data/vocab.txt',len(self.word_dic)))  
        
        #统计生成问题的情况
        self.statistics_quests()
        #抽取关键词信息
        self.jieba_extract_idf()
        
        #训练word2vec
        self.init_word_embedding()
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
                self.word_idf[str(word).upper()]=math.log(len(self.label_quest)/count)
        with open('data/words_idf.txt','w',encoding='utf-8') as f:
            for word,score in self.word_idf.items():
                f.write('%s %d\n'%(word,score))
    #清理字符串
    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string
    
    #扩充数据
    def extend_data(self):
        print('开始扩充数据')
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
                               
        with open('data/data.txt','w',encoding='utf-8') as f:
            for quest,values in self.quest_quests.items():
                label = self.quest_label[quest]
                if label =='':
                    continue
                #原始的quest也加入
                f.write(label+':'+quest+'\n')
                for value in values:
                    f.write(label+':'+value.upper()+'\n')
                if label not in self.labels:                    
                    self.labels.append(label)
                #维护label对应的quest
                self.label_quest[label].add(quest) 
    def get_words(self,line):
        quests=[]
        words=''
        i=0
        bFlag=False
            
        while i<len(line):
            if line[i] == '{':
                bFlag=True
            elif line[i] == '}':
                if(bFlag==False or words==''):
                    print("错误的{}",line)
                bFlag=False
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
        print('开始处理词汇表')
        #'<UNK>'是0
        with open('data/vocab.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split(' ')
                self.word_dic[line[0]]=int(line[1])
                
        #self.word_dic['<UNK>']=len(self.word_dic)
        if add==True:
            with open('data/data.txt','r',encoding='utf-8') as f:
                for line in f.readlines():
                    line=line.strip()
                    if line =='':
                        continue
                    line=self.clean_str(line.split(':')[1])
                    for word in jieba.lcut(line):
                        if word not in self.word_dic and word not in self.stop_words:
                            self.word_dic[word]=len(self.word_dic)
            with open('data/origin_test.txt','r',encoding='utf-8') as f:
                for line in f.readlines():
                    line=line.strip()
                    if line =='':
                        continue
                    line=self.clean_str(line.split('\t')[1]+line.split('\t')[2])
                    for word in jieba.lcut(line):
                        if word not in self.word_dic and word not in self.stop_words:
                            self.word_dic[word]=len(self.word_dic)                
        if '<UNK>' not in self.word_dic:
            self.word_dic['<UNK>'] =len(self.word_dic) 
    #对每句话向量化
    def build_vector(self,sentence):
        vector=[]
        for word in jieba.lcut(self.clean_str(sentence)):
            if word in self.word_dic:
                if word not in self.stop_words:
                    vector.append(self.word_dic[word])
            else:
                self.word_dic[word]=len(self.word_dic)
                vector.append(self.word_dic['<UNK>'])
                print("新增词汇:%s,%s"%(word,sentence))
                
        if(len(vector)>=self.max_document_lenth):
            vector=vector[0:self.max_document_lenth]
        else:
            num = self.max_document_lenth-len(vector)
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
        with open('data/data.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line =='':
                    continue
                line=line.split(':')
                try:
                    #将标签转为从0~xx的大小
                    self.train_test_data.append((self.build_vector(line[1]),self.one_hot(line[0])))
                except Exception as e:
                    print(e)
                    return               
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
                if label not in self.labels:
                    print('没有未此标签扩充语料:%s'%(line[0]))
                    continue                    
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
    #获取批次数据
    def get_batch_data(self,batch_size,num_epochs):
        #打乱顺序
        random.shuffle(self.train_test_data)
        train_data, train_target = zip(*self.train_test_data)
        self.train_test_data=[]
        num_batches_per_epoch = int((len(train_data)-1)/batch_size) + 1
        with open('data/origin_test.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip().split('\t') 
                label=self.quest_label[line[0]]
                if label=='':
                    print('无法找到标签:%s'%(line[0]))
                    continue
                if label not in self.labels:
                    print('没有未此标签扩充语料:%s'%(line[0]))
                    continue                
                #print(line[1],line[2],label,len(self.labels))
                self.train_test_data.append((self.build_vector(line[1]),self.one_hot(label)))
                self.train_test_data.append((self.build_vector(line[2]),self.one_hot(label))) 
                self.test_quests[len(self.test_quests)]=(line[1],label)
                self.test_quests[len(self.test_quests)]=(line[2],label)   
        
        test_data, test_target = zip(*self.train_test_data)
        train_data,train_target,test_data,test_target=np.array(train_data),np.array(train_target),np.array(test_data),np.array(test_target)
        print("训练数据维度:%s,训练标签维度:%s,测试数据维度:%s,测试标签维度:%s,标签数量:%s"%(train_data.shape,train_target.shape,test_data.shape,test_target.shape,len(self.labels)))
        
        for epoch in range(num_epochs):
            begin=time.clock()
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size  
                end_index = min((batch_num + 1) * batch_size, len(train_data))
                batch_x=train_data[start_index:end_index]
                batch_y=train_target[start_index:end_index]
                yield batch_x,batch_y,test_data,test_target
                
            with open(os.path.join('data','process_rate.txt'),'w',encoding='utf-8') as f:
                f.write("%d:%d:%d\n"%(epoch,num_epochs-1,time.clock()-begin))         
    #统计每个原始的每个问题扩充的情况
    def statistics_quests(self):
        key=''
        labels={}
        label_quest={}
        with open('data/data.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line == '': 
                    continue
                line=line.split(':')
                if line[0] not in labels:
                    labels[line[0]]=set()
                    labels[line[0]].add(line[1])
                else:
                    labels[line[0]].add(line[1])
                
        with open('data/quest_label.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                if line=='':
                    continue
                line=line.split(':')
                label_quest[line[0]]=line[1]
               
        with open('data/statistics_quests_less.txt','w',encoding='utf-8') as f_l:
            with open('data/statistics_quests_more.txt','w',encoding='utf-8') as f_m:
                for key in labels:
                    if len(labels[key])<=10:
                        f_l.write('%d:%s\n'%(len(labels[key]),label_quest[key]))
                        for quest in labels[key]:
                            f_l.write('%s\n'%(quest))
                    elif len(labels[key])>=500:
                        f_m.write('%d:%s\n'%(len(labels[key]),label_quest[key])) 
    def find_wrong_ans(self,correct_predictions): 
        correct_predictions = list(correct_predictions)
        num=0
        with open('data/accuracy.txt','w',encoding='utf-8') as f:
            for label in correct_predictions:
                num+=1 
                if(label==0):
                    f.write("答错的问题是:%d\n"%(num))                            
        #找出答错的问题
        wrong_quests=[]
        all_quests={}
        num=0
        with open('data/accuracy.txt','r',encoding='utf-8') as f1:           
            with open('data/origin_test.txt','r',encoding='utf-8') as f2:
                for line in f1.readlines():
                    wrong_quests.append(line.strip().split(':')[1])
                for line in f2.readlines():  
                    num+=1 
                    #存的是原始问题
                    all_quests[num]=line.strip().split('\t')[0]
                    num+=1
                    all_quests[num]=line.strip().split('\t')[0]
                
        with open('data/important2.txt','w',encoding='utf-8') as f1: 
            write=[]
            for wrong in wrong_quests:
                quest = all_quests[int(wrong)]
                if(quest not in write):
                    label=self.quest_label[quest]
                    if label=='':
                        #print('过滤没有标签的数据:%s'%(quest))
                        continue                
                    write.append(quest)
                    f1.write("#"+quest+'\n\n') 
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
        word_tf={}
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
            label_tfidf[label]=value*5
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
            
        print('Final Acuracy:%f\n'%(right/len(self.test_quests))) 
            
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
    my_data=data()