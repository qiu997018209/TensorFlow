#coding:utf-8
'''
Created on 2018年2月28日

@author: qiujiahao

@email:997018209@qq.com

'''
import re
import xlwt
from collections import defaultdict

class data(object):
    def __init__(self):
        self.data_process()
        
    def data_process(self):
        #初始化
        self.data_init()
        
        #扩充句子
        self.extend_data()
        
        #将句子安装机器人需要的格式写成excel格式
        self.write_to_java()
        
    def data_init(self):
        #加载同义词词表
        self.samewords=defaultdict(list)
        with open('data/sameword.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) <=1:
                    continue
                line=line.strip().split('\t')
                self.samewords[line[0]]=line[1:]
        #扩充数据
    def extend_data(self):
        self.quest_label=defaultdict(list)
        self.quest_quests=defaultdict(set)
        current_quest=''
        with open('data/data.txt','r',encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) <=1:
                    continue            
                if line.startswith('#'):
                    line=line[1:].split('\t')
                    self.quest_label[line[0]]=line[1]
                    current_quest=line[0]
                else:
                    self.quest_quests[current_quest]=self.extend_algorithm_rule(line.strip())
                    
    #安装java端的数据格式来生成
    def write_to_java(self):
        wb = xlwt.Workbook()
        sh=wb.add_sheet("Sheet1")
        #新增一个表单
        #表头
        sh.write(0, 0, '知识点')
        sh.write(0, 1, '问题内容')
        sh.write(0, 2, '语义内容')
        sh.write(0, 3, 'WEB应答内容')
        sh.write(0, 4, '微信应答内容')
        sh.write(0, 5, 'IVR应答内容')
        sh.write(0, 6, 'QAID')
        i=1
        total_num=0
        for quest,values in self.quest_quests.items():
            label = self.quest_label[quest]
            if(label==''):
                print('错误的原始quest:%s'%(quest))
                continue
            sh.write(i, 0, quest)
            quests=''
            for key in values:
                if quest not in values and quest not in quests:
                    quests+=quest+'&&&'
                quests+=key+'&&&'
                total_num+=1
            #去除最后三个符号
            quests=quests[:-3]
            sh.write(i, 1, quests)
            sh.write(i, 3, label)
            sh.write(i, 6, 0)
            i+=1
        print('扩充完毕,扩充后总共问题数量:{},结果写入文件:data/data.xls'.format(total_num))
        wb.save('data/data.xls')                    
                    
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
                
    def check_rules(self,flag,sentences):  
        #1代表{}里的检查
        if flag==1:
            for word in sentences: 
                if word in '{}()':
                    print("%s含有非法字符"%(sentences))
            #去掉<>里的东西
            sentences=re.sub(r'<.+>','',sentences)
            return sentences
 
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
                words=self.check_rules(1,words)
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
    
if __name__ == '__main__':
    data=data()
 