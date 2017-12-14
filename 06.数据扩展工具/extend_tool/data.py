#coding:utf-8
'''
Created on 2017年12月14日

@author: qiujiahao

@email:997018209@qq.com

'''
import os
import xlrd
import xlwt 
from collections import defaultdict
from extend_tool.conf import *
#文本扩充工具
class data(object):
    def __init__(self):
        self.quest_label=defaultdict(str)#问题和标签的对应关系
        self.label_quest=defaultdict(set)#标签和问题的对应关系
        self.quest_quests=defaultdict(set)#问题和对应的扩充后的句子
        self.quest_rules=defaultdict(list)#问题和对应的句式
        self.samewords=defaultdict(list)#同义词词表
        self.process_data()
        
    def process_data(self):
        #处理excel表单
        self.process_excels()
        
        #扩充数据
        self.extend_data()
        

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
        for quest,values in self.quest_quests.items():
            label = self.quest_label[quest]
            if(label==''):
                print('错误的原始quest:%s'%(quest))
            sh.write(i, 0, quest)
            quests=''
            for quest in values:
                quests+=quest+'&&&'
            #去除最后三个符号
            quests=quests[:-3]
            sh.write(i, 1, quests)
            sh.write(i, 3, label)
            sh.write(i, 6, 0)
            i+=1
        print('扩充后总共问题数量:%d'%(i))
        wb.save('data/data.xls')
                                           
    def process_excels(self):
        for file in filename.split(":"):
            self.process_one_excel(os.path.join(path,file))

    def process_one_excel(self,file):
        
        workbook = xlrd.open_workbook(file)
        sheet=workbook.sheet_by_name(sheetname)
        #第一列:问题
        quests = sheet.col_values(1)
        #第三列答案
        ans = sheet.col_values(3)
        if(len(quests)!=len(ans)):
            print('问题与答案数目不对')
            return
        
        for quest in quests:
            self.quest_label[quest.strip()]=ans[quests.index(quest)]   
            
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
        
if __name__=='__main__':
    data=data()    