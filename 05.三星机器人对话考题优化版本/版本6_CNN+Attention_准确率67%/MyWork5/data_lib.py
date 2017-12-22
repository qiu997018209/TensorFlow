#coding:utf-8
'''
Created on 2017年12月15日

@author: qiujiahao

@email:997018209@qq.com
曾使用过，后续可能继续使用的代码，放到lib文件里
'''
import xlrd
import re
from collections import defaultdict

class data_lib(object):
    def __init__(self):
        self.samewords=defaultdict(list)
        self.quest_quests=defaultdict(set)
        self.quest_ans={}
        
        
        #self.init(True)
        #self.extend_data()
    
    #数据预处理
    def init(self,bInit=True):
        if bInit==False:
            return 
        workbook = xlrd.open_workbook('data/三星机器人考题.xlsx')
        sheet=workbook.sheet_by_name('ask')
        #第一列
        quests = sheet.col_values(1)
        ans=sheet.col_values(2)
        with open('data/origin_train.txt','w',encoding='utf-8') as f:
            quests_info=dict(zip(quests[1:],ans[1:]))
            for key,value in quests_info.items():
                key,value = self.clean_str(key),self.clean_str(value)
                f.write('%s\t%s\n'%(key,value))
                self.quest_ans[key]=value
        
        sheet=workbook.sheet_by_name('Test')
        #第一列
        quests1 = sheet.col_values(1)[3:]
        quests2=sheet.col_values(2)[3:]

        with open('data/origin_test.txt','w',encoding='utf-8') as f:
            label=''
            for quest in quests1:
                if quest =='':
                    continue
                try:
                    label=self.quest_ans[self.clean_str(quest)]
                except:
                    print('init找不到quest')
                    continue
                key=quests2[quests1.index(quest)]
                f.write('%s\t%s\n'%(self.clean_str(key),label))   
                key=quests2[quests1.index(quest)+1]                 
                f.write('%s\t%s\n'%(self.clean_str(key),label)) 
    #数据扩展           
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
                        self.quest_quests[self.clean_str(key)].update(quests)                    
        num=0
        with open('data/train.txt','w',encoding='utf-8') as f:
            for quest in self.quest_quests:
                try:
                    f.write('%s\t%s\n'%(quest,self.quest_ans[quest]))
                except:
                    num+=1
                    print('extend_data异常',quest,num)
                    continue
                for q in self.quest_quests[quest]:
                    f.write('%s\t%s\n'%(self.clean_str(q),self.quest_ans[quest]))
 
 
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
    
    def clean_str(self,string):
        #去除空格,字母需要变为大写
        string=string.replace(' ','').strip().upper()
        return string   
    def get_top_k(self,score,k=10):
        result=[]
        score=list(score)
        num=-1
        for s in score:
            num+=1
            result.append((s,self.id_to_label[str(num)]))
        result=sorted(result,key=lambda x:x[0],reverse=True)
        #print("深度学习得分前10为:",result[0:k])
        return result[0:k]

        
if __name__=='__main__':
   lib=data_lib() 