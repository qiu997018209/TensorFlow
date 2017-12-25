#coding:utf-8
'''
Created on 2017年12月25日

@author: qiujiahao

@email:997018209@qq.com

'''
import xlrd
import numpy as np
from collections import defaultdict

#原始问题
def process_excel1():
    workbook = xlrd.open_workbook('data/知识管理.xls')
    sheet=workbook.sheet_by_name('问题答案')
    #第一列:问题
    quests = sheet.col_values(3)
    #第三列答案
    ans = sheet.col_values(4)
    if(len(quests)!=len(ans)):
        print('问题与答案数目不对')
    
    quest_ans=dict(zip(quests,ans))
    with open('data/origin_quest_label.txt','w',encoding='utf-8') as f:
        for quest,ans in quest_ans.items():
            f.write('%s\t%s\n'%(quest,ans))
            
#扩展问题            
def process_excel2(): 
    quest_label={}
    with open('data/origin_quest_label.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            quest,ans=line.strip().split('\t')
            quest_label[quest]=ans
    
    workbook = xlrd.open_workbook('data/知识管理.xls')
    sheet=workbook.sheet_by_name('关联问题')
    #第一列:问题
    quests = sheet.col_values(2)
    #第三列答案
    ans = sheet.col_values(3)
    if(len(quests)!=len(ans)):
        print('问题与答案数目不对')
    
    quest_ans=zip(quests,ans)
    writed_quest=set()
    with open('data/all_quest_label.txt','w',encoding='utf-8') as f:
        for quest,ans in quest_ans:
            try:
                label=quest_label[quest]
                if(quest not in writed_quest):
                    f.write('%s\t%s\n'%(quest,label))
                    writed_quest.add(quest) 
                f.write('%s\t%s\n'%(ans,label))
            except:
                print(quest)
#统计标签对应的问题数量           
def total_quest_label():
    label_quest=defaultdict(set)
    with open('data/all_quest_label.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            quest,ans=line.strip().split('\t')
            label_quest[ans].add(quest)
    with open('data/sentences.txt','w',encoding='utf-8') as f:
        for key,values in label_quest.items():
            if(len(values)>=10):
                continue
            for value in values:
                f.write('#%s\n\n'%(value))
#生成训练和测试数据集合
def get_train_test():
    label_quest=defaultdict(set)
    train_data=[]
    test_data=[]
    with open('data/all_quest_label.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            quest,ans=line.strip().split('\t')
            label_quest[ans].add(quest)    
        for label,values in label_quest.items():
            if len(values)<=10:
                for value in values:
                    train_data.append((value,label)) 
                continue
            values=list(values)
            np.random.shuffle(values)
            test_data.append((values[0],label))
            test_data.append((values[1],label))
            for value in values[2:]:
                train_data.append((value,label)) 
                            
    with open('data/train.txt','w',encoding='utf-8') as f:
        for w in train_data:
            f.write('%s\t%s\n'%(w[0],w[1]))
    with open('data/test.txt','w',encoding='utf-8') as f:
        for w in test_data:
            f.write('%s\t%s\n'%(w[0],w[1]))        
                    
get_train_test()        